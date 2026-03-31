#!/usr/bin/env python3
"""
ueba_trainer.py
───────────────
CyberSentinel UEBA — Model Training

Reads the HDF5 feature matrix produced by ueba_preprocessor.py and trains:
  Layer 1 — Isolation Forest      (global, per-event anomaly scoring)
  Layer 2 — Autoencoder(s)        (per-user behavioral anomaly)
              └─ global.pt        (fallback for users with < 500 events)
              └─ {username}.pt    (personal model for each heavy user)
  Layer 3 — FAISS index           (for RAG semantic retrieval at runtime)

Also saves:
  scaler.pkl                      (StandardScaler, must be applied at inference)
  thresholds.json                 (per-user autoencoder reconstruction thresholds)

Run once offline on your historical data. All models are saved to models/.
At runtime, ueba_engine.py loads these saved models — no retraining needed.

Usage:
  python3 ueba_trainer.py \
      --features  data/features.h5 \
      --config    ueba_config.yaml

  # Skip layers you've already trained (e.g. resume after crash)
  python3 ueba_trainer.py --features data/features.h5 --skip-if L2
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import yaml

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not installed — skipping FAISS index build. "
                    "Install with: pip install faiss-gpu or faiss-cpu")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/ueba_trainer.log"),
    ],
)
log = logging.getLogger("ueba.trainer")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── HDF5 Dataset (PyTorch) ────────────────────────────────────────────────────

class HDF5Dataset(Dataset):
    """
    PyTorch Dataset that reads feature vectors from an HDF5 file.
    Preloads all data into RAM for fast GPU feeding.
    With 178 GB RAM available this is always safe.
    Falls back to per-row reads only if RAM is insufficient.
    """

    def __init__(self, h5_path: str, indices: np.ndarray | None = None,
                 scaler: StandardScaler | None = None):
        self.h5_path = h5_path
        self.scaler  = scaler

        with h5py.File(h5_path, "r") as f:
            total = f["features"].shape[0]
            feat_dim = f["features"].shape[1]
            if indices is not None:
                self.indices = indices
            else:
                self.indices = np.arange(total)

            n = len(self.indices)
            log.info("  Preloading %d rows into RAM (%.0f MB)...",
                     n, n * feat_dim * 4 / 1024**2)

            # ALWAYS read full dataset sequentially then filter in numpy.
            # HDF5 fancy indexing (random positions) on compressed files
            # is extremely slow regardless of subset size.
            # Reading 1.7 GB sequentially takes ~3s; numpy indexing is instant.
            full = f["features"][:].astype(np.float32)
            if indices is None:
                data = full
            else:
                data = full[self.indices]
                del full

            log.info("  Preload complete: %.0f MB in RAM", data.nbytes / 1024**2)

        if scaler is not None:
            data = scaler.transform(data).astype(np.float32)

        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Autoencoder Architecture ──────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Encoder-decoder network for learning normal behavioral patterns.

    Architecture (default, feature_dim=35):
        Encoder: 35 → 24 → 12
        Decoder: 12 → 24 → 35

    High reconstruction error = event doesn't match learned normal behavior.

    The dropout layer prevents overfitting — forces the network to learn
    robust representations rather than memorizing exact values.
    """

    def __init__(self, input_dim: int = 35,
                 hidden_dims: list[int] | None = None,
                 dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [24, 12]

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x)
            # Mean across feature dimensions → one error value per sample
            return torch.mean((x - recon) ** 2, dim=1)


# ── Autoencoder Trainer ───────────────────────────────────────────────────────

class AutoencoderTrainer:
    """Trains a single autoencoder and computes its anomaly threshold."""

    def __init__(self, config: dict, device: torch.device):
        self.config = config["autoencoder"]
        self.device = device

    def train(self, dataset: HDF5Dataset,
              model_name: str = "global") -> tuple[Autoencoder, float]:
        """
        Train an autoencoder on the given dataset.
        Returns (trained_model, threshold).
        threshold = Nth percentile of training reconstruction errors.
        """
        arch_cfg  = self.config["architecture"]
        train_cfg = self.config["training"]

        # Train/val split — split the already-loaded tensor directly
        # No re-reading from HDF5, no index masking, instant
        n = len(dataset)
        val_size   = int(n * train_cfg["val_split"])
        train_size = n - val_size

        perm       = torch.randperm(n)
        train_data = dataset.data[perm[:train_size]]
        val_data   = dataset.data[perm[train_size:]]

        train_loader = DataLoader(
            torch.utils.data.TensorDataset(train_data),
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(val_data),
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # Model
        model = Autoencoder(
            input_dim=arch_cfg["input_dim"],
            hidden_dims=arch_cfg["hidden_dims"],
            dropout=arch_cfg.get("dropout", 0.1),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg["scheduler_step"],
            gamma=train_cfg["scheduler_gamma"],
        )
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        log.info("[%s] Training autoencoder | samples: %d | device: %s",
                 model_name, train_size, self.device)

        for epoch in range(1, train_cfg["epochs"] + 1):
            # Train
            model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch)
            train_loss /= train_size

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    output = model(batch)
                    val_loss += criterion(output, batch).item() * len(batch)
            val_loss /= val_size

            scheduler.step()

            if epoch % 10 == 0 or epoch == 1:
                log.info("[%s] Epoch %d/%d | train_loss: %.6f | val_loss: %.6f",
                         model_name, epoch, train_cfg["epochs"],
                         train_loss, val_loss)

            # Early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= train_cfg["early_stopping_patience"]:
                    log.info("[%s] Early stopping at epoch %d", model_name, epoch)
                    break

        # Restore best weights
        if best_state:
            model.load_state_dict(best_state)

        # Compute anomaly threshold on training data (use in-memory tensor)
        threshold = self._compute_threshold_tensor(model, train_data)
        log.info("[%s] Threshold (p%d): %.6f",
                 model_name,
                 self.config["threshold_percentile"],
                 threshold)

        return model, threshold

    def _compute_threshold_tensor(self, model: Autoencoder,
                                  data: torch.Tensor) -> float:
        """Compute threshold directly from an in-memory tensor."""
        model.eval()
        loader = DataLoader(
            torch.utils.data.TensorDataset(data),
            batch_size=2048, shuffle=False, num_workers=0
        )
        all_errors = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                errors = model.reconstruction_error(batch)
                all_errors.append(errors.cpu().numpy())
        all_errors = np.concatenate(all_errors)
        percentile = self.config["threshold_percentile"]
        return float(np.percentile(all_errors, percentile))

    def _compute_threshold(self, model: Autoencoder,
                           dataset: HDF5Dataset) -> float:
        """
        Compute the reconstruction error threshold using in-memory tensor.
        Events above this threshold at inference = behavioral anomaly.
        """
        model.eval()
        loader = DataLoader(
            torch.utils.data.TensorDataset(dataset.data),
            batch_size=2048, shuffle=False, num_workers=0
        )
        all_errors = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                errors = model.reconstruction_error(batch)
                all_errors.append(errors.cpu().numpy())

        all_errors = np.concatenate(all_errors)
        percentile = self.config["threshold_percentile"]
        return float(np.percentile(all_errors, percentile))


# ── Main Trainer ──────────────────────────────────────────────────────────────

class UEBATrainer:

    def __init__(self, config: dict):
        self.config = config

        # Paths
        self.models_dir       = Path(config["paths"]["models_dir"])
        self.autoencoder_dir  = Path(config["paths"]["autoencoder_dir"])
        self.if_path          = Path(config["paths"]["isolation_forest"])
        self.scaler_path      = Path(config["paths"]["scaler"])
        self.faiss_index_path = Path(config["paths"]["faiss_index"])
        self.faiss_meta_path  = Path(config["paths"]["faiss_metadata"])
        self.profile_db       = Path(config["paths"]["profile_db"])

        # Create directories
        for d in [self.models_dir, self.autoencoder_dir,
                  self.faiss_index_path.parent, Path("logs")]:
            d.mkdir(parents=True, exist_ok=True)

        # Device
        ae_cfg = config["autoencoder"]
        requested = ae_cfg.get("device", "cuda")
        if requested == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            log.info("GPU detected: %s", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            if requested == "cuda":
                log.warning("CUDA requested but not available — using CPU")
            log.info("Using device: CPU")

        # Thresholds store (saved to disk after training)
        self.thresholds: dict[str, float] = {}

    # ── Step 1: Fit StandardScaler ────────────────────────────────────────────

    def fit_scaler(self, h5_path: str) -> StandardScaler:
        """
        Fit StandardScaler on a sample of features.
        Reads sequentially for 10-20x faster HDF5 I/O on compressed files.
        Sequential reads are fine for scaling statistics.
        """
        sample_size = min(
            self.config["isolation_forest"]["training_sample_size"],
            500_000,
        )
        log.info("Fitting StandardScaler on %d rows (sequential read)...", sample_size)

        with h5py.File(h5_path, "r") as f:
            total = f["features"].shape[0]
            actual = min(sample_size, total)
            stride = max(1, total // actual)
            X_sample = f["features"][::stride].astype(np.float32)

        log.info("  Read %d rows (stride=%d, covers all 17 days)", len(X_sample), stride)
        scaler = StandardScaler()
        scaler.fit(X_sample)
        del X_sample

        joblib.dump(scaler, str(self.scaler_path))
        log.info("Scaler saved → %s", self.scaler_path)
        return scaler

    # ── Step 2: Train Isolation Forest ────────────────────────────────────────

    def train_isolation_forest(self, h5_path: str, scaler: StandardScaler):
        """
        Train global Isolation Forest on a representative sample.
        1M samples is statistically sufficient for anomaly boundary learning.
        """
        if_cfg = self.config["isolation_forest"]
        sample_size = if_cfg["training_sample_size"]

        log.info("Loading %d samples for Isolation Forest training (stride read)...", sample_size)

        with h5py.File(h5_path, "r") as f:
            total = f["features"].shape[0]
            actual_sample = min(sample_size, total)
            stride = max(1, total // actual_sample)
            X = f["features"][::stride].astype(np.float32)

        log.info("  Read %d rows (stride=%d, covers all 17 days)", len(X), stride)

        # Scale features
        X_scaled = scaler.transform(X)
        del X   # free RAM immediately

        log.info("Training Isolation Forest | samples: %d | estimators: %d",
                 actual_sample, if_cfg["n_estimators"])
        t0 = time.time()

        model = IsolationForest(
            n_estimators=if_cfg["n_estimators"],
            contamination=if_cfg["contamination"],
            max_samples=if_cfg["max_samples"],
            random_state=if_cfg["random_state"],
            n_jobs=if_cfg["n_jobs"],
        )
        model.fit(X_scaled)

        elapsed = time.time() - t0
        log.info("Isolation Forest trained in %.1f minutes", elapsed / 60)

        joblib.dump(model, str(self.if_path))
        log.info("Isolation Forest saved → %s", self.if_path)

        # Quick sanity check — what % of training data is flagged?
        labels = model.predict(X_scaled)
        anomaly_rate = (labels == -1).sum() / len(labels) * 100
        log.info("Training anomaly rate: %.2f%% (target: %.1f%%)",
                 anomaly_rate, if_cfg["contamination"] * 100)

        del X_scaled

    # ── Step 3: Train Autoencoders ────────────────────────────────────────────

    def train_autoencoders(self, h5_path: str, scaler: StandardScaler):
        """
        Train per-user autoencoders for users with enough data,
        plus a global autoencoder as fallback.
        """
        import sqlite3

        min_events = self.config["preprocessing"]["min_events_per_user"]
        ae_trainer = AutoencoderTrainer(self.config, self.device)

        # ── Global autoencoder (all data) ─────────────────────────────────────
        log.info("Training global autoencoder...")
        global_ds = HDF5Dataset(h5_path, scaler=scaler)
        global_model, global_threshold = ae_trainer.train(global_ds, "global")

        global_path = self.autoencoder_dir / "global.pt"
        torch.save(global_model.state_dict(), str(global_path))
        self.thresholds["global"] = global_threshold
        log.info("Global autoencoder saved → %s", global_path)

        # ── Per-user autoencoders ─────────────────────────────────────────────
        # Get list of users with enough events from profile store
        db_path = str(self.profile_db)
        if not Path(db_path).exists():
            log.warning("Profile store not found at %s — skipping per-user models",
                        db_path)
            self._save_thresholds()
            return

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT username, total_events FROM users WHERE total_events >= ?",
            (min_events,)
        ).fetchall()
        conn.close()

        eligible_users = [(r[0], r[1]) for r in rows]
        log.info("Users eligible for personal autoencoder: %d (>= %d events)",
                 len(eligible_users), min_events)

        if not eligible_users:
            log.info("No users meet the threshold — only global model will be used")
            self._save_thresholds()
            return

        # Load user → row mapping from HDF5 metadata
        log.info("Building user → row index map from HDF5 metadata...")
        user_indices: dict[str, list[int]] = {}

        with h5py.File(h5_path, "r") as f:
            meta_ds = f["metadata"]
            chunk = 100_000
            total = len(meta_ds)
            for start in range(0, total, chunk):
                end = min(start + chunk, total)
                batch_meta = meta_ds[start:end]
                for local_i, m_str in enumerate(batch_meta):
                    try:
                        m = json.loads(m_str)
                        user = m.get("user", "unknown")
                        if user not in user_indices:
                            user_indices[user] = []
                        user_indices[user].append(start + local_i)
                    except Exception:
                        pass

                if start % 1_000_000 == 0:
                    log.info("  Indexed %d / %d rows...", start, total)

        log.info("Unique users in dataset: %d", len(user_indices))

        # Train per-user models
        eligible_set = {u for u, _ in eligible_users}
        trained = 0
        for user, event_count in eligible_users:
            if user not in user_indices:
                log.debug("User %s in profile store but not in features — skipping",
                          user)
                continue

            # Resume support — skip if model already saved
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in user)
            model_path = self.autoencoder_dir / f"{safe_name}.pt"
            if model_path.exists():
                log.info("Skipping user '%s' — model already exists at %s", user, model_path)
                trained += 1
                continue

            indices = np.array(user_indices[user])
            log.info("Training autoencoder for user '%s' | events: %d",
                     user, len(indices))

            user_ds = HDF5Dataset(h5_path, indices=indices, scaler=scaler)
            user_model, user_threshold = ae_trainer.train(user_ds, user)

            # Save model — sanitize username for filename
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_"
                                for c in user)
            model_path = self.autoencoder_dir / f"{safe_name}.pt"
            torch.save(user_model.state_dict(), str(model_path))
            self.thresholds[user] = user_threshold
            trained += 1
            log.info("Saved user model → %s (threshold: %.6f)",
                     model_path, user_threshold)

            # Free GPU memory between users
            del user_model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        log.info("Per-user autoencoders trained: %d", trained)
        self._save_thresholds()

    def _save_thresholds(self):
        """Save all autoencoder thresholds to JSON."""
        path = self.models_dir / "thresholds.json"
        with open(path, "w") as f:
            json.dump(self.thresholds, f, indent=2)
        log.info("Thresholds saved → %s", path)

    # ── Step 4: Build FAISS Index ─────────────────────────────────────────────

    def build_faiss_index(self, h5_path: str, scaler: StandardScaler):
        """
        Build FAISS vector index from all training feature vectors.
        Used at runtime for RAG — retrieve similar past anomalies.
        At inference time: embed new anomaly → search index → top-K similar.
        """
        if not FAISS_AVAILABLE:
            log.warning("FAISS not available — skipping index build")
            return

        feature_dim = self.config["preprocessing"]["feature_dim"]
        max_size    = self.config["rag"]["max_index_size"]

        log.info("Building FAISS index | max_size: %d", max_size)

        with h5py.File(h5_path, "r") as f:
            total = f["features"].shape[0]
            actual = min(max_size, total)
            stride = max(1, total // actual)
            X = f["features"][::stride].astype(np.float32)
            stride_indices = list(range(0, total, stride))[:actual]
            metas = [json.loads(f["metadata"][i]) for i in stride_indices]

        log.info("  Read %d rows for FAISS (stride=%d)", len(X), stride)

        X_scaled = scaler.transform(X).astype(np.float32)
        del X

        # L2-normalize for cosine similarity (inner product search)
        faiss.normalize_L2(X_scaled)

        # IndexFlatIP = exact inner product search (cosine after L2 norm)
        index = faiss.IndexFlatIP(feature_dim)
        index.add(X_scaled)

        faiss.write_index(index, str(self.faiss_index_path))
        log.info("FAISS index saved → %s | vectors: %d",
                 self.faiss_index_path, index.ntotal)

        # Save metadata (needed to return human-readable similar events)
        with open(self.faiss_meta_path, "w") as f:
            for m in metas:
                f.write(json.dumps(m) + "\n")
        log.info("FAISS metadata saved → %s", self.faiss_meta_path)

        del X_scaled

    # ── Main Training Pipeline ────────────────────────────────────────────────

    def run(self, h5_path: str, skip_layers: list[str] | None = None):
        """Run the full training pipeline."""
        skip = set(skip_layers or [])
        start = time.time()

        log.info("=" * 60)
        log.info("UEBA TRAINER STARTING")
        log.info("  Features : %s", h5_path)
        log.info("  Device   : %s", self.device)
        log.info("=" * 60)

        # Verify HDF5 exists
        if not Path(h5_path).exists():
            log.error("Features file not found: %s", h5_path)
            log.error("Run ueba_preprocessor.py first.")
            sys.exit(1)

        with h5py.File(h5_path, "r") as f:
            total_rows = f["features"].shape[0]
            feature_dim = f["features"].shape[1]
        log.info("Feature matrix: %d rows × %d features", total_rows, feature_dim)

        # Step 1: Scaler (always needed)
        if "scaler" not in skip:
            scaler = self.fit_scaler(h5_path)
        else:
            log.info("Loading existing scaler...")
            scaler = joblib.load(str(self.scaler_path))

        # Step 2: Isolation Forest
        if "L1" not in skip:
            log.info("\n── Layer 1: Isolation Forest ─────────────────────────")
            self.train_isolation_forest(h5_path, scaler)
        else:
            log.info("Skipping L1 (Isolation Forest)")

        # Step 3: Autoencoders
        if "L2" not in skip:
            log.info("\n── Layer 2: Autoencoders ─────────────────────────────")
            self.train_autoencoders(h5_path, scaler)
        else:
            log.info("Skipping L2 (Autoencoders)")

        # Step 4: FAISS index
        if "L3" not in skip:
            log.info("\n── Layer 3: FAISS Index (RAG) ────────────────────────")
            self.build_faiss_index(h5_path, scaler)
        else:
            log.info("Skipping L3 (FAISS Index)")

        elapsed = time.time() - start
        log.info("=" * 60)
        log.info("TRAINING COMPLETE")
        log.info("  Total time : %.1f minutes", elapsed / 60)
        log.info("  Models dir : %s", self.models_dir)
        log.info("")
        log.info("  Files saved:")
        for p in sorted(Path(self.models_dir).rglob("*")):
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                log.info("    %-50s  %.1f MB", str(p), size_mb)
        log.info("")
        log.info("  Next step:")
        log.info("    python3 ueba_engine.py --config ueba_config.yaml")
        log.info("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UEBA Model Trainer — trains Isolation Forest, "
                    "Autoencoders, and FAISS index",
    )
    parser.add_argument("--features", default=None,
                        help="HDF5 feature matrix (default from config)")
    parser.add_argument("--config", default="ueba_config.yaml")
    parser.add_argument("--skip-if", nargs="+",
                        choices=["scaler", "L1", "L2", "L3"],
                        help="Skip specific layers (use to resume after crash)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    config = load_config(args.config)

    h5_path = args.features or config["paths"]["features_h5"]

    trainer = UEBATrainer(config)
    trainer.run(h5_path, skip_layers=args.skip_if)


if __name__ == "__main__":
    main()
