#!/usr/bin/env python3
"""
retrain.py
──────────
CyberSentinel UEBA — Daily Manual Retrain

USAGE (every day after dropping a new .json.gz into training_zips/):
    python3 retrain.py

WHAT IT DOES:
    Step 1 — Processes only NEW files in training_zips/ (skips already-done ones)
    Step 2 — Appends new enriched events to combined_training.jsonl
    Step 3 — Rebuilds features.h5 from ALL accumulated days
    Step 4 — Backs up old models
    Step 5 — Retrains all models (Isolation Forest + Autoencoders + FAISS)

IF IT CRASHES:
    Just re-run python3 retrain.py — it picks up where it left off.
"""

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PYTHON       = sys.executable
CONFIG       = Path("ueba_config.yaml")
PREPARE      = Path("ueba_prepare_training_data.py")
TRAINER      = Path("ueba_trainer.py")
MODELS_DIR   = Path("models")
FEATURES_H5  = Path("/data/ueba_training/features.h5")
TRAINING_DIR = Path("training_zips")
DONE_LOG     = Path(".prepare_done")

# ── Logging ───────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/retrain.log"),
    ],
)
log = logging.getLogger("ueba.retrain")


def run(cmd: list, label: str) -> bool:
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        log.error("%s FAILED (exit code %d)", label, result.returncode)
        return False
    log.info("%s completed successfully", label)
    return True


def count_done() -> list:
    if not DONE_LOG.exists():
        return []
    return [l.strip() for l in DONE_LOG.read_text().splitlines() if l.strip()]


def main():
    t_start = time.time()

    log.info("=" * 60)
    log.info("UEBA DAILY RETRAIN")
    log.info("=" * 60)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not CONFIG.exists():
        log.error("ueba_config.yaml not found. Are you in the right directory?")
        sys.exit(1)
    if not PREPARE.exists():
        log.error("ueba_prepare_training_data.py not found.")
        sys.exit(1)
    if not TRAINER.exists():
        log.error("ueba_trainer.py not found.")
        sys.exit(1)

    # Show what's in training_zips
    gz_files = sorted(TRAINING_DIR.glob("*.json.gz"))
    done_files = count_done()
    new_files = [f for f in gz_files if f.name not in done_files]

    log.info("training_zips/ contents : %d file(s)", len(gz_files))
    log.info("Already processed       : %d file(s)", len(done_files))
    log.info("New files to process    : %d file(s)", len(new_files))

    if not gz_files:
        log.error("No .json.gz files found in training_zips/")
        log.error("Add your log file and re-run.")
        sys.exit(1)

    if not new_files:
        log.warning("No new files detected in training_zips/")
        log.warning("If you added a new file, check its name matches *.json.gz")
        log.warning("Continuing — will still retrain models on existing data...")

    for f in new_files:
        log.info("  NEW → %s  (%.0f MB)", f.name, f.stat().st_size / 1024**2)

    # ── Step 1: Backup existing models ────────────────────────────────────────
    if MODELS_DIR.exists() and any(MODELS_DIR.iterdir()):
        from datetime import datetime
        backup_name = f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = Path(backup_name)
        log.info("Backing up existing models → %s/", backup_name)
        shutil.copytree(MODELS_DIR, backup_path)
        log.info("Backup complete")

        # Keep only the 3 most recent backups to save disk space
        backups = sorted(Path(".").glob("models_backup_*"), reverse=True)
        for old_backup in backups[3:]:
            shutil.rmtree(old_backup)
            log.info("Removed old backup: %s", old_backup)
    else:
        log.info("No existing models to backup (first run)")

    # ── Step 2: Data preparation (incremental — only new files processed) ─────
    log.info("")
    log.info("── STEP 1/2: Data Preparation ────────────────────────────")
    ok = run([PYTHON, str(PREPARE)], "Data preparation")
    if not ok:
        log.error("Data preparation failed. Check logs/prepare.log for details.")
        log.error("Re-run retrain.py to retry — already-processed files will be skipped.")
        sys.exit(1)

    # ── Step 3: Clear old models and retrain ──────────────────────────────────
    log.info("")
    log.info("── STEP 2/2: Model Training ───────────────────────────────")

    # Clear old models so trainer starts fresh
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.pkl"):
            f.unlink()
        ae_dir = MODELS_DIR / "autoencoders"
        if ae_dir.exists():
            shutil.rmtree(ae_dir)
        ae_dir.mkdir(parents=True, exist_ok=True)
        log.info("Cleared old model files")

    ok = run([PYTHON, str(TRAINER), "--config", str(CONFIG)], "Model training")
    if not ok:
        log.error("Model training failed. Check logs/ueba_trainer.log for details.")
        log.error("You can retry just the training step with:")
        log.error("  python3 ueba_trainer.py --config ueba_config.yaml")
        sys.exit(1)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    all_done = count_done()

    log.info("")
    log.info("=" * 60)
    log.info("RETRAIN COMPLETE")
    log.info("  Total time      : %.1f minutes", elapsed / 60)
    log.info("  Days in dataset : %d", len(all_done))
    for name in sorted(all_done):
        log.info("    ✓ %s", name)
    log.info("")
    log.info("  Models saved to : %s/", MODELS_DIR)
    log.info("  Features HDF5   : %s", FEATURES_H5)
    log.info("")
    log.info("  To run the UEBA engine:")
    log.info("    python3 ueba_engine.py --config ueba_config.yaml")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
