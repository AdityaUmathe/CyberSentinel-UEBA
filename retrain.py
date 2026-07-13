#!/usr/bin/env python3
"""
retrain.py
──────────
CyberSentinel UEBA — Retrain with staging + validation gate.

USAGE:
    python3 retrain.py

WHAT IT DOES (safe flow — added 2026-07-13):
    Step 1 — Processes only NEW files in training_zips/ (skips already-done ones)
    Step 2 — Appends new enriched events to combined_training.jsonl
    Step 3 — Rebuilds features.h5 from ALL accumulated days
    Step 4 — Trains all models into a STAGING dir (models_staging/, vector_db_staging/)
             — the LIVE models/ is never touched during training.
    Step 5 — VALIDATION GATE: scores a sample of real recent events through the
             engine's serve path with the staged models. If the staged models
             would flood (train/serve skew, threshold collapse, provenance
             mismatch), the gate FAILS and we exit non-zero WITHOUT swapping —
             the live models/ stays exactly as it was, and the wrapper restarts
             the engine on the known-good models.
    Step 6 — Only on PASS: backup live models, then atomically promote staging
             into models/ + vector_db/.

WHY: the old flow deleted models/ then retrained in place with no check, so a
skewed retrain went straight live and pinned ~99% of events to highly_anomalous.
See ueba_validate_models.py and the 2026-07-13 incident notes.

IF IT CRASHES / VALIDATION FAILS:
    Live models/ is untouched. Re-run python3 retrain.py — already-processed
    training files are skipped. Investigate logs/retrain.log +
    logs/validate.log before re-enabling the weekly cron.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ── Config ────────────────────────────────────────────────────────────────────

PYTHON        = sys.executable
CONFIG        = Path("ueba_config.yaml")
STAGING_CONFIG = Path("ueba_config.staging.yaml")  # base dir — must NOT live inside a staged dir (would get promoted)
PREPARE       = Path("ueba_prepare_training_data.py")
TRAINER       = Path("ueba_trainer.py")
VALIDATOR     = Path("ueba_validate_models.py")
MODELS_DIR    = Path("models")
STAGING_DIR   = Path("models_staging")
VECTOR_DB     = Path("vector_db")
VECTOR_STAGING = Path("vector_db_staging")
FEATURES_H5   = Path("/data/ueba_training/features.h5")
TRAINING_DIR  = Path("training_zips")
DONE_LOG      = Path(".prepare_done")

# Validation gate thresholds
VAL_SAMPLE_N       = 30000
VAL_MAX_CRIT_RATE  = 0.15   # fail if staged models flag >15% as highly_anomalous
VAL_BASELINE_MULT  = 4.0    # ...or >4x the current live models' rate on same sample

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


def write_staging_config() -> None:
    """Copy ueba_config.yaml but repoint every model + FAISS output path into the
    staging dirs, so the trainer writes there and never touches live models/."""
    cfg = yaml.safe_load(CONFIG.read_text())
    p = cfg["paths"]
    p["models_dir"]        = "models_staging/"
    p["isolation_forest"]  = "models_staging/isolation_forest.pkl"
    p["scaler"]            = "models_staging/scaler.pkl"
    p["autoencoder_dir"]   = "models_staging/autoencoders/"
    p["global_autoencoder"] = "models_staging/autoencoders/global.pt"
    p["faiss_index"]       = "vector_db_staging/faiss.index"
    p["faiss_metadata"]    = "vector_db_staging/metadata.jsonl"
    STAGING_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log.info("Wrote staging config → %s", STAGING_CONFIG)


def clean_staging() -> None:
    for d in (STAGING_DIR, VECTOR_STAGING):
        if d.exists():
            shutil.rmtree(d)
    (STAGING_DIR / "autoencoders").mkdir(parents=True, exist_ok=True)
    VECTOR_STAGING.mkdir(parents=True, exist_ok=True)


def promote() -> None:
    """Backup live models, then atomically swap staging → live for BOTH the model
    dir and the FAISS vector_db (kept in lock-step so RAG matches the models)."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup current live models (the known-good set we are replacing).
    if MODELS_DIR.exists() and any(MODELS_DIR.iterdir()):
        backup = Path(f"models_backup_{ts}")
        log.info("Backing up live models → %s/", backup)
        shutil.copytree(MODELS_DIR, backup)
        # keep only 3 most recent backups
        for old in sorted(Path(".").glob("models_backup_*"), reverse=True)[3:]:
            shutil.rmtree(old, ignore_errors=True)
            log.info("Removed old backup: %s", old)

    # Atomic same-filesystem swaps. os.replace is atomic; the brief gap between
    # the two dirs is irrelevant because the engine is stopped by the wrapper
    # during retrain and only restarted after this returns.
    for live, staged in ((MODELS_DIR, STAGING_DIR), (VECTOR_DB, VECTOR_STAGING)):
        if not staged.exists():
            log.warning("staged dir %s missing — skipping its swap", staged)
            continue
        old_tmp = Path(f"{live}.old_swap_{ts}")
        if live.exists():
            os.replace(live, old_tmp)
        os.replace(staged, live)
        if old_tmp.exists():
            shutil.rmtree(old_tmp, ignore_errors=True)
        log.info("Promoted %s → %s", staged, live)


def main():
    t_start = time.time()

    log.info("=" * 60)
    log.info("UEBA RETRAIN (staging + validation gate)")
    log.info("=" * 60)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    for req in (CONFIG, PREPARE, TRAINER, VALIDATOR):
        if not req.exists():
            log.error("%s not found. Are you in the right directory?", req)
            sys.exit(1)

    gz_files   = sorted(TRAINING_DIR.glob("*.json.gz"))
    done_files = count_done()
    new_files  = [f for f in gz_files if f.name not in done_files]

    log.info("training_zips/ contents : %d file(s)", len(gz_files))
    log.info("Already processed       : %d file(s)", len(done_files))
    log.info("New files to process    : %d file(s)", len(new_files))

    if not gz_files:
        log.error("No .json.gz files found in training_zips/ — add data and re-run.")
        sys.exit(1)
    if not new_files:
        log.warning("No new files detected — retraining on existing data anyway.")
    for f in new_files:
        log.info("  NEW → %s  (%.0f MB)", f.name, f.stat().st_size / 1024**2)

    # ── Step 1/3: Data preparation (incremental) ──────────────────────────────
    log.info("")
    log.info("── STEP 1/3: Data Preparation ────────────────────────────")
    if not run([PYTHON, str(PREPARE)], "Data preparation"):
        log.error("Data preparation failed. See logs/prepare.log. Live models untouched.")
        sys.exit(1)

    # ── Step 2/3: Train into STAGING (live models/ untouched) ─────────────────
    log.info("")
    log.info("── STEP 2/3: Model Training → staging ─────────────────────")
    clean_staging()
    write_staging_config()
    if not run([PYTHON, str(TRAINER), "--config", str(STAGING_CONFIG)],
               "Model training (staging)"):
        log.error("Model training failed. See logs/ueba_trainer.log. Live models untouched.")
        sys.exit(1)

    # ── Step 3/3: VALIDATION GATE ─────────────────────────────────────────────
    log.info("")
    log.info("── STEP 3/3: Validation gate ──────────────────────────────")
    val_cmd = [
        PYTHON, str(VALIDATOR),
        "--candidate", str(STAGING_DIR),
        "--config", str(CONFIG),
        "--n", str(VAL_SAMPLE_N),
        "--max-critical-rate", str(VAL_MAX_CRIT_RATE),
        "--baseline", str(MODELS_DIR),
        "--baseline-multiple", str(VAL_BASELINE_MULT),
    ]
    log.info("Running: %s", " ".join(str(c) for c in val_cmd))
    with open("logs/validate.log", "a") as vlog:
        vlog.write(f"\n===== validation {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        vlog.flush()
        val = subprocess.run(val_cmd, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        vlog.write(val.stdout or "")
    for line in (val.stdout or "").splitlines():
        if line.startswith(("CANDIDATE", "BASELINE", "VALIDATION", "  scored",
                            "  highly", "  crit")):
            log.info("  [validate] %s", line.strip())

    if val.returncode != 0:
        log.error("=" * 60)
        log.error("VALIDATION GATE FAILED (exit %d) — staged models would flood.",
                  val.returncode)
        log.error("NOT promoting. Live models/ kept as-is; engine will restart on them.")
        log.error("Staged (rejected) models left in %s/ for inspection.", STAGING_DIR)
        log.error("=" * 60)
        sys.exit(1)

    # ── Promote (atomic swap) ─────────────────────────────────────────────────
    log.info("")
    log.info("── Validation PASSED — promoting staging → live ───────────")
    promote()

    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 60)
    log.info("RETRAIN COMPLETE (validated + promoted)")
    log.info("  Total time      : %.1f minutes", elapsed / 60)
    log.info("  Days in dataset : %d", len(count_done()))
    log.info("  Models promoted → %s/  (+ %s/)", MODELS_DIR, VECTOR_DB)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
