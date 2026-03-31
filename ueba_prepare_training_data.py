#!/usr/bin/env python3
"""
ueba_prepare_training_data.py
─────────────────────────────
CyberSentinel UEBA — Training Data Preparation (Incremental)

HOW IT WORKS:
  - Tracks which .json.gz files have already been processed in .prepare_done
  - On each run, ONLY processes NEW files added to training_zips/
  - Appends new enriched events to the existing combined_training.jsonl
  - Re-runs preprocessor on the full combined file → fresh features.h5
  - OLD FILES ARE NEVER RE-PROCESSED — only the new day's file is enriched

DAILY USAGE:
  1. Drop new .json.gz into training_zips/
  2. Run:  python3 ueba_prepare_training_data.py
  3. Then: python3 ueba_trainer.py --config ueba_config.yaml

Resume support: if the script crashes mid-run, just re-run it.
Already-processed files are skipped automatically.
"""

import gzip
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

PIPELINE_DIR = Path("/root/NEW_DRIVE/CyberSentinel-Event-Correlation")
TRAINING_DIR = Path("training_zips")
OUTPUT       = Path("/data/ueba_training/combined_training.jsonl")
DONE_LOG     = Path(".prepare_done")
TMP_BASE     = Path("/dev/shm")

PREPROCESSOR = Path("ueba_preprocessor.py")
CONFIG_FILE  = Path("ueba_config.yaml")
FEATURES_H5  = Path("/data/ueba_training/features.h5")

GEOIP_DB = PIPELINE_DIR / "databases/GeoLite2-City.mmdb"
ASN_DB   = PIPELINE_DIR / "databases/GeoLite2-ASN.mmdb"
TOR_LIST = PIPELINE_DIR / "databases/tor-exit-nodes.txt"
REP_DB   = PIPELINE_DIR / "databases/malicious-ips.txt"

PYTHON = sys.executable
N_ENRICH_WORKERS = 40

# ── Logging ───────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/prepare.log"),
    ],
)
log = logging.getLogger("ueba.prepare")


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_stage(cmd: list) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Command failed:\n%s", result.stderr[-3000:])
        return False
    return True


def count_lines(path: Path) -> int:
    result = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    if result.returncode == 0:
        try:
            return int(result.stdout.split()[0])
        except (ValueError, IndexError):
            return 0
    return 0


def load_done() -> set:
    if DONE_LOG.exists():
        return set(line.strip() for line in DONE_LOG.read_text().splitlines() if line.strip())
    return set()


def mark_done(filename: str):
    with open(DONE_LOG, "a") as f:
        f.write(filename + "\n")


def shm_free_gb() -> float:
    st = os.statvfs("/dev/shm")
    return (st.f_bavail * st.f_frsize) / 1024**3


# ── Per-file processor ────────────────────────────────────────────────────────

def process_gz(gz_path: Path) -> int:
    """
    Process one .json.gz through L1 + L2 using /dev/shm as temp space.
    Appends enriched output to the combined file on disk.
    Returns number of enriched events appended, or -1 on failure.
    """
    log.info("─" * 55)
    log.info("Processing : %s  (%.0f MB compressed)",
             gz_path.name, gz_path.stat().st_size / 1024**2)
    log.info("  /dev/shm free: %.1f GB", shm_free_gb())
    t0 = time.time()

    tmp = Path(tempfile.mkdtemp(prefix=f"ueba_{gz_path.stem}_", dir=TMP_BASE))
    try:
        # ── Decompress → /dev/shm ────────────────────────────────────────────
        raw_json = tmp / "alerts.json"
        log.info("  [1/3] Decompressing to /dev/shm...")
        with gzip.open(gz_path, "rb") as fi, open(raw_json, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        raw_mb = raw_json.stat().st_size / 1024**2
        log.info("        Decompressed: %.0f MB  |  /dev/shm free: %.1f GB",
                 raw_mb, shm_free_gb())

        # ── L1: Normalize ─────────────────────────────────────────────────────
        normalized = tmp / "normalized.jsonl"
        state_file = tmp / "normalizer.state"
        log.info("  [2/3] Running L1 normalizer.py...")
        ok = run_stage([
            PYTHON,
            str(PIPELINE_DIR / "normalizer.py"),
            "--input",          str(raw_json),
            "--output",         str(normalized),
            "--state-file",     str(state_file),
            "--window-minutes", "999999",
        ])
        if not ok:
            raise RuntimeError("L1 normalizer failed")

        raw_json.unlink()
        n_norm = count_lines(normalized)
        log.info("        L1 done: %d events  |  /dev/shm free: %.1f GB",
                 n_norm, shm_free_gb())

        if n_norm == 0:
            raise RuntimeError("L1 produced 0 events — check alerts.json format")

        # ── L2: Enrich (parallel) ─────────────────────────────────────────────
        enriched = tmp / "enriched.jsonl"
        log.info("  [3/3] Running L2 enrich_parallel.py (%d workers)...", N_ENRICH_WORKERS)

        enrich_cmd = [
            PYTHON,
            str(Path(__file__).parent / "enrich_parallel.py"),
            "--input",   str(normalized),
            "--output",  str(enriched),
            "--workers", str(N_ENRICH_WORKERS),
        ]
        if GEOIP_DB.exists(): enrich_cmd += ["--geoip-db",     str(GEOIP_DB)]
        if ASN_DB.exists():   enrich_cmd += ["--asn-db",       str(ASN_DB)]
        if TOR_LIST.exists(): enrich_cmd += ["--tor-list",      str(TOR_LIST)]
        if REP_DB.exists():   enrich_cmd += ["--reputation-db", str(REP_DB)]

        ok = run_stage(enrich_cmd)
        if not ok:
            raise RuntimeError("L2 enricher failed")

        normalized.unlink()
        n_enrich = count_lines(enriched)
        log.info("        L2 done: %d events  |  /dev/shm free: %.1f GB",
                 n_enrich, shm_free_gb())

        # ── Append to combined file on disk ───────────────────────────────────
        # Always APPEND — never overwrite. Old days stay in the file.
        appended = 0
        with open(enriched, "r", encoding="utf-8", errors="replace") as src, \
             open(OUTPUT, "a", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if line:
                    dst.write(line + "\n")
                    appended += 1

        elapsed = time.time() - t0
        log.info("  Done: %d events appended | %.0f sec total", appended, elapsed)
        return appended

    except Exception as e:
        log.error("FAILED %s: %s", gz_path.name, e)
        return -1

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("UEBA Training Data Preparation — Incremental Mode")
    log.info("  Pipeline : %s", PIPELINE_DIR)
    log.info("  Temp dir : %s  (%.0f GB free)", TMP_BASE, shm_free_gb())
    log.info("  Output   : %s", OUTPUT)
    log.info("=" * 55)

    # Clean up any stale normalizer state files
    for stale in [Path("wazuh_normalizer.state"), Path(".wazuh_normalizer.state")]:
        if stale.exists():
            stale.unlink()
            log.info("Removed stale state file: %s", stale)

    # Clean up any leftover /dev/shm dirs from previous crashed runs
    for d in TMP_BASE.glob("ueba_*"):
        shutil.rmtree(d, ignore_errors=True)
        log.info("Cleaned up leftover temp dir: %s", d)

    # Ensure output directory exists
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Discover all training files
    gz_files = sorted(TRAINING_DIR.glob("*.json.gz"))
    if not gz_files:
        log.error("No .json.gz files found in %s", TRAINING_DIR)
        sys.exit(1)

    log.info("Found %d file(s) in training_zips/:", len(gz_files))
    for f in gz_files:
        log.info("  %-40s  %.0f MB", f.name, f.stat().st_size / 1024**2)

    # Load already-processed files
    done = load_done()

    # Determine which files are new
    new_files = [f for f in gz_files if f.name not in done]

    if not new_files:
        log.info("=" * 55)
        log.info("No new files to process.")
        log.info("All %d file(s) already processed:", len(done))
        for name in sorted(done):
            log.info("  ✓ %s", name)
        log.info("")
        log.info("To add more data: drop a new .json.gz into training_zips/")
        log.info("Then re-run this script.")
        log.info("=" * 55)

        # Even if no new files, re-run preprocessor + trainer if features.h5 is missing
        if not FEATURES_H5.exists() and OUTPUT.exists():
            log.info("features.h5 missing — re-running preprocessor on existing combined file...")
        else:
            log.info("Re-running preprocessor to refresh features.h5 with all accumulated data...")
    else:
        log.info("")
        log.info("Already processed (%d):", len(done))
        for name in sorted(done):
            log.info("  ✓ %s", name)
        log.info("")
        log.info("NEW files to process (%d):", len(new_files))
        for f in new_files:
            log.info("  → %s", f.name)
        log.info("")

        # Process only new files — append to existing combined_training.jsonl
        start_time   = time.time()
        total_events = 0
        failed       = []

        for i, gz_path in enumerate(new_files, 1):
            log.info("New file %d / %d", i, len(new_files))
            n = process_gz(gz_path)

            if n >= 0:
                total_events += n
                mark_done(gz_path.name)
                out_gb = OUTPUT.stat().st_size / 1024**3 if OUTPUT.exists() else 0
                log.info("Running total (this session): %d events | Combined file: %.2f GB",
                         total_events, out_gb)
            else:
                failed.append(gz_path.name)
                log.warning("Skipping failed file, continuing with next...")

        elapsed = time.time() - start_time
        log.info("\n" + "=" * 55)
        log.info("ENRICHMENT COMPLETE")
        log.info("  New files processed : %d / %d",
                 len(new_files) - len(failed), len(new_files))
        log.info("  New events added    : %d", total_events)
        log.info("  Time elapsed        : %.1f minutes", elapsed / 60)
        if OUTPUT.exists():
            total_lines = count_lines(OUTPUT)
            log.info("  Combined file total : %d events  (%.2f GB)",
                     total_lines, OUTPUT.stat().st_size / 1024**3)

        if failed:
            log.warning("  Failed files (%d):", len(failed))
            for f in failed:
                log.warning("    ✗ %s", f)
            log.warning("  Re-run this script to retry failed files.")
            sys.exit(1)

    # ── Re-run preprocessor on the FULL combined file ─────────────────────────
    # This always rebuilds features.h5 from scratch using ALL accumulated days.
    # Fast enough: preprocessor reads JSONL sequentially, no re-enrichment needed.
    if not OUTPUT.exists():
        log.error("combined_training.jsonl not found at %s", OUTPUT)
        log.error("Something went wrong — no data to preprocess.")
        sys.exit(1)

    total_lines = count_lines(OUTPUT)
    log.info("=" * 55)
    log.info("STAGE 2: Rebuilding features.h5 from full combined file")
    log.info("  Input  : %s", OUTPUT)
    log.info("  Events : %d  (all accumulated days)", total_lines)
    log.info("  Output : %s", FEATURES_H5)

    if not PREPROCESSOR.exists():
        log.error("Cannot find %s", PREPROCESSOR)
        log.error("Run manually:")
        log.error("  python3 ueba_preprocessor.py --input %s --config %s --output %s",
                  OUTPUT, CONFIG_FILE, FEATURES_H5)
        sys.exit(1)

    os.makedirs(str(FEATURES_H5.parent), exist_ok=True)

    # Delete old features.h5 before rebuilding (avoid stale data)
    if FEATURES_H5.exists():
        log.info("  Removing old features.h5 before rebuild...")
        FEATURES_H5.unlink()

    preproc_cmd = [
        PYTHON,
        str(PREPROCESSOR),
        "--input",  str(OUTPUT),
        "--output", str(FEATURES_H5),
        "--config", str(CONFIG_FILE),
    ]
    log.info("  Running: %s", " ".join(preproc_cmd))
    t_pp = time.time()
    ok = run_stage(preproc_cmd)
    pp_elapsed = time.time() - t_pp

    if not ok:
        log.error("Preprocessor FAILED!")
        log.error("combined_training.jsonl is preserved at: %s", OUTPUT)
        log.error("Run manually:")
        log.error("  python3 ueba_preprocessor.py --input %s --config %s --output %s",
                  OUTPUT, CONFIG_FILE, FEATURES_H5)
        sys.exit(1)

    log.info("  Preprocessor done in %.1f minutes", pp_elapsed / 60)
    if FEATURES_H5.exists():
        log.info("  features.h5 : %.2f GB → %s", FEATURES_H5.stat().st_size / 1024**3, FEATURES_H5)

    log.info("=" * 55)
    log.info("ALL DONE — Ready to train models")
    log.info("")

    # Print summary of all days processed so far
    all_done = load_done()
    log.info("  Total days in combined dataset : %d", len(all_done))
    for name in sorted(all_done):
        log.info("    ✓ %s", name)

    log.info("")
    log.info("  Next step:")
    log.info("    python3 ueba_trainer.py --config ueba_config.yaml")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
