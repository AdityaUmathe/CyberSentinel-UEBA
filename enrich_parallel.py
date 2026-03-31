#!/usr/bin/env python3
"""
enrich_parallel.py
──────────────────
Drop-in parallel replacement for enrich_json.py batch mode.

Splits the input JSONL into N equal chunks, runs N parallel workers
(each an independent LogEnricher instance), then merges the outputs
in order. Rolling-counter state is NOT shared across workers — events
at chunk boundaries will have slightly inaccurate 5-minute counters,
which is acceptable for UEBA model training.

Usage (same flags as enrich_json.py):
    python3 enrich_parallel.py \
        --input  normalized.jsonl \
        --output enriched.jsonl  \
        --workers 40             \
        --geoip-db  databases/GeoLite2-City.mmdb \
        --asn-db    databases/GeoLite2-ASN.mmdb  \
        --tor-list  databases/tor-exit-nodes.txt \
        --reputation-db databases/malicious-ips.txt

If --workers is not given, defaults to min(cpu_count, 40).
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def count_lines(path: Path) -> int:
    result = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True)
    if result.returncode == 0:
        try:
            return int(result.stdout.split()[0])
        except (ValueError, IndexError):
            return 0
    return 0


def split_file(input_path: Path, n_chunks: int, tmp_dir: Path) -> list[Path]:
    """Split input JSONL into n_chunks roughly equal parts. Returns list of chunk paths."""
    total = count_lines(input_path)
    if total == 0:
        return []

    chunk_size = max(1, (total + n_chunks - 1) // n_chunks)
    actual_chunks = (total + chunk_size - 1) // chunk_size

    print(f"  Splitting {total:,} lines into {actual_chunks} chunks of ~{chunk_size:,} lines each...",
          file=sys.stderr)

    chunk_paths = []
    chunk_idx = 0
    lines_written = 0
    current_file = None
    current_path = None

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if lines_written == 0:
                current_path = tmp_dir / f"chunk_{chunk_idx:04d}.jsonl"
                current_file = open(current_path, "w", encoding="utf-8")
                chunk_paths.append(current_path)

            current_file.write(line)
            lines_written += 1

            if lines_written >= chunk_size:
                current_file.close()
                chunk_idx += 1
                lines_written = 0
                current_file = None

    if current_file and not current_file.closed:
        current_file.close()

    print(f"  Split complete: {len(chunk_paths)} chunks", file=sys.stderr)
    return chunk_paths


def run_worker(args_tuple):
    """Run enrich_json.py on one chunk. Returns (chunk_idx, output_path, success, n_lines)."""
    chunk_idx, chunk_path, output_path, enrich_script, python_exe, extra_args = args_tuple

    cmd = [
        python_exe,
        enrich_script,
        "--input",  str(chunk_path),
        "--output", str(output_path),
    ] + extra_args

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n  [Worker {chunk_idx}] FAILED:\n{result.stderr[-1000:]}", file=sys.stderr)
        return (chunk_idx, output_path, False, 0)

    n = count_lines(output_path)
    return (chunk_idx, output_path, True, n)


def merge_chunks(chunk_output_paths: list[Path], output_path: Path) -> int:
    """Merge chunk outputs in order into final output file."""
    total = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for chunk_path in chunk_output_paths:
            if chunk_path.exists():
                with open(chunk_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            out.write(line + "\n")
                            total += 1
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Parallel drop-in replacement for enrich_json.py batch mode"
    )
    parser.add_argument("--input",  required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: min(nproc, 40))")
    # Pass-through flags to enrich_json.py
    parser.add_argument("--geoip-db",      default=None)
    parser.add_argument("--asn-db",        default=None)
    parser.add_argument("--tor-list",      default=None)
    parser.add_argument("--reputation-db", default=None)
    parser.add_argument("--overwrite",     action="store_true")
    parser.add_argument("--counter-window-seconds", type=int, default=300)

    args = parser.parse_args()

    # Resolve paths
    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Find enrich_json.py (same directory as this script, or use explicit path)
    script_dir   = Path(__file__).parent
    enrich_script = script_dir / "enrich_json.py"
    if not enrich_script.exists():
        # Try the known pipeline location
        enrich_script = Path("/root/NEW_DRIVE/CyberSentinel-Event-Correlation/enrich_json.py")
    if not enrich_script.exists():
        print(f"✗ Cannot find enrich_json.py", file=sys.stderr)
        sys.exit(1)

    python_exe = sys.executable

    # Worker count
    n_workers = args.workers
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 4, 40)
    print(f"  Using {n_workers} parallel workers", file=sys.stderr)

    # Build pass-through args list
    extra_args = []
    if args.geoip_db:            extra_args += ["--geoip-db",              args.geoip_db]
    if args.asn_db:              extra_args += ["--asn-db",                args.asn_db]
    if args.tor_list:            extra_args += ["--tor-list",               args.tor_list]
    if args.reputation_db:       extra_args += ["--reputation-db",         args.reputation_db]
    if args.overwrite:           extra_args += ["--overwrite"]
    if args.counter_window_seconds != 300:
        extra_args += ["--counter-window-seconds", str(args.counter_window_seconds)]

    t0 = time.time()

    # Use /dev/shm for temp chunk files if available, else system tmp
    tmp_base = Path("/dev/shm") if Path("/dev/shm").exists() else Path(tempfile.gettempdir())
    tmp_dir  = Path(tempfile.mkdtemp(prefix="enrich_parallel_", dir=tmp_base))

    try:
        # Step 1: Split input
        chunk_paths = split_file(input_path, n_workers, tmp_dir)
        if not chunk_paths:
            print("✗ Input file is empty", file=sys.stderr)
            sys.exit(1)

        actual_workers = len(chunk_paths)

        # Step 2: Build worker args
        worker_args = []
        output_chunk_paths = []
        for i, chunk_path in enumerate(chunk_paths):
            out_path = tmp_dir / f"enriched_{i:04d}.jsonl"
            output_chunk_paths.append(out_path)
            worker_args.append((i, chunk_path, out_path, str(enrich_script), python_exe, extra_args))

        # Step 3: Run workers in parallel using multiprocessing.Pool
        print(f"  Launching {actual_workers} enrichment workers...", file=sys.stderr)

        from multiprocessing import Pool

        completed = 0
        failed    = 0
        total_enriched = 0

        with Pool(processes=actual_workers) as pool:
            for chunk_idx, out_path, success, n_lines in pool.imap_unordered(run_worker, worker_args):
                completed += 1
                if success:
                    total_enriched += n_lines
                    elapsed = time.time() - t0
                    rate = total_enriched / elapsed if elapsed > 0 else 0
                    print(
                        f"  [{completed:3d}/{actual_workers}] chunk {chunk_idx:04d} done "
                        f"({n_lines:,} events) | total so far: {total_enriched:,} | "
                        f"{rate:,.0f} ev/s",
                        file=sys.stderr
                    )
                else:
                    failed += 1
                    print(f"  [{completed:3d}/{actual_workers}] chunk {chunk_idx:04d} FAILED", file=sys.stderr)

        if failed > 0:
            print(f"\n✗ {failed} worker(s) failed. Aborting.", file=sys.stderr)
            sys.exit(1)

        # Step 4: Merge in order
        print(f"\n  Merging {actual_workers} chunks into {output_path}...", file=sys.stderr)
        final_count = merge_chunks(output_chunk_paths, output_path)

        elapsed = time.time() - t0
        rate    = final_count / elapsed if elapsed > 0 else 0

        print(f"\n✓ Parallel enrichment complete:", file=sys.stderr)
        print(f"  Events  : {final_count:,}", file=sys.stderr)
        print(f"  Workers : {actual_workers}", file=sys.stderr)
        print(f"  Time    : {elapsed:.0f}s ({elapsed/60:.1f} min)", file=sys.stderr)
        print(f"  Rate    : {rate:,.0f} events/sec", file=sys.stderr)
        print(f"  Output  : {output_path}", file=sys.stderr)

    finally:
        # Clean up chunk files from /dev/shm
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
