#!/bin/bash
# ueba_rotate.sh — daily rotation of ueba_alerts.jsonl
#
# Runs from cron at 00:05 daily on the GPU server (98). Archives the
# previous day's alerts to /root/NEW_DRIVE/aditya_ueba/archive/ as a
# date-stamped zip, leaves the engine's output file empty so the
# dashboard "flushes" to a fresh state for the new day, and restarts
# the engine so it reopens the new file.
#
# Manual invocation is safe — running mid-day will rotate whatever's
# currently in ueba_alerts.jsonl and label the archive with "yesterday".
# Override the label by exporting ROTATE_DATE=YYYY-MM-DD before running.

set -euo pipefail

BASE=/root/NEW_DRIVE/aditya_ueba
ALERTS=$BASE/ueba_alerts.jsonl
ARCHIVE_DIR=$BASE/archive
LOG=$BASE/logs/rotate.log
VENV_PY=/data/aditya_ueba/venv/bin/python3

ROTATE_DATE=${ROTATE_DATE:-$(date -d "yesterday" +%Y-%m-%d)}

mkdir -p "$ARCHIVE_DIR" "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo "==================================================================="
echo "$(date '+%Y-%m-%d %H:%M:%S')  rotation start (archive label: ${ROTATE_DATE})"

if [ ! -s "$ALERTS" ]; then
  echo "  alerts file empty or missing — nothing to rotate"
  exit 0
fi

ALERTS_SIZE=$(stat -c %s "$ALERTS")
ALERTS_LINES=$(wc -l < "$ALERTS")
echo "  size  : $(numfmt --to=iec "$ALERTS_SIZE")  ($ALERTS_LINES alerts)"

# ── 1. Stop engine so the file move is safe (engine has the file open) ──
ENGINE_PID=$(pgrep -f "python3 ueba_engine\.py" | head -1 || true)
if [ -n "$ENGINE_PID" ]; then
  echo "  stopping engine PID=$ENGINE_PID"
  kill "$ENGINE_PID" 2>/dev/null || true
  for i in $(seq 1 25); do
    sleep 1
    kill -0 "$ENGINE_PID" 2>/dev/null || { echo "  engine exited after ${i}s"; break; }
  done
  if kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "  engine didn't exit in 25s — SIGKILL"
    kill -9 "$ENGINE_PID" || true
    sleep 1
  fi
else
  echo "  no running engine (skipping stop)"
fi

# ── 2. Move + zip ─────────────────────────────────────────────────────────
DATED="ueba_alerts_${ROTATE_DATE}.jsonl"
DATED_PATH="$ARCHIVE_DIR/$DATED"

# If a previous run already created the same dated archive, append a suffix.
if [ -e "${DATED_PATH}.zip" ]; then
  SUFFIX=$(date '+%H%M%S')
  DATED="ueba_alerts_${ROTATE_DATE}_${SUFFIX}.jsonl"
  DATED_PATH="$ARCHIVE_DIR/$DATED"
  echo "  note: archive for ${ROTATE_DATE} exists, using ${DATED}.zip"
fi

mv "$ALERTS" "$DATED_PATH"
touch "$ALERTS"
chmod 644 "$ALERTS"

(
  cd "$ARCHIVE_DIR"
  if zip -q "${DATED}.zip" "$DATED"; then
    rm -f "$DATED"
  else
    echo "  WARNING: zip failed; leaving uncompressed $DATED in $ARCHIVE_DIR"
  fi
)
echo "  archived → $ARCHIVE_DIR/${DATED}.zip ($(stat -c %s "$ARCHIVE_DIR/${DATED}.zip" 2>/dev/null) bytes)"

# ── 3. Restart engine so it reopens the now-empty alerts file ────────────
cd "$BASE"
setsid "$VENV_PY" ueba_engine.py >> engine.log 2>&1 < /dev/null &
NEW_PID=$!
echo "  engine restart launched (new PID: $NEW_PID, takes ~35s to be fully ready)"
echo "$(date '+%Y-%m-%d %H:%M:%S')  rotation done"
