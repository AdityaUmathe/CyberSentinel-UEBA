#!/bin/bash
# ueba_rotate.sh — daily maintenance for the UEBA pipeline on the GPU server.
#
# Runs from cron at 00:05 daily. Does three independent jobs in one pass:
#   1. Archives the previous day's ueba_alerts.jsonl to a date-stamped zip.
#   2. Truncates enriched.jsonl when its last flush was ≥ENRICHED_FLUSH_DAYS
#      ago, so the engine's input buffer doesn't grow unbounded on disk.
#      Engine state file is cleared so the restart starts cleanly at EOF.
#   3. Deletes archived alert zips older than ARCHIVE_RETENTION_DAYS.
#
# The engine is stopped at the top and restarted at the bottom regardless of
# which jobs actually fire, so any in-place file mutation is safe.
#
# Manual invocation is safe — running mid-day will rotate whatever's
# currently in ueba_alerts.jsonl and label the archive with "yesterday".
# Override the label by exporting ROTATE_DATE=YYYY-MM-DD before running.

set -euo pipefail

BASE=/root/NEW_DRIVE/aditya_ueba
ALERTS=$BASE/ueba_alerts.jsonl
ENRICHED=$BASE/enriched.jsonl
UEBA_STATE=$BASE/.state/ueba.state
ENRICHED_FLUSH_MARKER=$BASE/.state/enriched_last_flushed
ARCHIVE_DIR=$BASE/archive
LOG=$BASE/logs/rotate.log
VENV_PY=/data/aditya_ueba/venv/bin/python3

# Retention knobs
ENRICHED_FLUSH_DAYS=${ENRICHED_FLUSH_DAYS:-120}
ARCHIVE_RETENTION_DAYS=${ARCHIVE_RETENTION_DAYS:-120}

ROTATE_DATE=${ROTATE_DATE:-$(date -d "yesterday" +%Y-%m-%d)}

mkdir -p "$ARCHIVE_DIR" "$(dirname "$LOG")" "$(dirname "$UEBA_STATE")"
exec >> "$LOG" 2>&1

echo "==================================================================="
echo "$(date '+%Y-%m-%d %H:%M:%S')  rotation start (archive label: ${ROTATE_DATE})"

# Decide what work to do — but always stop+restart the engine so file
# mutations are safe even when only the enriched flush fires.
if [ -s "$ALERTS" ]; then
  ALERTS_SIZE=$(stat -c %s "$ALERTS")
  ALERTS_LINES=$(wc -l < "$ALERTS")
  echo "  alerts: $(numfmt --to=iec "$ALERTS_SIZE") (${ALERTS_LINES} alerts) — will archive"
  HAVE_ALERTS=1
else
  echo "  alerts: empty — skipping archive step"
  HAVE_ALERTS=0
fi

# ── 1. Stop engine so file mutations are safe (engine holds these open) ──
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

# ── 2. Archive ueba_alerts.jsonl (if non-empty) ──────────────────────────
if [ "$HAVE_ALERTS" = "1" ]; then
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
fi

# ── 3. Maybe flush enriched.jsonl (cap input buffer to N days) ───────────
# The bridge appends remote logs here continuously; without periodic
# truncation it grows by tens of GB. We use a marker file's mtime to
# decide when N days have elapsed since the last flush.
if [ -e "$ENRICHED_FLUSH_MARKER" ]; then
  MARKER_AGE_DAYS=$(( ($(date +%s) - $(stat -c %Y "$ENRICHED_FLUSH_MARKER")) / 86400 ))
else
  MARKER_AGE_DAYS=999  # no marker yet → treat as overdue, flush now
fi

if [ "$MARKER_AGE_DAYS" -ge "$ENRICHED_FLUSH_DAYS" ] && [ -e "$ENRICHED" ]; then
  ENRICHED_SIZE=$(stat -c %s "$ENRICHED" 2>/dev/null || echo 0)
  echo "  enriched: ${MARKER_AGE_DAYS}d since last flush ≥ ${ENRICHED_FLUSH_DAYS}d — truncating $(numfmt --to=iec "$ENRICHED_SIZE")"
  : > "$ENRICHED"
  # Clear engine state so the restart seeks to EOF of the now-empty file
  # rather than a stale 65 GB offset.
  rm -f "$UEBA_STATE"
  touch "$ENRICHED_FLUSH_MARKER"
else
  echo "  enriched: ${MARKER_AGE_DAYS}d since last flush (< ${ENRICHED_FLUSH_DAYS}d) — skip"
fi

# ── 4. Restart engine so it reopens the alerts file (and the truncated
#       enriched.jsonl, if it was flushed) ──────────────────────────────────
cd "$BASE"
setsid "$VENV_PY" ueba_engine.py >> engine.log 2>&1 < /dev/null &
NEW_PID=$!
echo "  engine restart launched (new PID: $NEW_PID, takes ~35s to be fully ready)"

# ── 5. Cleanup archived alert zips older than ARCHIVE_RETENTION_DAYS ─────
# Uses file mtime — created/copied with mv+zip above, so the timestamp
# reflects the day the archive was made.
DELETED_LIST=$(find "$ARCHIVE_DIR" -maxdepth 1 -name "ueba_alerts_*.jsonl.zip" \
                 -mtime "+${ARCHIVE_RETENTION_DAYS}" -print -delete 2>/dev/null || true)
if [ -n "$DELETED_LIST" ]; then
  DELETED_COUNT=$(echo "$DELETED_LIST" | wc -l)
  echo "  retention: deleted ${DELETED_COUNT} archive(s) older than ${ARCHIVE_RETENTION_DAYS}d"
  echo "$DELETED_LIST" | sed 's/^/    × /'
else
  echo "  retention: no archives older than ${ARCHIVE_RETENTION_DAYS}d"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S')  rotation done"
