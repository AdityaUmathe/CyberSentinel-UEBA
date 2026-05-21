#!/bin/bash
# ueba_weekly_retrain.sh — weekly model retraining for the UEBA pipeline.
#
# Runs from cron every Sunday at 03:30 IST. Each run:
#   1. Pulls new enriched_*.jsonl.gz chunks from 222 (since last successful
#      pull, OR last 7 days on the very first run).
#   2. Bundles them into one dated snapshot in training_zips/ — named so
#      retrain.py picks it up as a "new" training file.
#   3. Stops the engine.
#   4. Runs retrain.py (which backs up models, prepares features, retrains
#      Isolation Forest + global AE + per-user AEs + FAISS).
#   5. Restarts the engine.
#   6. Prunes weekly snapshots older than SNAPSHOT_RETENTION_DAYS.
#
# Engine downtime per run: 30–90 min (depends on features.h5 size).
# If retrain.py crashes, the engine is restarted with the previous models
# (retrain.py's own backup-then-rebuild flow ensures models/ is never half-empty).
#
# Manual invocation is safe — running mid-week pulls any new chunks since
# the last run and rebuilds normally. State tracked in .state/last_train_pull.

set -euo pipefail

BASE=/root/NEW_DRIVE/aditya_ueba
TRAINING_DIR=$BASE/training_zips
STATE_DIR=$BASE/.state
LAST_PULL_STATE=$STATE_DIR/last_train_pull
LOG=$BASE/logs/weekly_retrain.log
VENV_PY=/data/aditya_ueba/venv/bin/python3

# 222 SSH config (reuses the bridge's reverse-tunnel key)
SSH_OPTS="-p 2222 -i /root/.ssh/id_ueba -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
SOC_HOST="soc@localhost"
SOC_ENRICHED_DIR="/home/soc/correlation_kafka/CyberSentinel-Event-Correlation-Kafka/data/enriched"

SNAPSHOT_RETENTION_DAYS=${SNAPSHOT_RETENTION_DAYS:-90}

mkdir -p "$TRAINING_DIR" "$STATE_DIR" "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo "==================================================================="
echo "$(date '+%Y-%m-%d %H:%M:%S')  Weekly retrain start"

# ── 1. Discover new chunks on 222 ────────────────────────────────────────
# Filenames are enriched_YYYY-MM-DD_HH-MM-SS.jsonl.gz, so lexicographic
# comparison == chronological. State file holds the last-pulled basename.
LAST_PULLED=$(cat "$LAST_PULL_STATE" 2>/dev/null || echo "")
if [ -n "$LAST_PULLED" ]; then
  echo "  Last pulled: $LAST_PULLED"
  REMOTE_FILES=$(ssh $SSH_OPTS $SOC_HOST \
    "ls $SOC_ENRICHED_DIR/enriched_*.jsonl.gz 2>/dev/null | sort" \
    | awk -F/ -v last="$LAST_PULLED" '$NF > last { print $NF }')
else
  echo "  First run: pulling last 7 days of .gz from 222"
  REMOTE_FILES=$(ssh $SSH_OPTS $SOC_HOST \
    "find $SOC_ENRICHED_DIR -maxdepth 1 -name 'enriched_*.jsonl.gz' -mtime -7 -printf '%f\n' | sort")
fi

if [ -z "$REMOTE_FILES" ]; then
  echo "  No new .gz on 222 — skipping retrain (engine left running)"
  exit 0
fi

CHUNK_COUNT=$(echo "$REMOTE_FILES" | wc -l)
NEWEST=$(echo "$REMOTE_FILES" | tail -1)
echo "  Found $CHUNK_COUNT new chunk(s); newest: $NEWEST"

# ── 2. Bundle into a single weekly snapshot ──────────────────────────────
WEEK_TS=$(date +%Y-%m-%d)
SNAPSHOT="enriched-weekly-${WEEK_TS}.json.gz"
SNAPSHOT_PATH="$TRAINING_DIR/$SNAPSHOT"

if [ -e "$SNAPSHOT_PATH" ]; then
  TS_SUFFIX=$(date +%H%M%S)
  SNAPSHOT="enriched-weekly-${WEEK_TS}_${TS_SUFFIX}.json.gz"
  SNAPSHOT_PATH="$TRAINING_DIR/$SNAPSHOT"
  echo "  Today's snapshot exists; using ${SNAPSHOT}"
fi

echo "  Building $SNAPSHOT (zcat $CHUNK_COUNT remote chunks → gzip)..."
REMOTE_PATHS=$(echo "$REMOTE_FILES" | awk -v dir="$SOC_ENRICHED_DIR" '{print dir"/"$0}' | tr '\n' ' ')
if ! ssh $SSH_OPTS $SOC_HOST "zcat $REMOTE_PATHS" | gzip > "$SNAPSHOT_PATH"; then
  echo "  ERROR: snapshot build failed — aborting (engine left running)"
  rm -f "$SNAPSHOT_PATH"
  exit 1
fi
echo "  Snapshot built: $(numfmt --to=iec "$(stat -c %s "$SNAPSHOT_PATH")")"

# State updated only after a successful snapshot, so a failed pull is replayed.
echo "$NEWEST" > "$LAST_PULL_STATE"

# ── 3. Stop engine ───────────────────────────────────────────────────────
ENGINE_PID=$(pgrep -f "python3 ueba_engine\.py" | head -1 || true)
if [ -n "$ENGINE_PID" ]; then
  echo "  Stopping engine PID=$ENGINE_PID"
  kill "$ENGINE_PID" 2>/dev/null || true
  for i in $(seq 1 30); do
    sleep 1
    kill -0 "$ENGINE_PID" 2>/dev/null || { echo "  engine exited after ${i}s"; break; }
  done
  if kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "  engine didn't exit in 30s — SIGKILL"
    kill -9 "$ENGINE_PID" || true
    sleep 1
  fi
else
  echo "  No engine running"
fi

# ── 4. Run retrain.py ────────────────────────────────────────────────────
echo "  Running retrain.py..."
cd "$BASE"
RETRAIN_OK=1
if ! "$VENV_PY" retrain.py; then
  echo "  retrain.py FAILED — restarting engine with previous models"
  RETRAIN_OK=0
fi

# ── 5. Restart engine ────────────────────────────────────────────────────
cd "$BASE"
setsid "$VENV_PY" ueba_engine.py >> engine.log 2>&1 < /dev/null &
NEW_PID=$!
echo "  Engine restart launched (new PID: $NEW_PID, takes ~35s to be fully ready)"

# ── 6. Prune snapshots older than N days ─────────────────────────────────
DELETED=$(find "$TRAINING_DIR" -maxdepth 1 -name 'enriched-weekly-*.json.gz' \
            -mtime "+${SNAPSHOT_RETENTION_DAYS}" -print -delete 2>/dev/null || true)
if [ -n "$DELETED" ]; then
  DCOUNT=$(echo "$DELETED" | wc -l)
  echo "  Pruned $DCOUNT old weekly snapshot(s) >${SNAPSHOT_RETENTION_DAYS}d"
fi

if [ "$RETRAIN_OK" = "1" ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S')  Weekly retrain done"
else
  echo "$(date '+%Y-%m-%d %H:%M:%S')  Weekly retrain FAILED — engine restored with old models"
  exit 1
fi
