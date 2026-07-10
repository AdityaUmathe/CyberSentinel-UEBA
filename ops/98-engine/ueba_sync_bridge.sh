#!/bin/bash
# ueba_sync_bridge.sh — Sync enriched logs from 222 to 98
# Handles rotated gzipped chunks + live active file
# New format: ~/correlation_kafka/.../data/enriched/enriched_YYYY-MM-DD_HH-MM-SS.jsonl[.gz]

set -euo pipefail

# ── Config
SOC_HOST="soc@localhost"
SOC_PORT="2222"
SOC_ENRICHED_DIR="/home/soc/correlation_kafka/CyberSentinel-Event-Correlation-Kafka/data/enriched"
LOCAL_ENRICHED="/root/NEW_DRIVE/aditya_ueba/enriched.jsonl"
ALERTS_SRC="/root/NEW_DRIVE/aditya_ueba/ueba_alerts.jsonl"
ALERTS_DST="soc@localhost:/home/soc/ueba/ueba_alerts.jsonl"
STATE_FILE="/root/NEW_DRIVE/aditya_ueba/.state/sync_bridge.state"
LOG="/root/NEW_DRIVE/aditya_ueba/logs/sync_bridge.log"

SSH_OPTS="-p ${SOC_PORT} -i /root/.ssh/id_ueba -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

mkdir -p "$(dirname "$STATE_FILE")" "$(dirname "$LOG")"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$LOG"; }

get_last_processed() { [ -f "$STATE_FILE" ] && cat "$STATE_FILE" || echo ""; }
set_last_processed() { echo "$1" > "$STATE_FILE"; }

TAIL_PID=""
cleanup() {
    log "Shutting down sync bridge..."
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null || true
    kill "$PUSH_PID" 2>/dev/null || true
    kill "$STATS_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

log "=== UEBA Sync Bridge Started (rotated-file mode) ==="
log "  SOURCE DIR : ${SOC_HOST}:${SOC_ENRICHED_DIR}"
log "  LOCAL FILE : ${LOCAL_ENRICHED}"

# ── PUSH loop — DISABLED 2026-07-10: the dashboard runs on 98 itself now, so
# there is no need to ship ueba_alerts.jsonl back to the decommissioned SOC
# box. The push only ever produced perpetual "PUSH ERROR: rsync to 222
# failed" log noise (the target was unwritable). Kept PUSH_PID defined so the
# set -u cleanup/log references stay valid.
PUSH_PID=""

# ── STATS loop
# Note: TAIL_PID is set by the main loop *after* this subshell forks, so the
# inherited copy stays empty. Query pgrep each tick to report the real PID.
stats_loop() {
    while true; do
        sleep 60
        sz_bytes=$(wc -c < "$LOCAL_ENRICHED" 2>/dev/null || echo 0)
        sz_human=$(numfmt --to=iec --suffix=B "$sz_bytes" 2>/dev/null || echo "${sz_bytes}B")
        tail_pid=$(pgrep -f "ssh .* tail -n 0 -f ${SOC_ENRICHED_DIR}" | head -1)
        log "Stats | enriched.jsonl: ${sz_human} | tail PID: ${tail_pid:-none}"
    done
}
stats_loop &
STATS_PID=$!

log "Push PID: ${PUSH_PID} | Stats PID: ${STATS_PID}"

# ── Main loop
while true; do
    REMOTE_FILES=$(ssh ${SSH_OPTS} ${SOC_HOST} \
        "ls ${SOC_ENRICHED_DIR}/enriched_*.jsonl* 2>/dev/null | sort" 2>/dev/null || echo "")

    if [ -z "$REMOTE_FILES" ]; then
        log "WARNING: No enriched files found on 222 — retrying in 30s"
        sleep 30
        continue
    fi

    GZ_FILES=$(echo "$REMOTE_FILES" | grep '\.gz$' || true)
    ACTIVE_FILE=$(echo "$REMOTE_FILES" | grep -v '\.gz$' | tail -1 || true)
    LAST_PROCESSED=$(get_last_processed)

    # Process new gz chunks
    while IFS= read -r gz_file; do
        [ -z "$gz_file" ] && continue
        fname=$(basename "$gz_file")
        if [ -z "$LAST_PROCESSED" ] || [[ "$fname" > "$LAST_PROCESSED" ]]; then
            log "Processing new chunk: ${fname}"
            ssh ${SSH_OPTS} ${SOC_HOST} "zcat ${gz_file}" >> "$LOCAL_ENRICHED" 2>/dev/null
            set_last_processed "$fname"
            log "  Appended ${fname}"
        fi
    done <<< "$GZ_FILES"

    # Tail active file
    if [ -n "$ACTIVE_FILE" ]; then
        ACTIVE_FNAME=$(basename "$ACTIVE_FILE")
        log "PULL: tailing active file ${ACTIVE_FNAME}..."
        [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null || true

        # -n 0: stream from EOF only; otherwise tail re-emits the last 10 lines
        # of the file on every bridge restart, which the engine then re-processes.
        ssh ${SSH_OPTS} ${SOC_HOST} "tail -n 0 -f ${ACTIVE_FILE}" >> "$LOCAL_ENRICHED" &
        TAIL_PID=$!
        log "  Tail PID: ${TAIL_PID}"

        while kill -0 "$TAIL_PID" 2>/dev/null; do
            sleep 60
            NEW_ACTIVE=$(ssh ${SSH_OPTS} ${SOC_HOST} \
                "ls ${SOC_ENRICHED_DIR}/enriched_*.jsonl 2>/dev/null | sort | tail -1" 2>/dev/null || echo "")
            if [ -n "$NEW_ACTIVE" ] && [ "$(basename "$NEW_ACTIVE")" != "$ACTIVE_FNAME" ]; then
                log "File rotated → $(basename "$NEW_ACTIVE") — restarting tail"
                kill "$TAIL_PID" 2>/dev/null || true
                TAIL_PID=""
                break
            fi
        done
    else
        log "No active file — waiting 30s"
        sleep 30
    fi
done
