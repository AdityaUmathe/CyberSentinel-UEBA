#!/usr/bin/env bash
# CyberSentinel UEBA — engine liveness watchdog (heartbeat-based).
#
# Catches the "silent stall": the engine process stays alive (so systemd
# Restart= never fires) but its feed-processing loop wedges, freezing output
# while fresh logs keep arriving. That is the 2026-08-01 ~6h incident.
#
# LIVENESS SIGNAL — the engine's Stats heartbeat, NOT alert recency.
# The engine emits a "Stats | processed: N ..." line every ~60s WHENEVER it is
# looping, independent of whether any alert fired. So:
#   * healthy but quiet (few/no alerts)  -> heartbeat keeps advancing  -> OK
#   * genuinely stalled (loop wedged)    -> heartbeat goes stale       -> RESTART
# A previous version keyed off newest_alert_time (<300s) and thus false-restarted
# a healthy engine during every quiet alert period. Do NOT reintroduce that.
#
# Confirmed real stall requires ALL of:
#   1. engine process is active (else systemd is already handling it — but we
#      still restart to be safe if it's not active),
#   2. the FEED is fresh (enriched.jsonl mtime < FEED_FRESH_SECS) — there IS work,
#   3. the Stats heartbeat is stale (last "Stats |" log line > STALE_SECS old),
#   4. the processed counter has NOT advanced since the previous probe.
#
# Flap protection: <=1 restart / MIN_RESTART_GAP. Forensic snapshot before acting.

set -euo pipefail

BASE="/root/NEW_DRIVE/aditya_ueba"
ENGINE_LOG="$BASE/logs/ueba_engine.stdout.log"
FEED_FILE="$BASE/enriched.jsonl"
FORENSIC_DIR="$BASE/logs/watchdog"
STAMP_RESTART="/run/ueba-engine-watchdog.last-restart"
STATE="/run/ueba-engine-watchdog.state"     # remembers last processed count
WD_LOG="$FORENSIC_DIR/watchdog.log"

STALE_SECS=240          # >4 missed 60s heartbeats = stalled
FEED_FRESH_SECS=300     # feed considered "arriving" if mtime younger than this
MIN_RESTART_GAP=900     # never auto-restart more than once / 15 min

mkdir -p "$FORENSIC_DIR"
now="$(date +%s)"
log() { echo "$(date -Is) $*" >>"$WD_LOG"; }

# --- 0. process must exist; if systemd says it's dead, restart outright ---
if ! systemctl is-active --quiet ueba-engine; then
  log "engine not active -> restart"
  echo "$now" >"$STAMP_RESTART"
  systemctl restart ueba-engine
  exit 0
fi

# --- 1. is there work? (feed fresh) ---
if [ ! -f "$FEED_FILE" ]; then log "SKIP: feed file missing"; exit 0; fi
feed_age=$(( now - $(stat -c %Y "$FEED_FILE") ))
if [ "$feed_age" -ge "$FEED_FRESH_SECS" ]; then
  # No fresh feed → engine may be legitimately idle. Not a stall.
  exit 0
fi

# --- 2. heartbeat: age of the last "Stats |" line + its processed count ---
last_stats="$(grep -a 'Stats |' "$ENGINE_LOG" 2>/dev/null | tail -1 || true)"
if [ -z "$last_stats" ]; then
  log "SKIP: no Stats heartbeat line found in engine log yet"
  exit 0
fi
# Timestamp is the leading "YYYY-MM-DD HH:MM:SS" (box-local time).
hb_ts="$(printf '%s' "$last_stats" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' || true)"
hb_epoch="$(date -d "$hb_ts" +%s 2>/dev/null || echo 0)"
hb_age=$(( now - hb_epoch ))
processed="$(printf '%s' "$last_stats" | grep -oE 'processed: [0-9]+' | grep -oE '[0-9]+' || echo -1)"

# --- 3. compare processed counter vs previous probe ---
prev_processed=-1
[ -f "$STATE" ] && prev_processed="$(cat "$STATE" 2>/dev/null || echo -1)"
echo "$processed" >"$STATE"

# Healthy: heartbeat is recent. (Advancing processed OR fresh Stats line both ok.)
if [ "$hb_age" -lt "$STALE_SECS" ]; then
  exit 0
fi

# Heartbeat stale AND processed has not moved since last probe -> real stall.
# (If processed advanced despite an old Stats line, treat as alive: log rotation edge.)
if [ "$processed" != "$prev_processed" ] || [ "$prev_processed" = "-1" ]; then
  log "WATCH: heartbeat ${hb_age}s old but processed moved ($prev_processed->$processed); not restarting yet"
  exit 0
fi

# --- flap protection ---
if [ -f "$STAMP_RESTART" ]; then
  gap=$(( now - $(cat "$STAMP_RESTART" 2>/dev/null || echo 0) ))
  if [ "$gap" -lt "$MIN_RESTART_GAP" ]; then
    log "SUPPRESS restart (heartbeat ${hb_age}s, processed frozen @${processed}): only ${gap}s since last (<${MIN_RESTART_GAP}s)"
    exit 0
  fi
fi

# --- forensics then restart ---
snap="$FORENSIC_DIR/stall_$(date +%Y%m%d_%H%M%S).log"
{
  echo "=== UEBA engine watchdog auto-restart ==="
  echo "when:      $(date -Is)"
  echo "reason:    heartbeat ${hb_age}s old (>${STALE_SECS}s), processed frozen at ${processed}, feed_age ${feed_age}s"
  echo "last stats: $last_stats"
  echo "=== last 200 lines of engine log ==="
  tail -n 200 "$ENGINE_LOG" 2>/dev/null || echo "(engine log unreadable)"
} >"$snap" 2>&1

echo "$now" >"$STAMP_RESTART"
log "RESTART (heartbeat ${hb_age}s old, processed frozen @${processed}) — snapshot: $snap"
systemctl restart ueba-engine
log "restart issued; new active-state: $(systemctl is-active ueba-engine)"
