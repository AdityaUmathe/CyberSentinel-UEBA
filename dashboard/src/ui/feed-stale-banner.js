// Global stale-feed banner.
//
// The recurring operational failure here is the 222→98 reverse-SSH feed tunnel
// dying (often overnight): the alert stream silently freezes while the rest of
// the dashboard keeps looking healthy — it just shows stale data. Operators
// only ever caught it by manually eyeballing the newest timestamp each morning.
//
// This banner makes a frozen feed impossible to miss. It watches the newest
// alert's age — reported window-independently by /api/health.newest_alert_time,
// so it's correct no matter which time window the user has selected — and warns
// when the feed stalls, escalating to a red "DOWN" state. It piggybacks on the
// existing 60s health poll (renderHealth), so it adds no extra network traffic.

const WARN_MIN = 15;   // newest alert older than this → amber "stalling"
const DOWN_MIN = 60;   // …older than this → red "appears down"

function _fmtAge(mins) {
  if (mins < 60) return `${Math.round(mins)} min`;
  const h = Math.floor(mins / 60);
  const m = Math.round(mins % 60);
  if (h < 24) return m ? `${h}h ${m}m` : `${h}h`;
  const d = Math.floor(h / 24);
  return `${d}d ${h % 24}h`;
}

// Drive the banner from a /api/health payload. Hidden whenever the feed is
// fresh (or the timestamp is missing/unparseable — we don't cry wolf on a
// momentarily empty payload).
export function updateFeedStaleBanner(health) {
  const el = document.getElementById("feed-stale-banner");
  if (!el) return;

  const iso = health && health.newest_alert_time;
  if (!iso) {
    el.hidden = true;
    return;
  }

  let ageMin;
  try {
    // processed_at carries a UTC offset; Date.now() is UTC ms — the diff is
    // timezone-correct regardless of the browser's local zone.
    ageMin = (Date.now() - new Date(iso).getTime()) / 60000;
  } catch {
    el.hidden = true;
    return;
  }

  if (!(ageMin >= WARN_MIN)) {
    el.hidden = true;
    el.classList.remove("warn", "down");
    return;
  }

  const down = ageMin >= DOWN_MIN;
  el.classList.toggle("down", down);
  el.classList.toggle("warn", !down);
  const when = new Date(iso).toLocaleTimeString("en-IN");
  el.innerHTML = down
    ? `⚠ FEED APPEARS DOWN — no new alerts for <strong>${_fmtAge(ageMin)}</strong> ` +
      `(last at ${when}). Check the 222→98 feed tunnel.`
    : `⚠ Feed may be stalling — newest alert is <strong>${_fmtAge(ageMin)}</strong> ` +
      `old (last at ${when}).`;
  el.hidden = false;
}
