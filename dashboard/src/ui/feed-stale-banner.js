// Global stale-feed banner.
//
// The recurring operational failure here is the 222→98 reverse-SSH feed tunnel
// dying (often overnight): the alert stream silently freezes while the rest of
// the dashboard keeps looking healthy — it just shows stale data. Operators
// only ever caught it by manually eyeballing the newest timestamp each morning.
//
// This banner makes a frozen feed impossible to miss. It keys on FEED liveness —
// whether the engine's input (enriched.jsonl) is still growing, reported by
// /api/health.feed_age_secs — NOT on alert recency. A quiet period with no new
// alerts is normal (the engine may legitimately find nothing anomalous for a
// while) and must NOT raise a "feed down" alarm; only an actually-frozen feed
// (the 222→98 tunnel/bridge dying) should. It piggybacks on the existing 60s
// health poll (renderHealth), so it adds no extra network traffic.

const WARN_MIN = 5;    // feed hasn't grown in this long → amber "stalling"
const DOWN_MIN = 15;   // …no growth this long → red "appears down"

function _fmtAge(mins) {
  if (mins < 60) return `${Math.round(mins)} min`;
  const h = Math.floor(mins / 60);
  const m = Math.round(mins % 60);
  if (h < 24) return m ? `${h}h ${m}m` : `${h}h`;
  const d = Math.floor(h / 24);
  return `${d}d ${h % 24}h`;
}

// Drive the banner from a /api/health payload. Hidden whenever the feed is
// growing normally (or the field is missing — we don't cry wolf on a momentarily
// incomplete payload).
export function updateFeedStaleBanner(health) {
  const el = document.getElementById("feed-stale-banner");
  if (!el) return;

  // feed_age_secs = how long since enriched.jsonl last grew. This is the real
  // feed-health signal; alert recency is NOT used (no alerts ≠ feed down).
  const ageSec = health && typeof health.feed_age_secs === "number"
    ? health.feed_age_secs : null;
  if (ageSec == null) {
    el.hidden = true;
    el.classList.remove("warn", "down");
    return;
  }

  const ageMin = ageSec / 60;
  if (ageMin < WARN_MIN) {
    el.hidden = true;
    el.classList.remove("warn", "down");
    return;
  }

  const down = ageMin >= DOWN_MIN;
  el.classList.toggle("down", down);
  el.classList.toggle("warn", !down);
  el.innerHTML = down
    ? `⚠ FEED APPEARS DOWN — no new events for <strong>${_fmtAge(ageMin)}</strong>. ` +
      `Check the 222→98 feed tunnel/bridge.`
    : `⚠ Feed may be stalling — no new events for <strong>${_fmtAge(ageMin)}</strong>.`;
  el.hidden = false;
}
