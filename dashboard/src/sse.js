// Server-Sent Events client.
//
//  - Opens EventSource('/api/stream') and listens for 'alert' / 'hello' / 'error'.
//  - Prepends each new alert into state.feedData + state.allFeedData (capped),
//    refreshes the ticker, bumps the in-memory stats counters, and repaints
//    the feed panel. The 60s polling fallback reconciles aggregate panels
//    (gauge, users, agents, campaigns) so we don't repaint them on every alert.
//  - Drives the header status pill: green "LIVE" on open, yellow "RECONNECTING"
//    on transient errors.

import { state } from "./state.js";
import { prependFeedRow } from "./panels/feed.js";
import { updateTicker } from "./panels/overview.js";

const FEED_CAP = 10000;
let _es = null;

function _setPill(text, color) {
  const dot      = document.querySelector(".status-dot");
  const pill     = document.querySelector(".status-pill");
  const pillSpan = document.querySelector(".status-pill span");
  if (dot)  dot.style.background = color;
  if (pill) pill.style.color     = color;
  if (pillSpan) pillSpan.textContent = text;
}

function _bumpStats(verdict) {
  const s = state.lastStats;
  if (!s) return;
  s.total_alerts = (s.total_alerts || 0) + 1;
  if      (verdict === "highly_anomalous") s.highly_anomalous = (s.highly_anomalous || 0) + 1;
  else if (verdict === "anomalous")        s.anomalous        = (s.anomalous        || 0) + 1;
  else if (verdict === "suspicious")       s.suspicious       = (s.suspicious       || 0) + 1;
  // Re-render the banner stat values (these come from overview.renderStats /
  // renderDashMetrics; we just nudge the headline numbers).
  const el = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v.toLocaleString(); };
  el("s-total",      s.total_alerts);
  el("s-highly",     s.highly_anomalous || 0);
  el("s-anomalous",  s.anomalous        || 0);
  el("s-suspicious", s.suspicious       || 0);
}

function _onAlert(item) {
  if (!item || !item.event_id) return;
  // De-dup: if this event_id is already on top of the list, ignore.
  if (state.allFeedData[0] && state.allFeedData[0].event_id === item.event_id) return;

  state.allFeedData.unshift(item);
  state.feedData.unshift(item);
  if (state.allFeedData.length > FEED_CAP) state.allFeedData.length = FEED_CAP;
  if (state.feedData.length    > FEED_CAP) state.feedData.length    = FEED_CAP;

  _bumpStats(item.verdict);
  updateTicker(state.feedData);

  // Append-only DOM update — only paints the new row at the top of the feed
  // table. Preserves scroll position, input focus, and existing expanded
  // panels. Aggregate panels (gauge, users, agents, campaigns) refresh on
  // the 60-second poll, not on every alert.
  prependFeedRow(item);
}

export function initSse() {
  if (_es) return;
  try {
    _es = new EventSource("/api/stream");
  } catch (e) {
    _setPill("STREAM OFFLINE", "var(--red)");
    return;
  }

  _es.addEventListener("open", () => {
    _setPill("ENGINE LIVE", "var(--green)");
  });

  _es.addEventListener("hello", () => {
    _setPill("ENGINE LIVE", "var(--green)");
  });

  _es.addEventListener("alert", (e) => {
    try {
      _onAlert(JSON.parse(e.data));
    } catch (err) {
      // ignore malformed payload
    }
  });

  _es.addEventListener("error", () => {
    // EventSource auto-reconnects; surface that to the analyst.
    _setPill("RECONNECTING", "var(--yellow)");
  });
}
