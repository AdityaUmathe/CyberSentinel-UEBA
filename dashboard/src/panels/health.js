// Engine Health panel — small 6-KPI card on the Overview tab.
//
// Backed by GET /api/health. Called from renderDashboard so it refreshes
// alongside the rest of the Overview (every 60s polling cycle).

import { updateFeedStaleBanner } from "../ui/feed-stale-banner.js";

let _cache = null;

function _fmtBytes(n) {
  if (!n) return "0 B";
  if (n < 1024)             return n + " B";
  if (n < 1024 * 1024)      return (n / 1024).toFixed(1) + " KB";
  if (n < 1024 ** 3)        return (n / (1024 * 1024)).toFixed(1) + " MB";
  return (n / 1024 ** 3).toFixed(2) + " GB";
}

function _fmtUptime(secs) {
  if (secs < 60)     return `${secs}s`;
  if (secs < 3600)   return `${Math.floor(secs / 60)}m`;
  if (secs < 86400)  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
  return `${Math.floor(secs / 86400)}d ${Math.floor((secs % 86400) / 3600)}h`;
}

function _fmtAgo(iso) {
  if (!iso) return "—";
  try {
    const t = new Date(iso).getTime();
    const diff = Math.max(0, (Date.now() - t) / 1000);
    if (diff < 60)    return `${Math.round(diff)}s ago`;
    if (diff < 3600)  return `${Math.round(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
    return `${Math.round(diff / 86400)}d ago`;
  } catch {
    return "—";
  }
}

function _paint(h) {
  const body  = document.getElementById("health-body");
  const badge = document.getElementById("health-badge");
  if (!body) return;
  if (badge) {
    badge.textContent = h.engine_live ? "LIVE" : "IDLE";
    badge.style.color = h.engine_live ? "var(--green)" : "var(--text3)";
    badge.style.borderColor = h.engine_live ? "rgba(6,214,160,0.45)" : "var(--border)";
    badge.style.background = h.engine_live ? "rgba(6,214,160,0.08)" : "transparent";
  }

  const cards = [
    { label: "ALERTS / HOUR",   value: (h.alerts_1h || 0).toLocaleString(),  color: "var(--accent)" },
    { label: "ALERTS / 24H",    value: (h.alerts_24h || 0).toLocaleString(), color: "var(--accent)" },
    { label: "TOTAL ALERTS",    value: (h.total_alerts || 0).toLocaleString(), color: "var(--text)" },
    { label: "LAST ALERT",      value: _fmtAgo(h.newest_alert_time),         color: h.engine_live ? "var(--green)" : "var(--orange)" },
    { label: "FP RATE",         value: `${(h.fp_rate_pct || 0).toFixed(2)}%`,
                                sub:   `${(h.fp_count || 0).toLocaleString()} flagged`,
                                color: "var(--yellow)" },
    { label: "ALERTS FILE",     value: _fmtBytes(h.alerts_file_size_bytes || 0), color: "var(--text2)" },
    { label: "UPTIME",          value: _fmtUptime(h.uptime_secs || 0),       color: "var(--text2)" },
  ];

  body.innerHTML = `<div class="health-grid">
    ${cards.map((c) => `
      <div class="health-card">
        <div class="health-card-label">${c.label}</div>
        <div class="health-card-value" style="color:${c.color}">${c.value}</div>
        ${c.sub ? `<div class="health-card-sub">${c.sub}</div>` : ""}
      </div>`).join("")}
  </div>`;
}

export async function renderHealth() {
  // Paint cached data immediately to avoid a spinner flash on every refresh.
  if (_cache) {
    _paint(_cache);
    updateFeedStaleBanner(_cache);
  }
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    _cache = h;
    _paint(h);
    // Global stale-feed banner rides the same payload — newest_alert_time is
    // window-independent, so it's correct on every tab and time window.
    updateFeedStaleBanner(h);
  } catch {
    // Keep the cached view if the fetch fails.
  }
}
