// Overview tab: ticker, stats banner, dashboard metric cards, and the master
// renderDashboard() that orchestrates all the charts.

import { state } from "../state.js";
import { updateGauge } from "../charts/gauge.js";
import { drawRadar } from "../charts/radar.js";
import { drawTimeline } from "../charts/timeline.js";
import { renderScoreDist } from "../charts/scoredist.js";
import { renderHeatmap } from "../charts/heatmap.js";
import { renderTopSigs } from "../charts/topsigs.js";
import { renderRecommendedActions } from "./recommendations.js";
import { generateAIAnalysis } from "./ai-analyst.js";
import { goToFeedWithFilter } from "./agents.js";
import { renderMitre } from "./mitre.js";
import { renderHealth } from "./health.js";

export function updateTicker(alerts) {
  if (!alerts || !alerts.length) return;
  const sigs = alerts.slice(0, 12).map((a) => {
    const v = (a.verdict || "").replace(/_/g, " ").toUpperCase();
    const u = a.user || "unknown";
    const s = (a.signature || "").slice(0, 50);
    const score = (a.score || 0).toFixed(3);
    return `${u}  ·  ${s}  ·  ${v}  ·  SCORE ${score}`;
  }).join("     ⬦     ");
  const full = sigs + "     ⬦     " + sigs;
  const el = document.getElementById("ticker-text");
  if (el) el.textContent = full;
}

export function renderStats(s) {
  document.getElementById("s-highly").textContent    = s.highly_anomalous?.toLocaleString() || "0";
  document.getElementById("s-anomalous").textContent = s.anomalous?.toLocaleString() || "0";
  document.getElementById("s-suspicious").textContent = s.suspicious?.toLocaleString() || "0";
  document.getElementById("s-total").textContent     = s.total_alerts?.toLocaleString() || "0";
  document.getElementById("s-1h").textContent        = `${s.alert_rate_1h || 0} in last hour`;
  document.getElementById("s-users").textContent     = `${s.unique_users || 0} users`;
  const maxC = s.top_reasons?.[0]?.count || 1;
  document.getElementById("reasons-list").innerHTML = (s.top_reasons || []).map((r) => `
    <div class="reason-row">
      <div class="reason-label">${r.reason.replace(/_/g, " ")}</div>
      <div class="reason-bar-wrap"><div class="reason-bar" style="width:${Math.round((r.count / maxC) * 100)}%"></div></div>
      <div class="reason-count">${r.count}</div>
    </div>`).join("");
  // Legacy ".stat-card" bindings — kept for parity with the original;
  // current HTML uses ".banner-stat" which is wired in ui/tabs.js.
  const cardMap = { "s-highly": "highly_anomalous", "s-anomalous": "anomalous", "s-suspicious": "suspicious", "s-total": "all" };
  Object.entries(cardMap).forEach(([elId, filter]) => {
    const card = document.getElementById(elId)?.closest(".stat-card");
    if (card) {
      card.style.cursor = "pointer";
      card.onclick = () => goToFeedWithFilter(filter);
    }
  });
}

export function renderDashMetrics(stats) {
  const total = stats.total_alerts || 0;
  const campaigns = stats.campaigns || 0;
  const users = stats.unique_users || 0;
  const critPct = total > 0 ? ((stats.highly_anomalous || 0) / total * 100).toFixed(1) + "%" : "0.0%";

  const el = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
  el("s-total",         total.toLocaleString());
  el("dm-campaigns",    campaigns);
  el("dm-users-count",  users);
  el("dm-critical-pct", critPct);
  el("s-highly",        (stats.highly_anomalous || 0).toLocaleString());
  el("s-anomalous",     (stats.anomalous       || 0).toLocaleString());
  el("s-suspicious",    (stats.suspicious      || 0).toLocaleString());
  el("s-users",         users);
  const s1h = document.getElementById("s-1h");
  if (s1h) s1h.textContent = (stats.alert_rate_1h || 0) + " in last hour";
}

export function renderDashboard(stats, feedData) {
  state.lastStats = stats;
  // Also expose on window for parity — some legacy paths read window._lastStats.
  window._lastStats = stats;
  renderDashMetrics(stats);
  const score = updateGauge(stats, feedData);
  drawRadar(stats.top_reasons || []);
  drawTimeline(feedData);
  renderScoreDist(feedData);
  renderHeatmap(feedData);
  renderTopSigs(feedData);
  renderRecommendedActions(stats, score);
  generateAIAnalysis(stats, feedData, score);
  renderMitre();
  renderHealth();
}
