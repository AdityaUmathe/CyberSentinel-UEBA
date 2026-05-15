// Per-user drilldown — mirrors the agent drilldown but keyed on the user
// identity instead of the host. Loaded when an analyst clicks a row in the
// User Risk leaderboard.

import { state } from "../state.js";
import { scoreColor } from "../helpers.js";
import { feedRows } from "./feed.js";

export async function loadUserDetail(userName, { silent = false } = {}) {
  state.selectedUser = userName;

  // Highlight the clicked row in the leaderboard (visual feedback).
  document.querySelectorAll(".user-row.clickable").forEach((el) =>
    el.classList.toggle("active", el.querySelector(".user-name")?.textContent === userName)
  );

  const panel = document.getElementById("user-detail-panel");
  const body  = document.getElementById("user-detail-body");
  const title = document.getElementById("user-detail-title");
  const badge = document.getElementById("user-detail-badge");
  if (!panel || !body || !title) return;
  panel.removeAttribute("hidden");
  title.textContent = userName;

  if (!silent) {
    body.innerHTML = '<div class="loading"><div class="spinner"></div> Loading…</div>';
  }

  let alerts = [];
  try {
    alerts = await fetch("/api/user/" + encodeURIComponent(userName)).then((r) => r.json());
  } catch {
    if (!silent) body.innerHTML = '<div class="empty-state"><p>FAILED TO LOAD USER</p></div>';
    return;
  }

  badge.textContent = alerts.length + " alerts";
  badge.style.display = "";

  if (!alerts.length) {
    body.innerHTML = '<div class="empty-state" style="padding:30px"><p>NO ALERTS FOR THIS USER</p></div>';
    return;
  }

  const critCount = alerts.filter((a) => a.verdict === "highly_anomalous").length;
  const anomCount = alerts.filter((a) => a.verdict === "anomalous").length;
  const suspCount = alerts.filter((a) => a.verdict === "suspicious").length;
  const maxScore  = Math.max(...alerts.map((a) => a.score || 0));
  const total     = alerts.length;
  const score     = Math.min(100, Math.round(
    (critCount / Math.max(total, 1)) * 60 +
    (anomCount / Math.max(total, 1)) * 20 +
    maxScore * 15
  ));

  // Host distribution
  const hostMap = {};
  alerts.forEach((a) => { const h = a.host || "—"; hostMap[h] = (hostMap[h] || 0) + 1; });
  const topHosts = Object.entries(hostMap).sort((a, b) => b[1] - a[1]).slice(0, 6);

  // Signature distribution
  const sigMap = {};
  alerts.forEach((a) => { const s = (a.signature || "Unknown").slice(0, 60); sigMap[s] = (sigMap[s] || 0) + 1; });
  const topSigs = Object.entries(sigMap).sort((a, b) => b[1] - a[1]).slice(0, 6);

  // Day-bucket timeline (last 30 days max)
  const dayMap = {};
  alerts.forEach((a) => {
    const t = a.processed_at || a.event_time || "";
    if (!t) return;
    const d = t.slice(0, 10);
    if (!dayMap[d]) dayMap[d] = { total: 0, critical: 0 };
    dayMap[d].total++;
    if (a.verdict === "highly_anomalous") dayMap[d].critical++;
  });
  const days = Object.keys(dayMap).sort();
  const maxDay = Math.max(...days.map((d) => dayMap[d].total), 1);

  function hBar(label, val, maxVal, color) {
    const pct = maxVal > 0 ? Math.max(2, Math.round((val / maxVal) * 100)) : 0;
    return `<div style="margin-bottom:11px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px">
        <span style="font-family:var(--mono2);font-size:11px;color:var(--text2);max-width:78%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${label}">${label}</span>
        <span style="font-family:var(--mono);font-size:12px;color:${color};margin-left:8px;flex-shrink:0">${val}</span>
      </div>
      <div style="height:8px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden">
        <div style="height:100%;width:${pct}%;background:${color};border-radius:4px;opacity:0.7"></div>
      </div>
    </div>`;
  }

  function buildTimeline() {
    if (!days.length) return '<div class="empty-state" style="padding:30px"><p>NO TIMELINE DATA</p></div>';
    const W = 560, H = 130, pL = 36, pR = 8, pT = 10, pB = 24;
    const cW = W - pL - pR, cH = H - pT - pB, n = days.length;
    const spacing = cW / n, bW = Math.max(3, Math.floor(spacing - 2));
    const bars = days.map((d, i) => {
      const v = dayMap[d];
      const x = pL + i * spacing + (spacing - bW) / 2;
      const totalH = Math.max(2, Math.round((v.total / maxDay) * cH));
      const critH  = Math.max(0, Math.round((v.critical / maxDay) * cH));
      const yTot = pT + cH - totalH;
      const yCrit = pT + cH - critH;
      return `
        <rect x="${x}" y="${yTot}" width="${bW}" height="${totalH}" fill="#00d4ff" opacity="0.65" rx="1.5">
          <title>${d} — ${v.total} alerts</title></rect>
        <rect x="${x}" y="${yCrit}" width="${bW}" height="${critH}" fill="#ff3b5c" opacity="0.95" rx="1.5">
          <title>${d} — ${v.critical} critical</title></rect>`;
    }).join("");
    const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => {
      const y = pT + cH - Math.round(f * cH);
      return `<line x1="${pL}" y1="${y}" x2="${W - pR}" y2="${y}" stroke="#1a2838" stroke-width="0.5" stroke-dasharray="3,3"/>
              <text x="${pL - 4}" y="${y + 3}" font-size="8" fill="#3d5a72" text-anchor="end">${Math.round(f * maxDay)}</text>`;
    }).join("");
    const step = Math.max(1, Math.ceil(n / 8));
    const xLabels = days.filter((_, i) => i % step === 0 || i === n - 1).map((d) => {
      const i = days.indexOf(d);
      const x = pL + i * spacing + spacing / 2;
      return `<text x="${x}" y="${H - 4}" font-size="8" fill="#3d5a72" text-anchor="middle">${d.slice(5)}</text>`;
    }).join("");
    return `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:${H}px;display:block">
      ${grid}<line x1="${pL}" y1="${pT}" x2="${pL}" y2="${pT + cH}" stroke="#243447" stroke-width="1"/>
      <line x1="${pL}" y1="${pT + cH}" x2="${W - pR}" y2="${pT + cH}" stroke="#243447" stroke-width="1"/>
      ${bars}${xLabels}</svg>`;
  }

  body.innerHTML = `
    <div class="agent-summary">
      <div class="agent-stat"><div class="agent-stat-val" style="color:${scoreColor(maxScore)}">${score}</div><div class="agent-stat-lbl">Risk Score / 100</div></div>
      <div class="agent-stat"><div class="agent-stat-val" style="color:var(--red)">${critCount}</div><div class="agent-stat-lbl">Critical</div></div>
      <div class="agent-stat"><div class="agent-stat-val" style="color:var(--orange)">${anomCount}</div><div class="agent-stat-lbl">Anomalous</div></div>
      <div class="agent-stat"><div class="agent-stat-val" style="color:var(--yellow)">${suspCount}</div><div class="agent-stat-lbl">Suspicious</div></div>
    </div>

    <div style="padding:14px;">
      <div class="ep-chart-card" style="margin-bottom:14px;">
        <div class="ep-chart-title" style="display:flex;justify-content:space-between;align-items:center;">
          <span>Daily activity</span>
          <div style="display:flex;gap:14px;font-size:9px">
            <span style="color:#00d4ff"><span style="display:inline-block;width:10px;height:4px;background:#00d4ff;opacity:0.65;border-radius:2px;margin-right:4px"></span>Total</span>
            <span style="color:#ff3b5c"><span style="display:inline-block;width:10px;height:4px;background:#ff3b5c;border-radius:2px;margin-right:4px"></span>Critical</span>
          </div>
        </div>
        ${buildTimeline()}
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
        <div class="ep-chart-card">
          <div class="ep-chart-title">Top Hosts</div>
          <div style="padding:4px 0">${topHosts.map(([h, c]) => hBar(h, c, topHosts[0][1], "var(--accent)")).join("") || '<div style="color:var(--text3);font-family:var(--mono2);font-size:11px;padding:12px">No data</div>'}</div>
        </div>
        <div class="ep-chart-card">
          <div class="ep-chart-title">Top Triggered Rules</div>
          <div style="padding:4px 0">${topSigs.map(([s, c]) => hBar(s, c, topSigs[0][1], "var(--purple)")).join("") || '<div style="color:var(--text3);font-family:var(--mono2);font-size:11px;padding:12px">No data</div>'}</div>
        </div>
      </div>

      <div class="ep-chart-card" style="margin-top:14px;">
        <div class="ep-chart-title">All Events (${alerts.length})</div>
        <div class="feed-table-wrap" style="max-height:420px;">
          <table>
            <thead><tr><th>Time</th><th>Host</th><th>Verdict</th><th>Score</th><th>Reasons</th><th>Campaign</th><th>Signature</th></tr></thead>
            <tbody>${feedRows(alerts, 7, "usr-")}</tbody>
          </table>
        </div>
      </div>
    </div>`;

  // Collapse any auto-expanded evidence rows
  alerts.forEach((a) => {
    const id = "ev-usr-" + (a.event_id || a.processed_at + a.user);
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });
}

export function closeUserDetail() {
  state.selectedUser = null;
  const panel = document.getElementById("user-detail-panel");
  if (panel) panel.setAttribute("hidden", "");
  document.querySelectorAll(".user-row.clickable").forEach((el) => el.classList.remove("active"));
}

export function initUserDetail() {
  const close = document.getElementById("user-detail-close");
  if (close) close.addEventListener("click", closeUserDetail);
}
