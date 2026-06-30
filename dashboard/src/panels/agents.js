// Endpoints panel — agent list, agent detail drilldown (gauge + timeline +
// verdict donut + top sigs + top users + events table), agent risk gauge.

import { state } from "../state.js";
import { navigate } from "../router.js";
import { scoreColor } from "../helpers.js";
import { renderFeed, feedRows } from "./feed.js";

export function renderOverviewAgents() {
  const el = document.getElementById("overview-agents");
  const badge = document.getElementById("ov-agents-badge");
  if (!el) return;
  badge.textContent = state.agentsData.length;
  if (!state.agentsData.length) {
    el.innerHTML = '<div class="empty-state" style="padding:20px 10px"><p>NO AGENTS</p></div>';
    return;
  }
  el.innerHTML = state.agentsData.map((a) => {
    const hasCrit = a.highly_anomalous > 0, hasAnom = a.anomalous > 0;
    const nameColor = hasCrit ? "var(--red)" : hasAnom ? "var(--orange)" : "var(--green)";
    const cntCls = hasCrit ? "cnt-red" : hasAnom ? "cnt-orange" : "";
    const safeName = a.agent.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
    const dot = hasCrit
      ? `<span style="width:6px;height:6px;border-radius:50%;background:var(--red);display:inline-block;margin-right:6px;flex-shrink:0;"></span>`
      : hasAnom
      ? `<span style="width:6px;height:6px;border-radius:50%;background:var(--orange);display:inline-block;margin-right:6px;flex-shrink:0"></span>`
      : `<span style="width:6px;height:6px;border-radius:50%;background:var(--green);opacity:0.4;display:inline-block;margin-right:6px;flex-shrink:0"></span>`;
    return `<div class="agent-item" onclick="goToAgent('${safeName}')" style="padding:10px 14px;">
      <div style="display:flex;align-items:center;flex:1;min-width:0;">${dot}
        <div style="min-width:0;">
          <div class="agent-name" style="color:${nameColor};font-size:12px">${a.agent}</div>
          <div class="agent-ip">${a.ip || ""}${a.os ? " · " + a.os : ""}</div>
        </div>
      </div>
      <div style="text-align:right;flex-shrink:0;">
        <span class="agent-alert-count ${cntCls}">${a.alert_count}</span>
        ${a.highly_anomalous > 0 ? `<div style="font-family:var(--mono2);font-size:9px;color:var(--red);margin-top:2px">${a.highly_anomalous} critical</div>` : ""}
      </div>
    </div>`;
  }).join("");
}

export function renderAgents() {
  document.getElementById("agents-badge").textContent = state.agentsData.length;
  if (!state.agentsData.length) {
    document.getElementById("agents-list").innerHTML =
      '<div class="empty-state" style="padding:30px 10px"><p>NO AGENTS</p></div>';
    return;
  }
  document.getElementById("agents-list").innerHTML = state.agentsData.map((a) => {
    const hasCrit = a.highly_anomalous > 0, hasAnom = a.anomalous > 0;
    const cls = hasCrit ? "has-critical" : hasAnom ? "has-anomalous" : "";
    const cntCls = hasCrit ? "cnt-red" : hasAnom ? "cnt-orange" : "";
    const isActive = a.agent === state.selectedAgent ? "active" : "";
    const safeName = a.agent.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
    return `<div class="agent-item ${cls} ${isActive}" onclick="loadAgentDetail('${safeName}')">
      <div>
        <div class="agent-name">${a.agent}${a.agent_id ? ` <span style="color:var(--text3);font-size:9px">#${a.agent_id}</span>` : ""}</div>
        <div class="agent-ip">${a.ip || ""}${a.os ? " · " + a.os : ""}</div>
      </div>
      <span class="agent-alert-count ${cntCls}">${a.alert_count}</span>
    </div>`;
  }).join("");
}

export function goToFeedWithFilter(filter) {
  navigate("feed");
  state.feedFilter = filter;
  document
    .querySelectorAll(".filter-btn")
    .forEach((b) => b.classList.toggle("active", b.dataset.filter === filter));
  renderFeed();
}

export function goToAgent(agentName) {
  navigate("endpoints");
  loadAgentDetail(agentName);
}

export async function loadAgentDetail(agentName, { silent = false } = {}) {
  state.selectedAgent = agentName;
  document.querySelectorAll(".agent-item").forEach((el) =>
    el.classList.toggle("active", el.querySelector(".agent-name")?.textContent === agentName)
  );
  const agentMeta = state.agentsData.find((a) => a.agent === agentName) || {};
  document.getElementById("agent-detail-title").innerHTML =
    `${agentName} <span style="font-size:10px;color:var(--text3);font-family:var(--mono2);font-weight:400">${agentMeta.ip || ""} · ${agentMeta.os || ""}</span>`;
  const body = document.getElementById("agent-detail-body");
  // On an explicit click we show the spinner; on a 30s auto-refresh
  // (silent=true) we keep the current view until the new data arrives.
  if (!silent) {
    body.innerHTML = '<div class="loading"><div class="spinner"></div> Loading...</div>';
  }
  let alerts = [];
  try {
    alerts = await fetch("/api/agent/" + encodeURIComponent(agentName) + "?hours=" + (state.timelineHours || 0)).then((r) => r.json());
  } catch (e) {
    if (!silent) body.innerHTML = '<div class="empty-state"><p>FAILED TO LOAD</p></div>';
    return;
  }
  const badge = document.getElementById("agent-detail-badge");
  badge.textContent = alerts.length + " alerts";
  badge.style.display = "";

  if (!alerts.length) {
    body.innerHTML = `<div class="agent-summary" style="grid-template-columns:1fr 1fr;">
      <div class="agent-stat"><div class="agent-stat-val" style="color:var(--green)">0</div><div class="agent-stat-lbl">Alerts</div></div>
      <div class="agent-stat"><div class="agent-stat-val" style="color:var(--green);font-size:16px">CLEAN</div><div class="agent-stat-lbl">Status</div></div>
    </div>
    <div class="empty-state"><svg width="40" height="40" viewBox="0 0 40 40" fill="none"><circle cx="20" cy="20" r="16" stroke="#06d6a0" stroke-width="1.5" fill="none" opacity="0.4"/><path d="M13 20l5 5 9-9" stroke="#06d6a0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    <p style="color:var(--green)">NO ANOMALIES DETECTED</p></div>`;
    return;
  }

  const critCount = alerts.filter((a) => a.verdict === "highly_anomalous").length;
  const anomCount = alerts.filter((a) => a.verdict === "anomalous").length;
  const suspCount = alerts.filter((a) => a.verdict === "suspicious").length;
  // reduce, not Math.max(...spread): a noisy host can have 80k+ alerts and
  // spreading that many args overflows the call stack.
  const maxScore  = alerts.reduce((m, a) => Math.max(m, a.score || 0), 0);
  // Render only the newest ROW_CAP rows (aggregates/charts still use the full
  // set) so the table never lays out tens of thousands of <tr>. Evidence is
  // lazy-loaded per row on expand, so the cap drops no data.
  const ROW_CAP   = 5000;
  const shownRows = alerts.length > ROW_CAP ? alerts.slice(0, ROW_CAP) : alerts;

  // The Overview gauge stays as the all-agents average (set by updateGauge in
  // renderDashboard) — we don't override it with the selected agent's score.
  // The per-agent mini gauge below is drawn by drawAgentGauge.

  const firstT = alerts[alerts.length - 1]?.processed_at || "";
  const lastT  = alerts[0]?.processed_at || "";
  const spanH  = firstT && lastT ? (new Date(lastT) - new Date(firstT)) / 3600000 : 0;
  const groupByDay = spanH > 48;
  const bucketMap = {};
  alerts.forEach((a) => {
    const t = a.processed_at || a.event_time || "";
    if (!t) return;
    const key = groupByDay ? t.slice(0, 10) : t.slice(0, 13);
    if (!bucketMap[key]) bucketMap[key] = { critical: 0, anomalous: 0, suspicious: 0, total: 0 };
    bucketMap[key].total++;
    if (a.verdict === "highly_anomalous") bucketMap[key].critical++;
    else if (a.verdict === "anomalous")   bucketMap[key].anomalous++;
    else                                  bucketMap[key].suspicious++;
  });
  const bKeys = Object.keys(bucketMap).sort();

  const sigMap = {};
  alerts.forEach((a) => {
    const s = (a.signature || "Unknown").slice(0, 60);
    sigMap[s] = (sigMap[s] || 0) + 1;
  });
  const topSigs = Object.entries(sigMap).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const userMap = {};
  alerts.forEach((a) => {
    const u = a.user || "unknown";
    userMap[u] = (userMap[u] || 0) + 1;
  });
  const topUsers = Object.entries(userMap).sort((a, b) => b[1] - a[1]).slice(0, 5);

  function hBar(label, val, maxVal, color) {
    const pct = maxVal > 0 ? Math.max(2, Math.round((val / maxVal) * 100)) : 0;
    return `<div style="margin-bottom:11px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px">
        <span style="font-family:var(--mono2);font-size:11px;color:var(--text2);max-width:78%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${label}">${label}</span>
        <span style="font-family:var(--mono);font-size:12px;font-weight:400;color:${color};margin-left:8px;flex-shrink:0">${val}</span>
      </div>
      <div style="height:8px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden">
        <div style="height:100%;width:${pct}%;background:${color};border-radius:4px;opacity:0.7"></div>
      </div>
    </div>`;
  }

  function buildTimeline() {
    if (!bKeys.length) return `<div style="text-align:center;padding:50px 20px;color:var(--text3);font-family:var(--mono2);font-size:11px">NO TIMELINE DATA</div>`;
    const W = 560, H = 160, pL = 36, pB = 28, pT = 10, pR = 8;
    const cW = W - pL - pR, cH = H - pB - pT, n = bKeys.length;
    const cats = [
      { key: "total",      color: "#00d4ff", label: "Total" },
      { key: "critical",   color: "#ff3b5c", label: "Critical" },
      { key: "anomalous",  color: "#ff8c42", label: "Anomalous" },
      { key: "suspicious", color: "#ffd166", label: "Suspicious" },
    ];
    const maxVal = Math.max(...bKeys.map((k) => bucketMap[k].total), 1);
    const spacing = cW / n;
    const bW = Math.max(3, Math.floor(spacing / cats.length) - 1);
    const groupW = bW * cats.length + (cats.length - 1);
    const bars = bKeys.map((k, i) => {
      const d = bucketMap[k];
      const cx = pL + i * spacing + spacing / 2;
      const startX = cx - groupW / 2;
      return cats.map((c, ci) => {
        const val = d[c.key] || 0;
        const x = startX + ci * (bW + 1);
        const h = Math.max(val > 0 ? 3 : 0, Math.round((val / maxVal) * cH));
        const y = pT + cH - h;
        return `<rect x="${x}" y="${y}" width="${bW}" height="${h}" fill="${c.color}" opacity="0.8" rx="1"><title>${k} · ${c.label}: ${val}</title></rect>`;
      }).join("");
    }).join("");
    const gridLines = [0, 0.25, 0.5, 0.75, 1].map((f) => {
      const y = pT + cH - Math.round(f * cH);
      return `<line x1="${pL}" y1="${y}" x2="${W - pR}" y2="${y}" stroke="#1a2838" stroke-width="0.5" stroke-dasharray="3,3"/>
              <text x="${pL - 3}" y="${y + 3}" font-size="9" fill="#3d5a72" text-anchor="end">${Math.round(f * maxVal)}</text>`;
    }).join("");
    const step = Math.max(1, Math.ceil(n / 7));
    const xLabels = bKeys.filter((_, i) => i % step === 0 || i === n - 1).map((k) => {
      const i = bKeys.indexOf(k);
      const x = pL + i * spacing + spacing / 2;
      const lbl = groupByDay ? k.slice(5) : k.slice(5).replace("T", " ") + "h";
      return `<text x="${x}" y="${H}" font-size="9" fill="#3d5a72" text-anchor="middle">${lbl}</text>`;
    }).join("");
    return `<svg viewBox="0 0 ${W} ${H + 4}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:${H + 4}px;display:block;vertical-align:bottom;">
      ${gridLines}<line x1="${pL}" y1="${pT}" x2="${pL}" y2="${pT + cH}" stroke="#243447" stroke-width="1"/>
      <line x1="${pL}" y1="${pT + cH}" x2="${W - pR}" y2="${pT + cH}" stroke="#243447" stroke-width="1"/>
      ${bars}${xLabels}</svg>
    <div style="display:flex;gap:16px;margin-top:6px;flex-wrap:wrap;padding-bottom:2px">
      ${cats.map((c) => `<span style="font-family:var(--mono2);font-size:10px;color:${c.color};display:flex;align-items:center;gap:5px"><span style="display:inline-block;width:10px;height:8px;background:${c.color};opacity:0.8;border-radius:2px"></span>${c.label}</span>`).join("")}
    </div>`;
  }

  function buildDonut() {
    const total = alerts.length;
    if (!total) return "";
    const sliceData = [
      { val: critCount, color: "#ff3b5c", label: "Critical" },
      { val: anomCount, color: "#ff8c42", label: "Anomalous" },
      { val: suspCount, color: "#ffd166", label: "Suspicious" },
    ].filter((d) => d.val > 0);
    if (!sliceData.length) return "";
    const cx = 80, cy = 80, r = 62, ir = 40;
    let angle = -Math.PI / 2;
    const slices = sliceData.map((d) => {
      const sweep = (d.val / total) * Math.PI * 2;
      const x1 = cx + r * Math.cos(angle), y1 = cy + r * Math.sin(angle);
      angle += sweep;
      const x2 = cx + r * Math.cos(angle), y2 = cy + r * Math.sin(angle);
      const ix1 = cx + ir * Math.cos(angle - sweep), iy1 = cy + ir * Math.sin(angle - sweep);
      const ix2 = cx + ir * Math.cos(angle), iy2 = cy + ir * Math.sin(angle);
      const large = sweep > Math.PI ? 1 : 0;
      const pct = Math.round((d.val / total) * 100);
      return `<path d="M${x1.toFixed(1)},${y1.toFixed(1)} A${r},${r} 0 ${large} 1 ${x2.toFixed(1)},${y2.toFixed(1)} L${ix2.toFixed(1)},${iy2.toFixed(1)} A${ir},${ir} 0 ${large} 0 ${ix1.toFixed(1)},${iy1.toFixed(1)} Z" fill="${d.color}" opacity="0.8"><title>${d.label}: ${d.val} (${pct}%)</title></path>`;
    }).join("");
    const legend = sliceData.map((d, i) => `<rect x="168" y="${10 + i * 22}" width="10" height="10" fill="${d.color}" rx="2" opacity="0.8"/>
      <text x="182" y="${20 + i * 22}" font-size="11" fill="#c8d8e8">${d.label}</text>
      <text x="240" y="${20 + i * 22}" font-size="11" fill="${d.color}" font-weight="600">${d.val}</text>`).join("");
    return `<svg viewBox="0 0 280 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:160px;display:block">
      ${slices}<circle cx="${cx}" cy="${cy}" r="${ir - 1}" fill="#070b12"/>
      <text x="${cx}" y="${cy - 8}" font-size="18" fill="#c8d8e8" text-anchor="middle">${total}</text>
      <text x="${cx}" y="${cy + 10}" font-size="9" fill="#3d5a72" text-anchor="middle">TOTAL</text>
      <text x="${cx}" y="${cy + 24}" font-size="9" fill="#3d5a72" text-anchor="middle">ALERTS</text>
      ${legend}</svg>`;
  }

  body.innerHTML = `
    <div class="agent-summary">
      <div class="agent-stat"><div class="agent-stat-val" style="color:${scoreColor(maxScore)}">${maxScore.toFixed(3)}</div><div class="agent-stat-lbl">Max Score</div></div>
      <div class="agent-stat"><div class="agent-stat-val" style="color:var(--red)">${critCount}</div><div class="agent-stat-lbl">Critical</div></div>
      <div class="agent-stat"><div class="agent-stat-val" style="color:var(--orange)">${anomCount}</div><div class="agent-stat-lbl">Anomalous</div></div>
      <div class="agent-stat"><div class="agent-stat-val" style="color:var(--yellow)">${suspCount}</div><div class="agent-stat-lbl">Suspicious</div></div>
    </div>
    <div class="agent-tabs">
      <div class="agent-tab active" onclick="switchAgentTab('analytics')">◈ ANALYTICS</div>
      <div class="agent-tab" onclick="switchAgentTab('events')">◈ EVENTS</div>
    </div>
    <div id="agent-pane-analytics" class="agent-pane" style="padding:14px;">

      <div style="display:grid;grid-template-columns:200px 1fr;gap:12px;margin-bottom:14px;align-items:stretch;">
        <div class="ep-chart-card" style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:12px 10px;gap:6px;">
          <div style="font-family:var(--mono2);font-size:8px;color:var(--text3);letter-spacing:2px;text-transform:uppercase;align-self:flex-start;">⬡ Risk Meter</div>
          <canvas id="agent-risk-gauge" style="width:180px;height:100px;display:block;"></canvas>
          <div style="text-align:center;margin-top:-6px;">
            <div id="agent-risk-score-val" style="font-family:var(--mono);font-size:26px;line-height:1;color:var(--green);">—</div>
            <div id="agent-risk-label" style="font-family:var(--mono2);font-size:9px;padding:2px 10px;border-radius:3px;border:1px solid rgba(6,214,160,0.4);background:rgba(6,214,160,0.07);color:var(--green);letter-spacing:1px;margin-top:5px;display:inline-block;">Calculating...</div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:10px;">
          <div class="agent-kpi-card" style="background:rgba(255,59,92,0.06);border:1px solid rgba(255,59,92,0.2);border-radius:6px;padding:12px 14px;">
            <div style="font-family:var(--mono2);font-size:8px;color:var(--text3);letter-spacing:2px;margin-bottom:4px;">CRITICAL</div>
            <div style="display:flex;align-items:baseline;gap:8px;">
              <div style="font-family:var(--mono);font-size:26px;color:var(--red);line-height:1;">${critCount}</div>
              <div style="font-family:var(--mono2);font-size:10px;color:rgba(255,59,92,0.55);">${alerts.length > 0 ? Math.round((critCount / alerts.length) * 100) : 0}%</div>
            </div>
          </div>
          <div class="agent-kpi-card" style="background:rgba(255,140,66,0.06);border:1px solid rgba(255,140,66,0.2);border-radius:6px;padding:12px 14px;">
            <div style="font-family:var(--mono2);font-size:8px;color:var(--text3);letter-spacing:2px;margin-bottom:4px;">ANOMALOUS</div>
            <div style="display:flex;align-items:baseline;gap:8px;">
              <div style="font-family:var(--mono);font-size:26px;color:var(--orange);line-height:1;">${anomCount}</div>
              <div style="font-family:var(--mono2);font-size:10px;color:rgba(255,140,66,0.55);">${alerts.length > 0 ? Math.round((anomCount / alerts.length) * 100) : 0}%</div>
            </div>
          </div>
          <div class="agent-kpi-card" style="background:rgba(255,209,102,0.05);border:1px solid rgba(255,209,102,0.18);border-radius:6px;padding:12px 14px;">
            <div style="font-family:var(--mono2);font-size:8px;color:var(--text3);letter-spacing:2px;margin-bottom:4px;">SUSPICIOUS</div>
            <div style="display:flex;align-items:baseline;gap:8px;">
              <div style="font-family:var(--mono);font-size:26px;color:var(--yellow);line-height:1;">${suspCount}</div>
              <div style="font-family:var(--mono2);font-size:10px;color:rgba(255,209,102,0.45);">${alerts.length > 0 ? Math.round((suspCount / alerts.length) * 100) : 0}%</div>
            </div>
          </div>
          <div class="agent-kpi-card" style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.15);border-radius:6px;padding:12px 14px;">
            <div style="font-family:var(--mono2);font-size:8px;color:var(--text3);letter-spacing:2px;margin-bottom:4px;">MAX SCORE</div>
            <div style="display:flex;align-items:baseline;gap:8px;">
              <div style="font-family:var(--mono);font-size:26px;color:${scoreColor(maxScore)};line-height:1;">${maxScore.toFixed(3)}</div>
            </div>
          </div>
        </div>
      </div>

      <div class="ep-chart-card" style="margin-bottom:14px;">
        <div class="ep-chart-title" style="display:flex;justify-content:space-between;align-items:center;">
          <span>Alert Timeline (${groupByDay ? "by day" : "by hour"})</span>
          <div style="display:flex;gap:14px;">
            ${[
              { c: "#00d4ff", l: "Total" },
              { c: "#ff3b5c", l: "Critical" },
              { c: "#ff8c42", l: "Anomalous" },
              { c: "#ffd166", l: "Suspicious" },
            ].map((x) => `<span style="display:flex;align-items:center;gap:5px;font-size:9px;color:${x.c}"><span style="display:inline-block;width:10px;height:4px;background:${x.c};border-radius:2px;opacity:0.8"></span>${x.l}</span>`).join("")}
          </div>
        </div>
        ${buildTimeline()}
      </div>

      <div style="display:grid;grid-template-columns:220px 1fr 1fr;gap:14px;">
        <div class="ep-chart-card" style="display:flex;flex-direction:column;">
          <div class="ep-chart-title">Verdict Breakdown</div>
          <div style="display:flex;justify-content:center;">${buildDonut()}</div>
          <div style="padding:4px 0;margin-top:6px;">
            ${[
              { label: "CRITICAL",  val: critCount, color: "#ff3b5c" },
              { label: "ANOMALOUS", val: anomCount, color: "#ff8c42" },
              { label: "SUSPICIOUS",val: suspCount, color: "#ffd166" },
            ].map((it) => {
              const pct = alerts.length > 0 ? Math.round((it.val / alerts.length) * 100) : 0;
              return `<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 6px;border-bottom:1px solid rgba(26,40,56,0.4);">
                <div style="display:flex;align-items:center;gap:7px;"><div style="width:7px;height:7px;border-radius:50%;background:${it.color};flex-shrink:0;"></div><span style="font-family:var(--mono2);font-size:9px;color:var(--text3);letter-spacing:1.5px;">${it.label}</span></div>
                <div style="display:flex;align-items:baseline;gap:8px;"><span style="font-family:var(--mono);font-size:18px;color:${it.color};">${it.val}</span><span style="font-family:var(--mono2);font-size:10px;color:${it.color};opacity:0.5;">${pct}%</span></div>
              </div>`;
            }).join("")}
          </div>
        </div>
        <div class="ep-chart-card"><div class="ep-chart-title">Top Triggered Rules</div><div style="padding:4px 0">${topSigs.length ? topSigs.map(([s, c]) => hBar(s, c, topSigs[0][1], "var(--accent)")).join("") : '<div style="color:var(--text3);font-family:var(--mono2);font-size:11px;padding:12px">No data</div>'}</div></div>
        <div class="ep-chart-card"><div class="ep-chart-title">Top Users</div><div style="padding:4px 0">${topUsers.length ? topUsers.map(([u, c]) => hBar(u, c, topUsers[0][1], "var(--purple)")).join("") : '<div style="color:var(--text3);font-family:var(--mono2);font-size:11px;padding:12px">No data</div>'}</div></div>
      </div>
    </div>
    <div id="agent-pane-events" class="agent-pane" style="display:none;">
      ${shownRows.length < alerts.length
        ? `<div style="color:var(--text3);font-family:var(--mono2);font-size:11px;padding:6px 2px">Showing newest ${shownRows.length.toLocaleString()} of ${alerts.length.toLocaleString()} events</div>` : ""}
      <div class="feed-table-wrap" style="max-height:520px;">
        <table>
          <thead><tr><th>Time</th><th>User</th><th>Verdict</th><th>Score</th><th>Reasons</th><th>Campaign</th><th>Signature</th></tr></thead>
          <tbody>${feedRows(shownRows, 7, "ag-")}</tbody>
        </table>
      </div>
    </div>`;

  shownRows.forEach((a) => {
    const id = "ev-ag-" + (a.event_id || a.processed_at + a.user);
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });

  requestAnimationFrame(() => drawAgentGauge(critCount, anomCount, alerts.length, maxScore));
}

// ── Per-agent risk gauge (rendered inside the Endpoints drilldown) ──
function drawAgentGauge(critCount, anomCount, total, maxScore) {
  const canvas = document.getElementById("agent-risk-gauge");
  if (!canvas) return;

  const dpr  = window.devicePixelRatio || 1;
  const cssW = 220, cssH = 120;
  canvas.width  = cssW * dpr;
  canvas.height = cssH * dpr;
  canvas.style.width  = cssW + "px";
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const score = Math.min(100, Math.round(
    (critCount / Math.max(total, 1)) * 60 +
    (anomCount / Math.max(total, 1)) * 20 +
    maxScore * 15
  ));

  const cx = cssW / 2, cy = cssH - 10, r = 80;

  const segs = [
    { min: 0,  max: 30,  color: "#06d6a0" },
    { min: 30, max: 60,  color: "#ffd166" },
    { min: 60, max: 80,  color: "#ff8c42" },
    { min: 80, max: 100, color: "#ff3b5c" },
  ];
  segs.forEach((seg) => {
    const a1 = Math.PI + (seg.min / 100) * Math.PI;
    const a2 = Math.PI + (seg.max / 100) * Math.PI;
    ctx.beginPath(); ctx.arc(cx, cy, r, a1, a2);
    ctx.lineWidth = 14; ctx.strokeStyle = seg.color;
    ctx.globalAlpha = 0.2; ctx.stroke();
  });
  ctx.globalAlpha = 1;

  const fillCol =
    score < 30 ? "#06d6a0" :
    score < 60 ? "#ffd166" :
    score < 80 ? "#ff8c42" : "#ff3b5c";
  const fillEnd = Math.PI + (Math.min(score, 100) / 100) * Math.PI;
  ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, fillEnd);
  ctx.lineWidth = 14; ctx.strokeStyle = fillCol;
  ctx.globalAlpha = 0.9; ctx.lineCap = "round"; ctx.stroke();
  ctx.globalAlpha = 1;

  for (let i = 0; i <= 10; i++) {
    const a = Math.PI + (i / 10) * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx + (r - 9) * Math.cos(a), cy + (r - 9) * Math.sin(a));
    ctx.lineTo(cx + (r + 2) * Math.cos(a), cy + (r + 2) * Math.sin(a));
    ctx.strokeStyle = "#1a2838"; ctx.lineWidth = 1.5; ctx.stroke();
    if (i % 5 === 0) {
      ctx.fillStyle = "#3d5a72";
      ctx.font = "8px JetBrains Mono,monospace";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(i * 10, cx + (r + 14) * Math.cos(a), cy + (r + 14) * Math.sin(a));
    }
  }

  const needleAngle = Math.PI + (Math.min(score, 100) / 100) * Math.PI;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + (r - 8) * Math.cos(needleAngle), cy + (r - 8) * Math.sin(needleAngle));
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.lineCap = "round"; ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fillStyle = "#fff"; ctx.fill();

  const scoreEl = document.getElementById("agent-risk-score-val");
  const labelEl = document.getElementById("agent-risk-label");
  if (scoreEl) {
    scoreEl.textContent = score;
    scoreEl.style.color = fillCol;
  }
  if (labelEl) {
    const [lbl, brd] =
      score < 20 ? ["Minimal Risk",  "rgba(6,214,160,0.4)"] :
      score < 50 ? ["Moderate Risk", "rgba(255,209,102,0.4)"] :
      score < 75 ? ["High Risk",     "rgba(255,140,66,0.4)"] :
                   ["Critical Risk", "rgba(255,59,92,0.4)"];
    labelEl.textContent     = lbl;
    labelEl.style.borderColor = brd;
    labelEl.style.color       = fillCol;
    labelEl.style.background  = fillCol.replace(")", ",0.07)").replace("rgb", "rgba");
  }
}

export function switchAgentTab(tab) {
  document.querySelectorAll(".agent-tab").forEach((t) =>
    t.classList.toggle("active", t.textContent.toLowerCase().includes(tab))
  );
  document.getElementById("agent-pane-analytics").style.display = tab === "analytics" ? "block" : "none";
  document.getElementById("agent-pane-events").style.display    = tab === "events"    ? "block" : "none";
}
