// Campaigns panel — attack campaign cards with mini timeline bars.

import { state } from "../state.js";
import { fmtDate } from "../helpers.js";

export function renderCampaigns() {
  document.getElementById("camp-badge").textContent = state.campaignsData.length;
  if (!state.campaignsData.length) {
    document.getElementById("campaigns-list").innerHTML =
      '<div class="empty-state"><p>NO CAMPAIGNS DETECTED YET</p></div>';
    return;
  }
  document.getElementById("campaigns-list").innerHTML = state.campaignsData.map((c) => {
    const verdColor =
      c.top_verdict === "highly_anomalous"
        ? "var(--red)"
        : c.top_verdict === "anomalous"
        ? "var(--orange)"
        : "var(--yellow)";
    const tl = c.timeline || [];
    const maxS = Math.max(...tl.map((t) => t.score || 0), 0.1);
    const tlBars = tl.map((t) => {
      const h = Math.max(3, Math.round((t.score / maxS) * 36));
      const col =
        t.verdict === "highly_anomalous"
          ? "var(--red)"
          : t.verdict === "anomalous"
          ? "var(--orange)"
          : "var(--yellow)";
      return `<div class="timeline-bar" style="height:${h}px;background:${col}" title="${t.user} ${(t.score || 0).toFixed(3)}"></div>`;
    }).join("");
    const cidSafe = (c.campaign_id || "").replace(/'/g, "\\'").replace(/\\/g, "\\\\");
    return `<div class="campaign-card">
      <div class="campaign-header">
        <span class="campaign-id">${c.campaign_id}</span>
        <div class="campaign-meta">
          <span style="color:${verdColor}">${(c.top_verdict || "").replace(/_/g, " ").toUpperCase()}</span>
          <span>${c.alert_count} alerts</span><span>${c.users.length} users</span>
          <span>${fmtDate(c.first_seen)} → ${fmtDate(c.last_seen)}</span>
        </div>
        <button class="fp-camp-btn" title="Mark every alert in this campaign as a false positive"
          onclick="window.__markCampaignFpPrompt('${cidSafe}', ${c.alert_count})">⚑ Mark Campaign FP</button>
      </div>
      <div class="campaign-body">
        <div class="campaign-users">
          ${c.users.map((u) => `<span class="user-chip">${u}</span>`).join("")}
          ${c.hosts.map((h) => `<span class="host-chip">${h}</span>`).join("")}
        </div>
        ${c.signatures.length ? `<div class="campaign-sigs">${c.signatures.map((s) => `<div class="sig-item">⚡ ${s}</div>`).join("")}</div>` : ""}
        <div class="campaign-timeline">${tlBars}</div>
        <div style="margin-top:6px;font-family:var(--mono2);font-size:10px;color:var(--text3)">Top reason: <span style="color:var(--text2)">${(c.top_reason || "—").replace(/_/g, " ")}</span></div>
      </div>
    </div>`;
  }).join("");
}
