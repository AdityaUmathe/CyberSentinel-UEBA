// Incidents panel — the "calm" UEBA view.
//
// Each card is ONE incident: alerts rolled up by (entity, time-window) server-
// side (/api/incidents). A brute-force burst that is 1,000+ rows in the Alert
// Feed collapses into a single incident here with an event count. Clicking the
// entity opens its existing User Risk detail for full per-event drill-down, so
// nothing in the feed/evidence/FP workflow is duplicated or lost.

import { state } from "../state.js";
import { fmtDate } from "../helpers.js";

const VERD_COLOR = {
  highly_anomalous: "var(--red)",
  anomalous: "var(--orange)",
  suspicious: "var(--yellow)",
};

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function renderIncidents() {
  const list  = document.getElementById("incidents-list");
  const data  = state.incidentsData || [];
  const navB  = document.getElementById("inc-badge");
  const panB  = document.getElementById("inc-badge2");
  if (navB) navB.textContent = data.length;
  if (panB) panB.textContent = data.length;
  if (!list) return;

  if (!data.length) {
    list.innerHTML =
      '<div class="empty-state"><p>NO INCIDENTS IN THIS WINDOW</p></div>';
    return;
  }

  list.innerHTML = data.map((inc) => {
    const vc      = VERD_COLOR[inc.top_verdict] || "var(--yellow)";
    const ent     = String(inc.entity || "unknown");
    // Safe for a single-quoted onclick string arg.
    const entArg  = ent.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
    const reasons = (inc.reasons || [])
      .map((r) => `<span class="user-chip">${esc(r.replace(/_/g, " "))}</span>`).join("");
    const hosts   = (inc.hosts || [])
      .map((h) => `<span class="host-chip">${esc(h)}</span>`).join("");
    const camps   = (inc.campaign_ids || [])
      .map((c) => `<span class="host-chip">⛓ ${esc(c)}</span>`).join("");
    const sigs    = (inc.signatures || [])
      .map((s) => `<div class="sig-item">⚡ ${esc(s)}</div>`).join("");
    return `<div class="campaign-card">
      <div class="campaign-header">
        <span class="campaign-id" style="cursor:pointer" title="Open ${esc(ent)} detail"
              onclick="window.__goToEntity('${entArg}')">▸ ${esc(ent)}</span>
        <div class="campaign-meta">
          <span style="color:${vc}">${esc((inc.top_verdict || "").replace(/_/g, " ").toUpperCase())}</span>
          <span><b>${inc.count}</b> events</span>
          <span>peak ${(inc.max_score || 0).toFixed(3)}</span>
          <span>${fmtDate(inc.first_seen)} → ${fmtDate(inc.last_seen)}</span>
        </div>
      </div>
      <div class="campaign-body">
        <div class="campaign-users">${reasons}${hosts}${camps}</div>
        ${sigs ? `<div class="campaign-sigs">${sigs}</div>` : ""}
      </div>
    </div>`;
  }).join("");
}
