// MITRE ATT&CK coverage heatmap.
//
// Aggregates state.feedData into { tactic -> { technique -> count } } and
// renders one row per tactic with one cell per observed technique. Cells are
// colored by row-local count so a "busy" tactic doesn't wash out the others.
// Clicking a cell navigates to /feed and applies the matching tactic filter.

import { state } from "../state.js";
import { navigate } from "../router.js";
import { renderFeed } from "./feed.js";

// MITRE Enterprise tactics in their canonical kill-chain order. Tactics that
// don't appear in the data are hidden; tactics in the data but not on this
// list still render (appended at the end).
const TACTIC_ORDER = [
  "Reconnaissance",
  "Resource Development",
  "Initial Access",
  "Execution",
  "Persistence",
  "Privilege Escalation",
  "Defense Evasion",
  "Credential Access",
  "Discovery",
  "Lateral Movement",
  "Collection",
  "Command and Control",
  "Exfiltration",
  "Impact",
];

function _aggregate(alerts) {
  const matrix = {};   // tactic -> technique -> count
  let coverageCount = 0;
  alerts.forEach((a) => {
    const tactics    = a.mitre_tactic || [];
    const techniques = (a.evidence && a.evidence.signature && a.evidence.signature.mitre_techniques) || [];
    if (!tactics.length || !techniques.length) return;
    coverageCount++;
    tactics.forEach((tac) => {
      if (!matrix[tac]) matrix[tac] = {};
      techniques.forEach((tech) => {
        matrix[tac][tech] = (matrix[tac][tech] || 0) + 1;
      });
    });
  });
  return { matrix, coverageCount };
}

function _sortedTactics(matrix) {
  const present = new Set(Object.keys(matrix));
  const ordered = TACTIC_ORDER.filter((t) => present.has(t));
  const extras  = [...present].filter((t) => !TACTIC_ORDER.includes(t)).sort();
  return [...ordered, ...extras];
}

function _intensity(count, rowMax) {
  if (count <= 0 || rowMax <= 0) return 0;
  // Square-root scaling so a single huge cell doesn't dominate.
  return Math.min(1, 0.18 + 0.82 * Math.sqrt(count / rowMax));
}

export function renderMitre() {
  const body = document.getElementById("mitre-matrix-body");
  if (!body) return;
  const { matrix, coverageCount } = _aggregate(state.feedData);
  const tactics = _sortedTactics(matrix);

  const badge = document.getElementById("mitre-matrix-badge");
  if (badge) {
    badge.textContent = coverageCount
      ? `${coverageCount.toLocaleString()} alerts mapped`
      : "no MITRE coverage";
  }

  if (!tactics.length) {
    body.innerHTML = `<div class="empty-state" style="padding:30px 12px;">
      <p>NO MITRE ATT&CK MAPPINGS IN CURRENT DATA</p>
    </div>`;
    return;
  }

  body.innerHTML = tactics.map((tac) => {
    const techs = matrix[tac] || {};
    const entries = Object.entries(techs).sort((a, b) => b[1] - a[1]);
    const rowMax = entries.length ? entries[0][1] : 0;
    const total  = entries.reduce((s, [, c]) => s + c, 0);
    const cells = entries.map(([tech, count]) => {
      const intensity = _intensity(count, rowMax);
      const bg = `rgba(0,212,255,${intensity.toFixed(3)})`;
      const tacSafe  = _attr(tac);
      const techSafe = _attr(tech);
      return `<button class="mitre-cell" data-tactic="${tacSafe}" data-tech="${techSafe}"
                style="background:${bg};"
                title="${_attr(tac)} · ${_attr(tech)} — ${count} alert${count === 1 ? "" : "s"}">
        <span class="mitre-cell-id">${_text(tech)}</span>
        <span class="mitre-cell-count">${count}</span>
      </button>`;
    }).join("");
    return `<div class="mitre-row-grid">
      <div class="mitre-tac-label">
        <div class="mitre-tac-name">${_text(tac)}</div>
        <div class="mitre-tac-meta">${entries.length} technique${entries.length === 1 ? "" : "s"} · ${total} alert${total === 1 ? "" : "s"}</div>
      </div>
      <div class="mitre-row-cells">${cells}</div>
    </div>`;
  }).join("");

  // Wire click → navigate to /feed with the tactic filter applied.
  body.querySelectorAll(".mitre-cell").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tac = btn.dataset.tactic;
      state.feedTactics.clear();
      if (tac) state.feedTactics.add(tac);
      state.feedRenderLimit = 200;
      navigate("feed");
      renderFeed();
    });
  });
}

function _text(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function _attr(s) {
  return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
