// False Positives tab — lists every FP-marked alert with the original alert
// data merged in, plus an "Unmark" button to restore each one to the feed.
// Also renders the pattern-FP table — one row per rule description that
// auto-suppresses every future alert with the same description.

import { state } from "../state.js";
import { fmtTime, fmtDate, scoreColor, verdictClass, verdictLabel } from "../helpers.js";

export function renderFalsePositives() {
  renderFpPatterns();
  renderFpEvents();
}

function renderFpPatterns() {
  const tbody = document.getElementById("fp-pat-tbody");
  const badge = document.getElementById("fp-pat-badge");
  if (!tbody) return;

  const pats = state.fpPatterns || [];
  if (badge) badge.textContent = pats.length;

  if (!pats.length) {
    tbody.innerHTML = `<tr><td colspan="5"><div class="empty-state" style="padding:20px 12px">
      <p>NO AUTO-SUPPRESSION PATTERNS</p>
      <p style="font-family:var(--mono2);font-size:10px;color:var(--text3);margin-top:8px;letter-spacing:0.6px">
        Click <b>Mark FP</b> on any alert to auto-suppress every future alert with the same rule description.
      </p>
    </div></td></tr>`;
    return;
  }

  tbody.innerHTML = pats.map((p) => {
    const idSafe = (p.id || "").replace(/'/g, "\\'");
    const desc = p.rule_description || "—";
    return `<tr class="fp-row-record">
      <td class="sig-cell"><span class="fp-pat-desc">${escapeHtml(desc)}</span></td>
      <td class="fp-reason-cell">${p.reason ? `<span class="fp-reason-text">${escapeHtml(p.reason)}</span>` : '<span class="fp-reason-empty">—</span>'}</td>
      <td><span class="fp-pat-matched">${(p.matched || 0).toLocaleString()}</span></td>
      <td class="time-cell">${fmtDate(p.marked_at)}</td>
      <td><button class="fp-restore-btn" onclick="unmarkPatternFP('${idSafe}')">× Remove</button></td>
    </tr>`;
  }).join("");
}

function renderFpEvents() {
  const tbody = document.getElementById("fp-tbody");
  const badge = document.getElementById("fp-tab-badge");
  const navBadge = document.getElementById("fp-nav-badge");
  if (!tbody) return;

  const fps = state.falsePositives || [];
  if (badge)    badge.textContent    = fps.length;
  if (navBadge) navBadge.textContent = fps.length + ((state.fpPatterns || []).length);

  if (!fps.length) {
    tbody.innerHTML = `<tr><td colspan="8"><div class="empty-state" style="padding:30px 12px">
      <p>NO FALSE POSITIVES YET</p>
      <p style="font-family:var(--mono2);font-size:10px;color:var(--text3);margin-top:8px;letter-spacing:0.6px">
        Click the <span class="fp-btn-inline" style="cursor:default;pointer-events:none">FP</span> button on any alert in the feed to start.
      </p>
    </div></td></tr>`;
    return;
  }

  tbody.innerHTML = fps.map((rec) => {
    const a = rec.alert || {};
    const eid = rec.event_id;
    const eidSafe = (eid || "").replace(/'/g, "\\'");
    return `<tr class="fp-row-record">
      <td class="time-cell">${fmtDate(rec.marked_at)}</td>
      <td>${fmtTime(a.processed_at)}</td>
      <td class="user-cell">${a.user || "—"}</td>
      <td>${a.host || "—"}</td>
      <td><span class="verdict-badge ${verdictClass(a.verdict)}">${verdictLabel(a.verdict)}</span></td>
      <td><span class="score-num" style="color:${scoreColor(a.score || 0)}">${(a.score || 0).toFixed(3)}</span></td>
      <td class="sig-cell">${a.signature || "—"}</td>
      <td class="fp-reason-cell">${rec.reason ? `<span class="fp-reason-text">${escapeHtml(rec.reason)}</span>` : '<span class="fp-reason-empty">—</span>'}</td>
      <td><button class="fp-restore-btn" onclick="unmarkFP('${eidSafe}')">↺ Restore</button></td>
    </tr>`;
  }).join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
