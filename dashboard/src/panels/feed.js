// Alert feed + evidence panel + jump-to-alert behavior.

import { state } from "../state.js";
import { navigate } from "../router.js";
import { fmtTime, scoreColor, verdictClass, verdictLabel, reasonTag } from "../helpers.js";

function evRow(label, value, highlight) {
  if (value === null || value === undefined || value === "") return "";
  const val = Array.isArray(value) ? value.join(", ") : value;
  return `<div class="ev-row"><span class="ev-label">${label}</span><span class="${
    highlight ? "ev-val ev-highlight" : "ev-val"
  }">${val}</span></div>`;
}

export function buildEvidencePanel(ev, alertId, colSpan, eventId, signatureId, alertUser, alertHost) {
  colSpan = colSpan || 9;
  // Persist expanded/collapsed state across re-renders.
  const expanded = state.expandedRows.has(alertId);
  const evStyle  = expanded ? "" : ' style="display:none"';
  if (!ev || !Object.keys(ev).length) {
    return `<tr class="ev-panel-row" id="ev-${alertId}"${evStyle}><td colspan="${colSpan}"><div class="ev-empty">No evidence data — restart engine to generate evidence.</div></td></tr>`;
  }
  const sig = ev.signature || {}, raw = ev.raw_event || {}, base = ev.baseline || {}, hist = ev.history || {};
  const mitreIds     = (sig.mitre_ids || []).map((t) => `<span class="mitre-tag mitre-id">${t}</span>`).join("");
  const mitreTactics = (sig.mitre_tactics || []).map((t) => `<span class="mitre-tag">${t}</span>`).join("");
  const mitreTechs   = (sig.mitre_techniques || []).map((t) => `<span class="mitre-tag mitre-tech">${t}</span>`).join("");
  const privs = raw.privileges
    ? raw.privileges.map((p) => `<span class="priv-tag">${p}</span>`).join("")
    : null;
  const errH = base.error_vs_threshold && parseFloat(base.error_vs_threshold) > 5;
  const eidSafe = (eventId || "").replace(/'/g, "\\'").replace(/\\/g, "\\\\");
  // `evStyle` is set above based on state.expandedRows so an expanded panel
  // stays visible after a full re-render.
  // Build the inferred fingerprint that the server will auto-create as a
  // suppression pattern when this alert is marked FP. Shown to the analyst
  // upfront so the "blast radius" is obvious before they click.
  const sigForFp  = signatureId || sig.rule_id || "";
  const userForFp = (alertUser || "").trim() || "*";
  const hostForFp = (alertHost || "").trim() || "*";
  const escFp = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const fpHint = (eventId && sigForFp) ? `
      <div class="fp-form-hint">
        Marking this alert as FP will <b>also auto-suppress</b> future alerts where
        rule is <span class="fp-pat-chip"><b>${escFp(sigForFp)}</b></span>
        <b>AND</b> user is <span class="fp-pat-chip"><b>${escFp(userForFp)}</b></span>
        (host doesn't matter). Other rules from this user are unaffected. Manage in the False Positives tab.
      </div>` : "";
  const fpForm = eventId ? `
    <div class="fp-form" onclick="event.stopPropagation()">
      <div class="fp-form-title">Mark this alert as a false positive</div>
      <div class="fp-form-row">
        <input id="fp-reason-${alertId}" class="fp-form-input" placeholder="Reason (optional) — e.g. expected RDP from admin desk" maxlength="240"/>
        <button class="fp-form-submit" onclick="window.__markFpFromForm('${eidSafe}','${alertId}')">Mark FP</button>
      </div>
      ${fpHint}
    </div>` : "";
  // Event ID header bar — the first thing analysts need to pivot into the
  // SIEM. Click-to-copy for one-step paste.
  const eidDisplay = eventId || "—";
  const eidHeader = eventId ? `
    <div class="ev-eid-bar" onclick="event.stopPropagation()">
      <div class="ev-eid-block">
        <span class="ev-eid-label">SIEM Event ID</span>
        <code class="ev-eid-value" id="eid-${alertId}">${eidDisplay}</code>
      </div>
      <button class="ev-eid-copy" type="button" aria-label="Copy event ID"
              onclick="window.__copyEvId('${eidSafe}', this)">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <rect x="4.5" y="4.5" width="8" height="9" rx="1.4" stroke="currentColor" stroke-width="1.4"/>
          <path d="M11.5 4.5V3a1 1 0 0 0-1-1H3.5a1 1 0 0 0-1 1v7.5a1 1 0 0 0 1 1H5" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" fill="none"/>
        </svg>
        <span>Copy</span>
      </button>
    </div>` : "";
  return `<tr class="ev-panel-row" id="ev-${alertId}"${evStyle}>
    <td colspan="${colSpan}">
      <div class="ev-panel">
        ${eidHeader}
        <div class="ev-grid">
          <div class="ev-section">
            <div class="ev-section-title">◈ Signature &amp; Rule</div>
            ${evRow("Rule ID", sig.rule_id)}${evRow("Description", sig.description)}${evRow("Severity", sig.severity_level)}
            ${sig.mitre_ids?.length ? `<div class="ev-row"><span class="ev-label">MITRE ATT&CK</span><span class="ev-val">${mitreIds}</span></div>` : ""}
            ${sig.mitre_tactics?.length ? `<div class="ev-row"><span class="ev-label">Tactics</span><span class="ev-val">${mitreTactics}</span></div>` : ""}
            ${sig.mitre_techniques?.length ? `<div class="ev-row"><span class="ev-label">Techniques</span><span class="ev-val">${mitreTechs}</span></div>` : ""}
          </div>
          <div class="ev-section">
            <div class="ev-section-title">◈ Raw Event Details</div>
            ${evRow("Category", raw.event_category)}${evRow("Outcome", raw.event_outcome)}
            ${evRow("Host", raw.host)}${evRow("Host IP", raw.host_ip)}${evRow("Source IP", raw.source_ip)}
            ${evRow("Process", raw.process_name)}${evRow("Logon Type", raw.logon_type)}
            ${evRow("Windows Event ID", raw.event_id_windows)}
            ${raw.failures_5m ? evRow("Failures (5m)", raw.failures_5m, raw.failures_5m > 50) : ""}
            ${raw.user_events_5m ? evRow("User Events (5m)", raw.user_events_5m, raw.user_events_5m > 500) : ""}
            ${raw.unique_dests_1h ? evRow("Unique Dests (1h)", raw.unique_dests_1h, raw.unique_dests_1h > 10) : ""}
            ${raw.is_tor ? evRow("TOR Exit Node", "⚠ YES", true) : ""}
            ${raw.threat_detected ? evRow("Threat Intel", "⚠ MATCH", true) : ""}
            ${privs ? `<div class="ev-row ev-row-block"><span class="ev-label">Privileges</span><div class="ev-privs">${privs}</div></div>` : ""}
          </div>
          <div class="ev-section">
            <div class="ev-section-title">◈ Baseline vs Now</div>
            ${base.note ? `<div class="ev-note">${base.note}</div>` : ""}
            ${evRow("Typical Hour", base.typical_hour)}${evRow("Current Hour", base.current_hour)}
            ${evRow("Hour Deviation", base.hour_deviation)}
            ${evRow("Typical Risk Score", base.typical_risk_score)}${evRow("Current Risk Score", base.current_risk_score)}
            ${evRow("Risk Deviation", base.risk_deviation, parseFloat(base.risk_deviation || 0) > 20)}
            ${evRow("Event Rate", base.events_multiplier, (base.events_multiplier || "").includes("x") && parseFloat(base.events_multiplier) > 3)}
            ${evRow("Business Hours", base.is_business_hours ? "Yes" : "No")}${evRow("Day", base.day_of_week)}
            <div class="ev-divider"></div>
            ${evRow("AE Recon Error", base.autoencoder_error)}${evRow("AE Threshold", base.autoencoder_threshold)}
            ${evRow("Error vs Threshold", base.error_vs_threshold, errH)}
            ${evRow("Model Used", base.model_used)}${evRow("IF Raw Score", base.if_raw_score)}
          </div>
          <div class="ev-section">
            <div class="ev-section-title">◈ Historical Context</div>
            ${evRow("First Seen", hist.first_seen ? hist.first_seen.replace("T", " ").slice(0, 19) + " UTC" : "First time")}
            ${evRow("Last Seen", hist.last_seen ? hist.last_seen.replace("T", " ").slice(0, 19) + " UTC" : "—")}
            ${evRow("Total Events Seen", hist.total_events_seen?.toLocaleString())}
            ${evRow("Alerts Today", hist.alerts_today, hist.alerts_today > 10)}
            <div class="ev-divider"></div>
            <div class="ev-label" style="margin-bottom:6px">Anomaly Reasons</div>
            <div>${(hist.anomaly_reasons || []).map((r) => `<span class="reason-tag">${r.replace(/_/g, " ")}</span>`).join("")}</div>
            ${hist.campaign_id ? evRow("Campaign", hist.campaign_id) : ""}
          </div>
        </div>
        ${fpForm}
      </div>
    </td>
  </tr>`;
}

export function toggleEvidence(alertId) {
  const row = document.getElementById("ev-" + alertId);
  if (!row) return;
  if (state.expandedRows.has(alertId)) {
    row.style.display = "none";
    state.expandedRows.delete(alertId);
    const btn = document.querySelector(`tr[data-id="${alertId}"] .ev-toggle`);
    if (btn) btn.textContent = "▶";
  } else {
    row.style.display = "table-row";
    state.expandedRows.add(alertId);
    const btn = document.querySelector(`tr[data-id="${alertId}"] .ev-toggle`);
    if (btn) btn.textContent = "▼";
  }
}

export function feedRows(alerts, colSpan, idPrefix) {
  idPrefix = idPrefix || "";
  const rows = [];
  alerts.forEach((a) => {
    const alertId = idPrefix + (a.event_id || a.processed_at + a.user);
    const isFp = !!a.fp;
    const rowCls = "feed-row" + (isFp ? " fp-row" : "");
    // Chevron reflects current expanded state — set by toggleEvidence and
    // preserved across SSE/full re-renders via state.expandedRows.
    const expanded = state.expandedRows.has(alertId);
    const chev = expanded ? "▼" : "▶";
    const eidSafe = (a.event_id || "").replace(/'/g, "\\'").replace(/\\/g, "\\\\");
    const fpBtn = a.event_id
      ? (isFp
          ? `<button class="fp-btn-inline fp-btn-restore" title="Restore from false positives — was marked: ${escapeAttr(a.fp.reason || "(no reason)")}" onclick="event.stopPropagation();unmarkFP('${eidSafe}')">FP ↺</button>`
          : `<button class="fp-btn-inline" title="Mark as false positive (no reason). Use the evidence panel to add a reason." onclick="event.stopPropagation();markFP('${eidSafe}')">FP</button>`)
      : "";
    rows.push(`
      <tr class="${rowCls}" data-id="${alertId}" style="cursor:pointer">
        <td class="time-cell"><span class="ev-toggle" style="color:var(--text3);margin-right:6px;font-size:10px">${chev}</span>${fmtTime(a.processed_at)}</td>
        <td class="user-cell">${a.user || "—"}</td>
        ${colSpan > 7 ? `<td class="host-cell">${a.host || "—"}</td>` : ""}
        <td><span class="verdict-badge ${verdictClass(a.verdict)}">${verdictLabel(a.verdict)}</span></td>
        <td><div class="score-bar"><span class="score-num" style="color:${scoreColor(a.score)}">${(a.score || 0).toFixed(3)}</span><div class="score-track"><div class="score-fill" style="width:${Math.round((a.score || 0) * 100)}%;background:${scoreColor(a.score)}"></div></div></div></td>
        <td>${(a.reasons || []).map(reasonTag).join("")}</td>
        <td>${a.campaign_id ? `<span class="camp-tag">${a.campaign_id}</span>` : "—"}</td>
        <td class="sig-cell"><div class="sig-cell-inner"><span class="sig-text">${a.signature || "—"}</span>${fpBtn}</div></td>
      </tr>`);
    rows.push(buildEvidencePanel(a.evidence, alertId, colSpan, a.event_id, a.signature_id, a.user, a.host));
  });
  return rows.join("");
}

function escapeAttr(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// Apply every active feed filter (verdict + free-text search + MITRE tactic +
// minimum severity) to state.feedData and return the matching subset.
export function filteredFeed() {
  const needle = (state.feedSearch || "").trim().toLowerCase();
  const tacticFilter = state.feedTactics && state.feedTactics.size > 0;
  const sevMin = state.feedSeverityMin || 0;

  return state.feedData.filter((a) => {
    if (state.feedFilter !== "all" && a.verdict !== state.feedFilter) return false;
    if (sevMin > 0 && (a.severity ?? 0) < sevMin) return false;
    if (tacticFilter) {
      const tactics = a.mitre_tactic || [];
      if (!tactics.some((t) => state.feedTactics.has(t))) return false;
    }
    if (needle) {
      const hay = [
        a.user, a.host, a.host_ip, a.signature, a.signature_id, a.event_id, a.campaign_id,
        ...(a.reasons || []),
        ...(a.mitre_tactic || []),
      ].filter(Boolean).join(" ").toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    return true;
  });
}

// Build the unique sorted list of MITRE tactics present in the current feed.
export function availableTactics() {
  const set = new Set();
  state.feedData.forEach((a) => (a.mitre_tactic || []).forEach((t) => t && set.add(t)));
  return [...set].sort();
}

export function renderFeed() {
  const matching = filteredFeed();
  const total    = state.feedData.length;
  const limit    = state.feedRenderLimit || matching.length;
  const visible  = matching.slice(0, limit);

  const fb = document.getElementById("feed-badge");
  if (fb) fb.textContent = matching.length;
  const ov = document.getElementById("ov-badge");
  if (ov) ov.textContent = total;

  document.getElementById("overview-tbody").innerHTML =
    state.feedData.slice(0, 20).map((a) => {
      const alertId = a.event_id || a.processed_at + a.user;
      return `<tr class="ov-row" onclick="jumpToAlert('${alertId}','${a.verdict || ""}')">
        <td class="time-cell" style="padding-left:16px">${fmtTime(a.processed_at)}</td>
        <td class="user-cell">${a.user || "—"}</td>
        <td><span class="verdict-badge ${verdictClass(a.verdict)}">${verdictLabel(a.verdict)}</span></td>
        <td><div class="score-bar"><span class="score-num" style="color:${scoreColor(a.score)}">${(a.score || 0).toFixed(3)}</span><div class="score-track"><div class="score-fill" style="width:${Math.round((a.score || 0) * 100)}%;background:${scoreColor(a.score)}"></div></div></div></td>
        <td class="sig-cell">${a.signature || "—"}</td>
      </tr>`;
    }).join("") || '<tr><td colspan="5" class="loading">No alerts yet</td></tr>';

  // Footer (Showing X of Y matching · Load more)
  _renderFeedFooter(visible.length, matching.length, total);

  if (!visible.length) {
    document.getElementById("feed-tbody").innerHTML =
      '<tr><td colspan="9"><div class="empty-state"><p>NO ALERTS MATCHING CURRENT FILTERS</p></div></td></tr>';
    return;
  }
  // Each feedRows row already emits the correct chevron + display style based
  // on state.expandedRows — no post-pass needed.
  document.getElementById("feed-tbody").innerHTML = feedRows(visible, 9);
}

// Insert a single new alert at the top of the live feed table without
// rebuilding the rest of the tbody. Preserves scroll position, input focus,
// and existing expanded panels. Used by the SSE handler so the page doesn't
// "refresh" every time a new alert arrives.
export function prependFeedRow(alert) {
  if (!alert) return;

  // Filter check — only paint the row if it matches current filter state.
  if (!_alertMatchesFilters(alert)) {
    _refreshFooterFromState();
    return;
  }

  const tbody = document.getElementById("feed-tbody");
  if (!tbody) return;

  // If the empty-state row is showing, swap it out for a real tbody first.
  const empty = tbody.querySelector(".empty-state");
  if (empty) {
    tbody.innerHTML = "";
  }

  const html = feedRows([alert], 9);
  tbody.insertAdjacentHTML("afterbegin", html);

  // Cap the rendered DOM size — each alert contributes 2 <tr>s (data + ev).
  const max = (state.feedRenderLimit || 200) * 2;
  while (tbody.children.length > max) {
    tbody.removeChild(tbody.lastElementChild);
  }

  _refreshFooterFromState();
}

// Recompute Showing X of Y / badges from current state without touching rows.
function _refreshFooterFromState() {
  const matched = filteredFeed().length;
  const total   = state.feedData.length;
  const fb = document.getElementById("feed-badge");
  if (fb) fb.textContent = matched;
  const ov = document.getElementById("ov-badge");
  if (ov) ov.textContent = total;

  const tbody = document.getElementById("feed-tbody");
  const rendered = tbody ? tbody.querySelectorAll("tr.feed-row").length : 0;
  _renderFeedFooter(rendered, matched, total);
}

// Lightweight version of filteredFeed() that tests a single alert.
function _alertMatchesFilters(a) {
  if (state.feedFilter !== "all" && a.verdict !== state.feedFilter) return false;
  if ((state.feedSeverityMin || 0) > 0 && (a.severity ?? 0) < state.feedSeverityMin) return false;
  if (state.feedTactics && state.feedTactics.size > 0) {
    const tactics = a.mitre_tactic || [];
    if (!tactics.some((t) => state.feedTactics.has(t))) return false;
  }
  const needle = (state.feedSearch || "").trim().toLowerCase();
  if (needle) {
    const hay = [
      a.user, a.host, a.host_ip, a.signature, a.signature_id, a.event_id, a.campaign_id,
      ...(a.reasons || []),
      ...(a.mitre_tactic || []),
    ].filter(Boolean).join(" ").toLowerCase();
    if (!hay.includes(needle)) return false;
  }
  return true;
}

function _renderFeedFooter(shown, matched, total) {
  const el = document.getElementById("feed-footer");
  if (!el) return;
  if (shown === 0) {
    el.innerHTML = `<span class="feed-footer-count">${matched.toLocaleString()} of ${total.toLocaleString()} alerts</span>`;
    return;
  }
  const hasMore = shown < matched;
  const filtered = matched < total;
  el.innerHTML = `
    <span class="feed-footer-count">
      Showing <strong>${shown.toLocaleString()}</strong> of
      <strong>${matched.toLocaleString()}</strong> matching
      ${filtered ? `<span class="feed-footer-faint">(${total.toLocaleString()} total)</span>` : ""}
    </span>
    ${hasMore ? `
      <button class="feed-loadmore-btn" onclick="window.__loadMoreFeed()">↓ Load 200 more</button>
      <button class="feed-loadall-btn" onclick="window.__loadAllFeed()">Show all (${matched.toLocaleString()})</button>
    ` : ""}
  `;
}

function scrollAndExpand(row, alertId) {
  row.scrollIntoView({ behavior: "smooth", block: "center" });

  row.classList.remove("jump-highlight");
  void row.offsetWidth; // force reflow
  row.classList.add("jump-highlight");
  setTimeout(() => row.classList.remove("jump-highlight"), 2200);

  const evRowEl = document.getElementById("ev-" + alertId);
  if (evRowEl && !state.expandedRows.has(alertId)) {
    evRowEl.style.display = "table-row";
    state.expandedRows.add(alertId);
    const btn = row.querySelector(".ev-toggle");
    if (btn) btn.textContent = "▼";
  }
}

export function jumpToAlert(alertId, verdict) {
  navigate("feed");

  state.feedFilter = verdict || "all";
  const hasFilter = ["highly_anomalous", "anomalous", "suspicious"].includes(state.feedFilter);
  document.querySelectorAll(".filter-btn").forEach((b) => {
    b.classList.remove("active");
    if (hasFilter && b.dataset.filter === state.feedFilter) b.classList.add("active");
    if (!hasFilter && b.dataset.filter === "all") b.classList.add("active");
  });

  renderFeed();

  requestAnimationFrame(() => {
    setTimeout(() => {
      const row = document.querySelector(`tr[data-id="${CSS.escape(alertId)}"]`);
      if (!row) {
        state.feedFilter = "all";
        document
          .querySelectorAll(".filter-btn")
          .forEach((b) => b.classList.toggle("active", b.dataset.filter === "all"));
        renderFeed();
        requestAnimationFrame(() => {
          setTimeout(() => {
            const r2 = document.querySelector(`tr[data-id="${CSS.escape(alertId)}"]`);
            if (r2) scrollAndExpand(r2, alertId);
          }, 60);
        });
        return;
      }
      scrollAndExpand(row, alertId);
    }, 80);
  });
}
