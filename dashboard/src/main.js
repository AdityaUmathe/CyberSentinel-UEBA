// CyberSentinel UEBA dashboard — module entry point.
//
//  - Imports the stylesheet so Vite includes it in the build.
//  - Initialises UI subsystems (theme, footer, info-tooltip, tabs).
//  - Exposes inline-handler functions on `window` for `onclick="..."` attrs.
//  - Kicks off the initial data fetch and starts the 30-second auto-refresh.

import "./styles.css";

import { state } from "./state.js";
import { fetchAll, fetchIncidents, manualRefresh, markFP, unmarkFP, markCampaignFP, unmarkPatternFP } from "./api.js";
import { initSse } from "./sse.js";
import { initRouter, navigate } from "./router.js";
import { showSkeletons } from "./ui/skeleton.js";
import { initTheme } from "./ui/theme.js";
import { initFooter, openFooterPage, closeFooterPage } from "./ui/footer.js";
import { initInfoTooltip } from "./ui/info-tooltip.js";
import { initTabs } from "./ui/tabs.js";
import { initShortcuts } from "./ui/shortcuts.js";
import { initThreatMap } from "./panels/threatmap.js";
import { initFeedToolbar } from "./panels/feed-toolbar.js";
import { jumpToAlert, toggleEvidence } from "./panels/feed.js";
import { goToAgent, loadAgentDetail, switchAgentTab, goToFeedWithFilter } from "./panels/agents.js";
import { loadUserDetail, closeUserDetail, initUserDetail } from "./panels/user-detail.js";
import { drawTimeline } from "./charts/timeline.js";
import { drawRadar } from "./charts/radar.js";

// Submit an FP mark from the form inside an expanded evidence panel.
function __markFpFromForm(eventId, alertId) {
  const input = document.getElementById("fp-reason-" + alertId);
  const reason = (input && input.value || "").trim();
  markFP(eventId, reason);
}

// Prompt the analyst for confirmation + an optional reason before bulk-marking
// every alert in a campaign as a false positive.
function __markCampaignFpPrompt(campaignId, alertCount) {
  const reason = window.prompt(
    `Mark every alert in campaign ${campaignId} (${alertCount} alerts) as a false positive?\n\n` +
    `Optional: enter a reason (e.g. "scheduled vulnerability scan").\n` +
    `Leave blank and press OK to proceed without a reason. Press Cancel to abort.`,
    ""
  );
  if (reason === null) return;
  markCampaignFP(campaignId, reason.trim());
}

// Copy SIEM event ID to clipboard with a brief visual confirmation on the
// button that was clicked.
function __copyEvId(eid, btn) {
  if (!eid) return;
  const done = (ok) => {
    if (!btn) return;
    const span = btn.querySelector("span");
    const orig = span ? span.textContent : null;
    btn.classList.add(ok ? "copied" : "copy-failed");
    if (span) span.textContent = ok ? "Copied" : "Failed";
    setTimeout(() => {
      btn.classList.remove("copied", "copy-failed");
      if (span && orig != null) span.textContent = orig;
    }, 1200);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(eid).then(() => done(true)).catch(() => done(false));
  } else {
    // Fallback for non-HTTPS / older browsers.
    try {
      const ta = document.createElement("textarea");
      ta.value = eid;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      done(ok);
    } catch (e) { done(false); }
  }
}

// Open an entity's existing User Risk detail from an incident card — reuses the
// full per-entity drill-down instead of duplicating it in the Incidents tab.
function __goToEntity(entity) {
  navigate("users");
  if (entity) loadUserDetail(entity);
}

// Change the incident rollup window (minutes) and refresh just the Incidents tab.
function __setIncidentWindow(min) {
  state.incidentWindow = parseInt(min, 10) || 60;
  fetchIncidents().catch(() => {});
}

// ── Expose inline-handler functions on window ────────────────────────────────
Object.assign(window, {
  __goToEntity,
  __setIncidentWindow,
  manualRefresh,
  openFooterPage,
  closeFooterPage,
  jumpToAlert,
  goToAgent,
  loadAgentDetail,
  switchAgentTab,
  goToFeedWithFilter,
  toggleEvidence,
  markFP,
  unmarkFP,
  markCampaignFP,
  unmarkPatternFP,
  loadUserDetail,
  closeUserDetail,
  __markFpFromForm,
  __markCampaignFpPrompt,
  __copyEvId,
});

// ── Init subsystems (order matters for theme + footer DOM access) ────────────
initTheme();
initFooter();
initInfoTooltip();
initRouter();  // must run before initTabs so the initial URL drives tab state
initTabs();
initFeedToolbar();
initUserDetail();
initShortcuts();
initThreatMap();   // live firewall geoIP globe (lazy — builds on first tab open)

// ── Show FPs toggle in the feed header ───────────────────────────────────────
const fpToggle = document.getElementById("fp-toggle-cb");
if (fpToggle) {
  fpToggle.addEventListener("change", () => {
    state.showFps = fpToggle.checked;
    fetchAll().catch(() => {});
  });
}

// ── Incidents tab: lazy-load on open (heavy entity×window rollup, kept off the
// global refresh so it never slows the rest of the dashboard) ────────────────
const incTab = document.querySelector('.tab[data-tab="incidents"]');
if (incTab) incTab.addEventListener("click", () => fetchIncidents().catch(() => {}));
// Direct deep-link to /incidents: load it once after the initial paint.
if (location.pathname.replace(/^\/+/, "").split("/")[0] === "incidents") {
  fetchIncidents().catch(() => {});
}

// ── First paint + initial fetch, then attach the live SSE stream ────────────
showSkeletons();
fetchAll().finally(() => initSse());

// ── Background reconciliation poll (60s) — SSE delivers freshness, this only
// keeps aggregates (gauge / users / agents / campaigns) in sync. ────────────
const AUTO_REFRESH_MS = 60_000;
setInterval(() => {
  if (!document.hidden) fetchAll().catch(() => {});
}, AUTO_REFRESH_MS);

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && !state.feedData.length) fetchAll().catch(() => {});
});

// ── Redraw canvas charts on resize (debounced) ───────────────────────────────
let _resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    if (state.feedData.length) {
      drawTimeline(state.feedData);
      if (state.allFeedData.length) {
        const stats = state.lastStats;
        if (stats) drawRadar(stats.top_reasons || []);
      }
    }
  }, 200);
});
