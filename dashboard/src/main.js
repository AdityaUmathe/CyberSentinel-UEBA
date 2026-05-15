// CyberSentinel UEBA dashboard — module entry point.
//
//  - Imports the stylesheet so Vite includes it in the build.
//  - Initialises UI subsystems (theme, footer, info-tooltip, tabs).
//  - Exposes inline-handler functions on `window` for `onclick="..."` attrs.
//  - Kicks off the initial data fetch and starts the 30-second auto-refresh.

import "./styles.css";

import { state } from "./state.js";
import { fetchAll, manualRefresh, markFP, unmarkFP, markCampaignFP } from "./api.js";
import { initSse } from "./sse.js";
import { initRouter } from "./router.js";
import { showSkeletons } from "./ui/skeleton.js";
import { initTheme } from "./ui/theme.js";
import { initFooter, openFooterPage, closeFooterPage } from "./ui/footer.js";
import { initInfoTooltip } from "./ui/info-tooltip.js";
import { initTabs } from "./ui/tabs.js";
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

// ── Expose inline-handler functions on window ────────────────────────────────
Object.assign(window, {
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
  loadUserDetail,
  closeUserDetail,
  __markFpFromForm,
  __markCampaignFpPrompt,
});

// ── Init subsystems (order matters for theme + footer DOM access) ────────────
initTheme();
initFooter();
initInfoTooltip();
initRouter();  // must run before initTabs so the initial URL drives tab state
initTabs();
initFeedToolbar();
initUserDetail();

// ── Show FPs toggle in the feed header ───────────────────────────────────────
const fpToggle = document.getElementById("fp-toggle-cb");
if (fpToggle) {
  fpToggle.addEventListener("change", () => {
    state.showFps = fpToggle.checked;
    fetchAll().catch(() => {});
  });
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
