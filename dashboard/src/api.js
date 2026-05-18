// API layer — wraps /api/* fetches and orchestrates a full refresh.

import { state } from "./state.js";
import { renderStats, updateTicker, renderDashboard } from "./panels/overview.js";
import { applyTimelineFilter } from "./panels/timeline-filter.js";
import { renderCampaigns } from "./panels/campaigns.js";
import { loadAgentDetail } from "./panels/agents.js";
import { loadUserDetail } from "./panels/user-detail.js";
import { renderFalsePositives } from "./panels/false-positives.js";
import { showToast } from "./ui/toast.js";

const feedURL = () => "/api/feed" + (state.showFps ? "?include_fp=1" : "");

export async function fetchAll() {
  const [stats, feed, users, camps, agents, fps, fpPatterns] = await Promise.all([
    fetch("/api/stats").then((r) => r.json()),
    fetch(feedURL()).then((r) => r.json()),
    fetch("/api/users").then((r) => r.json()),
    fetch("/api/campaigns").then((r) => r.json()),
    fetch("/api/agents").then((r) => r.json()),
    fetch("/api/false-positives").then((r) => r.json()),
    fetch("/api/false-positive-patterns").then((r) => r.json()).catch(() => []),
  ]);
  state.feedData       = feed;
  state.usersData      = users;
  state.campaignsData  = camps;
  state.agentsData     = agents;
  state.allFeedData    = feed;
  state.allUsersData   = users;
  state.allAgentsData  = agents;
  state.falsePositives = fps;
  state.fpPatterns     = Array.isArray(fpPatterns) ? fpPatterns : [];

  renderStats(stats);
  updateTicker(feed);
  renderDashboard(stats, feed);
  applyTimelineFilter();
  renderCampaigns();
  renderFalsePositives();

  if (state.selectedAgent) {
    const found = state.agentsData.find((a) => a.agent === state.selectedAgent);
    // silent:true → don't flash the spinner; refresh data in place when ready.
    if (found) loadAgentDetail(state.selectedAgent, { silent: true });
  }
  if (state.selectedUser) {
    loadUserDetail(state.selectedUser, { silent: true });
  }

  const lu = document.getElementById("last-update");
  if (lu) lu.textContent = "Updated " + new Date().toLocaleTimeString("en-IN");
  const dot  = document.querySelector(".status-dot");
  const pill = document.querySelector(".status-pill");
  if (dot)  dot.style.background = "var(--green)";
  if (pill) pill.style.color = "var(--green)";
  const pillSpan = document.querySelector(".status-pill span");
  if (pillSpan) pillSpan.textContent = "ENGINE LIVE";
}

// Refresh feels instant: button shows a brief spinner state, an immediate
// toast confirms the click, and fetchAll runs in the background. The
// button only stays disabled long enough to prevent double-clicks (600ms),
// then becomes clickable again — the actual data render finishes when
// it finishes, but the UI never feels frozen.
let _refreshInFlight = false;
export function manualRefresh() {
  if (_refreshInFlight) return;
  _refreshInFlight = true;

  const btn = document.getElementById("refresh-btn");
  if (btn) {
    btn.style.opacity = "0.6";
    btn.style.pointerEvents = "none";
    // Spin the glyph; keep the "REFRESH" word so width doesn't jump.
    btn.classList.add("refreshing");
  }
  // Immediate toast so the click feels acknowledged.
  showToast("Refreshing dashboard…", "info");

  state.aiGenerated = false;

  // Run the heavy fetch+render fully asynchronously; don't await.
  fetchAll()
    .then(() => {
      showToast("Dashboard refreshed", "success");
    })
    .catch(() => {
      const dot  = document.querySelector(".status-dot");
      const pill = document.querySelector(".status-pill");
      const pillSpan = document.querySelector(".status-pill span");
      if (dot)  dot.style.background = "var(--red)";
      if (pill) pill.style.color = "var(--red)";
      if (pillSpan) pillSpan.textContent = "OFFLINE";
      const lu = document.getElementById("last-update");
      if (lu) lu.textContent = "Connection error";
      showToast("Connection failed — engine offline?", "error");
    });

  // Re-enable the button after a short cooldown — independent of the
  // network call. This prevents accidental double-refreshes but doesn't
  // leave the UI feeling frozen if the request is slow.
  setTimeout(() => {
    if (btn) {
      btn.style.opacity = "1";
      btn.style.pointerEvents = "auto";
      btn.classList.remove("refreshing");
    }
    _refreshInFlight = false;
  }, 600);
}

// ── False-positive mutators ───────────────────────────────────────────────────
//
// Optimistic strategy: mutate local state + repaint the visible panels first,
// then POST/DELETE to the server in the background. The next 30-second
// auto-refresh reconciles the gauge/users/agents aggregates with the server.
// On failure we toast the error and force a fetchAll to undo the bad state.

function _applyLocalMark(eventId, reason) {
  const now = new Date().toISOString();
  const rec = { event_id: eventId, reason, marked_at: now };
  // Pull the alert out of the local lists so it disappears from the feed.
  const ix = state.allFeedData.findIndex((a) => a.event_id === eventId);
  const taken = ix >= 0 ? state.allFeedData[ix] : null;
  if (taken) {
    state.allFeedData.splice(ix, 1);
    state.feedData = state.feedData.filter((a) => a.event_id !== eventId);
  }
  // Append to falsePositives (with the original alert if we still have it).
  state.falsePositives = [
    { ...rec, alert: taken || null },
    ...state.falsePositives.filter((r) => r.event_id !== eventId),
  ];
}

function _applyLocalUnmark(eventId) {
  const rec = state.falsePositives.find((r) => r.event_id === eventId);
  state.falsePositives = state.falsePositives.filter((r) => r.event_id !== eventId);
  if (rec && rec.alert) {
    // Put the alert back into the feed (newest-first ordering preserved).
    state.allFeedData = [rec.alert, ...state.allFeedData.filter((a) => a.event_id !== eventId)];
    state.feedData    = [rec.alert, ...state.feedData.filter((a) => a.event_id !== eventId)];
  }
}

function _repaintFpAffectedPanels() {
  // Imports are inside the function to avoid a circular import at module load.
  import("./panels/feed.js").then(({ renderFeed }) => renderFeed());
  import("./panels/false-positives.js").then(({ renderFalsePositives }) =>
    renderFalsePositives()
  );
}

export async function markFP(eventId, reason = "") {
  if (!eventId) return;
  _applyLocalMark(eventId, reason);
  _repaintFpAffectedPanels();
  // Initial toast — replaced with a richer "+ auto-suppression" message once
  // the server responds (the server also creates a pattern from the alert's
  // fingerprint so future similar alerts disappear).
  showToast("Marked as false positive", "success");
  try {
    const r = await fetch("/api/false-positive", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ event_id: eventId, reason }),
    });
    const data = await r.json();
    if (!r.ok || data.ok === false) {
      throw new Error((data && data.error) || `HTTP ${r.status}`);
    }
    if (data.pattern) {
      const p = data.pattern;
      const verb = data.pattern_new ? "Auto-suppressing new" : "Already auto-suppressing new";
      showToast(`${verb} alerts with rule "${p.rule_description}"`, "success");
    }
    // Don't await — background reconciliation of stats/users/agents/campaigns
    // and refresh of the pattern list.
    fetchAll().catch(() => {});
  } catch (e) {
    showToast("Failed to mark FP: " + e.message, "error");
    fetchAll().catch(() => {});
  }
}

export async function unmarkFP(eventId) {
  if (!eventId) return;
  _applyLocalUnmark(eventId);
  _repaintFpAffectedPanels();
  showToast("Restored from false positives", "success");
  try {
    const r = await fetch("/api/false-positive/" + encodeURIComponent(eventId), {
      method: "DELETE",
    });
    const data = await r.json();
    if (!r.ok || data.ok === false) {
      throw new Error((data && data.error) || `HTTP ${r.status}`);
    }
    fetchAll().catch(() => {});
  } catch (e) {
    showToast("Failed to unmark FP: " + e.message, "error");
    fetchAll().catch(() => {});
  }
}

// Pattern-FP mutators — patterns auto-suppress ALL future alerts whose rule
// description matches. Server filter applies on every load_alerts(), so the
// next fetchAll() reflects the new pattern without needing an engine restart.
export async function markPatternFP({ rule_description, reason = "" }) {
  if (!rule_description) {
    showToast("Cannot mark pattern: missing rule description", "error");
    return;
  }
  try {
    const r = await fetch("/api/false-positive-pattern", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ rule_description, reason }),
    });
    const data = await r.json();
    if (!r.ok || data.ok === false) {
      throw new Error((data && data.error) || `HTTP ${r.status}`);
    }
    if (data.deduped) {
      showToast("Pattern already exists — nothing to do", "info");
    } else {
      showToast(`Pattern saved — new alerts with rule "${rule_description}" will be auto-suppressed`, "success");
    }
    fetchAll().catch(() => {});
  } catch (e) {
    showToast("Failed to mark pattern: " + e.message, "error");
  }
}

export async function unmarkPatternFP(patternId) {
  if (!patternId) return;
  try {
    const r = await fetch("/api/false-positive-pattern/" + encodeURIComponent(patternId), {
      method: "DELETE",
    });
    const data = await r.json();
    if (!r.ok || data.ok === false) {
      throw new Error((data && data.error) || `HTTP ${r.status}`);
    }
    showToast("Pattern removed", "success");
    fetchAll().catch(() => {});
  } catch (e) {
    showToast("Failed to remove pattern: " + e.message, "error");
  }
}

export async function markCampaignFP(campaignId, reason = "") {
  if (!campaignId) return;
  // Optimistic: drop every alert with this campaign_id from local lists, drop
  // the campaign itself, append synthetic FP records.
  const now = new Date().toISOString();
  const affected = state.allFeedData.filter((a) => a.campaign_id === campaignId);
  state.allFeedData = state.allFeedData.filter((a) => a.campaign_id !== campaignId);
  state.feedData    = state.feedData.filter((a) => a.campaign_id !== campaignId);
  state.campaignsData = state.campaignsData.filter((c) => c.campaign_id !== campaignId);
  state.falsePositives = [
    ...affected.map((a) => ({
      event_id: a.event_id,
      reason: reason || `campaign ${campaignId} marked as FP`,
      marked_at: now,
      alert: a,
    })),
    ...state.falsePositives.filter((r) => !affected.some((a) => a.event_id === r.event_id)),
  ];
  _repaintFpAffectedPanels();
  // Campaign panel needs an explicit repaint since it isn't in _repaintFpAffectedPanels.
  import("./panels/campaigns.js").then(({ renderCampaigns }) => renderCampaigns());
  showToast(`Campaign ${campaignId} marked (${affected.length} alerts)`, "success");

  try {
    const r = await fetch(
      "/api/false-positive/campaign/" + encodeURIComponent(campaignId),
      {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ reason }),
      }
    );
    const data = await r.json();
    if (!r.ok || data.ok === false) {
      throw new Error((data && data.error) || `HTTP ${r.status}`);
    }
    if (data.patterns_new) {
      showToast(`Auto-suppressing ${data.patterns_new} new pattern${data.patterns_new === 1 ? "" : "s"}`, "success");
    }
    fetchAll().catch(() => {});
  } catch (e) {
    showToast("Failed to mark campaign: " + e.message, "error");
    fetchAll().catch(() => {});
  }
}
