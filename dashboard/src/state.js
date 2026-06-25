// Shared dashboard state.
//
// Modules import this object by reference; mutating a property here is visible
// to every other module. This replaces the top-level `let` variables that lived
// in the original single-file <script>.

// Restore the last-used timeline window (24/72/168/720/1440/2160/0=All) from a
// previous visit so a browser refresh stays on the same window instead of
// snapping back to the default. Persisted by the timeline buttons in ui/tabs.js.
//
// Default is 24h, NOT All-time: on All-time /api/feed serializes+ships every
// in-history row (tens of MB / several seconds) and the client re-aggregates it
// all, so a fresh visit appeared to "hang" on LOADING. 24h is ~1.5MB/0.6s. An
// explicit "All" the analyst picked is still honored (0 is in the allow-list).
const TIMELINE_ALLOWED = [0, 24, 72, 168, 720, 1440, 2160];
const TIMELINE_DEFAULT = 24;
function restoreTimelineHours() {
  try {
    const v = parseInt(localStorage.getItem("cs-timeline-hours"), 10);
    return TIMELINE_ALLOWED.includes(v) ? v : TIMELINE_DEFAULT;
  } catch {
    return TIMELINE_DEFAULT;
  }
}

export const state = {
  feedFilter: "all",
  feedData: [],
  usersData: [],
  campaignsData: [],
  incidentsData: [],
  incidentWindow: 60,   // entity-incident rollup bucket, minutes (Incidents tab)
  agentsData: [],
  selectedAgent: null,
  selectedUser:  null,
  timelineHours: restoreTimelineHours(),
  allFeedData: [],
  allUsersData: [],
  allAgentsData: [],
  expandedRows: new Set(),
  aiGenerated: false,
  lastStats: null,

  // False positives — list of {event_id, reason, marked_at, alert:{...}}
  // populated from GET /api/false-positives.
  falsePositives: [],
  // Pattern FPs — list of {id, signature_id, user, agent, reason, marked_at, matched}
  // populated from GET /api/false-positive-patterns.
  fpPatterns: [],
  // When true, the Alert Feed sends include_fp=1 and renders FP-tagged rows
  // inline with strikethrough styling.
  showFps: false,

  // ── Analyst workflow filters (Phase 4) ──
  feedSearch: "",          // free-text needle, matched case-insensitively across user/host/sig/reasons/campaign/event_id
  feedTactics: new Set(),  // selected MITRE tactic names; empty Set = all tactics
  feedSeverityMin: 0,      // minimum security.severity to include
  feedRenderLimit: 200,    // how many filtered rows to actually paint into the DOM
};
