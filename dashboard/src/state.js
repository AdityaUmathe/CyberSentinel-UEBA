// Shared dashboard state.
//
// Modules import this object by reference; mutating a property here is visible
// to every other module. This replaces the top-level `let` variables that lived
// in the original single-file <script>.

// Restore the last-used timeline window (24/72/168/720/1440/2160/0=All) from a
// previous visit so a browser refresh stays on the same window instead of
// snapping back to "All". Persisted by the timeline buttons in ui/tabs.js.
const TIMELINE_ALLOWED = [0, 24, 72, 168, 720, 1440, 2160];
function restoreTimelineHours() {
  try {
    const v = parseInt(localStorage.getItem("cs-timeline-hours"), 10);
    return TIMELINE_ALLOWED.includes(v) ? v : 0;
  } catch {
    return 0;
  }
}

export const state = {
  feedFilter: "all",
  feedData: [],
  usersData: [],
  campaignsData: [],
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
