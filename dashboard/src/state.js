// Shared dashboard state.
//
// Modules import this object by reference; mutating a property here is visible
// to every other module. This replaces the top-level `let` variables that lived
// in the original single-file <script>.

export const state = {
  feedFilter: "all",
  feedData: [],
  usersData: [],
  campaignsData: [],
  agentsData: [],
  selectedAgent: null,
  selectedUser:  null,
  timelineHours: 0,
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
