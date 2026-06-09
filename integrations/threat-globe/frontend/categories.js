// Traffic categories + small formatting helpers shared by the globe engine and
// the React component. Self-contained — no external dependencies.
//
// Each firewall event is classified (server-side) into one of these categories;
// `dir` ("in" = external → DC, "out" = DC → external) drives the on-globe flow.

export const CATEGORY = {
  incoming_threat: { color: "#ff3b5c", label: "Incoming threat",      dir: "in"  },
  normal_incoming: { color: "#3d7bff", label: "Normal incoming",      dir: "in"  },
  outgoing:        { color: "#22c55e", label: "Outgoing from server", dir: "out" },
  external_conn:   { color: "#facc15", label: "External connection",  dir: "in"  },
};

// Display order for the legend / filter chips.
export const CATEGORY_ORDER = ["outgoing", "incoming_threat", "normal_incoming", "external_conn"];

export const DC_COLOR = "#7cffcb";

export function catColor(cat) { return (CATEGORY[cat] || CATEGORY.normal_incoming).color; }
export function catLabel(cat) { return (CATEGORY[cat] || CATEGORY.normal_incoming).label; }

export function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
  return m ? `${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)}` : "255,59,92";
}
export function rgba(hex, a) { return `rgba(${hexToRgb(hex)},${a})`; }

export function flagEmoji(cc) {
  if (!cc || cc.length !== 2 || cc === "??") return "🏴";
  const A = 0x1f1e6;
  return String.fromCodePoint(A + (cc.charCodeAt(0) - 65), A + (cc.charCodeAt(1) - 65));
}

export function shortSig(s) { return String(s || "").replace(/^Fortigate:\s*/i, "").slice(0, 42); }

export function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return "";
  return d.toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
