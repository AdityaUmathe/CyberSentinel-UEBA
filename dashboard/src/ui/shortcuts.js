// Global keyboard shortcuts for analyst workflow.
//
//   /            focus the feed search input (navigates to /feed if needed)
//   j / k        next / prev visible feed row (only on feed page)
//   Enter        toggle evidence panel on the focused row
//   f            mark the focused row as a false positive
//   g f|o|u|c|e|x  vim-style tab jump (feed/overview/users/campaigns/endpoints/fps)
//   ?            toggle keyboard-shortcut help overlay
//   Esc          close popovers/overlays; clear search when focused
//
// All handlers no-op while the user is typing in an input, textarea, select,
// or contenteditable region — except for Esc, which always closes overlays.

import { navigate, TAB_PATHS } from "../router.js";
import { toggleEvidence } from "../panels/feed.js";
import { markFP } from "../api.js";
import { toggleHelpOverlay, hideHelpOverlay, isHelpOverlayOpen } from "./help-overlay.js";

const TAB_FOR_KEY = {
  o: "overview",
  f: "feed",
  u: "users",
  c: "campaigns",
  e: "endpoints",
  x: "false-positives",
};

let _gPending = false;
let _gTimer   = null;

function _isTyping(el) {
  if (!el) return false;
  const tag = el.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (el.isContentEditable) return true;
  return false;
}

function _currentTab() {
  return (location.pathname.replace(/^\/+/, "").split("/")[0]) || "overview";
}

function _visibleFeedRows() {
  // Feed page rows that are not collapsed/hidden by the table layout.
  return Array.from(document.querySelectorAll("#page-feed tr.feed-row"))
    .filter((tr) => tr.style.display !== "none");
}

function _focusedRowIndex(rows) {
  return rows.findIndex((r) => r.classList.contains("kbd-focus"));
}

function _setRowFocus(rows, ix) {
  rows.forEach((r) => r.classList.remove("kbd-focus"));
  if (ix < 0 || ix >= rows.length) return;
  const row = rows[ix];
  row.classList.add("kbd-focus");
  // Scroll the row into view if it's outside the viewport.
  const r = row.getBoundingClientRect();
  if (r.top < 80 || r.bottom > window.innerHeight - 20) {
    row.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

function _moveRow(delta) {
  const rows = _visibleFeedRows();
  if (!rows.length) return;
  const cur  = _focusedRowIndex(rows);
  const next = cur < 0
    ? (delta > 0 ? 0 : rows.length - 1)
    : Math.max(0, Math.min(rows.length - 1, cur + delta));
  _setRowFocus(rows, next);
}

function _focusedRowId() {
  const row = document.querySelector("#page-feed tr.feed-row.kbd-focus");
  return row ? row.getAttribute("data-id") : null;
}

function _focusSearch() {
  if (_currentTab() !== "feed") navigate("feed");
  // Defer to next frame so the tab is active and the input is in flow.
  requestAnimationFrame(() => {
    const inp = document.getElementById("feed-search-input");
    if (inp) { inp.focus(); inp.select(); }
  });
}

function _closeAllOverlays() {
  // MITRE popover
  const pop  = document.getElementById("mitre-pop");
  const wrap = document.getElementById("mitre-dropdown");
  if (pop && wrap && !pop.hasAttribute("hidden")) {
    pop.setAttribute("hidden", "");
    wrap.classList.remove("open");
  }
  // FP overlays (per-row reason popovers)
  document.querySelectorAll(".fp-overlay.open").forEach((o) => o.classList.remove("open"));
  // Help overlay
  hideHelpOverlay();
}

export function initShortcuts() {
  document.addEventListener("keydown", (e) => {
    // Esc is always live — closes overlays / blurs search.
    if (e.key === "Escape") {
      if (isHelpOverlayOpen()) { hideHelpOverlay(); e.preventDefault(); return; }
      // If the search input has focus, blur it (don't clear — that's the ✕ button's job).
      const ae = document.activeElement;
      if (ae && ae.id === "feed-search-input") {
        ae.blur();
        return;
      }
      _closeAllOverlays();
      return;
    }

    // Suppress shortcuts while the user is typing.
    if (_isTyping(e.target) || _isTyping(document.activeElement)) return;
    // Modifier combos belong to the browser / OS.
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    // ── g + <letter> tab jump ──────────────────────────────────────────────
    if (_gPending) {
      const dest = TAB_FOR_KEY[e.key];
      _gPending = false;
      clearTimeout(_gTimer);
      if (dest && TAB_PATHS.includes(dest)) {
        e.preventDefault();
        navigate(dest);
      }
      return;
    }
    if (e.key === "g") {
      _gPending = true;
      _gTimer = setTimeout(() => { _gPending = false; }, 1200);
      return;
    }

    // ── Single-key shortcuts ───────────────────────────────────────────────
    switch (e.key) {
      case "/":
        e.preventDefault();
        _focusSearch();
        return;
      case "?":
        e.preventDefault();
        toggleHelpOverlay();
        return;
      case "j":
        if (_currentTab() === "feed") { e.preventDefault(); _moveRow(+1); }
        return;
      case "k":
        if (_currentTab() === "feed") { e.preventDefault(); _moveRow(-1); }
        return;
      case "Enter": {
        if (_currentTab() !== "feed") return;
        const id = _focusedRowId();
        if (id) { e.preventDefault(); toggleEvidence(id); }
        return;
      }
      case "f": {
        // Lowercase f on its own (not part of g+f) marks the focused row FP.
        if (_currentTab() !== "feed") return;
        const id = _focusedRowId();
        if (id) { e.preventDefault(); markFP(id, "marked via keyboard shortcut"); }
        return;
      }
    }
  });
}
