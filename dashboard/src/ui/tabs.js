// Top-level navigation: main tabs, banner-stat clicks, timeline-window
// buttons, feed-filter buttons, and feed-row click delegation (which expands
// the evidence panel).
//
// All tab navigation routes through src/router.js so the URL stays in sync
// (clean URLs: /overview, /feed, /users, /campaigns, /endpoints).

import { state } from "../state.js";
import { navigate } from "../router.js";
import { applyTimelineFilter } from "../panels/timeline-filter.js";
import { renderFeed, toggleEvidence } from "../panels/feed.js";

export function initTabs() {
  // ── Main tabs (Overview / Alert Feed / User Risk / Campaigns / Endpoints) ──
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => {
      navigate(t.dataset.tab);
    });
  });

  // ── Banner stat card click → navigate to correct tab/filter ──
  document.querySelectorAll(".banner-stat").forEach((card) => {
    card.addEventListener("click", () => {
      const nav = card.dataset.nav;
      const filter = card.dataset.filter;
      if (!nav) return;
      navigate(nav);
      if (nav === "feed" && filter) {
        state.feedFilter = filter;
        document
          .querySelectorAll(".filter-btn")
          .forEach((b) => b.classList.toggle("active", b.dataset.filter === filter));
        renderFeed();
      }
      card.style.transition = "background 0.15s";
      card.style.background = "rgba(0,212,255,0.08)";
      setTimeout(() => {
        card.style.background = "";
      }, 300);
    });
  });

  // ── Timeline filter buttons ──
  document.querySelectorAll(".tl-btn").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".tl-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.timelineHours = parseInt(b.dataset.hours);
      applyTimelineFilter();
    });
  });

  // ── Feed verdict filters ──
  document.querySelectorAll(".filter-btn").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.feedFilter = b.dataset.filter;
      state.feedRenderLimit = 200;   // reset pagination when filter changes
      renderFeed();
    });
  });

  // ── Event delegation for feed rows (click anywhere in row → toggle ev) ──
  document.addEventListener("click", (e) => {
    const row = e.target.closest("tr.feed-row");
    if (!row) return;
    const alertId = row.getAttribute("data-id");
    if (alertId) toggleEvidence(alertId);
  });
}
