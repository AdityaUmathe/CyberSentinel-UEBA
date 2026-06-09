// Tiny client-side router for clean URLs (/overview, /feed, /users, /campaigns, /endpoints).
//
//  - On initial load, reads location.pathname and activates the matching tab.
//  - navigate(path) updates the URL via history.pushState and activates the tab.
//  - Browser back/forward (popstate) reads the new URL and re-activates.
//
// Flask serves the same index.html for every route in TAB_PATHS, so direct
// links like http://host:3026/feed work as well as in-app navigation.

export const TAB_PATHS = ["overview", "feed", "users", "campaigns", "endpoints", "false-positives", "threatmap"];
const DEFAULT_TAB = "overview";

function pathToTab(pathname) {
  const seg = (pathname || "/").replace(/^\/+/, "").split("/")[0];
  return TAB_PATHS.includes(seg) ? seg : DEFAULT_TAB;
}

export function activateTab(name) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  document.querySelectorAll(".page").forEach((x) => x.classList.remove("active"));
  const tab  = document.querySelector(`.tab[data-tab="${name}"]`);
  const page = document.getElementById(`page-${name}`);
  if (tab)  tab.classList.add("active");
  if (page) page.classList.add("active");
}

/** Navigate to /<name> without a full page reload. */
export function navigate(name, { replace = false } = {}) {
  if (!TAB_PATHS.includes(name)) name = DEFAULT_TAB;
  const url = "/" + name;
  if (location.pathname !== url) {
    if (replace) history.replaceState({ tab: name }, "", url);
    else         history.pushState({ tab: name }, "", url);
  }
  activateTab(name);
}

export function initRouter() {
  // Activate the right tab for the URL the user landed on.
  const initial = pathToTab(location.pathname);
  // replaceState so a direct visit to /feed leaves a sensible history entry.
  history.replaceState({ tab: initial }, "", "/" + initial);
  activateTab(initial);

  // Browser back/forward
  window.addEventListener("popstate", (e) => {
    const tab = (e.state && e.state.tab) || pathToTab(location.pathname);
    activateTab(tab);
  });
}
