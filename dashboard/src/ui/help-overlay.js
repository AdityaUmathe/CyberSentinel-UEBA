// Keyboard-shortcut help overlay — triggered by `?`.
//
// Renders a single-element modal at body level the first time it's shown,
// then just toggles visibility. ESC closes it (handled in shortcuts.js).

const SHORTCUTS = [
  { keys: ["/"],         desc: "Focus the search box" },
  { keys: ["j"],         desc: "Move to next alert (feed page)" },
  { keys: ["k"],         desc: "Move to previous alert (feed page)" },
  { keys: ["Enter"],     desc: "Expand / collapse evidence on focused alert" },
  { keys: ["f"],         desc: "Mark focused alert as false positive" },
  { keys: ["g", "o"],    desc: "Go to Overview" },
  { keys: ["g", "f"],    desc: "Go to Alert Feed" },
  { keys: ["g", "u"],    desc: "Go to Users" },
  { keys: ["g", "c"],    desc: "Go to Campaigns" },
  { keys: ["g", "e"],    desc: "Go to Endpoints" },
  { keys: ["g", "x"],    desc: "Go to False Positives" },
  { keys: ["?"],         desc: "Show / hide this help" },
  { keys: ["Esc"],       desc: "Close popovers / overlays" },
];

let _el = null;

function _build() {
  const overlay = document.createElement("div");
  overlay.id = "kbd-help-overlay";
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  overlay.setAttribute("aria-labelledby", "kbd-help-title");
  overlay.hidden = true;

  const rows = SHORTCUTS.map(({ keys, desc }) => {
    const k = keys.map((s) => `<kbd>${s}</kbd>`).join(" <span class=\"kbd-sep\">then</span> ");
    return `<div class="kbd-row"><div class="kbd-keys">${k}</div><div class="kbd-desc">${desc}</div></div>`;
  }).join("");

  overlay.innerHTML = `
    <div class="kbd-help-card" role="document">
      <div class="kbd-help-header">
        <h2 id="kbd-help-title">Keyboard shortcuts</h2>
        <button type="button" id="kbd-help-close" aria-label="Close help">✕</button>
      </div>
      <div class="kbd-help-body">${rows}</div>
      <div class="kbd-help-footer">Press <kbd>?</kbd> or <kbd>Esc</kbd> to close</div>
    </div>`;

  overlay.addEventListener("click", (e) => {
    // Click on the dimmed backdrop (not the card) closes the overlay.
    if (e.target === overlay) hideHelpOverlay();
  });
  overlay.querySelector("#kbd-help-close").addEventListener("click", hideHelpOverlay);

  document.body.appendChild(overlay);
  return overlay;
}

function _ensure() {
  if (!_el) _el = _build();
  return _el;
}

export function showHelpOverlay() {
  const el = _ensure();
  el.hidden = false;
}

export function hideHelpOverlay() {
  if (_el) _el.hidden = true;
}

export function toggleHelpOverlay() {
  const el = _ensure();
  el.hidden = !el.hidden;
}

export function isHelpOverlayOpen() {
  return !!(_el && !_el.hidden);
}
