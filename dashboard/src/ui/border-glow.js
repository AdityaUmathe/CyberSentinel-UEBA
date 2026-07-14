// BorderGlow — cursor-tracking gradient border glow (vanilla port of the
// reactbits <BorderGlow>). A masked gradient ring is attached to each card;
// the ring lights up where the pointer is, strongest near the edges
// (edgeSensitivity). Purely cosmetic — pairs with the `.bg-ring` rules in
// liquid-glass.css. Remove the import in main.js to revert.

const SEL = ".panel, .banner-stat, .banner-reasons";
const EDGE = 78;            // px band from an edge where the glow reaches full strength

let raf = 0;
let pending = null;
let current = null;

function ensureRing(card) {
  let ring = card.__bgRing;
  if (ring && ring.isConnected) return ring;
  ring = document.createElement("span");
  ring.className = "bg-ring";
  ring.setAttribute("aria-hidden", "true");
  card.appendChild(ring);
  card.__bgRing = ring;
  return ring;
}

function clear(card) {
  if (card) card.style.setProperty("--glow-o", "0");
}

function apply() {
  raf = 0;
  const e = pending;
  if (!e) return;
  const card = e.target && e.target.closest ? e.target.closest(SEL) : null;

  if (card !== current) {
    clear(current);
    current = card;
  }
  if (!card) return;

  ensureRing(card);
  const r = card.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  // distance to the nearest edge → glow ramps up as the pointer nears any edge
  const d = Math.min(x, y, r.width - x, r.height - y);
  const o = Math.max(0, Math.min(1, 1 - d / EDGE));

  card.style.setProperty("--glow-x", x + "px");
  card.style.setProperty("--glow-y", y + "px");
  card.style.setProperty("--glow-o", o.toFixed(3));
}

function onMove(e) {
  pending = e;
  if (!raf) raf = requestAnimationFrame(apply);
}

function init() {
  document.addEventListener("pointermove", onMove, { passive: true });
  // pointer left the window entirely → kill the active glow
  window.addEventListener("pointerout", (e) => {
    if (!e.relatedTarget && current) {
      clear(current);
      current = null;
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
