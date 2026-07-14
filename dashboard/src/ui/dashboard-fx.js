// Dashboard FX — small cosmetic polish, all additive & self-contained:
//   1. Count-up animation on the KPI banner numbers (roll up on load/change).
//   2. Severity accent bars on Alert-Feed rows (tags each row from its badge).
//   3. Sliding underline that glides between the active nav tabs.
// Purely cosmetic — remove the import in main.js to revert.

/* ── 1. KPI count-up ──────────────────────────────────────────────────────── */
const KPI_IDS = [
  "s-highly", "s-anomalous", "s-suspicious", "s-total",
  "dm-campaigns", "dm-users-count", "dm-critical-pct", "s-users",
];
const DUR = 900;

function parseNum(txt) {
  const m = String(txt).replace(/,/g, "").match(/^(\D*)(-?\d+(?:\.\d+)?)(.*)$/);
  if (!m) return null;
  const dec = m[2].includes(".") ? m[2].split(".")[1].length : 0;
  return { prefix: m[1] || "", num: parseFloat(m[2]), suffix: m[3] || "", dec };
}
function fmtNum(n, dec, prefix, suffix) {
  const parts = n.toFixed(dec).split(".");
  parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return prefix + parts.join(".") + suffix;
}
function animateEl(el, toText) {
  const t = parseNum(toText);
  if (!t) { el.textContent = toText; el.__lastNum = null; return; }
  const from = el.__lastNum != null ? el.__lastNum : 0;
  if (from === t.num) { el.textContent = toText; el.__lastNum = t.num; return; }
  el.__animating = true;
  el.__toText = toText;
  const start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / DUR, 1);
    const eased = 1 - Math.pow(1 - p, 3); // easeOutCubic
    if (p < 1) {
      el.textContent = fmtNum(from + (t.num - from) * eased, t.dec, t.prefix, t.suffix);
      requestAnimationFrame(step);
    } else {
      el.textContent = toText;            // land exactly on the app's own string
      el.__lastNum = t.num;
      el.__animating = false;
    }
  }
  requestAnimationFrame(step);
}
function initCountUp() {
  KPI_IDS.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    const seed = parseNum(el.textContent.trim());
    el.__lastNum = seed ? seed.num : null;
    const ob = new MutationObserver(() => {
      if (el.__animating) return;
      const txt = el.textContent.trim();
      if (txt === el.__toText) return;
      animateEl(el, txt);
    });
    ob.observe(el, { childList: true, characterData: true, subtree: true });
  });
}

/* ── 2. Severity accent bars on feed rows ─────────────────────────────────── */
const SEV = {
  "v-highly_anomalous": "sev-critical",
  "v-anomalous": "sev-anom",
  "v-suspicious": "sev-susp",
};
function tagRows() {
  document.querySelectorAll("tr.feed-row:not([data-sev])").forEach((tr) => {
    const badge = tr.querySelector(".verdict-badge");
    let sev = "none";
    if (badge) badge.classList.forEach((c) => { if (SEV[c]) sev = SEV[c]; });
    tr.setAttribute("data-sev", sev);
  });
}
function initSeverityRows() {
  let pending = false;
  const obs = new MutationObserver(() => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; tagRows(); });
  });
  obs.observe(document.querySelector("main") || document.body, { childList: true, subtree: true });
  tagRows();
}

/* ── 3. Sliding nav underline ─────────────────────────────────────────────── */
function initNavUnderline() {
  const nav = document.querySelector("nav");
  if (!nav) return;
  const ind = document.createElement("div");
  ind.className = "tab-underline";
  ind.setAttribute("aria-hidden", "true");
  nav.appendChild(ind);
  const move = () => {
    const active = nav.querySelector(".tab.active");
    if (!active) { ind.style.opacity = "0"; return; }
    ind.style.opacity = "1";
    ind.style.transform = `translateX(${active.offsetLeft}px)`;
    ind.style.width = active.offsetWidth + "px";
  };
  nav.querySelectorAll(".tab").forEach((t) =>
    new MutationObserver(move).observe(t, { attributes: true, attributeFilter: ["class"] })
  );
  window.addEventListener("resize", move);
  requestAnimationFrame(move); // after first layout
}

/* ── 4. Live browser-tab badge — critical count in the <title> ────────────── */
function initTitleBadge() {
  const base = "CyberSentinel UEBA";
  const el = document.getElementById("s-highly"); // HIGHLY ANOMALOUS / critical count
  if (!el) return;
  const update = () => {
    const n = parseInt(String(el.textContent).replace(/,/g, ""), 10);
    document.title = n > 0 ? `⚠ ${n.toLocaleString()} · ${base}` : base;
  };
  new MutationObserver(update).observe(el, { childList: true, characterData: true, subtree: true });
  update();
}

function init() {
  initCountUp();
  initSeverityRows();
  initNavUnderline();
  initTitleBadge();
}
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
