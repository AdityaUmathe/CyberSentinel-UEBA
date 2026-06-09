// Live Threat Map — a 3D globe (globe.gl / Three.js) that plots firewall attacks
// in real time. Each arc rises from the geolocated SOURCE IP of a firewall event
// to our datacenter; the source IP is geolocated server-side from a local MaxMind
// GeoLite2 DB (see /api/geofeed in ueba_dashboard_server.py) so the browser only
// ever receives lat/lng — the whole feature works offline.
//
// The globe is built lazily the first time the tab is opened — globe.gl/Three.js
// (~1.5 MB) is loaded via dynamic import() so it's code-split into its own chunk
// and never weighs down the other tabs' initial load. The live poll only runs
// while the tab is visible, so it costs nothing elsewhere.

// ── Tunables ──────────────────────────────────────────────────────────────────
const POLL_MS        = 6000;   // how often to pull /api/geofeed while visible
const MAX_ARCS       = 70;     // rolling on-screen arc budget (animated)
const MAX_POINTS     = 90;     // rolling attacker-point budget
const ARC_FLY_MS     = 2200;   // arc dash travel time (source → DC)

// Severity → colour. Matches the dashboard palette (red / orange / cyan).
const WEIGHT_COLOR = {
  high: "#ff3b5c",   // failed login / blocked / denied  → attack
  med:  "#ffa94d",   // other firewall activity
  low:  "#27d3ff",   // vpn / passed / informational
};
const DC_COLOR = "#7CFFcb";

// ── Module state ──────────────────────────────────────────────────────────────
let globe       = null;
let mounted     = false;
let building    = false;
let pollTimer   = null;
let observer    = null;
let dc          = null;

const arcs      = [];          // rolling display buffer
const points    = new Map();   // ip -> { lat, lng, color, count, ts }
const rings     = [];          // transient impact rings
const seenKeys  = new Set();   // dedupe across polls (ip|time|sig)
let   seenOrder = [];          // FIFO for trimming seenKeys
let   lastError = null;
let   firstLoad = true;

function el(id) { return document.getElementById(id); }

function keyOf(a) { return `${a.ip}|${a.time}|${a.sig}`; }

// ── Globe construction (once, lazy — dynamic import keeps three.js out of the
//    main bundle) ─────────────────────────────────────────────────────────────
async function buildGlobe() {
  const host = el("threatmap-globe");
  if (!host || globe || building) return;
  building = true;

  let Globe;
  try {
    ({ default: Globe } = await import("globe.gl"));
  } catch (e) {
    building = false;
    lastError = "Failed to load globe library: " + (e.message || e);
    return;
  }

  globe = new Globe(host)
    .backgroundColor("#01030a")
    .backgroundImageUrl("/textures/night-sky.png")
    .globeImageUrl("/textures/earth-night.jpg")
    .showAtmosphere(true)
    .atmosphereColor("#2b6cff")
    .atmosphereAltitude(0.18)
    // Arcs: attacker → datacenter, animated dash "tracer"
    .arcsData([])
    .arcStartLat((d) => d.srcLat)
    .arcStartLng((d) => d.srcLng)
    .arcEndLat(() => dc.lat)
    .arcEndLng(() => dc.lng)
    .arcColor((d) => {
      const c = WEIGHT_COLOR[d.weight] || WEIGHT_COLOR.med;
      return [c, DC_COLOR];
    })
    .arcStroke((d) => (d.weight === "high" ? 0.55 : 0.35))
    .arcDashLength(0.45)
    .arcDashGap(1.4)
    .arcDashInitialGap(() => Math.random())
    .arcDashAnimateTime(ARC_FLY_MS)
    .arcAltitudeAutoScale(0.45)
    // Glowing attacker points
    .pointsData([])
    .pointLat((d) => d.lat)
    .pointLng((d) => d.lng)
    .pointColor((d) => d.color)
    .pointAltitude(0.01)
    .pointRadius((d) => Math.min(0.8, 0.18 + Math.log2(1 + d.count) * 0.12))
    .pointsMerge(false)
    // Impact rings (DC pulse + transient source pings)
    .ringsData([])
    .ringLat((d) => d.lat)
    .ringLng((d) => d.lng)
    .ringColor((d) => (t) => {
      // d.rgb like "255,59,92"; fade alpha out as the ring expands
      return `rgba(${d.rgb},${(1 - t).toFixed(3)})`;
    })
    .ringMaxRadius((d) => d.maxR)
    .ringPropagationSpeed((d) => d.speed)
    .ringRepeatPeriod((d) => d.period);

  // Camera + auto-rotate
  globe.pointOfView({ lat: 22, lng: 78, altitude: 2.4 }, 0);
  const controls = globe.controls();
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.45;
  controls.enableDamping = true;

  sizeGlobe();
  window.addEventListener("resize", sizeGlobe);
  mounted = true;
  building = false;
}

function sizeGlobe() {
  const host = el("threatmap-globe");
  if (!host || !globe) return;
  const r = host.getBoundingClientRect();
  if (r.width > 0 && r.height > 0) {
    globe.width(r.width).height(r.height);
  }
}

// ── Data application ──────────────────────────────────────────────────────────
function rememberKey(k) {
  seenKeys.add(k);
  seenOrder.push(k);
  if (seenOrder.length > 4000) {
    const drop = seenOrder.splice(0, 2000);
    drop.forEach((x) => seenKeys.delete(x));
  }
}

function applyFeed(data) {
  dc = data.dc || dc || { lat: 21.1463, lng: 79.0849, label: "Datacenter" };

  if (!data.geo_ok) {
    showOffline();
    return;
  }
  hideOffline();

  const incoming = Array.isArray(data.arcs) ? data.arcs : [];
  const fresh = [];
  for (const a of incoming) {
    const k = keyOf(a);
    if (seenKeys.has(k)) continue;
    rememberKey(k);
    fresh.push(a);
  }

  // On the very first load, seed the display with the most-recent slice so the
  // map isn't empty, but don't fire a ring storm for historical events.
  const ringFor = firstLoad ? [] : fresh;

  for (const a of fresh) {
    arcs.push(a);
    // attacker point (accumulate count per IP)
    const p = points.get(a.ip);
    const color = WEIGHT_COLOR[a.weight] || WEIGHT_COLOR.med;
    if (p) { p.count += 1; p.ts = Date.now(); if (a.weight === "high") p.color = color; }
    else   { points.set(a.ip, { lat: a.srcLat, lng: a.srcLng, color, count: 1, ts: Date.now() }); }
  }
  if (firstLoad && fresh.length) {
    // keep only the freshest slice visible at startup
    const seed = fresh.slice(-MAX_ARCS);
    arcs.length = 0;
    arcs.push(...seed);
  }

  // trim rolling buffers
  while (arcs.length > MAX_ARCS) arcs.shift();
  if (points.size > MAX_POINTS) {
    const oldest = [...points.entries()].sort((a, b) => a[1].ts - b[1].ts);
    for (let i = 0; i < points.size - MAX_POINTS; i++) points.delete(oldest[i][0]);
  }

  // transient impact rings for the new attacks (source ping)
  for (const a of ringFor.slice(-12)) {
    const rgb = hexToRgb(WEIGHT_COLOR[a.weight] || WEIGHT_COLOR.med);
    rings.push({ lat: a.srcLat, lng: a.srcLng, rgb, maxR: 3, speed: 2.5, period: 1e9, born: Date.now() });
  }
  // age out transient rings (~1.4s each)
  const now = Date.now();
  for (let i = rings.length - 1; i >= 0; i--) {
    if (rings[i].dc) continue;
    if (now - rings[i].born > 1500) rings.splice(i, 1);
  }

  // persistent DC pulse ring (kept at index 0)
  const dcRing = { lat: dc.lat, lng: dc.lng, rgb: "124,255,203", maxR: 5, speed: 3.2, period: 900, dc: true };
  const ringData = [dcRing, ...rings];

  // push to globe
  globe.arcsData([...arcs]);
  globe.pointsData([...points.values()]);
  globe.ringsData(ringData);

  paintSidebar(data, fresh);
  firstLoad = false;
}

// ── Sidebar / HUD ─────────────────────────────────────────────────────────────
function paintSidebar(data, fresh) {
  const s = data.stats || {};
  setText("tm-stat-events", (s.events || 0).toLocaleString());
  setText("tm-stat-mapped", (s.mapped || 0).toLocaleString());
  setText("tm-stat-ips", (s.unique_ips || 0).toLocaleString());
  setText("tm-stat-countries", (s.countries || 0).toLocaleString());

  // top countries
  const cc = el("tm-countries");
  if (cc) {
    const rows = (s.top_countries || []);
    const max = rows.length ? rows[0].count : 1;
    cc.innerHTML = rows.map((r) => `
      <div class="tm-row">
        <span class="tm-flag">${flagEmoji(r.country)}</span>
        <span class="tm-cn" title="${esc(r.country_name)}">${esc(r.country_name || r.country)}</span>
        <span class="tm-bar"><i style="width:${Math.max(4, (r.count / max) * 100)}%"></i></span>
        <span class="tm-ct">${r.count.toLocaleString()}</span>
      </div>`).join("") || `<div class="tm-empty">No external sources in window</div>`;
  }

  // top signatures
  const sg = el("tm-sigs");
  if (sg) {
    sg.innerHTML = (s.top_signatures || []).map((r) => `
      <div class="tm-sig"><span>${esc(r.sig)}</span><b>${r.count.toLocaleString()}</b></div>`
    ).join("") || `<div class="tm-empty">—</div>`;
  }

  // live ticker — prepend newest attacks
  const tk = el("tm-ticker");
  if (tk && fresh.length) {
    const items = fresh.slice(-8).reverse().map((a) => `
      <div class="tm-tick tm-${a.weight}">
        <span class="tm-dot"></span>
        <span class="tm-tip">${esc(a.country_name || a.country)}${a.city ? " · " + esc(a.city) : ""}</span>
        <span class="tm-tsig">${esc(shortSig(a.sig))}</span>
        <span class="tm-tip2">${esc(a.ip)}</span>
      </div>`).join("");
    tk.insertAdjacentHTML("afterbegin", items);
    while (tk.children.length > 40) tk.removeChild(tk.lastElementChild);
  }
}

function showOffline() {
  const o = el("tm-offline");
  if (o) o.style.display = "flex";
}
function hideOffline() {
  const o = el("tm-offline");
  if (o) o.style.display = "none";
}

// ── Poll loop, gated on tab + document visibility ─────────────────────────────
async function poll() {
  try {
    const r = await fetch("/api/geofeed");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    lastError = null;
    applyFeed(data);
  } catch (e) {
    lastError = e.message || String(e);
  }
}

function isActive() {
  const page = el("page-threatmap");
  return page && page.classList.contains("active") && document.visibilityState === "visible";
}

async function start() {
  if (!mounted) await buildGlobe();
  if (!globe) return;              // import failed — leave a clean no-op
  sizeGlobe();
  globe.controls().autoRotate = true;
  poll();                          // immediate
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => { if (isActive()) poll(); }, POLL_MS);
}

function stop() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  if (globe) globe.controls().autoRotate = false;   // save CPU when hidden
}

export function initThreatMap() {
  const page = el("page-threatmap");
  if (!page) return;

  // React to this tab becoming active / inactive (works for clicks, URL, back/fwd).
  observer = new MutationObserver(() => { isActive() ? start() : stop(); });
  observer.observe(page, { attributes: true, attributeFilter: ["class"] });

  document.addEventListener("visibilitychange", () => { isActive() ? start() : stop(); });

  // If the user deep-linked straight to /threatmap, it's already active.
  if (isActive()) start();
}

// ── tiny utils ────────────────────────────────────────────────────────────────
function setText(id, v) { const e = el(id); if (e) e.textContent = v; }
function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function shortSig(s) { return String(s || "").replace(/^Fortigate:\s*/i, "").slice(0, 38); }
function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
  return m ? `${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)}` : "255,59,92";
}
function flagEmoji(cc) {
  if (!cc || cc.length !== 2 || cc === "??") return "🏴";
  const A = 0x1f1e6;
  return String.fromCodePoint(A + (cc.charCodeAt(0) - 65), A + (cc.charCodeAt(1) - 65));
}
