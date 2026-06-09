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
// Traffic categories — colour + label + arc flow direction. Mirrors the analyst
// legend / filter chips. "in" = external → DC, "out" = DC → external.
const CATEGORY = {
  incoming_threat: { color: "#ff3b5c", label: "Incoming threat",       dir: "in"  },
  normal_incoming: { color: "#3d7bff", label: "Normal incoming",       dir: "in"  },
  outgoing:        { color: "#22c55e", label: "Outgoing from server",  dir: "out" },
  external_conn:   { color: "#facc15", label: "External connection",   dir: "in"  },
};
const CATEGORY_ORDER = ["outgoing", "incoming_threat", "normal_incoming", "external_conn"];
function catColor(cat) { return (CATEGORY[cat] || CATEGORY.normal_incoming).color; }
function catLabel(cat) { return (CATEGORY[cat] || CATEGORY.normal_incoming).label; }

const DC_COLOR = "#7CFFcb";

// Active category filters (all on by default). Toggled by the legend/filter chips.
const activeCats = new Set(CATEGORY_ORDER);

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
let   hoveredPoly  = null;     // country polygon currently under the cursor
let   hoveredArc   = null;     // attack arc (alert) under the cursor
let   hoveredPoint = null;     // attacker point (aggregated source IP) under the cursor
const lastMouse = { x: 0, y: 0 };

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
    // Arcs: direction-aware. Outgoing flows DC → external; everything else
    // flows external → DC. The dash animates start→end so the flow visibly
    // shows which way the traffic is going. Colour = traffic category.
    .arcsData([])
    .arcStartLat((d) => (d.direction === "out" ? dc.lat : d.srcLat))
    .arcStartLng((d) => (d.direction === "out" ? dc.lng : d.srcLng))
    .arcEndLat((d) => (d.direction === "out" ? d.srcLat : dc.lat))
    .arcEndLng((d) => (d.direction === "out" ? d.srcLng : dc.lng))
    .arcColor((d) => {
      const c = catColor(d.category);
      return [rgba(c, 0.12), c];   // faint tail → bright head (the destination)
    })
    // Stroke also defines the arc's tube radius = the hover hit-target, so it's
    // kept a bit chunky to make the lines easy to hover.
    .arcStroke((d) => (d.category === "incoming_threat" ? 0.7 : 0.6))
    .arcDashLength(0.45)
    .arcDashGap(1.4)
    .arcDashInitialGap(() => Math.random())
    .arcDashAnimateTime(ARC_FLY_MS)
    .arcAltitudeAutoScale(0.45)
    .onArcHover((arc) => { hoveredArc = arc; refreshHover(); })
    // Glowing attacker points
    .pointsData([])
    .pointLat((d) => d.lat)
    .pointLng((d) => d.lng)
    .pointColor((d) => d.color)
    .pointAltitude(0.01)
    .pointRadius((d) => Math.min(0.8, 0.18 + Math.log2(1 + d.count) * 0.12))
    .pointsMerge(false)
    .onPointHover((pt) => { hoveredPoint = pt; refreshHover(); })
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
    .ringRepeatPeriod((d) => d.period)
    // Country polygons — invisible until hovered; drive the cursor tooltip.
    .polygonsData([])
    .polygonCapColor((d) => (d === hoveredPoly ? "rgba(0,212,255,0.22)" : "rgba(0,0,0,0)"))
    .polygonSideColor(() => "rgba(0,0,0,0)")
    .polygonStrokeColor((d) => (d === hoveredPoly ? "rgba(0,212,255,0.9)" : "rgba(0,0,0,0)"))
    .polygonAltitude((d) => (d === hoveredPoly ? 0.014 : 0.006))
    .onPolygonHover((poly) => {
      hoveredPoly = poly;
      // re-evaluate the polygon accessors so the highlight repaints
      globe
        .polygonCapColor((d) => (d === hoveredPoly ? "rgba(0,212,255,0.22)" : "rgba(0,0,0,0)"))
        .polygonStrokeColor((d) => (d === hoveredPoly ? "rgba(0,212,255,0.9)" : "rgba(0,0,0,0)"))
        .polygonAltitude((d) => (d === hoveredPoly ? 0.014 : 0.006));
      refreshHover();
    });

  // Camera + auto-rotate
  globe.pointOfView({ lat: 22, lng: 78, altitude: 2.4 }, 0);
  const controls = globe.controls();
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.45;
  controls.enableDamping = true;

  // Load offline country polygons for hover detection (non-fatal if missing).
  try {
    const res = await fetch("/textures/countries-110m.geojson");
    const gj = await res.json();
    globe.polygonsData((gj && gj.features) || []);
  } catch (e) {
    lastError = "country polygons unavailable: " + (e.message || e);
  }

  // Cursor tracking for the hover tooltip.
  const area = el("threatmap-globe");
  if (area) {
    area.addEventListener("mousemove", (e) => {
      const r = area.getBoundingClientRect();
      lastMouse.x = e.clientX - r.left;
      lastMouse.y = e.clientY - r.top;
      if (hoveredArc || hoveredPoint || hoveredPoly) positionTooltip();
    });
    area.addEventListener("mouseleave", () => {
      hoveredPoly = hoveredArc = hoveredPoint = null;
      refreshHover();
    });
  }

  sizeGlobe();
  window.addEventListener("resize", sizeGlobe);
  mounted = true;
  building = false;
}

// ── Hover tooltip (arcs = alerts, points = attacker IPs, polygons = country) ──
// Follows the cursor. Precedence: attacker point > attack arc > country.
function refreshHover() {
  const tip = el("tm-tooltip");
  if (!tip) return;
  let html = null;
  if (hoveredPoint)      html = pointTooltipHTML(hoveredPoint);
  else if (hoveredArc)   html = arcTooltipHTML(hoveredArc);
  else if (hoveredPoly)  html = `<div class="tmtt-title" style="color:var(--accent)">${esc((hoveredPoly.properties || {}).name || "Unknown")}</div>`;

  if (!html) {
    tip.classList.remove("show");
  } else {
    tip.innerHTML = html;
    tip.classList.add("show");
    positionTooltip();
  }
  // Pause auto-rotate while the analyst is inspecting anything.
  if (globe) globe.controls().autoRotate = !(hoveredPoint || hoveredArc || hoveredPoly) && isActive();
}

function positionTooltip() {
  const tip = el("tm-tooltip"); const area = el("threatmap-globe");
  if (!tip || !area) return;
  const pad = 14;
  let x = lastMouse.x + 16, y = lastMouse.y + 16;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  if (x + tw + pad > area.clientWidth)  x = lastMouse.x - tw - 16;
  if (y + th + pad > area.clientHeight) y = lastMouse.y - th - 16;
  tip.style.left = Math.max(4, x) + "px";
  tip.style.top  = Math.max(4, y) + "px";
}

function _row(label, value) {
  if (!value && value !== 0) return "";
  return `<div class="tmtt-row"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
}

function arcTooltipHTML(a) {
  const color = catColor(a.category);
  const loc = [a.country_name || a.country, a.city].filter(Boolean).join(" · ");
  const net = [a.asn ? "AS" + a.asn : "", a.org].filter(Boolean).join(" ");
  const when = [fmtTime(a.time), a.outcome].filter(Boolean).join(" · ");
  const out = a.direction === "out";
  const dirLabel = out ? `Outgoing — DC → ${esc(a.country_name || a.country || "external")}`
                       : `Incoming — ${esc(a.country_name || a.country || "external")} → DC`;
  return `<div class="tmtt-title" style="color:${color}">${flagEmoji(a.country)} ${esc(shortSig(a.sig)) || "Firewall event"}</div>`
       + `<div class="tmtt-cat" style="color:${color}">${esc(catLabel(a.category))}</div>`
       + `<div class="tmtt-row"><span>Direction</span><b style="color:${color}">${dirLabel}</b></div>`
       + _row(out ? "Destination IP" : "Source IP", a.ip)
       + _row("Location", loc)
       + _row("Network", net)
       + _row("When", when);
}

function pointTooltipHTML(p) {
  const color = catColor(p.category);
  const loc = [p.country_name || p.country, p.city].filter(Boolean).join(" · ");
  const net = [p.asn ? "AS" + p.asn : "", p.org].filter(Boolean).join(" ");
  return `<div class="tmtt-title" style="color:${color}">${flagEmoji(p.country)} ${esc(p.ip || "Endpoint")}</div>`
       + `<div class="tmtt-cat" style="color:${color}">${esc(catLabel(p.category))}</div>`
       + _row("Location", loc)
       + _row("Network", net)
       + _row("Events", (p.count || 1).toLocaleString())
       + _row("Last hit", [shortSig(p.lastSig), fmtTime(p.lastTime)].filter(Boolean).join(" · "));
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
    // external endpoint point (accumulate count per IP, carry details for hover)
    const p = points.get(a.ip);
    const color = catColor(a.category);
    if (p) {
      p.count += 1; p.ts = Date.now();
      p.lastSig = a.sig; p.lastTime = a.time;
      p.category = a.category; p.color = color;   // reflect most recent category
    } else {
      points.set(a.ip, {
        lat: a.srcLat, lng: a.srcLng, color, count: 1, ts: Date.now(),
        ip: a.ip, country: a.country, country_name: a.country_name,
        city: a.city, org: a.org, asn: a.asn, category: a.category,
        lastSig: a.sig, lastTime: a.time,
      });
    }
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

  // transient impact rings for the new events (ping at the external endpoint),
  // honouring the active category filters.
  for (const a of ringFor.slice(-12)) {
    if (!activeCats.has(a.category)) continue;
    rings.push({ lat: a.srcLat, lng: a.srcLng, rgb: hexToRgb(catColor(a.category)),
                 maxR: 3, speed: 2.5, period: 1e9, born: Date.now() });
  }
  // age out transient rings (~1.5s each)
  const now = Date.now();
  for (let i = rings.length - 1; i >= 0; i--) {
    if (rings[i].dc) continue;
    if (now - rings[i].born > 1500) rings.splice(i, 1);
  }

  paintGlobe();
  paintSidebar(data, fresh);
  firstLoad = false;
}

// Push the (filtered) arcs / points / rings to the globe. Called on each feed
// and whenever a category filter is toggled.
function paintGlobe() {
  if (!globe) return;
  const visArcs   = arcs.filter((a) => activeCats.has(a.category));
  const visPoints = [...points.values()].filter((p) => activeCats.has(p.category));
  const dcRing = { lat: dc.lat, lng: dc.lng, rgb: "124,255,203",
                   maxR: 5, speed: 3.2, period: 900, dc: true };
  globe.arcsData(visArcs);
  globe.pointsData(visPoints);
  globe.ringsData([dcRing, ...rings]);
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

  // category filter-chip counts
  const cats = s.categories || {};
  for (const c of CATEGORY_ORDER) setText("tmf-" + c, (cats[c] || 0).toLocaleString());

  // live ticker — prepend newest events (respecting the active filters)
  const tk = el("tm-ticker");
  if (tk) {
    const shown = fresh.filter((a) => activeCats.has(a.category));
    if (shown.length) {
      const items = shown.slice(-8).reverse().map((a) => {
        const arrow = a.direction === "out" ? "↑" : "↓";
        return `<div class="tm-tick">
          <span class="tm-dot" style="background:${catColor(a.category)};color:${catColor(a.category)}"></span>
          <span class="tm-tip">${arrow} ${esc(a.country_name || a.country)}${a.city ? " · " + esc(a.city) : ""}</span>
          <span class="tm-tsig">${esc(shortSig(a.sig))}</span>
          <span class="tm-tip2">${esc(a.ip)}</span>
        </div>`;
      }).join("");
      tk.insertAdjacentHTML("afterbegin", items);
      while (tk.children.length > 40) tk.removeChild(tk.lastElementChild);
    }
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

// ── Category filter chips (legend doubles as a filter) ───────────────────────
function initFilters() {
  document.querySelectorAll(".tm-filt").forEach((btn) => {
    const cat = btn.dataset.cat;
    btn.classList.toggle("off", !activeCats.has(cat));
    btn.addEventListener("click", () => {
      if (activeCats.has(cat)) activeCats.delete(cat);
      else activeCats.add(cat);
      btn.classList.toggle("off", !activeCats.has(cat));
      paintGlobe();                 // re-render immediately, no refetch
    });
  });
}

export function initThreatMap() {
  const page = el("page-threatmap");
  if (!page) return;

  initFilters();

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
function shortSig(s) { return String(s || "").replace(/^Fortigate:\s*/i, "").slice(0, 42); }
function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return "";
  return d.toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}
function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
  return m ? `${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)}` : "255,59,92";
}
function rgba(hex, a) { return `rgba(${hexToRgb(hex)},${a})`; }
function flagEmoji(cc) {
  if (!cc || cc.length !== 2 || cc === "??") return "🏴";
  const A = 0x1f1e6;
  return String.fromCodePoint(A + (cc.charCodeAt(0) - 65), A + (cc.charCodeAt(1) - 65));
}
