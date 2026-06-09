// Framework-agnostic 3D threat globe (globe.gl / Three.js).
//
// This owns the WebGL canvas + the cursor-following hover tooltip and nothing
// else — no app/framework assumptions. Wrap it in React/Vue/vanilla as you like
// (see ThreatGlobe.jsx). globe.gl/three are loaded via dynamic import() so the
// ~1.5 MB three.js chunk is only fetched when a globe is actually created.
//
//   const engine = await createGlobeEngine(hostEl, {
//     assetBase: "/threat-globe",         // where the textures + geojson live
//     onUpdate: ({ geoOk, stats, fresh }) => { ...drive your HUD... },
//   });
//   engine.applyFeed(geofeedJson);        // call on each poll / push
//   engine.setActiveCategories(new Set([...]));
//   engine.setRunning(true|false);        // pause auto-rotate / when hidden
//   engine.resize();
//   engine.destroy();

import {
  CATEGORY_ORDER, catColor, catLabel, hexToRgb, rgba,
  flagEmoji, shortSig, fmtTime, esc,
} from "./categories.js";

const MAX_ARCS   = 70;     // rolling on-screen arc budget (animated)
const MAX_POINTS = 90;     // rolling endpoint-point budget
const ARC_FLY_MS = 2200;   // arc dash travel time

export async function createGlobeEngine(host, options = {}) {
  const opts = {
    assetBase: "/threat-globe",
    autoRotate: true,
    onUpdate: () => {},
    ...options,
  };
  const A = String(opts.assetBase).replace(/\/+$/, "");

  // ── instance state ──
  let globe = null;
  let dc = opts.dc || { lat: 21.1463, lng: 79.0849, label: "Datacenter" };
  let running = true;
  let destroyed = false;
  const activeCats = new Set(CATEGORY_ORDER);

  const arcs = [];                 // rolling display buffer
  const points = new Map();        // ip -> aggregate
  const rings = [];                // transient impact rings
  const seenKeys = new Set();
  let seenOrder = [];
  let firstLoad = true;

  let hoveredPoly = null, hoveredArc = null, hoveredPoint = null;
  const lastMouse = { x: 0, y: 0 };

  // ── tooltip element (engine-owned, lives inside the host) ──
  const tip = document.createElement("div");
  tip.className = "tg-tooltip";
  host.appendChild(tip);

  // ── build the globe ──
  let Globe;
  try {
    ({ default: Globe } = await import("globe.gl"));
  } catch (e) {
    return _stub("Failed to load globe.gl: " + (e && e.message || e));
  }
  if (destroyed) return _stub("destroyed before init");

  globe = new Globe(host)
    .backgroundColor("#01030a")
    .backgroundImageUrl(`${A}/night-sky.png`)
    .globeImageUrl(`${A}/earth-night.jpg`)
    .showAtmosphere(true)
    .atmosphereColor("#2b6cff")
    .atmosphereAltitude(0.18)
    // Arcs: direction-aware. Outgoing flows DC → external; everything else
    // flows external → DC. Dash animates start→end so the flow shows direction.
    .arcsData([])
    .arcStartLat((d) => (d.direction === "out" ? dc.lat : d.srcLat))
    .arcStartLng((d) => (d.direction === "out" ? dc.lng : d.srcLng))
    .arcEndLat((d) => (d.direction === "out" ? d.srcLat : dc.lat))
    .arcEndLng((d) => (d.direction === "out" ? d.srcLng : dc.lng))
    .arcColor((d) => { const c = catColor(d.category); return [rgba(c, 0.12), c]; })
    .arcStroke((d) => (d.category === "incoming_threat" ? 0.7 : 0.6))
    .arcDashLength(0.45)
    .arcDashGap(1.4)
    .arcDashInitialGap(() => Math.random())
    .arcDashAnimateTime(ARC_FLY_MS)
    .arcAltitudeAutoScale(0.45)
    .onArcHover((arc) => { hoveredArc = arc; refreshHover(); })
    // Endpoint points
    .pointsData([])
    .pointLat((d) => d.lat)
    .pointLng((d) => d.lng)
    .pointColor((d) => d.color)
    .pointAltitude(0.01)
    .pointRadius((d) => Math.min(0.8, 0.18 + Math.log2(1 + d.count) * 0.12))
    .pointsMerge(false)
    .onPointHover((pt) => { hoveredPoint = pt; refreshHover(); })
    // Impact rings (DC pulse + transient pings)
    .ringsData([])
    .ringLat((d) => d.lat)
    .ringLng((d) => d.lng)
    .ringColor((d) => (t) => `rgba(${d.rgb},${(1 - t).toFixed(3)})`)
    .ringMaxRadius((d) => d.maxR)
    .ringPropagationSpeed((d) => d.speed)
    .ringRepeatPeriod((d) => d.period)
    // Country polygons — invisible until hovered; drive the tooltip
    .polygonsData([])
    .polygonCapColor((d) => (d === hoveredPoly ? "rgba(0,212,255,0.22)" : "rgba(0,0,0,0)"))
    .polygonSideColor(() => "rgba(0,0,0,0)")
    .polygonStrokeColor((d) => (d === hoveredPoly ? "rgba(0,212,255,0.9)" : "rgba(0,0,0,0)"))
    .polygonAltitude((d) => (d === hoveredPoly ? 0.014 : 0.006))
    .onPolygonHover((poly) => {
      hoveredPoly = poly;
      globe
        .polygonCapColor((d) => (d === hoveredPoly ? "rgba(0,212,255,0.22)" : "rgba(0,0,0,0)"))
        .polygonStrokeColor((d) => (d === hoveredPoly ? "rgba(0,212,255,0.9)" : "rgba(0,0,0,0)"))
        .polygonAltitude((d) => (d === hoveredPoly ? 0.014 : 0.006));
      refreshHover();
    });

  globe.pointOfView({ lat: 22, lng: 78, altitude: 2.4 }, 0);
  const controls = globe.controls();
  controls.autoRotate = opts.autoRotate;
  controls.autoRotateSpeed = 0.45;
  controls.enableDamping = true;

  // offline country polygons for hover detection (non-fatal if missing)
  try {
    const res = await fetch(`${A}/countries-110m.geojson`);
    const gj = await res.json();
    if (!destroyed) globe.polygonsData((gj && gj.features) || []);
  } catch (e) { /* hover-by-country disabled, harmless */ }

  // cursor tracking
  const onMove = (e) => {
    const r = host.getBoundingClientRect();
    lastMouse.x = e.clientX - r.left;
    lastMouse.y = e.clientY - r.top;
    if (hoveredArc || hoveredPoint || hoveredPoly) positionTooltip();
  };
  const onLeave = () => { hoveredPoly = hoveredArc = hoveredPoint = null; refreshHover(); };
  host.addEventListener("mousemove", onMove);
  host.addEventListener("mouseleave", onLeave);

  resize();

  // ── hover tooltip ──
  function refreshHover() {
    let html = null;
    if (hoveredPoint)     html = pointTooltipHTML(hoveredPoint);
    else if (hoveredArc)  html = arcTooltipHTML(hoveredArc);
    else if (hoveredPoly) html = `<div class="tg-tt-title" style="color:#00d4ff">${esc((hoveredPoly.properties || {}).name || "Unknown")}</div>`;
    if (!html) { tip.classList.remove("show"); }
    else { tip.innerHTML = html; tip.classList.add("show"); positionTooltip(); }
    if (globe) globe.controls().autoRotate = !(hoveredPoint || hoveredArc || hoveredPoly) && running;
  }
  function positionTooltip() {
    const pad = 14;
    let x = lastMouse.x + 16, y = lastMouse.y + 16;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    if (x + tw + pad > host.clientWidth)  x = lastMouse.x - tw - 16;
    if (y + th + pad > host.clientHeight) y = lastMouse.y - th - 16;
    tip.style.left = Math.max(4, x) + "px";
    tip.style.top  = Math.max(4, y) + "px";
  }
  function _row(label, value) {
    if (!value && value !== 0) return "";
    return `<div class="tg-tt-row"><span>${esc(label)}</span><b>${esc(value)}</b></div>`;
  }
  function arcTooltipHTML(a) {
    const color = catColor(a.category);
    const loc = [a.country_name || a.country, a.city].filter(Boolean).join(" · ");
    const net = [a.asn ? "AS" + a.asn : "", a.org].filter(Boolean).join(" ");
    const when = [fmtTime(a.time), a.outcome].filter(Boolean).join(" · ");
    const out = a.direction === "out";
    const dirLabel = out ? `Outgoing — DC → ${esc(a.country_name || a.country || "external")}`
                         : `Incoming — ${esc(a.country_name || a.country || "external")} → DC`;
    return `<div class="tg-tt-title" style="color:${color}">${flagEmoji(a.country)} ${esc(shortSig(a.sig)) || "Firewall event"}</div>`
         + `<div class="tg-tt-cat" style="color:${color}">${esc(catLabel(a.category))}</div>`
         + `<div class="tg-tt-row"><span>Direction</span><b style="color:${color}">${dirLabel}</b></div>`
         + _row(out ? "Destination IP" : "Source IP", a.ip)
         + _row("Location", loc) + _row("Network", net) + _row("When", when);
  }
  function pointTooltipHTML(p) {
    const color = catColor(p.category);
    const loc = [p.country_name || p.country, p.city].filter(Boolean).join(" · ");
    const net = [p.asn ? "AS" + p.asn : "", p.org].filter(Boolean).join(" ");
    return `<div class="tg-tt-title" style="color:${color}">${flagEmoji(p.country)} ${esc(p.ip || "Endpoint")}</div>`
         + `<div class="tg-tt-cat" style="color:${color}">${esc(catLabel(p.category))}</div>`
         + _row("Location", loc) + _row("Network", net)
         + _row("Events", (p.count || 1).toLocaleString())
         + _row("Last hit", [shortSig(p.lastSig), fmtTime(p.lastTime)].filter(Boolean).join(" · "));
  }

  // ── feed application ──
  function rememberKey(k) {
    seenKeys.add(k); seenOrder.push(k);
    if (seenOrder.length > 4000) { seenOrder.splice(0, 2000).forEach((x) => seenKeys.delete(x)); }
  }
  function applyFeed(data) {
    if (!data) return;
    dc = data.dc || dc;
    if (!data.geo_ok) { opts.onUpdate({ geoOk: false, stats: data.stats || {}, fresh: [] }); return; }

    const incoming = Array.isArray(data.arcs) ? data.arcs : [];
    const fresh = [];
    for (const a of incoming) {
      const k = `${a.ip}|${a.time}|${a.sig}`;
      if (seenKeys.has(k)) continue;
      rememberKey(k); fresh.push(a);
    }
    const ringFor = firstLoad ? [] : fresh;

    for (const a of fresh) {
      arcs.push(a);
      const color = catColor(a.category);
      const p = points.get(a.ip);
      if (p) {
        p.count += 1; p.ts = Date.now();
        p.lastSig = a.sig; p.lastTime = a.time; p.category = a.category; p.color = color;
      } else {
        points.set(a.ip, {
          lat: a.srcLat, lng: a.srcLng, color, count: 1, ts: Date.now(),
          ip: a.ip, country: a.country, country_name: a.country_name,
          city: a.city, org: a.org, asn: a.asn, category: a.category,
          lastSig: a.sig, lastTime: a.time,
        });
      }
    }
    if (firstLoad && fresh.length) { const seed = fresh.slice(-MAX_ARCS); arcs.length = 0; arcs.push(...seed); }
    while (arcs.length > MAX_ARCS) arcs.shift();
    if (points.size > MAX_POINTS) {
      const oldest = [...points.entries()].sort((a, b) => a[1].ts - b[1].ts);
      for (let i = 0; i < points.size - MAX_POINTS; i++) points.delete(oldest[i][0]);
    }

    for (const a of ringFor.slice(-12)) {
      if (!activeCats.has(a.category)) continue;
      rings.push({ lat: a.srcLat, lng: a.srcLng, rgb: hexToRgb(catColor(a.category)),
                   maxR: 3, speed: 2.5, period: 1e9, born: Date.now() });
    }
    const now = Date.now();
    for (let i = rings.length - 1; i >= 0; i--) {
      if (rings[i].dc) continue;
      if (now - rings[i].born > 1500) rings.splice(i, 1);
    }

    paintGlobe();
    firstLoad = false;
    opts.onUpdate({ geoOk: true, stats: data.stats || {}, fresh });
  }

  function paintGlobe() {
    if (!globe) return;
    globe.arcsData(arcs.filter((a) => activeCats.has(a.category)));
    globe.pointsData([...points.values()].filter((p) => activeCats.has(p.category)));
    const dcRing = { lat: dc.lat, lng: dc.lng, rgb: "124,255,203", maxR: 5, speed: 3.2, period: 900, dc: true };
    globe.ringsData([dcRing, ...rings]);
  }

  function resize() {
    if (!globe) return;
    const r = host.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) globe.width(r.width).height(r.height);
  }

  // ── public API ──
  return {
    applyFeed,
    setActiveCategories(set) {
      activeCats.clear(); for (const c of set) activeCats.add(c);
      paintGlobe();
    },
    setRunning(on) { running = !!on; if (globe) globe.controls().autoRotate = running && !(hoveredArc || hoveredPoint || hoveredPoly); },
    resize,
    getDC() { return dc; },
    destroy() {
      destroyed = true;
      host.removeEventListener("mousemove", onMove);
      host.removeEventListener("mouseleave", onLeave);
      try { tip.remove(); } catch (e) {}
      try { if (globe) globe._destructor && globe._destructor(); } catch (e) {}
      try { host.querySelectorAll("canvas").forEach((c) => c.remove()); } catch (e) {}
      globe = null;
    },
  };

  function _stub(err) {
    // graceful no-op engine when globe.gl can't load
    opts.onUpdate({ geoOk: false, stats: {}, fresh: [], error: err });
    return {
      applyFeed: () => {}, setActiveCategories: () => {}, setRunning: () => {},
      resize: () => {}, getDC: () => dc, destroy: () => { try { tip.remove(); } catch (e) {} }, error: err,
    };
  }
}
