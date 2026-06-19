// Live Threat Map — 2D flat-map renderer (pure SVG), a feature-parity alternative
// to the 3D globe in threatmap.js. It consumes the exact same /api/geofeed data,
// the same traffic categories/colours, the same hover tooltips and the same
// category filters — only the rendering surface differs.
//
// Rendering model: a single SVG drawn in a fixed 2000×1000 "map-unit" space
// (2:1 equirectangular) and scaled to the container via preserveAspectRatio, so
// it stays undistorted and crisp at any size. lat/lng project linearly into that
// box, which is pixel-aligned with the equirectangular earth-night.jpg drawn as
// the SVG backdrop — the same offline texture the globe uses. Lines use
// non-scaling-stroke so they stay sharp regardless of the on-screen scale.
//
// The controller (threatmap.js) owns all data/state; this module is a dumb
// renderer that reconciles the current frame onto persistent SVG nodes (so arc
// "comet" animations are preserved across the 6 s polls instead of restarting).

const VB_W = 2000, VB_H = 1000;
const SVGNS = "http://www.w3.org/2000/svg";
const XLINK = "http://www.w3.org/1999/xlink";

let _uid = 0;

function proj(lng, lat) {
  return [((lng + 180) / 360) * VB_W, ((90 - lat) / 180) * VB_H];
}
function svg(tag, attrs) {
  const e = document.createElementNS(SVGNS, tag);
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

// Build the flat-map renderer inside `host`. `cb` supplies the shared bits from
// the controller so hover/colour/keying behave identically to the globe:
//   colorFor(cat) · flyMs · arcKey(arc) · onArc/onPoint/onPoly(d|null) · onMove(x,y)
export function createFlatMap(host, cb) {
  const root = svg("svg", {
    viewBox: `0 0 ${VB_W} ${VB_H}`,
    preserveAspectRatio: "xMidYMid meet",
    class: "tm2-svg",
  });

  // defs: point/comet glow, the night-hemisphere clip region, sun gradient
  const defs = svg("defs");
  defs.innerHTML =
    '<filter id="tm2glow" x="-60%" y="-60%" width="220%" height="220%">' +
      '<feGaussianBlur stdDeviation="5" result="b"/>' +
      '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>' +
    '<clipPath id="tm2-nightclip"><path id="tm2-nightclip-path"/></clipPath>' +
    '<radialGradient id="tm2-sun" cx="50%" cy="50%" r="50%">' +
      '<stop offset="0%" stop-color="#fffbe8" stop-opacity="1"/>' +
      '<stop offset="28%" stop-color="#ffe79a" stop-opacity="0.85"/>' +
      '<stop offset="68%" stop-color="#ffc94d" stop-opacity="0.22"/>' +
      '<stop offset="100%" stop-color="#ffbe3c" stop-opacity="0"/>' +
    '</radialGradient>';
  root.appendChild(defs);

  // Equirectangular earth: day (blue-marble) base + night-lights clipped to the
  // dark hemisphere, so the lit side is daytime and the dark side shows city
  // lights — mirroring the 3D globe's day/night blend. Same offline textures.
  const dayImg = svg("image", { x: 0, y: 0, width: VB_W, height: VB_H,
    preserveAspectRatio: "none", class: "tm2-earth-day" });
  dayImg.setAttributeNS(XLINK, "href", "/textures/earth-day.jpg");
  dayImg.setAttribute("href", "/textures/earth-day.jpg");
  const nightImg = svg("image", { x: 0, y: 0, width: VB_W, height: VB_H,
    preserveAspectRatio: "none", class: "tm2-earth-night", "clip-path": "url(#tm2-nightclip)" });
  nightImg.setAttributeNS(XLINK, "href", "/textures/earth-night.jpg");
  nightImg.setAttribute("href", "/textures/earth-night.jpg");
  const termLine = svg("path", { class: "tm2-terminator", fill: "none" });
  root.append(dayImg, nightImg, termLine);
  const nightClip = defs.querySelector("#tm2-nightclip-path");

  const gCountries = svg("g", { class: "tm2-countries" });
  const gRings     = svg("g", { class: "tm2-rings" });
  const gArcs      = svg("g", { class: "tm2-arcs" });
  const sunG       = svg("g", { class: "tm2-sun" });
  sunG.innerHTML = '<circle r="40" fill="url(#tm2-sun)"/><circle r="8" class="tm2-sun-core"/>';
  const gPoints    = svg("g", { class: "tm2-points" });
  root.append(gCountries, gRings, gArcs, sunG, gPoints);
  host.appendChild(root);

  // Cursor tracking + hover clear (mirrors the globe's mouse handling)
  host.addEventListener("mousemove", (e) => {
    const r = host.getBoundingClientRect();
    cb.onMove(e.clientX - r.left, e.clientY - r.top);
  });
  host.addEventListener("mouseleave", () => { cb.onArc(null); cb.onPoint(null); cb.onPoly(null); });

  // ── zoom / pan ───────────────────────────────────────────────────────────────
  // Mutate the SVG viewBox: wheel zooms toward the cursor, drag pans, double-click
  // resets. getScreenCTM maps client px → SVG user units exactly (accounts for the
  // preserveAspectRatio letterboxing), so the point under the cursor stays put.
  const view = { x: 0, y: 0, w: VB_W, h: VB_H };
  const MIN_W = VB_W / 14;                 // max zoom ≈ 14×
  const _pt = root.createSVGPoint();
  function applyView() {
    root.setAttribute("viewBox",
      `${view.x.toFixed(2)} ${view.y.toFixed(2)} ${view.w.toFixed(2)} ${view.h.toFixed(2)}`);
  }
  function clientToSvg(cx, cy) {
    const m = root.getScreenCTM();
    if (!m) return { x: 0, y: 0 };
    _pt.x = cx; _pt.y = cy;
    const p = _pt.matrixTransform(m.inverse());
    return { x: p.x, y: p.y };
  }
  function clampPan() {
    if (view.w >= VB_W) { view.w = VB_W; view.h = VB_H; view.x = 0; view.y = 0; return; }
    view.x = Math.max(0, Math.min(VB_W - view.w, view.x));
    view.y = Math.max(0, Math.min(VB_H - view.h, view.y));
  }
  host.addEventListener("wheel", (e) => {
    e.preventDefault();
    const before = clientToSvg(e.clientX, e.clientY);
    const factor = e.deltaY < 0 ? 0.84 : 1 / 0.84;        // in : out
    let nw = Math.max(MIN_W, Math.min(VB_W, view.w * factor));
    const nh = nw * (VB_H / VB_W);
    const rx = (before.x - view.x) / view.w;
    const ry = (before.y - view.y) / view.h;
    view.w = nw; view.h = nh;
    view.x = before.x - rx * nw;
    view.y = before.y - ry * nh;
    clampPan(); applyView();
  }, { passive: false });

  // Drag to pan — keep the SVG point grabbed on mousedown pinned to the cursor.
  let grab = null;
  host.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    grab = clientToSvg(e.clientX, e.clientY);
    host.classList.add("tm2-grabbing");
  });
  window.addEventListener("mousemove", (e) => {
    if (!grab) return;
    const cur = clientToSvg(e.clientX, e.clientY);
    view.x += grab.x - cur.x;
    view.y += grab.y - cur.y;
    clampPan(); applyView();
  });
  window.addEventListener("mouseup", () => {
    grab = null; host.classList.remove("tm2-grabbing");
  });
  host.addEventListener("dblclick", (e) => {
    e.preventDefault();
    view.x = 0; view.y = 0; view.w = VB_W; view.h = VB_H; applyView();
  });

  // reconcile state
  const arcEls  = new Map();   // arcKey -> <g>
  const ptEls   = new Map();   // ip -> <circle>
  const ringEls = new Map();   // key -> <circle>
  let dc = null, dcEl = null;

  // ── country polygons (drawn once) ──────────────────────────────────────────
  function geoPath(geom) {
    if (!geom) return "";
    const polys = geom.type === "Polygon" ? [geom.coordinates]
      : geom.type === "MultiPolygon" ? geom.coordinates : null;
    if (!polys) return "";
    let d = "";
    for (const poly of polys) {
      for (const ring of poly) {
        let prevLng = null, started = false;
        for (const pt of ring) {
          const lng = pt[0], lat = pt[1];
          const xy = proj(lng, lat);
          // break the subpath across the antimeridian so it doesn't streak
          if (prevLng !== null && Math.abs(lng - prevLng) > 180) started = false;
          d += (started ? "L" : "M") + xy[0].toFixed(1) + "," + xy[1].toFixed(1) + " ";
          started = true; prevLng = lng;
        }
        d += "Z ";
      }
    }
    return d;
  }
  function setPolygons(features) {
    gCountries.textContent = "";
    const frag = document.createDocumentFragment();
    for (const f of features || []) {
      const d = geoPath(f.geometry);
      if (!d) continue;
      const p = svg("path", { d, class: "tm2-country" });
      p.addEventListener("mouseenter", () => { p.classList.add("hov"); cb.onPoly(f); });
      p.addEventListener("mouseleave", () => { p.classList.remove("hov"); cb.onPoly(null); });
      frag.appendChild(p);
    }
    gCountries.appendChild(frag);
  }

  // ── datacenter marker ───────────────────────────────────────────────────────
  function setDC(d) { dc = d; ensureDC(); }
  function ensureDC() {
    if (!dc) return;
    const xy = proj(dc.lng, dc.lat), x = xy[0], y = xy[1], s = 9;
    if (!dcEl) { dcEl = svg("path", { class: "tm2-dc" }); gPoints.appendChild(dcEl); }
    dcEl.setAttribute("d", `M${x},${y - s} L${x + s},${y} L${x},${y + s} L${x - s},${y} Z`);
  }

  // ── arcs ────────────────────────────────────────────────────────────────────
  // A faint route line plus a bright "comet" head travelling start→end along it,
  // so direction reads instantly. Outgoing flows DC→external; everything else
  // external→DC, matching the globe.
  function arcPathD(a) {
    const ext = proj(a.srcLng, a.srcLat), home = proj(dc.lng, dc.lat);
    const from = a.direction === "out" ? home : ext;
    const to   = a.direction === "out" ? ext  : home;
    const x1 = from[0], y1 = from[1], x2 = to[0], y2 = to[1];
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const lift = Math.min(len * 0.28, 260);
    let nx = -dy / len, ny = dx / len;
    if (ny > 0) { nx = -nx; ny = -ny; }          // always bow toward the top
    const cx = (x1 + x2) / 2 + nx * lift, cy = (y1 + y2) / 2 + ny * lift;
    return `M${x1.toFixed(1)},${y1.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`;
  }
  function reconcileArcs(list) {
    const seen = new Set();
    for (const a of list) {
      const k = cb.arcKey(a); seen.add(k);
      if (arcEls.has(k)) continue;
      const color = cb.colorFor(a.category);
      const d = arcPathD(a);
      const id = "tm2a" + (_uid++);
      const g = svg("g", { class: "tm2-arc" });
      const route = svg("path", { id, d, fill: "none", stroke: color, class: "tm2-route" });
      const hit = svg("path", { d, fill: "none", stroke: "transparent", "stroke-width": "16", class: "tm2-arc-hit" });
      hit.addEventListener("mouseenter", () => cb.onArc(a));
      hit.addEventListener("mouseleave", () => cb.onArc(null));
      const head = svg("circle", { r: "5", class: "tm2-arc-head", fill: color });
      head.style.setProperty("--c", color);
      const mot = svg("animateMotion", { dur: (cb.flyMs / 1000) + "s", repeatCount: "indefinite", begin: "0s" });
      const mp = svg("mpath");
      mp.setAttributeNS(XLINK, "href", "#" + id);
      mp.setAttribute("href", "#" + id);
      mot.appendChild(mp); head.appendChild(mot);
      g.append(route, hit, head);
      gArcs.appendChild(g);
      arcEls.set(k, g);
    }
    for (const [k, g] of arcEls) if (!seen.has(k)) { g.remove(); arcEls.delete(k); }
  }

  // ── attacker points ─────────────────────────────────────────────────────────
  function reconcilePoints(list) {
    const seen = new Set();
    for (const p of list) {
      seen.add(p.ip);
      let c = ptEls.get(p.ip);
      const xy = proj(p.lng, p.lat);
      const r = Math.min(16, 5 + Math.log2(1 + (p.count || 1)) * 3);
      const color = cb.colorFor(p.category);
      if (!c) {
        c = svg("circle", { class: "tm2-point" });
        c.addEventListener("mouseenter", () => cb.onPoint(c.__d));
        c.addEventListener("mouseleave", () => cb.onPoint(null));
        gPoints.appendChild(c);
        ptEls.set(p.ip, c);
      }
      c.__d = p;
      c.setAttribute("cx", xy[0].toFixed(1));
      c.setAttribute("cy", xy[1].toFixed(1));
      c.setAttribute("r", r.toFixed(1));
      c.setAttribute("fill", color);
      c.style.setProperty("--c", color);
    }
    for (const [ip, c] of ptEls) if (!seen.has(ip)) { c.remove(); ptEls.delete(ip); }
  }

  // ── impact rings ────────────────────────────────────────────────────────────
  function makeAnim(attr, from, to, dur, rep) {
    return svg("animate", { attributeName: attr, from: String(from), to: String(to),
      dur, repeatCount: rep, fill: "freeze" });
  }
  function reconcileRings(list) {
    const seen = new Set();
    for (const rg of list) {
      const key = rg.dc ? "dc" : `${rg.born}|${rg.lat}|${rg.lng}`;
      seen.add(key);
      if (ringEls.has(key)) continue;
      const xy = proj(rg.lng, rg.lat);
      const c = svg("circle", { cx: xy[0].toFixed(1), cy: xy[1].toFixed(1), r: "2",
        fill: "none", stroke: `rgb(${rg.rgb})`, class: rg.dc ? "tm2-ring tm2-ring-dc" : "tm2-ring" });
      const maxR = rg.dc ? 64 : 44;
      const dur  = rg.dc ? "2.4s" : "1.3s";
      const rep  = rg.dc ? "indefinite" : "1";
      c.append(makeAnim("r", 2, maxR, dur, rep),
               makeAnim("opacity", rg.dc ? 0.5 : 0.9, 0, dur, rep));
      gRings.appendChild(c);
      ringEls.set(key, c);
      if (!rg.dc) setTimeout(() => { c.remove(); ringEls.delete(key); }, 1600);
    }
    for (const [k, c] of ringEls) if (!seen.has(k) && k !== "dc") { c.remove(); ringEls.delete(k); }
  }

  function render(frame) {
    if (frame.dc) setDC(frame.dc);
    if (dc) ensureDC();
    reconcileRings(frame.rings || []);
    reconcileArcs(frame.arcs || []);
    reconcilePoints(frame.points || []);
  }

  // ── live day/night ──────────────────────────────────────────────────────────
  // Terminator latitude for a longitude: the great circle 90° from the subsolar
  // point. lat = atan(-cos(lng - sunLng) / tan(sunLat)).
  const RAD = Math.PI / 180;
  function terminatorPts(sub) {
    let tanLat = Math.tan(sub.lat * RAD);
    if (Math.abs(tanLat) < 0.01) tanLat = (tanLat < 0 ? -1 : 1) * 0.01;  // guard near equinox noon
    const pts = [];
    for (let lng = -180; lng <= 180; lng += 2) {
      const lat = Math.atan(-Math.cos((lng - sub.lng) * RAD) / tanLat) / RAD;
      pts.push(proj(lng, lat));
    }
    return pts;
  }
  function polyline(pts) {
    let d = "";
    for (let i = 0; i < pts.length; i++) d += (i ? "L" : "M") + pts[i][0].toFixed(1) + "," + pts[i][1].toFixed(1) + " ";
    return d;
  }
  function updateDayNight(sub) {
    if (!sub) return;
    const pts = terminatorPts(sub);
    // night region = terminator curve closed around the pole in darkness
    // (the hemisphere opposite the sun's declination)
    const poleY = sub.lat >= 0 ? proj(0, -90)[1] : proj(0, 90)[1];
    const night = polyline(pts) + `L${VB_W},${poleY.toFixed(1)} L0,${poleY.toFixed(1)} Z`;
    nightClip.setAttribute("d", night);
    termLine.setAttribute("d", polyline(pts));
    const s = proj(sub.lng, sub.lat);
    sunG.setAttribute("transform", `translate(${s[0].toFixed(1)},${s[1].toFixed(1)})`);
  }

  return { el: root, setPolygons, setDC, render, updateDayNight };
}
