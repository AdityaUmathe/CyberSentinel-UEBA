// Alert timeline — DPR-aware canvas with smooth bezier lines for
// Total / Anomalous / Critical, plus an interactive hover state showing a
// vertical guide line, dots on each series, and an HTML tooltip with the
// exact counts for the hovered bucket.

// Cached data for hover redraws (one set per canvas — there's only one).
let _last = null;

export function drawTimeline(feedData) {
  // Defer one frame so the canvas has its CSS layout dimensions.
  requestAnimationFrame(() => _compute(feedData));
}

function _compute(feedData) {
  const canvas = document.getElementById("timeline-chart");
  if (!canvas || !feedData || !feedData.length) return;

  const buckets = {};
  feedData.forEach((a) => {
    const t = a.processed_at || a.event_time || "";
    if (!t) return;
    const key = t.slice(0, 13);
    if (!buckets[key]) buckets[key] = { total: 0, critical: 0, anomalous: 0 };
    buckets[key].total++;
    if      (a.verdict === "highly_anomalous") buckets[key].critical++;
    else if (a.verdict === "anomalous")        buckets[key].anomalous++;
  });
  const keys = Object.keys(buckets).sort();
  if (!keys.length) return;

  _last = { canvas, buckets, keys };
  _bindHover(canvas);
  _render(null);
}

function _render(hoverIdx) {
  if (!_last) return;
  const { canvas, buckets, keys } = _last;

  const dpr  = window.devicePixelRatio || 1;
  const cssW = canvas.offsetWidth  || 700;
  const cssH = canvas.offsetHeight || 220;
  canvas.width  = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width  = cssW + "px";
  canvas.style.height = cssH + "px";

  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  const W = cssW, H = cssH;

  const pL = 50, pR = 18, pT = 16, pB = 46;
  const cW = W - pL - pR, cH = H - pT - pB;
  const n = keys.length;
  const maxVal = Math.max(...keys.map((k) => buckets[k].total), 1);

  ctx.clearRect(0, 0, W, H);

  // ── Horizontal grid + Y axis labels ──
  for (let i = 0; i <= 4; i++) {
    const y = pT + cH - (i / 4) * cH;
    ctx.beginPath(); ctx.moveTo(pL, y); ctx.lineTo(W - pR, y);
    ctx.strokeStyle = "rgba(26,40,56,0.55)"; ctx.lineWidth = 0.5;
    ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#3d5a72";
    ctx.font = '500 9px "JetBrains Mono", monospace';
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(Math.round((maxVal * i) / 4), pL - 6, y);
  }

  // ── Axes ──
  ctx.beginPath(); ctx.moveTo(pL, pT); ctx.lineTo(pL, pT + cH);
  ctx.strokeStyle = "rgba(26,40,56,0.9)"; ctx.lineWidth = 1; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pL, pT + cH); ctx.lineTo(W - pR, pT + cH);
  ctx.stroke();

  const getX = (i) => pL + (i / Math.max(n - 1, 1)) * cW;
  const getY = (v) => pT + cH - (v / maxVal) * cH;

  const buildPts = (key) => keys.map((k, i) => ({ x: getX(i), y: getY(buckets[k][key]) }));

  function drawSmooth(pts, close) {
    if (!pts.length) return;
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 0; i < pts.length - 1; i++) {
      const mx = (pts[i].x + pts[i + 1].x) / 2;
      ctx.bezierCurveTo(mx, pts[i].y, mx, pts[i + 1].y, pts[i + 1].x, pts[i + 1].y);
    }
    if (close) {
      ctx.lineTo(pts[pts.length - 1].x, pT + cH);
      ctx.lineTo(pts[0].x, pT + cH);
      ctx.closePath();
    }
  }

  const totalPts = buildPts("total");
  const anomPts  = buildPts("anomalous");
  const critPts  = buildPts("critical");

  // ── Filled areas ──
  const gTotal = ctx.createLinearGradient(0, pT, 0, pT + cH);
  gTotal.addColorStop(0,   "rgba(0,212,255,0.22)");
  gTotal.addColorStop(0.7, "rgba(0,212,255,0.04)");
  gTotal.addColorStop(1,   "rgba(0,0,0,0)");
  ctx.beginPath(); drawSmooth(totalPts, true);
  ctx.fillStyle = gTotal; ctx.fill();

  const gCrit = ctx.createLinearGradient(0, pT, 0, pT + cH);
  gCrit.addColorStop(0,   "rgba(255,59,92,0.28)");
  gCrit.addColorStop(0.7, "rgba(255,59,92,0.04)");
  gCrit.addColorStop(1,   "rgba(0,0,0,0)");
  ctx.beginPath(); drawSmooth(critPts, true);
  ctx.fillStyle = gCrit; ctx.fill();

  // ── Lines ──
  ctx.beginPath(); drawSmooth(totalPts, false);
  ctx.strokeStyle = "rgba(0,212,255,0.95)"; ctx.lineWidth = 2.5;
  ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke();

  ctx.beginPath(); drawSmooth(anomPts, false);
  ctx.strokeStyle = "rgba(255,140,66,0.82)"; ctx.lineWidth = 1.8; ctx.stroke();

  ctx.beginPath(); drawSmooth(critPts, false);
  ctx.strokeStyle = "rgba(255,59,92,0.95)"; ctx.lineWidth = 1.8; ctx.stroke();

  // ── Sparse "always-on" dots (every 1/14th of total) ──
  const dotStep = Math.max(1, Math.floor(n / 14));
  totalPts.forEach((p, i) => {
    if (i % dotStep !== 0 && i !== n - 1) return;
    if (hoverIdx !== null && i === hoverIdx) return; // hover dots drawn below
    ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#00d4ff"; ctx.fill();
    ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0,212,255,0.3)"; ctx.lineWidth = 1.5; ctx.stroke();
  });

  // ── X axis labels — prefer day boundaries when available ──
  const dayBoundaries = [];
  keys.forEach((k, i) => {
    if (i === 0 || k.slice(0, 10) !== keys[i - 1].slice(0, 10)) dayBoundaries.push(i);
  });
  const labelIndices =
    dayBoundaries.length >= 2 && dayBoundaries.length <= 18
      ? dayBoundaries
      : (() => {
          const step = Math.max(1, Math.ceil(n / 8));
          return keys.map((_, i) => i).filter((i) => i % step === 0 || i === n - 1);
        })();
  ctx.fillStyle = "#3d5a72";
  ctx.font = '500 9px "JetBrains Mono", monospace';
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  labelIndices.forEach((i) => {
    const x = getX(i);
    const k = keys[i];
    const lbl = dayBoundaries.length >= 2 && dayBoundaries.length <= 18
      ? k.slice(5, 10)                          // MM-DD on day boundaries
      : k.slice(5).replace("T", " ") + "h";     // MM-DDTHHh otherwise
    ctx.fillText(lbl, x, pT + cH + 6);
  });

  // ── Hover overlay: vertical guide line + bigger dots on each series ──
  if (hoverIdx !== null && hoverIdx >= 0 && hoverIdx < n) {
    const hx = getX(hoverIdx);
    ctx.beginPath(); ctx.moveTo(hx, pT); ctx.lineTo(hx, pT + cH);
    ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);

    const drawHoverDot = (pt, color) => {
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill();
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 1.5; ctx.stroke();
    };
    drawHoverDot(totalPts[hoverIdx], "#00d4ff");
    if (buckets[keys[hoverIdx]].anomalous > 0) drawHoverDot(anomPts[hoverIdx], "#ff8c42");
    if (buckets[keys[hoverIdx]].critical  > 0) drawHoverDot(critPts[hoverIdx],  "#ff3b5c");
  }

  // ── Legend (bottom-right) ──
  const leg = [
    { c: "#00d4ff", l: "Total" },
    { c: "#ff8c42", l: "Anomalous" },
    { c: "#ff3b5c", l: "Critical" },
  ];
  const legY = H - 14;
  leg.forEach((item, i) => {
    const lx = W - 230 + i * 76;
    ctx.beginPath();
    ctx.moveTo(lx, legY); ctx.lineTo(lx + 16, legY);
    ctx.strokeStyle = item.c; ctx.lineWidth = 2.5; ctx.stroke();
    ctx.beginPath(); ctx.arc(lx + 8, legY, 3, 0, Math.PI * 2);
    ctx.fillStyle = item.c; ctx.fill();
    ctx.fillStyle = "#7a9ab5";
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(item.l, lx + 21, legY);
  });

  // ── Date range badge ──
  const badge = document.getElementById("timeline-range");
  if (badge && keys.length) {
    badge.textContent = keys[0].slice(0, 10) + " → " + keys[keys.length - 1].slice(0, 10);
  }

  // Stash bucket geometry for the mousemove handler.
  _last.geom = { pL, pR, pT, pB, cW, cH, W, H, getX, getY, n, totalPts, anomPts, critPts };
}

// ── Hover wiring ──────────────────────────────────────────────────────────────
function _ensureTooltip(canvas) {
  let parent = canvas.parentElement;
  if (!parent) return null;
  if (getComputedStyle(parent).position === "static") parent.style.position = "relative";
  let tip = parent.querySelector(".timeline-tooltip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "timeline-tooltip";
    tip.style.display = "none";
    parent.appendChild(tip);
  }
  return tip;
}

function _fmtBucketKey(k) {
  // Input: "YYYY-MM-DDTHH"
  if (!k || k.length < 13) return k;
  const d = k.slice(0, 10);
  const h = k.slice(11, 13);
  try {
    const day = new Date(d + "T00:00:00Z").toLocaleDateString("en-IN", {
      day: "2-digit", month: "short", year: "2-digit",
    });
    return `${day} · ${h}:00`;
  } catch {
    return `${d} ${h}:00`;
  }
}

function _bindHover(canvas) {
  if (canvas.__hoverBound) return;
  canvas.__hoverBound = true;

  const onMove = (e) => {
    if (!_last || !_last.geom) return;
    const { geom, keys, buckets } = _last;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < geom.pL || x > geom.W - geom.pR || y < geom.pT || y > geom.pT + geom.cH) {
      _render(null);
      const tip = _ensureTooltip(canvas);
      if (tip) tip.style.display = "none";
      return;
    }

    const ratio = (x - geom.pL) / Math.max(geom.cW, 1);
    const idx = Math.max(0, Math.min(geom.n - 1, Math.round(ratio * (geom.n - 1))));
    _render(idx);

    const k = keys[idx];
    const d = buckets[k];
    const tip = _ensureTooltip(canvas);
    if (!tip) return;
    tip.innerHTML = `
      <div class="ttip-head">${_fmtBucketKey(k)}</div>
      <div class="ttip-row"><span class="ttip-sw" style="background:#00d4ff"></span>Total<span class="ttip-val">${d.total}</span></div>
      <div class="ttip-row"><span class="ttip-sw" style="background:#ff8c42"></span>Anomalous<span class="ttip-val" style="color:#ff8c42">${d.anomalous}</span></div>
      <div class="ttip-row"><span class="ttip-sw" style="background:#ff3b5c"></span>Critical<span class="ttip-val" style="color:#ff3b5c">${d.critical}</span></div>`;
    tip.style.display = "block";

    // Position: prefer right of cursor; flip to left near right edge
    const tipW = tip.offsetWidth  || 170;
    const tipH = tip.offsetHeight || 80;
    let left = geom.getX(idx) + 12;
    if (left + tipW > geom.W - 4) left = geom.getX(idx) - tipW - 12;
    let top = geom.totalPts[idx].y - tipH - 10;
    if (top < geom.pT) top = geom.totalPts[idx].y + 14;
    tip.style.left = left + "px";
    tip.style.top  = top + "px";
  };

  const onLeave = () => {
    _render(null);
    const tip = _ensureTooltip(canvas);
    if (tip) tip.style.display = "none";
  };

  canvas.addEventListener("mousemove", onMove);
  canvas.addEventListener("mouseleave", onLeave);
}
