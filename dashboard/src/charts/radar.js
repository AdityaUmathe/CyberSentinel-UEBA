// Radar / spider chart — SVG renderer for top anomaly reasons.
//
// Long reason names are word-wrapped onto up to two lines so the chart
// itself stays at full size (no shrinking the viewBox to make room).
// Each label group is positioned just past the spoke endpoint and grows
// outward — upward for top half labels, downward for bottom half — so it
// can't collide with the ring stack.

export function drawRadar(reasons) {
  if (!reasons || !reasons.length) return;

  const n = Math.min(reasons.length, 7);
  const maxVal = Math.max(...reasons.slice(0, n).map((r) => r.count), 1);

  const W = 480, H = 400;
  const cx = W / 2, cy = H / 2 - 10;
  const R = Math.min(cx, cy) - 80;
  const labelR = R + 30;

  function pt(i, radius) {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2;
    return { x: cx + radius * Math.cos(a), y: cy + radius * Math.sin(a), a };
  }

  let rings = "";
  for (let ring = 1; ring <= 4; ring++) {
    const rr = (R * ring) / 4;
    const pts = Array.from({ length: n }, (_, i) => pt(i, rr));
    const d =
      pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") +
      " Z";
    const isOuter = ring === 4;
    rings += `<path d="${d}" fill="none" stroke="${
      isOuter ? "rgba(0,212,255,0.22)" : "rgba(26,40,56,0.9)"
    }" stroke-width="${isOuter ? 1.5 : 0.8}"/>`;
  }

  let spokes = "";
  for (let i = 0; i < n; i++) {
    const p = pt(i, R);
    spokes += `<line x1="${cx}" y1="${cy}" x2="${p.x.toFixed(1)}" y2="${p.y.toFixed(
      1
    )}" stroke="rgba(0,212,255,0.12)" stroke-width="1"/>`;
  }

  const dataPts = reasons.slice(0, n).map((item, i) => {
    const val = (item.count / maxVal) * R;
    return { ...pt(i, val), item };
  });
  const polyD =
    dataPts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") +
    " Z";

  function escapeXml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // Wrap a label into 1 or 2 lines at a word boundary. If the text is one
  // very long token with no spaces, hard-break it. Hard cap at 36 chars
  // total — beyond that we ellipsize the second line.
  function wrapLabel(text, maxPerLine = 16) {
    if (text.length <= maxPerLine) return [text];
    // Try to split on the last space at or before maxPerLine.
    const breakAt = text.lastIndexOf(" ", maxPerLine);
    let first, rest;
    if (breakAt <= 2) {
      first = text.slice(0, maxPerLine);
      rest  = text.slice(maxPerLine);
    } else {
      first = text.slice(0, breakAt);
      rest  = text.slice(breakAt + 1);
    }
    if (rest.length > maxPerLine) rest = rest.slice(0, maxPerLine - 1) + "…";
    return [first, rest];
  }

  // Build labels AND dots together. Each "spoke" (reason) gets a single
  // <g> wrapper carrying a <title> with the full reason name + count, so
  // hovering anywhere over its label text OR its data dot shows the same
  // tooltip. A transparent rect behind the text widens the hover hit-area
  // so users don't have to land exactly on the glyphs.
  let spokeGroups = "";
  reasons.slice(0, n).forEach((item, i) => {
    const { x, y, a } = pt(i, labelR);
    const cosA = Math.cos(a);
    const sinA = Math.sin(a);
    const anchor = cosA > 0.2 ? "start" : cosA < -0.2 ? "end" : "middle";

    const raw = (item.reason || "").replace(/_/g, " ");
    const lines = wrapLabel(raw, 16).map(escapeXml);
    const escapedRaw = escapeXml(raw);

    // Vertical layout: build relative offsets, then shift the group so top
    // labels float above y, bottom labels float below y, sides centered.
    const lineH = 11;
    const fontReason = 10;
    const fontCount  = 13;
    const totalH = lines.length * lineH + fontCount + 2;
    let baseDy;
    if (sinA < -0.2)      baseDy = -totalH + fontReason;     // top half
    else if (sinA > 0.2)  baseDy = fontReason + 2;           // bottom half
    else                  baseDy = -totalH / 2 + fontReason; // sides

    // Reason text lines.
    let reasonTxt = "";
    let labelMinY = Infinity, labelMaxY = -Infinity;
    lines.forEach((ln, idx) => {
      const ly = y + baseDy + idx * lineH;
      labelMinY = Math.min(labelMinY, ly - fontReason);
      labelMaxY = Math.max(labelMaxY, ly + 2);
      reasonTxt += `<text x="${x.toFixed(1)}" y="${ly.toFixed(
        1
      )}" text-anchor="${anchor}" font-family="JetBrains Mono,monospace" font-size="${fontReason}" font-weight="500" fill="#a8c0d2">${ln}</text>`;
    });
    // Count line, sits one line below the last reason line.
    const cy_ = y + baseDy + lines.length * lineH + 2;
    labelMaxY = Math.max(labelMaxY, cy_ + 2);
    const countTxt = `<text x="${x.toFixed(1)}" y="${cy_.toFixed(
      1
    )}" text-anchor="${anchor}" font-family="JetBrains Mono,monospace" font-size="${fontCount}" font-weight="700" fill="#00d4ff">${item.count.toLocaleString()}</text>`;

    // Transparent hit-box covering the whole label stack. Width estimated
    // from the longest text line (monospace ≈ 6.2 px / char @ 10px font).
    const longest = Math.max(...lines.map((s) => s.length), String(item.count).length + 1);
    const hitW = longest * 6.6 + 8;
    let hitX;
    if (anchor === "start")      hitX = x - 4;
    else if (anchor === "end")   hitX = x - hitW + 4;
    else                          hitX = x - hitW / 2;
    const hitY = labelMinY - 2;
    const hitH = (labelMaxY - labelMinY) + 6;
    const hitBox = `<rect x="${hitX.toFixed(1)}" y="${hitY.toFixed(1)}" width="${hitW.toFixed(
      1
    )}" height="${hitH.toFixed(1)}" fill="transparent"/>`;

    // The dot for this spoke — wrapped in the SAME <g> so it shares the
    // hover tooltip with its label.
    const p = dataPts[i];
    const dotMarkup = `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(
      1
    )}" r="5" fill="#00d4ff"/>
        <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(
      1
    )}" r="8" fill="none" stroke="rgba(0,212,255,0.25)" stroke-width="3"/>`;

    // Tooltip is rendered by a JS handler attached after the SVG is in DOM
    // (see _attachTooltips below) — much faster than the browser's native
    // <title> which has a ~1.5s hover delay we can't override.
    const tooltip = `${escapedRaw} — ${item.count.toLocaleString()} alert${
      item.count === 1 ? "" : "s"
    }`;
    spokeGroups += `<g class="radar-spoke" data-tip="${tooltip.replace(/"/g, "&quot;")}" style="cursor:default">${hitBox}${reasonTxt}${countTxt}${dotMarkup}</g>`;
  });

  const labels = "";   // labels are now inside spokeGroups
  const dots = spokeGroups;

  // Small horizontal padding for label runway — keeps chart at near-full
  // size while giving side labels a few pixels of safe rendering space.
  const PAD_X = 50;
  const PAD_Y = 14;
  const svgHtml = `
  <svg viewBox="${-PAD_X} ${-PAD_Y} ${W + PAD_X * 2} ${
    H + PAD_Y * 2
  }" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;max-width:100%;max-height:100%;display:block">
    <defs>
      <radialGradient id="rg" cx="50%" cy="50%" r="50%">
        <stop offset="0%"   stop-color="#00d4ff" stop-opacity="0.22"/>
        <stop offset="100%" stop-color="#00d4ff" stop-opacity="0.03"/>
      </radialGradient>
    </defs>
    ${rings}${spokes}
    <path d="${polyD}" fill="url(#rg)" stroke="rgba(0,212,255,0.85)" stroke-width="2" stroke-linejoin="round"/>
    ${dots}${labels}
    <rect x="10" y="${
      H - 18
    }" width="12" height="8" fill="rgba(0,212,255,0.18)" stroke="rgba(0,212,255,0.7)" stroke-width="1" rx="1"/>
    <text x="26" y="${
      H - 10
    }" font-family="JetBrains Mono,monospace" font-size="9" fill="#7a9ab5">Alert Count</text>
  </svg>`;

  const existing = document.getElementById("radar-svg-wrap");
  if (existing) {
    existing.innerHTML = svgHtml;
    _attachTooltips(existing);
    return;
  }
  const canvas = document.getElementById("radar-chart");
  if (canvas) {
    const wrapper = canvas.parentElement;
    if (wrapper) {
      const div = document.createElement("div");
      div.id = "radar-svg-wrap";
      div.style.cssText =
        "width:100%;height:100%;display:flex;align-items:center;justify-content:center;" +
        "padding:6px 10px;box-sizing:border-box;overflow:hidden;";
      div.innerHTML = svgHtml;
      canvas.replaceWith(div);
      _attachTooltips(div);
    }
  }
}

// Singleton floating tooltip used by every spoke. Created on demand,
// pinned to <body> so it can render above the SVG without clipping.
let _tipEl = null;
function _ensureTipEl() {
  if (_tipEl) return _tipEl;
  _tipEl = document.createElement("div");
  _tipEl.className = "radar-tooltip";
  _tipEl.style.cssText = [
    "position:fixed",
    "pointer-events:none",
    "z-index:10000",
    "padding:6px 10px",
    "background:rgba(11,17,24,0.96)",
    "color:#c8d8e8",
    "font-family:'IBM Plex Mono',monospace",
    "font-size:11.5px",
    "letter-spacing:0.2px",
    "border:1px solid rgba(0,212,255,0.35)",
    "border-radius:4px",
    "box-shadow:0 4px 14px rgba(0,0,0,0.45)",
    "white-space:nowrap",
    "opacity:0",
    "transition:opacity 0.08s ease",
    "transform:translate(-50%, -120%)",
  ].join(";");
  document.body.appendChild(_tipEl);
  return _tipEl;
}

function _attachTooltips(root) {
  const tip = _ensureTipEl();
  root.querySelectorAll("g.radar-spoke").forEach((g) => {
    const text = g.getAttribute("data-tip") || "";
    if (!text) return;
    g.addEventListener("mouseenter", () => {
      tip.textContent = text;
      tip.style.opacity = "1";
    });
    g.addEventListener("mousemove", (e) => {
      tip.style.left = e.clientX + "px";
      tip.style.top  = (e.clientY - 8) + "px";
    });
    g.addEventListener("mouseleave", () => {
      tip.style.opacity = "0";
    });
  });
}
