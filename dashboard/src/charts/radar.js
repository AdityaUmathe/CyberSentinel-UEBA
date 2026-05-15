// Radar / spider chart — SVG renderer for top anomaly reasons.

export function drawRadar(reasons) {
  if (!reasons || !reasons.length) return;

  const n = Math.min(reasons.length, 7);
  const maxVal = Math.max(...reasons.slice(0, n).map((r) => r.count), 1);

  const W = 480, H = 400;
  const cx = W / 2, cy = H / 2 - 10;
  const R = Math.min(cx, cy) - 80;
  const labelR = R + 44;

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

  let labels = "";
  reasons.slice(0, n).forEach((item, i) => {
    const { x, y, a } = pt(i, labelR);
    const cosA = Math.cos(a);
    const anchor = cosA > 0.2 ? "start" : cosA < -0.2 ? "end" : "middle";
    const lbl = (item.reason || "").replace(/_/g, " ");
    labels += `
      <text x="${x.toFixed(1)}" y="${(y - 7).toFixed(
      1
    )}" text-anchor="${anchor}" font-family="JetBrains Mono,monospace" font-size="9" font-weight="500" fill="#a8c0d2">${lbl}</text>
      <text x="${x.toFixed(1)}" y="${(y + 8).toFixed(
      1
    )}" text-anchor="${anchor}" font-family="JetBrains Mono,monospace" font-size="11" font-weight="700" fill="#00d4ff">${item.count.toLocaleString()}</text>`;
  });

  let dots = "";
  dataPts.forEach((p) => {
    dots += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="5" fill="#00d4ff"/>
             <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(
      1
    )}" r="8" fill="none" stroke="rgba(0,212,255,0.25)" stroke-width="3"/>`;
  });

  const svgHtml = `
  <svg viewBox="-10 -10 ${W + 20} ${
    H + 20
  }" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;max-width:100%;max-height:100%;display:block;overflow:visible">
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
    return;
  }
  const canvas = document.getElementById("radar-chart");
  if (canvas) {
    const wrapper = canvas.parentElement;
    if (wrapper) {
      const div = document.createElement("div");
      div.id = "radar-svg-wrap";
      // Padding gives the external labels breathing room so they never touch
      // the panel border. box-sizing keeps the wrap from overflowing.
      div.style.cssText =
        "width:100%;height:100%;display:flex;align-items:center;justify-content:center;" +
        "padding:6px 14px;box-sizing:border-box;overflow:hidden;";
      div.innerHTML = svgHtml;
      canvas.replaceWith(div);
    }
  }
}
