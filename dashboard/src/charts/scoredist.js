// Score distribution histogram — 10 buckets, SVG output.

export function renderScoreDist(feedData) {
  const body  = document.getElementById("score-dist-body");
  const badge = document.getElementById("score-dist-badge");
  if (!body || !feedData || !feedData.length) return;

  const bins = Array(10).fill(0);
  const colorsMap = { 0:"#06d6a0",1:"#06d6a0",2:"#06d6a0",3:"#06d6a0",4:"#ffd166",5:"#ffd166",6:"#ff8c42",7:"#ff8c42",8:"#ff3b5c",9:"#ff3b5c" };
  feedData.forEach((a) => {
    const s = Math.min(Math.max(a.score || 0, 0), 0.9999);
    bins[Math.floor(s * 10)]++;
  });
  if (badge) badge.textContent = feedData.length;
  const maxB = Math.max(...bins, 1);
  const W = 460, H = 130, pL = 36, pR = 8, pT = 10, pB = 28;
  const cW = W - pL - pR, cH = H - pT - pB;
  const bW = Math.floor(cW / 10) - 3;

  let rects = "", labels = "", gridLines = "";
  [0.25, 0.5, 0.75, 1].forEach((f) => {
    const y = pT + cH - Math.round(f * cH);
    gridLines += `<line x1="${pL}" y1="${y}" x2="${W - pR}" y2="${y}" stroke="#1a2838" stroke-width="0.5" stroke-dasharray="3,3"/>
                  <text x="${pL - 4}" y="${y + 3}" font-size="8" fill="#3d5a72" text-anchor="end">${Math.round(f * maxB)}</text>`;
  });
  bins.forEach((count, i) => {
    const bH = Math.max(count > 0 ? 3 : 0, Math.round((count / maxB) * cH));
    const x  = pL + i * (cW / 10) + 2;
    const y  = pT + cH - bH;
    const color = colorsMap[i];
    rects  += `<rect x="${x}" y="${y}" width="${bW}" height="${bH}" fill="${color}" opacity="0.82" rx="2"><title>${(i/10).toFixed(1)}–${((i+1)/10).toFixed(1)}: ${count}</title></rect>`;
    labels += `<text x="${x + bW/2}" y="${H - 2}" font-size="8" fill="#3d5a72" text-anchor="middle">${(i/10).toFixed(1)}</text>`;
    if (count > 0) {
      rects += `<text x="${x + bW/2}" y="${y - 3}" font-size="8" fill="${color}" text-anchor="middle">${count}</text>`;
    }
  });

  body.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:${H}px;display:block;">
      ${gridLines}
      <line x1="${pL}" y1="${pT}" x2="${pL}" y2="${pT+cH}" stroke="#243447" stroke-width="1"/>
      <line x1="${pL}" y1="${pT+cH}" x2="${W-pR}" y2="${pT+cH}" stroke="#243447" stroke-width="1"/>
      ${rects}${labels}
    </svg>
    <div style="display:flex;gap:14px;margin-top:8px;flex-wrap:wrap;">
      <span style="font-family:var(--mono2);font-size:9px;color:var(--green);display:flex;align-items:center;gap:5px;"><span style="width:10px;height:6px;background:#06d6a0;border-radius:1px;display:inline-block;"></span>Low (0.0–0.3)</span>
      <span style="font-family:var(--mono2);font-size:9px;color:var(--yellow);display:flex;align-items:center;gap:5px;"><span style="width:10px;height:6px;background:#ffd166;border-radius:1px;display:inline-block;"></span>Moderate (0.4–0.5)</span>
      <span style="font-family:var(--mono2);font-size:9px;color:var(--orange);display:flex;align-items:center;gap:5px;"><span style="width:10px;height:6px;background:#ff8c42;border-radius:1px;display:inline-block;"></span>High (0.6–0.7)</span>
      <span style="font-family:var(--mono2);font-size:9px;color:var(--red);display:flex;align-items:center;gap:5px;"><span style="width:10px;height:6px;background:#ff3b5c;border-radius:1px;display:inline-block;"></span>Critical (0.8–1.0)</span>
    </div>`;
}
