// Hourly alert heatmap — day × hour grid, last 7 days. SVG output.

export function renderHeatmap(feedData) {
  const body = document.getElementById("heatmap-body");
  if (!body || !feedData || !feedData.length) return;

  const matrix = {};
  feedData.forEach((a) => {
    const t = a.processed_at || a.event_time || "";
    if (!t || t.length < 13) return;
    const day  = t.slice(0, 10);
    const hour = t.slice(11, 13);
    const key  = `${day}|${hour}`;
    if (!matrix[key]) matrix[key] = { total: 0, critical: 0 };
    matrix[key].total++;
    if (a.verdict === "highly_anomalous") matrix[key].critical++;
  });

  const allDays = [...new Set(Object.keys(matrix).map((k) => k.split("|")[0]))].sort().slice(-7);
  const hours   = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, "0"));
  const maxVal  = Math.max(...Object.values(matrix).map((v) => v.total), 1);

  const cellW = 22, cellH = 18, padL = 58, padT = 14, padB = 22;
  const W = padL + 24 * cellW + 8;
  const H = padT + allDays.length * cellH + padB;

  let cells = "", dayLabels = "", hourLabels = "";

  hours.forEach((h, hi) => {
    hourLabels += `<text x="${padL + hi * cellW + cellW / 2}" y="${
      padT - 3
    }" font-size="8" fill="#3d5a72" text-anchor="middle">${hi % 3 === 0 ? h : ""}</text>`;
  });

  allDays.forEach((day, di) => {
    const shortDay = day.slice(5);
    dayLabels += `<text x="${padL - 4}" y="${
      padT + di * cellH + cellH * 0.65
    }" font-size="9" fill="#7a9ab5" text-anchor="end">${shortDay}</text>`;
    hours.forEach((h, hi) => {
      const key = `${day}|${h}`;
      const v = matrix[key] || { total: 0, critical: 0 };
      const intensity = v.total / maxVal;
      const hasCrit = v.critical > 0;
      let fill = "rgba(26,40,56,0.5)";
      if (intensity > 0) {
        if (hasCrit) {
          fill = `rgba(255,59,92,${0.15 + intensity * 0.75})`;
        } else {
          fill = `rgba(0,212,255,${0.08 + intensity * 0.62})`;
        }
      }
      cells += `<rect x="${padL + hi * cellW + 1}" y="${
        padT + di * cellH + 1
      }" width="${cellW - 2}" height="${cellH - 2}" fill="${fill}" rx="2">
        <title>${day} ${h}:00 — ${v.total} alerts${
        v.critical > 0 ? ` (${v.critical} critical)` : ""
      }</title></rect>`;
    });
  });

  body.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:${H}px;display:block;">
      ${dayLabels}${hourLabels}${cells}
    </svg>
    <div style="display:flex;gap:16px;margin-top:8px;align-items:center;">
      <span style="font-family:var(--mono2);font-size:9px;color:var(--text3);">Low activity</span>
      <div style="display:flex;gap:2px;">
        ${[0.1, 0.3, 0.5, 0.7, 0.9]
          .map(
            (i) =>
              `<div style="width:14px;height:10px;background:rgba(0,212,255,${
                0.08 + i * 0.62
              });border-radius:2px;"></div>`
          )
          .join("")}
      </div>
      <span style="font-family:var(--mono2);font-size:9px;color:var(--accent);">High (normal)</span>
      <div style="width:14px;height:10px;background:rgba(255,59,92,0.7);border-radius:2px;"></div>
      <span style="font-family:var(--mono2);font-size:9px;color:var(--red);">Critical alerts</span>
    </div>`;
}
