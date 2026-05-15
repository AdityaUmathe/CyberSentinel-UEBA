// Top signatures — horizontal bar chart.

export function renderTopSigs(feedData) {
  const body  = document.getElementById("top-sigs-body");
  const badge = document.getElementById("top-sigs-badge");
  if (!body || !feedData) return;

  const sigMap = {};
  feedData.forEach((a) => {
    const s = (a.signature || "Unknown").slice(0, 65);
    sigMap[s] = (sigMap[s] || 0) + 1;
  });
  const top = Object.entries(sigMap).sort((a, b) => b[1] - a[1]).slice(0, 10);
  if (badge) badge.textContent = top.length;
  if (!top.length) {
    body.innerHTML = '<div class="empty-state" style="padding:20px"><p>NO DATA</p></div>';
    return;
  }

  const maxC = top[0][1];
  body.innerHTML = top.map(([sig, count]) => `
    <div class="sig-bar-row">
      <div class="sig-bar-label">
        <span class="sig-bar-name" title="${sig}">${sig}</span>
        <span class="sig-bar-count">${count.toLocaleString()}</span>
      </div>
      <div class="sig-bar-track">
        <div class="sig-bar-fill" style="width:${Math.round((count / maxC) * 100)}%"></div>
      </div>
    </div>`).join("");
}
