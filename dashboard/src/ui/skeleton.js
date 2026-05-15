export function showSkeletons() {
  const skels = `<div style="padding:16px;display:flex;flex-direction:column;gap:8px">
    <div class="skeleton skeleton-line medium"></div>
    <div class="skeleton skeleton-line full"></div>
    <div class="skeleton skeleton-line short"></div>
    <div class="skeleton skeleton-box" style="margin-top:4px"></div>
    <div class="skeleton skeleton-line full"></div>
    <div class="skeleton skeleton-line medium"></div>
  </div>`;
  ["overview-tbody", "feed-tbody", "users-list", "campaigns-list"].forEach((id) => {
    const el = document.getElementById(id);
    if (el && !el.innerHTML.trim()) el.innerHTML = skels;
  });
}
