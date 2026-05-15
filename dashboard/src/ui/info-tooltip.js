// Info-btn tooltip — fixed positioning anchored to the button's rect.
export function initInfoTooltip() {
  document.addEventListener("mouseover", (e) => {
    const btn = e.target.closest(".info-btn");
    if (!btn) return;
    const tip = btn.querySelector(".info-tooltip");
    if (!tip) return;
    const r = btn.getBoundingClientRect();
    tip.style.top = r.bottom + 8 + "px";
    let left = r.left;
    const tipW = 252;
    if (left + tipW > window.innerWidth - 12) left = window.innerWidth - tipW - 12;
    if (left < 8) left = 8;
    tip.style.left = left + "px";
  });
}
