// Theme toggle — restore preference and wire the button.
export function initTheme() {
  const html  = document.documentElement;
  const btn   = document.getElementById("theme-toggle-btn");
  const icon  = document.getElementById("theme-icon");
  const label = document.getElementById("theme-label");
  if (!btn) return;

  const saved = localStorage.getItem("cs-theme");
  if (saved === "light") {
    html.classList.add("light");
    if (icon)  icon.textContent  = "☀️";
    if (label) label.textContent = "LIGHT";
  }

  btn.addEventListener("click", () => {
    const isLight = html.classList.toggle("light");
    if (icon)  icon.textContent  = isLight ? "☀️" : "🌙";
    if (label) label.textContent = isLight ? "LIGHT" : "DARK";
    localStorage.setItem("cs-theme", isLight ? "light" : "dark");
  });
}
