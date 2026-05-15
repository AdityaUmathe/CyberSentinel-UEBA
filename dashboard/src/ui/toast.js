export function showToast(msg, type = "info", duration = 3000) {
  const icons = { info: "ℹ", success: "✓", warning: "⚠", error: "✕" };
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type] || "ℹ"}</span><span>${msg}</span>`;
  document.getElementById("toast-container").appendChild(el);
  setTimeout(() => el.remove(), duration);
}
