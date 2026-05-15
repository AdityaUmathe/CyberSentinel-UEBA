// Footer year stamp + footer-page modals (Support / Privacy).

export function openFooterPage(id) {
  const el = document.getElementById("fp-" + id);
  if (el) {
    el.classList.add("open");
    document.body.style.overflow = "hidden";
  }
}

export function closeFooterPage(id) {
  const el = document.getElementById("fp-" + id);
  if (el) {
    el.classList.remove("open");
    document.body.style.overflow = "";
  }
}

export function initFooter() {
  const y = document.getElementById("footer-year");
  if (y) y.textContent = new Date().getFullYear();

  // Close-on-overlay-click
  document.querySelectorAll(".fp-overlay").forEach((overlay) => {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) closeFooterPage(overlay.id.replace("fp-", ""));
    });
  });

  // Close-on-Escape
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      document.querySelectorAll(".fp-overlay.open").forEach((el) => el.classList.remove("open"));
    }
  });
}
