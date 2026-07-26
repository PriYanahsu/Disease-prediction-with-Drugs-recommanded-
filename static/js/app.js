(function () {
  const overlay = document.getElementById("predict-overlay");
  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("predict-submit");
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      const open = navLinks.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  if (!form || !overlay || !submitBtn) return;

  form.addEventListener("submit", function () {
    if (!form.checkValidity()) return;
    overlay.hidden = false;
    overlay.setAttribute("aria-hidden", "false");
    submitBtn.disabled = true;
    submitBtn.classList.add("is-loading");
    submitBtn.textContent = "Running inference…";
  });
})();
