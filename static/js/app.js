(function () {
  document.querySelectorAll("[data-pct]").forEach(function (el) {
    var pct = el.getAttribute("data-pct");
    if (pct == null || pct === "") return;
    if (el.classList.contains("ring")) {
      el.style.setProperty("--p", pct);
    } else {
      el.style.width = pct + "%";
    }
  });

  const overlay = document.getElementById("predict-overlay");
  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("predict-submit");
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");
  const pipelineSteps = overlay
    ? overlay.querySelectorAll(".overlay-pipeline li")
    : [];

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", function () {
      const open = navLinks.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });

    document.addEventListener("click", function (e) {
      if (!navLinks.classList.contains("open")) return;
      if (navToggle.contains(e.target) || navLinks.contains(e.target)) return;
      navLinks.classList.remove("open");
      navToggle.setAttribute("aria-expanded", "false");
    });
  }

  if (!form || !overlay || !submitBtn) return;

  form.addEventListener("submit", function () {
    if (!form.checkValidity()) return;
    overlay.hidden = false;
    overlay.setAttribute("aria-hidden", "false");
    submitBtn.disabled = true;
    submitBtn.classList.add("is-loading");
    submitBtn.textContent = "Running…";

    pipelineSteps.forEach(function (li, i) {
      li.classList.remove("on");
      window.setTimeout(function () {
        li.classList.add("on");
      }, 180 + i * 220);
    });
  });
})();
