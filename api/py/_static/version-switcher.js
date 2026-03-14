// Version switcher for multi-version gh-pages docs.
// Reads versions.json (mike-compatible format) from the site root and injects
// a <select> into the page.
// Format: [{"version": "0.22.0", "title": "0.22.0", "aliases": ["latest"]}, ...]
(function () {
  "use strict";

  // Resolve the site root from script src.
  // Script lives at <root>/<version>/_static/version-switcher.js
  var scriptSrc = document.currentScript && document.currentScript.src;
  if (!scriptSrc) return;

  var parts = scriptSrc.split("/");
  // pop "version-switcher.js?...", "_static", "<version>"
  parts.pop(); // filename
  parts.pop(); // _static
  var current = parts.pop(); // version directory name
  var siteRoot = parts.join("/");

  function injectSwitcher() {
    fetch(siteRoot + "/versions.json")
      .then(function (r) { return r.json(); })
      .then(function (versions) {
        var select = document.createElement("select");
        select.setAttribute("aria-label", "Version");
        select.style.cssText =
          "display:block;margin:0.5em auto;padding:4px 8px;" +
          "border:1px solid var(--color-sidebar-border,#ccc);" +
          "border-radius:4px;font-size:.85em;" +
          "background:var(--color-sidebar-background,#fff);" +
          "color:var(--color-sidebar-text,#333);cursor:pointer;";

        versions.forEach(function (v) {
          var opt = document.createElement("option");
          opt.value = siteRoot + "/" + v.version + "/";
          opt.textContent = v.title || v.version;
          if (v.version === current) opt.selected = true;
          select.appendChild(opt);
        });

        select.addEventListener("change", function () {
          window.location.href = this.value;
        });

        // Insert into Furo's sidebar brand area
        var target =
          document.querySelector(".sidebar-brand") ||
          document.querySelector(".sidebar-sticky") ||
          document.querySelector("header");
        if (target) {
          target.appendChild(select);
        }
      })
      .catch(function () {
        // No versions.json (local build) — silently skip.
      });
  }

  // Run immediately — script is at end of <body> so DOM is ready
  injectSwitcher();
})();
