// Version switcher for multi-version gh-pages docs.
// Reads versions.json (mike-compatible format) from the site root and injects
// a <select> into the page.
// Format: [{"version": "0.22.0", "title": "0.22.0", "aliases": ["latest"]}, ...]
(function () {
  "use strict";

  // Resolve the site root: we sit at <root>/<version>/_static/version-switcher.js
  var scripts = document.getElementsByTagName("script");
  var thisScript = scripts[scripts.length - 1];
  var staticDir = thisScript.src.replace(/\/[^/]*$/, "");
  var versionDir = staticDir.replace(/\/[^/]*$/, "");
  var siteRoot = versionDir.replace(/\/[^/]*$/, "");

  // Current version is the last path component of versionDir
  var current = versionDir.split("/").pop();

  fetch(siteRoot + "/versions.json")
    .then(function (r) { return r.json(); })
    .then(function (versions) {
      var select = document.createElement("select");
      select.setAttribute("aria-label", "Version");
      select.style.cssText =
        "margin-left:1em;padding:2px 4px;border:1px solid var(--color-sidebar-border,#ccc);" +
        "border-radius:4px;font-size:.85em;background:var(--color-sidebar-background,#fff);" +
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

      // Insert into Furo's sidebar brand area or fall back to the header
      var target =
        document.querySelector(".sidebar-brand") ||
        document.querySelector("header") ||
        document.body;
      target.appendChild(select);
    })
    .catch(function () {
      // No versions.json yet (local build) — silently skip.
    });
})();
