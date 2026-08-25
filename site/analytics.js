/* 1bit.monster — link-attribution + Web Analytics beacon loader.
 *
 * Two jobs:
 *  1. Attribution: capture UTM / campaign params on first load (a shared
 *     link), persist them in localStorage, and re-append them to every
 *     internal navigation so the attribution survives multi-page sessions.
 *     This is what lets us answer "who shared what link, from which team"
 *     — each shared link carries a campaign tag (see share.html).
 *  2. Beacon: load Cloudflare Web Analytics (RUM) if a token is configured.
 *     The token is injected at deploy time by the CF Web Analytics bootstrap
 *     workflow (it creates the RUM site and rewrites __CF_ANALYTICS_TOKEN__
 *     here). Without a token, attribution still works via localStorage;
 *     only the server-side RUM aggregation is unavailable.
 *
 * Attribution is intentionally dependency-free and synchronous-once.
 */
(function () {
  "use strict";

  var STORE_KEY = "1bit_attribution_v1";

  // ── 1. Attribution capture ──────────────────────────────────────────
  function readAttribution() {
    var p = new URLSearchParams(window.location.search);
    var out = {};
    ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
     "ref", "team", "share"].forEach(function (k) {
      var v = p.get(k);
      if (v) out[k] = v;
    });
    return out;
  }

  function loadStored() {
    try {
      return JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    } catch (e) { return {}; }
  }

  function store(a) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(a)); } catch (e) {}
  }

  function mergeAttribution(existing, fresh) {
    // First-touch: keep the earliest non-empty campaign/source.
    var merged = {};
    Object.keys(existing).forEach(function (k) { merged[k] = existing[k]; });
    Object.keys(fresh).forEach(function (k) {
      if (!merged[k] && fresh[k]) merged[k] = fresh[k];
    });
    // Always stamp when we first saw it.
    if (!merged.ts) merged.ts = new Date().toISOString();
    return merged;
  }

  var fresh = readAttribution();
  var stored = loadStored();
  if (Object.keys(fresh).length || !Object.keys(stored).length) {
    store(mergeAttribution(stored, fresh));
  }

  // Re-append attribution to internal link clicks so the session keeps it.
  var att = loadStored();
  var hasAtt = Object.keys(att).some(function (k) { return k !== "ts" && att[k]; });
  if (hasAtt) {
    var qs = [];
    ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
     "ref", "team", "share"].forEach(function (k) {
      if (att[k]) qs.push(encodeURIComponent(k) + "=" + encodeURIComponent(att[k]));
    });
    if (qs.length) {
      var suffix = qs.join("&");
      document.addEventListener("click", function (e) {
        var a = e.target && e.target.closest ? e.target.closest("a") : null;
        if (!a || !a.href || a.target === "_blank") return;
        var href = a.getAttribute("href") || "";
        // internal only
        if (!href || /^(https?:)?\/\//.test(href) && href.indexOf(location.host) < 0) return;
        if (href.charAt(0) === "#") return;
        var sep = href.indexOf("?") >= 0 ? "&" : "?";
        // avoid double-appending
        if (href.indexOf("utm_source=") >= 0) return;
        e.preventDefault();
        location.href = href + sep + suffix;
      });
    }
  }

  // ── 2. Web Analytics beacon ────────────────────────────────────────
  var TOKEN = "__CF_ANALYTICS_TOKEN__";  // rewritten at deploy by bootstrap
  if (TOKEN && TOKEN.indexOf("CF_ANALYTICS") < 0 && TOKEN.length > 10) {
    var s = document.createElement("script");
    s.defer = true;
    s.src = "https://static.cloudflareinsights.com/beacon.min.js";
    s.setAttribute("data-cf-beacon", '{"token": "' + TOKEN + '"}');
    document.head.appendChild(s);
  }
})();
