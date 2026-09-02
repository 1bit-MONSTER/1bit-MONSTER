/* 1bit.MONSTER theme modes: light (default) / auto / dark.
 *
 * Runs in <head> so data-theme is set on <html> before first paint (no flash).
 * Preference is persisted under "1bit-theme"; absent preference = light.
 * Auto follows the OS via CSS media queries (see the per-page style block);
 * the .theme-switch buttons only need aria-pressed highlighting here.
 */
(function () {
  'use strict';
  var KEY = '1bit-theme';
  var MODES = ['light', 'auto', 'dark'];
  var root = document.documentElement;
  var store = null;
  try { store = window.localStorage; } catch (e) { /* private mode */ }

  function stored() {
    if (!store) return null;
    try { return store.getItem(KEY); } catch (e) { return null; }
  }
  function save(m) {
    if (!store) return;
    try { store.setItem(KEY, m); } catch (e) { /* ignore */ }
  }

  var mode = stored();
  if (MODES.indexOf(mode) < 0) mode = 'light'; /* light mode is the default */

  function apply(m) { root.setAttribute('data-theme', m); }
  apply(mode);

  function wire() {
    var group = document.querySelector('.theme-switch');
    if (!group) return;
    var btns = group.querySelectorAll('.ts-btn');
    function mark(m) {
      for (var i = 0; i < btns.length; i++) {
        btns[i].setAttribute('aria-pressed', btns[i].getAttribute('data-choice') === m ? 'true' : 'false');
      }
    }
    function set(m) {
      mode = m;
      apply(m);
      mark(m);
      save(m);
    }
    for (var i = 0; i < btns.length; i++) {
      (function (btn) {
        btn.addEventListener('click', function () { set(btn.getAttribute('data-choice')); });
      })(btns[i]);
    }
    mark(mode);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }
})();
