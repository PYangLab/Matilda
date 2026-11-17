/* ─────────────────────────────────────────────────────────────────────────
   Small UI tweaks layered on Material for MkDocs.
   ───────────────────────────────────────────────────────────────────────── */

/* Clear the search field when the search overlay closes, so the collapsed
   search box never keeps showing the previous query. The header (and its
   search input) survive instant navigation, so wiring once on load is enough. */
(function () {
  function wireSearchClear() {
    var toggle = document.querySelector('#__search');
    var input = document.querySelector('.md-search__input');
    if (!toggle || !input || toggle.dataset.mtoSearchClear) return;
    toggle.dataset.mtoSearchClear = '1';
    toggle.addEventListener('change', function () {
      if (!toggle.checked) {
        input.value = '';
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wireSearchClear);
  } else {
    wireSearchClear();
  }
})();
