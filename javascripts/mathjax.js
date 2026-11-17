/* MathJax 3 config for Material for MkDocs + pymdownx.arithmatex (generic mode).
   Arithmatex wraps math in <span class="arithmatex">\( … \)</span> / \[ … \].
   Two things are required and were missing:
     1. tell MathJax to only typeset arithmatex spans, and
     2. RE-typeset on every page change: Material's navigation.instant swaps page
        content without a reload, so without this hook the math stays raw LaTeX on
        any page reached by clicking a nav link (exactly the "broken formulas" bug).
   Must be loaded BEFORE the MathJax CDN script (sets window.MathJax first). */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  if (!window.MathJax || !window.MathJax.typesetPromise) return;
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
