/* Robust Mermaid initialization for MkDocs Material.

   Why this is non-trivial:
   - Material's superfences-mermaid combo produces
       <pre class="mermaid"><code>SOURCE</code></pre>
     on first load. Material's own auto-init may then attempt to render
     this with a default theme and (with our custom CSS variables)
     occasionally leaves the result empty.
   - With navigation.instant, the JS init runs again on virtual page
     changes, which means we have to be re-entrant: not destroy source,
     not double-render, not infinite-loop.

   Strategy:
   - Normalise everything to <div class="mermaid"> with the original
     diagram source stored on `dataset.src` so we always have the truth
     even after mermaid replaces innerHTML with SVG.
   - On every init pass, reset processed divs from their stored source
     and let mermaid.run() reprocess. */

(function () {
  const CDN = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';

  const loadMermaid = () => {
    if (window.mermaid) return Promise.resolve(window.mermaid);
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = CDN;
      s.async = false;
      s.onload = () => resolve(window.mermaid);
      s.onerror = reject;
      document.head.appendChild(s);
    });
  };

  /* Collect all mermaid blocks regardless of original HTML shape and
     return a flat list of <div class="mermaid"> elements with their
     diagram source saved on dataset.src. Replaces <pre>-wrapped variants
     in place. */
  const collectMermaidDivs = () => {
    // Shape 1: <pre class="mermaid"><code>SRC</code></pre>
    document.querySelectorAll('pre.mermaid').forEach((pre) => {
      const src = (pre.querySelector('code') || pre).textContent;
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = src;
      div.dataset.src = src;
      pre.parentNode.replaceChild(div, pre);
    });

    // Shape 2: <pre><code class="language-mermaid">SRC</code></pre>
    document.querySelectorAll('pre > code.language-mermaid').forEach((code) => {
      const src = code.textContent;
      const pre = code.parentElement;
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = src;
      div.dataset.src = src;
      pre.parentNode.replaceChild(div, pre);
    });

    return Array.from(document.querySelectorAll('div.mermaid'));
  };

  const initMermaid = async () => {
    const divs = collectMermaidDivs();
    if (!divs.length) return;

    const mermaid = await loadMermaid();
    if (!mermaid) return;

    const scheme = document.body.getAttribute('data-md-color-scheme') ||
                   document.documentElement.getAttribute('data-md-color-scheme') || 'default';

    mermaid.initialize({
      startOnLoad: false,
      theme: scheme === 'slate' ? 'dark' : 'neutral',
      themeVariables: {
        // Solid colors, no gradients
        primaryColor: '#0f172a',        // dark navy primary nodes
        primaryTextColor: '#ffffff',
        primaryBorderColor: '#0f172a',
        secondaryColor: '#7c3aed',      // violet accent nodes
        secondaryTextColor: '#ffffff',
        secondaryBorderColor: '#7c3aed',
        tertiaryColor: '#f1f5f9',       // slate-100 muted nodes
        tertiaryTextColor: '#0f172a',
        tertiaryBorderColor: '#cbd5e1',
        lineColor: '#475569',           // slate-600 lines
        textColor: '#0f172a',
        mainBkg: scheme === 'slate' ? '#14142b' : '#ffffff',
        background: scheme === 'slate' ? '#14142b' : '#ffffff',
        fontFamily: 'Manrope, Inter, system-ui, sans-serif',
        fontSize: '14px',
      },
      flowchart: {
        curve: 'basis',
        padding: 18,
        nodeSpacing: 40,
        rankSpacing: 50,
      },
      securityLevel: 'loose',
    });

    // Reset every div from its stored source so mermaid.run can re-parse.
    divs.forEach((d) => {
      if (d.dataset.src) {
        d.textContent = d.dataset.src;
        d.removeAttribute('data-processed');
      }
    });

    try {
      await mermaid.run({ querySelector: 'div.mermaid' });
    } catch (e) {
      console.warn('Mermaid render failed:', e);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMermaid);
  } else {
    initMermaid();
  }
  if (typeof document$ !== 'undefined' && document$.subscribe) {
    document$.subscribe(initMermaid);
  }
})();
