/* Matilda docs: page polish JS.

   Two responsibilities:

   1. Scroll-triggered fade-up:
      - Content is visible by default (CSS).
      - JS finds animation candidates, tags below-fold ones with
        `.anim-pending` (which sets opacity:0 + translate via CSS),
        and uses IntersectionObserver to flip them to `.is-visible`
        as they enter the viewport.
      - If anything fails (JS disabled, iframe with innerHeight=0,
        broken observer), no element ever gets `.anim-pending` →
        content remains fully visible. Fail-open by design.

   2. Sticky header shadow on scroll. */

(function () {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const ANIM_SELECTOR =
    '.md-typeset .grid.cards > ul > li, ' +
    '.md-typeset .grid:not(.cards) > *, ' +
    '.anim-fade-up';

  const initFadeUp = () => {
    const targets = document.querySelectorAll(ANIM_SELECTOR);
    if (!targets.length) return;
    if (!('IntersectionObserver' in window)) return;

    const vh = window.innerHeight || document.documentElement.clientHeight || 0;
    // If viewport is degenerate (0 or unknown), bail and keep content visible.
    if (vh < 100) return;

    // Tag below-the-fold targets as pending (CSS hides them); keep above-fold
    // content fully visible to avoid any flash of empty space.
    targets.forEach(el => {
      const r = el.getBoundingClientRect();
      // r.top here is relative to viewport. If element's top is more than ~50px
      // below the visible area, animate. Otherwise leave visible.
      if (r.top > vh - 40) {
        el.classList.add('anim-pending');
      }
    });

    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          io.unobserve(entry.target);
        }
      });
    }, {
      rootMargin: '0px 0px 80px 0px',   // pre-reveal 80px before entering
      threshold: 0.01,
    });

    targets.forEach(el => {
      if (el.classList.contains('anim-pending')) io.observe(el);
    });
  };

  const stickyHeader = () => {
    const header = document.querySelector('.md-header');
    if (!header) return;
    const setShadow = () => header.setAttribute('data-md-state', window.scrollY > 8 ? 'shadow' : '');
    setShadow();                        // refresh state on every (re-)init
    if (window.__stickyWired) return;   // but attach the scroll listener only ONCE
    window.__stickyWired = true;        // (instant-nav re-runs init; don't stack handlers)
    let ticking = false;
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(() => { setShadow(); ticking = false; }); ticking = true; }
    }, { passive: true });
  };

  /* Scroll progress bar (Bioconductor reference): slim violet bar at top */
  const scrollProgress = () => {
    if (document.querySelector('.scroll-progress')) return;
    const bar = document.createElement('div');
    bar.className = 'scroll-progress';
    document.body.appendChild(bar);

    let ticking = false;
    const update = () => {
      const h = document.documentElement;
      const total = h.scrollHeight - h.clientHeight;
      const pct = total > 0 ? (window.scrollY / total) * 100 : 0;
      bar.style.width = pct + '%';
      bar.classList.toggle('is-scrolling', window.scrollY > 80);
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  };

  /* Back-to-top button: fastai-style, appears after 600px scroll */
  const backToTop = () => {
    if (document.querySelector('.back-to-top')) return;
    const btn = document.createElement('button');
    btn.className = 'back-to-top';
    btn.setAttribute('aria-label', 'Back to top');
    btn.innerHTML = '↑';
    btn.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    document.body.appendChild(btn);

    let ticking = false;
    const update = () => {
      btn.classList.toggle('is-visible', window.scrollY > 600);
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  };

  const enableSectionNumbersOnHome = () => {
    const path = (window.location.pathname || '').replace(/\/+$/, '/');
    // A landing page is any page carrying the `.lp` wrapper; detect that so the
    // hero styling is identical across all of them, not just at the site root.
    const isHome = !!document.querySelector('.lp') || path === '/' || /index\.html?$/i.test(path);
    if (isHome) document.body.classList.add('has-section-numbers');
  };

  /* === SWISS-WHITE v0.5 ADDITIONS === */

  /* Magnetic buttons: opt-in via .magnetic class only.
     (Get Started / .lp-cta intentionally excluded; plain button, no pull.) */
  const magneticButtons = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (window.matchMedia('(pointer: coarse)').matches) return; // touch devices
    const targets = document.querySelectorAll('.magnetic');
    targets.forEach((el) => {
      let raf = null;
      const STRENGTH = 0.18;   // 0..1, higher = more pull
      const RANGE   = 90;      // px of activation radius beyond button bounds
      el.addEventListener('mousemove', (e) => {
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          const r = el.getBoundingClientRect();
          const cx = r.left + r.width / 2;
          const cy = r.top + r.height / 2;
          const dx = e.clientX - cx;
          const dy = e.clientY - cy;
          const dist = Math.hypot(dx, dy);
          const max = Math.max(r.width, r.height) / 2 + RANGE;
          if (dist < max) {
            el.style.transform = `translate3d(${dx * STRENGTH}px, ${dy * STRENGTH}px, 0)`;
          }
        });
      });
      el.addEventListener('mouseleave', () => {
        if (raf) cancelAnimationFrame(raf);
        el.style.transform = '';
      });
    });
  };

  /* Number counter: count up to target on viewport entry */
  const numberCounters = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const els = document.querySelectorAll('.count-up');
    if (!els.length || !('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const target = parseFloat(el.getAttribute('data-target') || el.textContent || '0');
        const duration = 450;
        const start = performance.now();
        const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);
        const step = (now) => {
          const t = Math.min(1, (now - start) / duration);
          const value = Math.round(target * easeOutQuart(t));
          el.textContent = value.toLocaleString();
          if (t < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
        io.unobserve(el);
      });
    }, { threshold: 0.6 });
    els.forEach((el) => io.observe(el));
  };

  /* GSAP ScrollTrigger: tasteful landing choreography.
     Loads only if GSAP global present + motion allowed + on a landing page.
     Fail-open: if GSAP missing, elements render at natural CSS state (visible). */
  const gsapLanding = () => {
    const gsap = window.gsap;
    const ScrollTrigger = window.ScrollTrigger;
    if (!gsap || !ScrollTrigger) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (!document.querySelector('.lp')) return;

    gsap.registerPlugin(ScrollTrigger);
    /* instant-nav safety: clear triggers from a previous page */
    ScrollTrigger.getAll().forEach((t) => t.kill());

    const OUT = 'power2.out';

    /* (a) Hero text: stagger children in on load */
    const heroBits = document.querySelectorAll(
      '.lp-hero-text .lp-eyebrow, .lp-hero-text > h1, .lp-hero-text .lp-sub,' +
      '.lp-hero-text .highlight, .lp-hero-text .lp-cta'
    );
    const introTweens = [];
    if (heroBits.length) {
      introTweens.push(gsap.from(heroBits, {
        y: 0, duration: 0.5, ease: OUT,
        stagger: 0.09, delay: 0.05, clearProps: 'all',
      }));
    }

    /* (b) Hero diagram card: fade up on load */
    const card = document.querySelector('.lp-hero-card');
    if (card) {
      introTweens.push(gsap.from(card, {
        y: 0, duration: 0.6, ease: OUT,
        delay: 0.25, clearProps: 'all',
      }));
    }

    /* Safety net (truly fail-open): an on-load from() holds the above-the-fold hero
       at opacity:0 until the rAF ticker advances it, but rAF is PAUSED in a
       backgrounded tab, so the hero could stay blank for a never-focused tab, a
       crawler, or a prerender/screenshot. setTimeout still fires in the background,
       so force any unfinished intro tween to its visible end-state. No-op once the
       (sub-second) entrance has already completed in a normal foreground load. */
    setTimeout(() => {
      introTweens.forEach((tw) => { if (tw && tw.progress() < 1) tw.progress(1); });
      const introEls = [...heroBits, card].filter(Boolean);
      if (introEls.length) gsap.set(introEls, { clearProps: 'all' });
    }, 1600);

    /* Only animate blocks that start BELOW the fold; anything already in the
       initial viewport (e.g. the short ecosystem landing) stays visible, so a
       flaky/late ScrollTrigger or a re-init can never strand it at opacity:0. */
    const belowFold = (el) => el.getBoundingClientRect().top > (window.innerHeight || 800) * 0.9;

    /* (c) Section eyebrows (FEATURES / TUTORIALS): slide in from left on scroll */
    gsap.utils.toArray('.lp-section .lp-eyebrow').forEach((eb) => {
      if (!belowFold(eb)) return;
      gsap.from(eb, {
        scrollTrigger: { trigger: eb, start: 'top 88%' },
        x: -16, duration: 0.45, ease: OUT, clearProps: 'all',
      });
    });

    /* (d) Cards: staggered fade-up per grid on scroll */
    gsap.utils.toArray('.lp-grid-4').forEach((grid) => {
      const cards = grid.querySelectorAll('.lp-card');
      if (!cards.length) return;
      if (!belowFold(grid)) return;   // in view at setup → keep visible, don't hide
      gsap.from(cards, {
        scrollTrigger: { trigger: grid, start: 'top 85%' },
        y: 26, duration: 0.5, ease: OUT,
        stagger: 0.08, clearProps: 'all',
      });
    });

    /* (e) Feature columns: staggered fade-up on scroll */
    const featRow = document.querySelector('.lp-feature-row');
    if (featRow && belowFold(featRow)) {
      const feats = featRow.querySelectorAll('.lp-feat');
      if (feats.length) {
        gsap.from(feats, {
          scrollTrigger: { trigger: featRow, start: 'top 88%' },
          y: 22, duration: 0.5, ease: OUT,
          stagger: 0.08, clearProps: 'all',
        });
      }
    }

    /* (f) Footer bar: fade up on scroll */
    const footer = document.querySelector('.lp-footer-bar');
    if (footer && belowFold(footer)) {
      gsap.from(footer, {
        scrollTrigger: { trigger: footer, start: 'top 92%' },
        y: 16, duration: 0.5, ease: OUT, clearProps: 'all',
      });
    }

    ScrollTrigger.refresh();

    /* Fail-open safety net for the scroll-revealed blocks (cards / features /
       footer / eyebrows). gsap.from(...{scrollTrigger}) parks them at opacity:0
       until the trigger fires. On a SHORT landing they already sit in the initial
       viewport, so if ScrollTrigger doesn't fire on load (backgrounded tab,
       prerender, headless screenshot, or odd scroll metrics) they'd stay blank.
       After a beat, force any still-hidden, in-view block to its visible state.
       Below-the-fold blocks on a long page are untouched; they reveal on scroll
       as designed (a working foreground load makes this a no-op). */
    setTimeout(() => {
      const vh = window.innerHeight || 800;
      document.querySelectorAll(
        '.lp-card, .lp-feat, .lp-footer-bar, .lp-section .lp-eyebrow'
      ).forEach((el) => {
        const r = el.getBoundingClientRect();
        const inView = r.top < vh && r.bottom > 0;
        if (inView && parseFloat(getComputedStyle(el).opacity) < 0.5) {
          gsap.set(el, { clearProps: 'opacity,transform' });
          el.style.opacity = '';
          el.style.transform = '';
        }
      });
    }, 1700);
  };

  /* ── shared SVG helpers for the task-flow ── */
  const TF_NS = 'http://www.w3.org/2000/svg';
  const tfMk = (t, a) => { const e = document.createElementNS(TF_NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const tfScene = () => { const e = tfMk('g', { class: 'tf-scene', opacity: 0 }); return e; };
  const tfLabel = (x, y, str, anchor, size, fill, weight) => {
    const t = tfMk('text', { x, y, 'text-anchor': anchor || 'start', 'font-size': size || 8.5, fill: fill || 'currentColor' });
    if (weight) t.setAttribute('font-weight', weight);
    t.textContent = str; return t;
  };
  /* organic blob path (smooth closed curve around a centre) */
  const tfBlob = (cx, cy, rx, ry) => {
    const n = 12, pts = [];
    for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; const k = 0.93 + Math.random() * 0.12; pts.push([cx + Math.cos(a) * rx * k, cy + Math.sin(a) * ry * k]); }
    let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)} `;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
      const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6;
      const c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += `C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2[0].toFixed(1)} ${p2[1].toFixed(1)} `;
    }
    return d + 'Z';
  };

  /* right-side legend; appended to scene and registered as a fade-in morph */
  const tfLegend = (morphs, sceneG, x, yTop, rows, immediate) => {
    const g = tfMk('g', {});
    let y = yTop;
    rows.forEach((r) => {
      if (r.title) { g.appendChild(tfLabel(x, y, r.title, 'start', 9, 'currentColor', '700')); y += r.gap || 17; return; }
      if (r.swatch === 'tri') g.appendChild(tfMk('polygon', { points: `${x + 5},${y - 8} ${x + 10},${y} ${x},${y}`, fill: r.color || '#64748b' }));
      else if (r.swatch === 'ring') { const rc = tfMk('circle', { cx: x + 5, cy: y - 3, r: 4.5, fill: 'none', stroke: r.color || 'currentColor', 'stroke-width': 1.4 }); if (r.dashed) rc.setAttribute('stroke-dasharray', '2.5 1.8'); g.appendChild(rc); }
      else if (r.swatch === 'striped') { g.appendChild(tfMk('circle', { cx: x + 5, cy: y - 3, r: 5, fill: r.color })); g.appendChild(tfMk('line', { x1: x + 1.5, y1: y, x2: x + 8.5, y2: y - 7, stroke: '#fff', 'stroke-width': 1.1 })); }
      else if (r.swatch === 'sq') g.appendChild(tfMk('rect', { x, y: y - 8, width: 10, height: 10, rx: 1, fill: r.color }));
      else if (r.swatch === 'dot') g.appendChild(tfMk('circle', { cx: x + 5, cy: y - 3, r: 5, fill: r.color }));
      g.appendChild(tfLabel(x + 18, y, r.text, 'start', 8.5));
      y += r.gap || 15;
    });
    sceneG.appendChild(g);
    // immediate → legend rides the scene fade-in (visible from the start), no delayed reveal
    if (!immediate) morphs.push({ el: g, from: { opacity: 0 }, to: { opacity: 1 }, dur: 0.4, delay: 0.5 });
    return g;
  };

  /* Task-flow showcase: six downstream tasks, smooth in-place
     morphs + per-scene legends on the right. No arrows. GSAP master timeline. */
  const taskFlow = () => {
    const wrap = document.querySelector('.taskflow');
    // instant-nav: a previous page's hero card may have left its perpetual
    // repeat:-1 timeline running on now-detached nodes (burning rAF forever).
    // Kill it when the current page has no task-flow card.
    if (!wrap) { if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; } return; }
    const scenesG = wrap.querySelector('.tf-scenes');
    const headNum = wrap.querySelector('.tf-num');
    const headTitle = wrap.querySelector('.tf-title');
    const dots = wrap.querySelectorAll('.tf-dots span');
    if (!scenesG) return;
    if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; }
    scenesG.innerHTML = '';

    const mk = tfMk, label = tfLabel;
    const BLUES = ['#dbeafe', '#93c5fd', '#60a5fa', '#3b82f6', '#1d4ed8'];
    const CT = ['#3b82f6', '#10b981', '#ef4444'];
    const COND_A = '#f59e0b', COND_B = '#3b82f6', UNK = '#cbd5e1';
    const rnd = (a) => a[Math.floor(Math.random() * a.length)];
    const PERSON = 'M11 2.4a3.3 3.3 0 1 1 0 6.6 3.3 3.3 0 0 1 0-6.6zm0 8.1c3.7 0 6.6 1.8 6.6 4.1V17H4.4v-2.4c0-2.3 2.9-4.1 6.6-4.1z';
    const person = (cx, cy, h, fill) => {
      const s = h / 19;
      const outer = mk('g', {});
      const inner = mk('g', { transform: `translate(${(cx - 11 * s).toFixed(1)},${(cy - 9.5 * s).toFixed(1)}) scale(${s.toFixed(3)})` });
      const p = mk('path', { d: PERSON, fill });
      inner.appendChild(p); outer.appendChild(inner); outer.__path = p;
      return outer;
    };
    const cell = (cx, cy, color, tri, striped) => {
      const g = mk('g', {});
      if (tri) g.appendChild(mk('polygon', { points: `${cx},${cy - 7} ${cx + 6.4},${cy + 5.5} ${cx - 6.4},${cy + 5.5}`, fill: color }));
      else g.appendChild(mk('circle', { cx, cy, r: 6, fill: color }));
      if (striped) {
        g.appendChild(mk('line', { x1: cx - 3.5, y1: cy + 3, x2: cx + 3, y2: cy - 3.5, stroke: '#fff', 'stroke-width': 1.1, opacity: 0.85 }));
        g.appendChild(mk('line', { x1: cx - 0.5, y1: cy + 4.5, x2: cx + 4.5, y2: cy - 0.5, stroke: '#fff', 'stroke-width': 1.1, opacity: 0.85 }));
      }
      return g;
    };

    const reg = (arr, el, from, to, dur, delay) => arr.push({ el, from, to, dur: dur || 1.3, delay: delay || 0 });

    /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       Scene set for the task-flow card. The builder returns a fresh `scenes[]`
       array so render() can rebuild it cleanly.
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    const BUILDERS = {};

    /* ══════════════════════════ Matilda ══════════════════════════ */
    BUILDERS.matilda = () => {
    const scenes = [];

    /* ① Multimodal integration: three modality inputs → arrow → a centred shared latent z.
       Balanced left→centre→right layout (inputs left, z centre, legend right) so the
       frame never goes lopsided the way a lone z + right legend did. */
    (() => {
      const g = tfScene(); const m = [];
      const mods = [
        { color: '#7c3aed', y: 60 },   // RNA
        { color: '#0ea5e9', y: 100 },  // ADT
        { color: '#f59e0b', y: 140 },  // ATAC
      ];
      // three modality input strips on the left (colours are keyed in the legend; no inline text)
      const sN = 4, ssq = 14, sgap = 4, sx0 = 40;
      mods.forEach((mod, mi) => {
        for (let i = 0; i < sN; i++) {
          const rr = mk('rect', { x: sx0 + i * (ssq + sgap), y: mod.y, width: ssq, height: ssq, rx: 2, fill: mod.color, opacity: 0 });
          g.appendChild(rr);
          reg(m, rr, { opacity: 0 }, { opacity: 0.9 }, 0.4, 0.15 + mi * 0.08 + i * 0.05);
        }
      });
      // arrow: fades in WITH the modality strips (not before, in a blank frame)
      const arr1 = mk('line', { x1: 128, y1: 104, x2: 192, y2: 104, stroke: '#94a3b8', 'stroke-width': 2.2, opacity: 0 });
      g.appendChild(arr1);
      const arh1 = mk('polygon', { points: '192,99 203,104 192,109', fill: '#94a3b8', opacity: 0 });
      g.appendChild(arh1);
      reg(m, arr1, { opacity: 0 }, { opacity: 1 }, 0.4, 0.25);
      reg(m, arh1, { opacity: 0 }, { opacity: 1 }, 0.4, 0.3);
      // shared latent z: centred in the frame (column centre ≈ x230)
      const latX = 224;
      const lat = mk('g', {});
      for (let k = 0; k < 6; k++) lat.appendChild(mk('rect', { x: latX, y: 64 + k * 16, width: 14, height: 14, rx: 2, fill: '#6d28d9' }));
      g.appendChild(lat);
      reg(m, lat, { opacity: 0, scale: 0.5, svgOrigin: (latX + 7) + ' 112' }, { opacity: 1, scale: 1, svgOrigin: (latX + 7) + ' 112' }, 0.7, 0.55);
      const zlab = label(latX + 7, 56, 'z', 'middle', 11, '#6d28d9', '700'); zlab.setAttribute('opacity', 0);
      g.appendChild(zlab);
      reg(m, zlab, { opacity: 0 }, { opacity: 1 }, 0.5, 0.85);
      // legend fades in with the rest of the scene (synced, not ahead of the right side)
      tfLegend(m, g, 300, 58, [
        { title: 'Modalities' },
        { swatch: 'sq', color: '#7c3aed', text: 'RNA' },
        { swatch: 'sq', color: '#0ea5e9', text: 'ADT' },
        { swatch: 'sq', color: '#f59e0b', text: 'ATAC' },
        { title: 'VAE encoder' },
        { swatch: 'sq', color: '#6d28d9', text: 'Shared latent z' },
      ], false);
      scenes.push({ num: 1, title: 'Multimodal integration', g, morphs: m });
    })();

    /* ② Data simulation: a few real cells → an arrow → many neatly-tiled simulated cells */
    (() => {
      const g = tfScene(); const m = [];
      // a few real "seed" cells on the left, one per cell type; these fade in first
      CT.forEach((col, i) => {
        const rc = mk('circle', { cx: 60, cy: 78 + i * 36, r: 7, fill: col, opacity: 0 });
        g.appendChild(rc);
        reg(m, rc, { opacity: 0 }, { opacity: 1 }, 0.4, 0.1 + i * 0.05);
      });
      // arrow: appears WITH the real cells (not before), then the grid grows from it
      const arr2 = mk('line', { x1: 92, y1: 114, x2: 150, y2: 114, stroke: '#94a3b8', 'stroke-width': 2.4, opacity: 0 });
      g.appendChild(arr2);
      const arh2 = mk('polygon', { points: '150,109 161,114 150,119', fill: '#94a3b8', opacity: 0 });
      g.appendChild(arh2);
      reg(m, arr2, { opacity: 0 }, { opacity: 1 }, 0.4, 0.2);
      reg(m, arh2, { opacity: 0 }, { opacity: 1 }, 0.4, 0.25);
      // many VAE-simulated cells in a TIDY GRID (1 → many); tile in row by row
      const cols = 5, rows = 5, dx = 22, dy = 22, gx0 = 178, gy0 = 64;
      let idx = 0;
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const cx = gx0 + c * dx, cy = gy0 + r * dy;
        const ci = mk('circle', { cx, cy, r: 6, fill: CT[(r + c) % 3] });
        g.appendChild(ci);
        reg(m, ci, { opacity: 0, scale: 0, svgOrigin: cx + ' ' + cy }, { opacity: 1, scale: 1, svgOrigin: cx + ' ' + cy }, 0.45, 0.3 + idx * 0.03);
        idx++;
      }
      // legend matches the actual cells (coloured by type); real vs simulated is shown by the arrow
      tfLegend(m, g, 300, 70, [
        { title: 'Cell type' },
        { swatch: 'dot', color: CT[0], text: 'Type A' },
        { swatch: 'dot', color: CT[1], text: 'Type B' },
        { swatch: 'dot', color: CT[2], text: 'Type C' },
      ]);
      scenes.push({ num: 2, title: 'Data simulation', g, morphs: m });
    })();

    /* ③ Cell-type classification: query cells are coloured by their predicted type */
    (() => {
      const g = tfScene(); const m = [];
      const cols = 5, rows = 3, gx = 42, gy = 44, x0 = 60, y0 = 58;
      let idx = 0;
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const cx = x0 + c * gx, cy = y0 + r * gy;
        const col = CT[(c + r) % 3];
        const ci = mk('circle', { cx, cy, r: 6.5, fill: '#cbd5e1' });
        g.appendChild(ci);
        reg(m, ci, { attr: { fill: '#cbd5e1' } }, { attr: { fill: col } }, 0.5, 0.5 + idx * 0.045);
        idx++;
      }
      tfLegend(m, g, 300, 64, [
        { title: 'Predicted' },
        { swatch: 'dot', color: CT[0], text: 'Type A' },
        { swatch: 'dot', color: CT[1], text: 'Type B' },
        { swatch: 'dot', color: CT[2], text: 'Type C' },
      ]);
      scenes.push({ num: 3, title: 'Cell-type classification', g, morphs: m });
    })();

    /* ④ Feature selection: integrated gradients highlight markers across modalities */
    (() => {
      const g = tfScene(); const m = [];
      const IMP = '#be123c';
      const rows = [
        { name: 'RNA',  y: 56,  imp: [2, 5, 8] },
        { name: 'ADT',  y: 104, imp: [1, 4] },
        { name: 'ATAC', y: 152, imp: [0, 3, 6, 9] },
      ];
      const N = 11, sq = 13, gap = 4, x0 = 70;
      rows.forEach((row) => {
        g.appendChild(label(x0 - 30, row.y + 10, row.name, 'start', 8.5, 'currentColor', '700'));
        for (let i = 0; i < N; i++) {
          const fx = x0 + i * (sq + gap);
          const r = mk('rect', { x: fx, y: row.y, width: sq, height: sq, rx: 1.5, fill: '#e2e8f0' });
          g.appendChild(r);
          if (row.imp.includes(i)) reg(m, r, { attr: { fill: '#e2e8f0' } }, { attr: { fill: IMP } }, 0.4, 0.6 + i * 0.05);
        }
      });
      tfLegend(m, g, 300, 84, [
        { title: 'Integrated' },
        { title: 'gradients', gap: 18 },
        { swatch: 'sq', color: IMP, text: 'Important' },
        { swatch: 'sq', color: '#e2e8f0', text: 'Other' },
      ]);
      scenes.push({ num: 4, title: 'Feature selection', g, morphs: m });
    })();

    return scenes;
    };  /* end BUILDERS.matilda */

    /* ── Render the active scene-set; dots are rebuilt to match its length ── */
    const render = (key) => {
      if (!BUILDERS[key]) key = 'matilda';
      if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; }
      scenesG.innerHTML = '';
      const scenes = BUILDERS[key]();
      scenes.forEach((s) => scenesG.appendChild(s.g));

      const dotsWrap = wrap.querySelector('.tf-dots');
      if (dotsWrap) {
        dotsWrap.innerHTML = '';
        scenes.forEach((_, i) => {
          const sp = document.createElement('span');
          if (i === 0) sp.className = 'is-active';
          dotsWrap.appendChild(sp);
        });
      }
      const dots = wrap.querySelectorAll('.tf-dots span');

    const setHead = (i) => {
      headNum.textContent = scenes[i].num;
      headTitle.textContent = scenes[i].title;
      dots.forEach((d, j) => d.classList.toggle('is-active', j === i));
    };

    const gsap = window.gsap;
    if (!gsap) { scenes[0].g.setAttribute('opacity', 1); setHead(0); return; }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      scenes[0].g.setAttribute('opacity', 1);
      scenes[0].morphs.forEach((mo) => gsap.set(mo.el, mo.to));
      setHead(0); return;
    }

    setHead(0);
    const master = gsap.timeline({ repeat: -1 });
    window.__tfTl = master;
    const startTimes = [];                   // where each scene begins on the master
    scenes.forEach((s, idx) => {
      startTimes[idx] = master.duration();   // current end = this scene's start time
      const st = gsap.timeline();
      st.call(() => setHead(idx), null, 0);
      st.set(s.g, { opacity: 0 }, 0);
      s.morphs.forEach((mo) => st.set(mo.el, mo.from, 0));
      st.to(s.g, { opacity: 1, duration: 0.4, ease: 'power2.out' }, 0);
      let maxEnd = 0.45;
      s.morphs.forEach((mo) => {
        st.to(mo.el, Object.assign({ duration: mo.dur, ease: 'power2.inOut', delay: mo.delay }, mo.to), 0.45);
        maxEnd = Math.max(maxEnd, 0.45 + mo.delay + mo.dur);
      });
      st.to(s.g, { opacity: 0, duration: 0.45, ease: 'power2.in' }, maxEnd + 3.3); /* +2s hold */
      master.add(st);
    });

    /* Clickable dots: jump to any scene and keep auto-playing from there.
       The master is one declarative timeline, so seeking to a scene's start
       renders that scene's exact state; play() resumes the loop. Dots are
       rebuilt on every render(), so handlers are wired fresh each switch
       (old dot elements, and their listeners, are discarded). */
    dots.forEach((dot, i) => {
      dot.addEventListener('click', () => { setHead(i); master.seek(startTimes[i]).play(); });
    });
    };  /* end render() */

    render((wrap.dataset.taskflow) || 'matilda');

    /* Perf: pause the perpetual loop when the card is scrolled off-screen, so it
       isn't burning CPU/rAF the whole time you're reading further down the page. */
    if (!wrap.dataset.visWired && 'IntersectionObserver' in window) {
      wrap.dataset.visWired = '1';
      new IntersectionObserver((ents) => {
        ents.forEach((e) => { const tl = window.__tfTl; if (tl) { e.isIntersecting ? tl.play() : tl.pause(); } });
      }, { threshold: 0.01 }).observe(wrap);
    }

  };

  /* Sliding TOC indicator: one vertical bar that animates to the active heading,
     instead of a border jumping between items.
     Tracks Material's `.md-nav__link--active`, which the scrollspy toggles. */
  const tocSlider = () => {
    const getCtx = () => {
      const list = document.querySelector('.md-sidebar--secondary .md-nav--secondary > .md-nav__list');
      if (!list) return null;
      let bar = list.querySelector(':scope > .toc-indicator');
      if (!bar) { bar = document.createElement('div'); bar.className = 'toc-indicator'; list.appendChild(bar); }
      return { list, bar };
    };
    let raf = null;
    const move = () => {
      raf = null;
      const ctx = getCtx();
      if (!ctx) return;
      const active = document.querySelector('.md-sidebar--secondary .md-nav__link--active');
      // Keep the bar where it is during the brief moments the scrollspy has no
      // active link (it toggles classes in two steps); avoids flicker.
      if (!active) return;
      const a = active.getBoundingClientRect(), l = ctx.list.getBoundingClientRect();
      ctx.bar.style.height = Math.round(a.height) + 'px';
      ctx.bar.style.transform = 'translateY(' + Math.round(a.top - l.top) + 'px)';
      ctx.bar.style.opacity = '1';
    };
    const schedule = () => { if (raf == null) raf = requestAnimationFrame(move); };
    if (!window.__tocSliderWired) {
      window.__tocSliderWired = true;
      window.addEventListener('scroll', schedule, { passive: true });
      window.addEventListener('resize', schedule, { passive: true });
      // NOTE: a document.body-wide class MutationObserver used to re-run move()
      // (a getBoundingClientRect reflow) on EVERY class change Material makes
      // (scrollspy, header autohide, nav state), thrashing layout on every nav
      // click and every scroll frame. Removed; scroll/resize + the timed move()
      // calls below and the per-navigation re-init keep the indicator in sync.
    }
    move();                      // create + position the bar immediately
    setTimeout(move, 400);       // re-position after fonts/layout settle
    setTimeout(move, 1200);      // and once more after instant-nav / late render
  };

  const init = () => {
    enableSectionNumbersOnHome();
    initFadeUp();
    stickyHeader();
    scrollProgress();
    backToTop();
    magneticButtons();
    numberCounters();
    gsapLanding();
    taskFlow();
    tocSlider();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  if (typeof document$ !== 'undefined' && document$.subscribe) {
    document$.subscribe(init);
  }
})();
