---
hide:
  - navigation
  - toc
---

<div class="lp" markdown>

<section class="lp-hero" markdown>

<div class="lp-hero-text" markdown>

# Matilda

<p class="lp-sub">
  Matilda is a <strong>multi-task</strong> neural network for single-cell multimodal
  omics. One model, trained once over RNA, ADT, and ATAC, drives classification,
  dimension reduction, feature selection, and simulation from a single shared
  representation. It is available in both Python (<code>matilda-sc</code>) and
  R (<code>matilda</code>), with matching results from either.
</p>

```bash
pip install matilda-sc   # Python
# or, in R:  remotes::install_github("PYangLab/Matilda", subdir = "matilda-r")
```

<a href="quickstart/" class="md-button md-button--primary lp-cta">Get started</a>
<a href="overview/" class="md-button lp-cta">How it works</a>

</div>

<div class="lp-hero-card">
<div class="taskflow" data-taskflow="matilda">
<div class="tf-main">
<div class="tf-head">
<span class="tf-num">1</span>
<span class="tf-title">Multimodal integration</span>
</div>
<div class="tf-stage">
<svg viewBox="0 0 460 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" aria-hidden="true"><g class="tf-scenes"></g></svg>
</div>
<div class="tf-dots"><span class="is-active"></span></div>
</div>
</div>
</div>

</section>

<section class="lp-section" markdown>

<div class="lp-grid-4" markdown>

<div class="lp-card lp-card-link">
<a class="lp-card-cover" href="tutorial-python/">
<span class="lp-card-kicker">Python</span>
<span class="lp-card-title">Python tutorial</span>
<span class="lp-card-desc">Train and run every task with the object API, <code>matilda.train()</code> and <code>matilda.task()</code>, from the <code>matilda-sc</code> package.</span>
</a>
</div>

<div class="lp-card lp-card-link">
<a class="lp-card-cover" href="tutorial-r/">
<span class="lp-card-kicker">R</span>
<span class="lp-card-title">R tutorial</span>
<span class="lp-card-desc">Run the same model from a SingleCellExperiment with the R object API, reproducing the Python results bit-for-bit.</span>
</a>
</div>

<div class="lp-card lp-card-link">
<a class="lp-card-cover" href="overview/">
<span class="lp-card-kicker">Concepts</span>
<span class="lp-card-title">How it works</span>
<span class="lp-card-desc">See how per-modality encoders, the shared latent space, and the multi-task heads fit together in one network.</span>
</a>
</div>

<div class="lp-card lp-card-link">
<a class="lp-card-cover" href="api-python/">
<span class="lp-card-kicker">Reference</span>
<span class="lp-card-title">API reference</span>
<span class="lp-card-desc">Full details on the Python functions and the R object methods that expose the shared model.</span>
</a>
</div>

</div>

</section>

<section class="lp-section lp-section--band" markdown>

<div class="lp-feature-row">
<div class="lp-feat">
<span class="lp-feat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 3 7.5 12 12 21 7.5 12 3Z"/><path d="M3 12 12 16.5 21 12"/><path d="M3 16.5 12 21 21 16.5"/></svg></span>
<span class="lp-feat-title">Multimodal integration</span>
<span class="lp-feat-desc">Per-modality encoders for RNA, ADT, and ATAC feed a variational autoencoder whose shared latent space integrates every modality into one embedding.</span>
</div>
<div class="lp-feat">
<span class="lp-feat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="2.5"/><circle cx="5" cy="6" r="1.6"/><circle cx="19" cy="6" r="1.6"/><circle cx="5" cy="18" r="1.6"/><circle cx="19" cy="18" r="1.6"/><path d="M10 10.5 6.3 7.1M14 10.5l3.7-3.4M10 13.5l-3.7 3.4M14 13.5l3.7 3.4"/></svg></span>
<span class="lp-feat-title">One model, many tasks</span>
<span class="lp-feat-desc">Because the encoders and latent space are shared, a single trained network performs classification, dimension reduction, feature selection, and simulation, and the tasks reinforce one another.</span>
</div>
<div class="lp-feat">
<span class="lp-feat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6"/><path d="M19.5 19.5l-4.6-4.6"/><path d="M8.5 12v-1.5M10.5 12V8.5M12.5 12v-2.5"/></svg></span>
<span class="lp-feat-title">One tool, two interfaces</span>
<span class="lp-feat-desc">Call the same model from Python or R: the object API in <code>matilda-sc</code> and in the R <code>matilda</code> package give bit-identical results on the same hardware.</span>
</div>
</div>

</section>

</div>
