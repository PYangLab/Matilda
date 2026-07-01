# matilda tutorial

**`matilda-tutorial.Rmd`** — the complete Matilda workflow in **R** on TEA-seq. A parallel
Python tutorial with the same structure is available as a Jupyter notebook for `matilda-sc` users.

1. **Read your data** into a `SingleCellExperiment`, shown for four formats — native `.h5`
   (`rhdf5`), `.h5ad` (also `rhdf5`, reading the AnnData CSR matrix directly), 10x
   (`Seurat::Read10X`), and a `Seurat` `.rds`.
2. **Train** (`matilda_train`).
3. **Classification** of held-out query cells (+ per-cell-type accuracy plot).
4. **Dimension reduction** (+ latent-space UMAP).
5. **Feature selection** (+ marker-importance heatmap).
6. **Simulation** (+ real-vs-simulated UMAPs per modality).
7. **Modality combinations Matilda supports** (RNA only / RNA+ADT / RNA+ATAC / RNA+ADT+ATAC, + accuracy bar).
8. **Session info**.

## Requirements
- the `matilda` package (Python is provisioned automatically by basilisk);
- the TEA-seq demo data (see the data-loading chunk at the top of the tutorial);
- `rhdf5` (`.h5` + `.h5ad`) and `Seurat` (10x + `.rds`) for the data loaders;
  `scater` for the UMAPs; `ggplot2` for the plots.

## Run
```r
rmarkdown::render("matilda-tutorial.Rmd")
```

Exact figures can vary slightly between machines (a floating-point property of PyTorch); the
tutorial computes and prints the numbers for your run.
