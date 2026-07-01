# matilda

**Multi-task learning from single-cell multimodal omics (RNA + ADT + ATAC), in R.**

`matilda` is an R interface to [Matilda](https://github.com/PYangLab/Matilda). One multimodal
variational autoencoder + classifier is trained once and reused for five tasks —
**classification, dimension reduction, feature selection, data simulation**, and training-time
**augmentation**. The R package runs Matilda's **unchanged** PyTorch code through `basilisk`
+ `reticulate`, so you never install or manage Python, and results are **bit-identical** to the
original on the same hardware (`inst/scripts/parity_check.R`).

The API is Seurat-style: object in → object out. You pass a `SingleCellExperiment` /
`MultiAssayExperiment` / `Seurat`; the trained model is stored in `metadata()`, and results are
written back (`colData$matilda_pred`, `reducedDim "MATILDA"`).

```r
sce <- matilda_train(sce, label = "cell_type")   # train (model stored in the object)
sce <- matilda_reduce(sce)                        # reducedDim "MATILDA"
query <- matilda_classify(query, reference = sce) # colData$matilda_pred / $matilda_prob
mk  <- matilda_markers(sce)                        # per-cell-type feature importance
sim <- matilda_simulate(sce, celltype = "B.Naive", n = 200)
# different feature panels? the same matilda_classify() auto-takes the reference∩query
# intersection and retrains (no zero-padding) — no separate function needed:
out <- matilda_classify(query_diff_panel, reference = sce, label = "cell_type")
```

## Installation

```r
# install.packages("remotes")
remotes::install_github("PYangLab/Matilda", subdir = "matilda-r")
```

Python is provisioned automatically by `basilisk` on the first `matilda_train()` — you never
install or manage Python or CUDA yourself. The Bioconductor dependencies (`SingleCellExperiment`,
`SummarizedExperiment`, `MultiAssayExperiment`, `S4Vectors`, `rhdf5`, `HDF5Array`, `basilisk`)
install most smoothly with `BiocManager` if `install_github` doesn't resolve them automatically.

---

## Tutorial

[`inst/tutorials/matilda-tutorial.Rmd`](inst/tutorials/matilda-tutorial.Rmd) — the complete
workflow in **R** on real TEA-seq. A separate, identically-structured **Python** tutorial is
provided as a Jupyter notebook for Python users. Both are runnable on Google Colab:

- **R tutorial** — [https://colab.research.google.com/github/PYangLab/Matilda/blob/main/matilda-r/inst/colab/tutorial-r.ipynb](https://colab.research.google.com/github/PYangLab/Matilda/blob/main/matilda-r/inst/colab/tutorial-r.ipynb)
- **Python tutorial** — [https://colab.research.google.com/github/PYangLab/Matilda/blob/main/colab/tutorial-python.ipynb](https://colab.research.google.com/github/PYangLab/Matilda/blob/main/colab/tutorial-python.ipynb)

1. **Read your data** — load TEA-seq from `.h5`, `.h5ad`, 10x, or a `Seurat` `.rds` (each verified
   to give the same `SingleCellExperiment`).
2. **Train** (`matilda_train`).
3. **Classification** of held-out query cells (+ per-cell-type accuracy plot).
4. **Dimension reduction** (+ latent-space UMAP).
5. **Feature selection** (+ marker heatmap).
6. **Simulation** (+ real-vs-synthetic UMAP).
7. **Modality combinations Matilda supports** — RNA only / RNA+ADT / RNA+ATAC / RNA+ADT+ATAC on the
   *same* cells (a modality ablation; adding the ADT panel helps most).

---

## Repository structure

### R package — the binding (`R/`)

| File | Responsibility |
|------|----------------|
| `matilda-package.R` | package-level roxygen / imports |
| `train.R`   | `matilda_train()`, `matilda_train_files()` — train; build the `matilda_model` |
| `tasks.R`   | `matilda_classify/reduce/markers/simulate()`, `matilda_task_files()`, shared `.run_task()`, result write-backs |
| `model.R`   | the `matilda_model` S3 class: constructor, `print`, accessor, store/resolve |
| `convert.R` | `.as_modalities()` — SCE/MAE/Seurat/list → `rna/adt/atac` matrices + labels + mode |
| `io_write.R`| `.write_h5_matilda()`, `.write_cty_csv()` — R objects → Matilda `.h5`/`.csv` |
| `io_read.R` | parse Python outputs (classification / latent / markers / simulation) |
| `bridge.R`  | `.matilda_run()` — run a vendored script in the env (sets `sys.argv` + cwd + `py_run_file`) |
| `basilisk.R`| the bundled Python environment (torch 2.1.2, captum, scanpy, pandas, …) |
| `rundir.R`  | stage the temporary `../` tree the scripts expect; seed the trained checkpoint |
| `utils.R`   | `.mode_of()`, `.ncells()`, `.h5_features()`, `.pkg_py()` |
| `data.R`    | `matilda_example_teaseq()`, `matilda_example_sce()` — local / synthetic example data |
| `download.R`| `matilda_download_example()` — download + cache the public TEA-seq demo (~75 MB) |


## API reference

Each R verb has a one-to-one counterpart in the Python `matilda-sc` package
(`import matilda`). `device = c("auto","cpu","cuda")` on every call.

### Train

| R | Python (`matilda-sc`) | Effect |
|---|--------|--------|
| `matilda_train(x, label, assay="counts", adt_exp="ADT", atac_exp="ATAC", batch_size=64, epochs=30, lr=0.02, z_dim=100, hidden_rna=185, hidden_adt=30, hidden_atac=185, augmentation=TRUE, seed=1, device)` | `matilda.train(rna, adt=None, atac=None, labels=, batch_size=64, epochs=30, lr=0.02, z_dim=100, hidden_rna=185, hidden_adt=30, hidden_atac=185, augmentation=True, seed=1, device="auto")` | train on an SCE/MAE/Seurat (or matrices); store model in `metadata(x)$matilda`; return `x` |
| `matilda_train_files(rna, adt=NULL, atac=NULL, cty, …same hyperparams…, device)` | same — pass file paths as `rna`/`adt`/`atac` | train from file paths; return a `matilda_model` |

### Tasks — object API (model carried in the object, results written back)

| R | Python (`matilda-sc`) | Returns / writes back |
|---|--------|-----------------------|
| `matilda_classify(x, reference=NULL, label=NULL, assay, adt_exp, atac_exp, device)` | `matilda.classify(query, model=fit, reference=, labels=, query_labels=)` | `colData$matilda_pred`, `$matilda_prob` |
| `matilda_reduce(x, reference=NULL, …)` | `matilda.reduce(data, model=fit)` | `reducedDim(x, "MATILDA")` |
| `matilda_markers(x, reference=NULL, method=c("IntegratedGradient","Saliency"), …)` | `matilda.markers(data, model=fit, method=)` | `data.frame(celltype, feature, importance)` |
| `matilda_simulate(x, reference=NULL, celltype=NULL, n=100, label=NULL, …)` | `matilda.simulate(data, model=fit, celltype=, n=)` | a simulated `SingleCellExperiment` (`celltype=NULL` ⇒ all cells) |

### Tasks — combinable call

| R | Python (`matilda-sc`) | Effect |
|---|--------|--------|
| `matilda_task_files(model, rna, adt=NULL, atac=NULL, cty, classification=FALSE, fs=FALSE, dim_reduce=FALSE, simulation=FALSE, query=FALSE, fs_method="IntegratedGradient", simulation_ct=-1, simulation_num=100, outdir=".", device)` | engine `main_task(...)` / `rna_task(...)` (raw file paths; pre-trained model) | run any combination of tasks in one engine pass |

### Model handle & example data

| R | Effect |
|---|--------|
| `matilda_model(object)` | extract the stored `matilda_model` from an SCE/MAE (or pass one through) |
| `print(<matilda_model>)` | summary: mode, cell types, latent dim, per-modality feature counts, hyperparameters |
| `matilda_download_example()` | download + cache the TEA-seq demo dataset (~75 MB) used by the tutorial; returns the local dir |

**Modes** are auto-detected from which modalities are present: `TEAseq` (RNA+ADT+ATAC),
`CITEseq` (RNA+ADT), `SHAREseq` (RNA+ATAC), `rna_only` (RNA). The mode selects both the Python
script and the model architecture.


