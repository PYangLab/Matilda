---
tags:
  - reference
  - api
---

# R API

The R interface is an **object API**: you load the package with `library(matilda)` and pass a
`SingleCellExperiment` (or `MultiAssayExperiment` / `Seurat`) into each verb. The trained model is
stored inside the object (`metadata(x)$matilda`) and task results are written back onto the same
object (`colData$matilda_pred`, `reducedDim(x, "MATILDA")`, and so on), so a whole analysis pipes
together object-in → object-out. Python is provisioned for you automatically on first use, so you
never install or manage Python yourself.

Every R verb has a one-to-one counterpart in the [Python API](api-python.md), and the two interfaces
produce the same results. Use this page when you work in R; use the [Python API](api-python.md) when
you work in Python.

Conventions shared by every verb:

- `device = c("auto", "cpu", "cuda")` selects the compute backend (`"auto"` prefers a GPU).
- **Modes are auto-detected** from which modalities you supply: `TEAseq` (RNA+ADT+ATAC),
  `CITEseq` (RNA+ADT), `SHAREseq` (RNA+ATAC), `rna_only` (RNA). The mode selects both the
  Python script and the model architecture.
- Supplying `reference =` (a trained object/model) puts the task in **query mode** (the Python
  `--query True` branch): the query cells are scored against a separately trained reference.

---

## Training

| Function | Description |
|----------|-------------|
| [`matilda_train()`](#matilda_train) | Train the multimodal VAE + classifier from an object (or raw matrices); store the model in the object. |
| [`matilda_train_files()`](#matilda_train_files) | Train from on-disk Matilda `.h5` / `.csv` file paths; return a standalone `matilda_model`. |

### `matilda_train()`

```r
matilda_train(x = NULL, label = NULL, rna = NULL, adt = NULL, atac = NULL, cty = NULL,
              assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
              batch_size = 64L, epochs = 30L, lr = 0.02, z_dim = 100L,
              hidden_rna = 185L, hidden_adt = 30L, hidden_atac = 185L,
              augmentation = TRUE, seed = 1L, device = c("auto", "cpu", "cuda"))
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `x` | an SCE / MAE / Seurat object (object path), or `NULL` to use the matrix arguments instead |
| `label` | cell-type labels: a `colData` column name **or** a length-ncell vector (object path) |
| `rna`, `adt`, `atac` | genes × cells matrices (matrix path, used only when `x = NULL`) |
| `cty` | cell-type label vector (matrix path, used only when `x = NULL`) |
| `assay` | assay holding RNA counts (default `"counts"`) |
| `adt_exp`, `atac_exp` | `altExp` names for the ADT / ATAC modalities |
| `batch_size`, `epochs`, `lr`, `z_dim` | training hyperparameters (batch size, epochs, learning rate, latent dimension) |
| `hidden_rna`, `hidden_adt`, `hidden_atac` | encoder hidden widths per modality |
| `augmentation` | class-balancing VAE augmentation (the second training stage) |
| `seed` | random seed |
| `device` | one of `"auto"`, `"cpu"`, `"cuda"` |

**Returns:** the input object with the model stored in `metadata(x)$matilda` (object path, pipeable), or a `matilda_model` (matrix path).

Trains the one multimodal variational autoencoder + classifier that all four Matilda tasks reuse. Pass a `SingleCellExperiment` (or `MultiAssayExperiment` / `Seurat`) and the column name of its cell-type labels; the fitted model is tucked into `metadata(x)$matilda` and the object is returned, so it pipes straight into [`matilda_reduce()`](#matilda_reduce), [`matilda_classify()`](#matilda_classify), and the other task verbs. Defaults match upstream Matilda, so the result is bit-identical to the reference Python on the same hardware. Maps to Python `main_matilda_train.py` (or `main_matilda_rna_train.py` for `rna_only`).

### `matilda_train_files()`

```r
matilda_train_files(rna, adt = NULL, atac = NULL, cty,
                    batch_size = 64L, epochs = 30L, lr = 0.02, z_dim = 100L,
                    hidden_rna = 185L, hidden_adt = 30L, hidden_atac = 185L,
                    augmentation = TRUE, seed = 1L, device = c("auto", "cpu", "cuda"))
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `rna`, `adt`, `atac` | paths to per-modality Matilda `.h5` files (`adt`/`atac` may be `NULL` ⇒ `rna_only`) |
| `cty` | path to the cell-type label `.csv` |
| `batch_size`, `epochs`, `lr`, `z_dim` | training hyperparameters, as in `matilda_train()` |
| `hidden_rna`, `hidden_adt`, `hidden_atac` | encoder hidden widths per modality |
| `augmentation` | class-balancing VAE augmentation |
| `seed` | random seed |
| `device` | one of `"auto"`, `"cpu"`, `"cuda"` |

**Returns:** a `matilda_model`.

The power-user / engine entry point: trains directly from raw on-disk Matilda `.h5` and `.csv` paths, mirroring `main_matilda_train.py` argument-for-argument. Use it when your data is already in Matilda's native file format (e.g. the demo dataset from [`matilda_download_example()`](#matilda_download_example)) and you want to skip the object-conversion layer, or for parity validation against the Python CLI. It returns a standalone `matilda_model` that you then feed to [`matilda_task_files()`](#matilda_task_files). Maps to the same Python script as `matilda_train()`.

---

## Tasks

`matilda_task()` is the combinable **object API**: the model is carried inside the object and any
combination of tasks runs in one call, with results written back onto the object. The four verbs
(`matilda_classify` / `matilda_reduce` / `matilda_markers` / `matilda_simulate`) are single-task
shortcuts over it. `matilda_task_files()` is the **file-path API** for power users and validation.

| Function | Description |
|----------|-------------|
| [`matilda_task()`](#matilda_task) | Run any combination of tasks in one call; results written back onto the object. |
| [`matilda_classify()`](#matilda_classify) | Predict cell types; write `colData$matilda_pred` and `$matilda_prob`. |
| [`matilda_reduce()`](#matilda_reduce) | Project cells into the integrated latent space; write `reducedDim(x, "MATILDA")`. |
| [`matilda_markers()`](#matilda_markers) | Per-cell-type feature importance via integrated gradients / saliency. |
| [`matilda_simulate()`](#matilda_simulate) | Generate synthetic cells for a cell type (or all types). |
| [`matilda_task_files()`](#matilda_task_files) | Run one or more tasks from file paths; copy the `output/` tree to a directory. |

### `matilda_task()`

```r
matilda_task(x, reference = NULL,
             classification = FALSE, dim_reduce = FALSE, fs = FALSE, simulation = FALSE,
             fs_method = c("IntegratedGradient", "Saliency"),
             simulation_ct = NULL, simulation_num = 100L, label = NULL,
             assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
             device = c("auto", "cpu", "cuda"))
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `x` | the object (or matrix list) to run tasks on |
| `reference` | a trained object/`matilda_model`; `NULL` ⇒ use `x`'s own. Non-`NULL` ⇒ query mode |
| `classification`, `dim_reduce`, `fs`, `simulation` | task flags; any combination may be `TRUE` |
| `fs_method` | `"IntegratedGradient"` (default) or `"Saliency"` |
| `simulation_ct`, `simulation_num` | cell type to simulate (`NULL` = all types) and how many cells |
| `label` | cell-type labels (column name or vector); required for `fs` / `simulation` |
| `assay`, `adt_exp`, `atac_exp`, `device` | as for the other tasks |

**Returns:** `x` enriched with the requested results: `colData$matilda_pred`/`$matilda_prob`, `reducedDim(x, "MATILDA")`, `metadata(x)$matilda_markers`, and `metadata(x)$matilda_simulation` (a matrix list returns a named results list instead).

The combinable object-API entry point and the counterpart of Python's `matilda.task()`: switch on any mix of tasks and they run in a single engine pass (the model loads once). The four task verbs below are thin wrappers that each enable one flag. Like the verbs, R returns the **enriched object** (Python's `task()` returns a `TaskResult`, the one idiomatic difference). Maps to Python `main_matilda_task.py` with the corresponding flags.

### `matilda_classify()`

```r
matilda_classify(x, reference = NULL, label = NULL, query_label = NULL,
                 assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
                 epochs = 30L, seed = 1L, device = c("auto", "cpu", "cuda"))
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `x` | the object (or matrix list) to classify (the **query** when `reference` is given) |
| `reference` | a trained object/`matilda_model` to use; `NULL` ⇒ use `x`'s own stored model. **Non-`NULL` ⇒ query mode** (`--query True`) |
| `label` | reference cell-type label (column name or vector). Used only when a **retrain** is required (the query misses model features, or the reference is untrained); when `reference = NULL`, these are `x`'s own labels |
| `query_label` | optional ground-truth labels for the query (adds the per-cell-type accuracy report) |
| `epochs`, `seed` | training options, used only when a retrain is required |
| `assay`, `adt_exp`, `atac_exp` | assay / `altExp` selectors |
| `device` | one of `"auto"`, `"cpu"`, `"cuda"` |

**Returns:** `x` with `colData$matilda_pred` and `colData$matilda_prob` (object path), or a `list(pred, prob)` (matrix path). `metadata(.)$matilda_retrained` records whether it reused the model or retrained; when reconciled, `metadata(.)$matilda_common_features` records the per-modality feature counts kept.

Predicts a cell type for every cell and writes the hard predictions to `colData$matilda_pred` and the softmax class probabilities to `colData$matilda_prob`. Call it on a self-trained object to label its own cells, or pass `reference =` a separately trained object to score **query** cells against that reference. **It reconciles features automatically**, so the call is the same whether or not the panels match: if the query carries every feature the model was trained on (equal panel, or a superset) the query is sliced to the model's features and the model is **reused** (no retrain); if the query is missing some of them (the common cross-dataset case) the per-modality **intersection** of reference and query is taken (RNA + matching `altExp`s, in reference order, **real values only, never zero-padding**), a model is **retrained** on it, and the query is classified. Supplying `query_label` adds per-cell-type accuracy. Maps to Python `main_matilda_task.py --classification True [--query True]` (with a `matilda_train()` on the intersection first when a retrain is needed).

### `matilda_reduce()`

```r
matilda_reduce(x, reference = NULL, label = NULL,
               assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
               device = c("auto", "cpu", "cuda"))
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `x` | the object (or matrix list) to embed |
| `reference` | a trained object/`matilda_model` to use; `NULL` ⇒ use `x`'s own model. Non-`NULL` ⇒ query mode |
| `label` | optional cell-type labels passed through to the engine; `reduce` writes only the latent matrix, so no label column is returned |
| `assay`, `adt_exp`, `atac_exp` | assay / `altExp` selectors |
| `device` | one of `"auto"`, `"cpu"`, `"cuda"` |

**Returns:** `x` with `reducedDim(x, "MATILDA")` (cells × `z_dim`), or a list (matrix path).

Projects every cell into Matilda's integrated latent space (a single `z_dim`-dimensional embedding that fuses all available modalities) and stores it as `reducedDim(x, "MATILDA")`. Use it to get a modality-integrated coordinate system for visualization (e.g. a UMAP of the latent space) or downstream clustering, with the embedding sitting alongside any other `reducedDims` on the object. As with classification, passing `reference =` embeds query cells against a trained reference. Maps to Python `main_matilda_task.py --dim_reduce True [--query True]`.

### `matilda_markers()`

```r
matilda_markers(x, reference = NULL, label = NULL,
                method = c("IntegratedGradient", "Saliency"),
                assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
                device = c("auto", "cpu", "cuda"))
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `x` | the object (or matrix list) to score |
| `reference` | a trained object/`matilda_model` to use; `NULL` ⇒ use `x`'s own model. Non-`NULL` ⇒ query mode |
| `label` | cell-type labels; if omitted, falls back to the label column name recorded at train time (`model$label_col`) resolved against `x`'s `colData` — available only when the model was object-trained with a `colData` column name. Errors if no label can be resolved |
| `method` | attribution method: `"IntegratedGradient"` (default) or `"Saliency"` |
| `assay`, `adt_exp`, `atac_exp` | assay / `altExp` selectors |
| `device` | one of `"auto"`, `"cpu"`, `"cuda"` |

**Returns:** a tidy `data.frame(celltype, feature, importance)`.

Computes per-cell-type feature importance by attributing the classifier's decision back onto the input features with Captum: integrated gradients (default) or plain saliency. The result is a long-format `data.frame` with one row per (cell type, feature) pair, ready to filter for each type's top markers or to feed a marker heatmap. Because importance is defined per cell type, the call needs labels; if omitted it falls back to the label column name recorded at train time (`model$label_col`, resolved against `x`'s `colData`) and errors if none can be resolved. Maps to Python `main_matilda_task.py --fs True --fs_method <method> [--query True]`.

### `matilda_simulate()`

```r
matilda_simulate(x, reference = NULL, celltype = NULL, n = 100L, label = NULL,
                 assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
                 device = c("auto", "cpu", "cuda"))
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `x` | the object (or matrix list) providing the reference cells |
| `reference` | a trained object/`matilda_model` to use; `NULL` ⇒ use `x`'s own model. Non-`NULL` ⇒ query mode |
| `celltype` | the cell type to simulate; `NULL` ⇒ `-1` = reconstruct **all** cells |
| `n` | number of cells to simulate (`--simulation_num`) |
| `label` | cell-type labels; if omitted, falls back to the label column name recorded at train time (`model$label_col`) resolved against `x`'s `colData` — available only for object-trained models with a `colData` column name |
| `assay`, `adt_exp`, `atac_exp` | assay / `altExp` selectors |
| `device` | one of `"auto"`, `"cpu"`, `"cuda"` |

**Returns:** a new `SingleCellExperiment` of simulated cells (counts + ADT/ATAC `altExp`s + `colData$label`); `metadata(.)$real` holds the real reference cells (per-modality matrices + `label`) in the same feature space, for real-vs-simulated UMAPs.

Generates synthetic cells by sampling the trained VAE's latent space and decoding back to expression: `n` cells of one named `celltype`, or (`celltype = NULL`) a reconstruction of all cell types. The returned object is a fresh `SingleCellExperiment` carrying every modality the model knows about, plus the matched real reference cells under `metadata(.)$real` so you can overlay real and simulated cells in the same plot to judge fidelity. Use it to augment rare populations or to sanity-check that the generative model reproduces the data's structure. Maps to Python `main_matilda_task.py --simulation True --simulation_ct <celltype | -1> --simulation_num <n> [--query True]`.

### `matilda_task_files()`

```r
matilda_task_files(model, rna, adt = NULL, atac = NULL, cty,
                   classification = FALSE, fs = FALSE, dim_reduce = FALSE, simulation = FALSE,
                   query = FALSE, fs_method = "IntegratedGradient",
                   simulation_ct = -1, simulation_num = 100L,
                   outdir = ".", device = c("auto", "cpu", "cuda"))
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `model` | a trained `matilda_model` |
| `rna`, `adt`, `atac`, `cty` | input file paths |
| `classification`, `fs`, `dim_reduce`, `simulation` | which task(s) to run, **combinable in one call** |
| `query` | query vs reference (`--query True`) |
| `fs_method` | attribution method for feature selection (`"IntegratedGradient"` / `"Saliency"`) |
| `simulation_ct` | cell type to simulate; `-1` = all cells |
| `simulation_num` | number of cells to simulate |
| `outdir` | directory the produced `output/` tree is copied into |
| `device` | one of `"auto"`, `"cpu"`, `"cuda"` |

**Returns:** the output directory, invisibly; results are written as files.

The power-user / validation counterpart to the four object-API task verbs: it drives the unchanged task script on raw `.h5` / `.csv` inputs and copies the produced `output/` tree into `outdir`. Unlike the object verbs, you can switch on several tasks (`classification`, `fs`, `dim_reduce`, `simulation`) in a single call, exactly as the Python CLI combines its flags. Reach for it when your data already lives in Matilda's file format, when you want the raw on-disk output tree, or to reproduce a Python CLI run for parity checking. Maps to Python `main_matilda_task.py` with the combinable task flags.

---

## Model and example data

| Function | Description |
|----------|-------------|
| [`matilda_model()`](#matilda_model) | Extract the `matilda_model` stored in (or passed as) an object. |
| [`print(<matilda_model>)`](#printmatilda_model) | Print a one-screen summary of a model handle. |
| [`matilda_download_example()`](#matilda_download_example) | Download and cache the TEA-seq demo dataset used by the tutorial. |

### `matilda_model()`

```r
matilda_model(object)
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `object` | an SCE / MAE trained with [`matilda_train()`](#matilda_train), or a `matilda_model` itself |

**Returns:** a `matilda_model`, or `NULL` if none is stored.

Extracts the trained model that `matilda_train()` tucked into `metadata(object)$matilda`, returning a standalone `matilda_model` handle (or passing a `matilda_model` straight through). Use it to pull a model out of one object to use as the `reference =` for classifying / embedding / simulating a different object, or to inspect a fitted model with `print()`. The handle holds the raw checkpoint bytes plus the model's `mode`, `label_levels`, per-modality `features` and `dims`, and the training `hyper`-parameters.

### `print(<matilda_model>)`

```r
## S3 method for class 'matilda_model'
print(x, ...)
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `x` | a `matilda_model` |
| `...` | unused |

**Returns:** `x`, invisibly.

The S3 `print` method for a model handle: prints a compact one-screen summary of the detected mode, the cell types (`label_levels`), the latent dimension, the per-modality feature counts, and the training hyperparameters (epochs / learning rate / batch size / seed / augmentation). Use it to confirm at a glance what a model was trained on before you reuse it as a reference.

### `matilda_download_example()`

```r
matilda_download_example(dest = tools::R_user_dir("matilda", "cache"),
                         url = getOption("matilda.demo_url", .matilda_demo_url),
                         force = FALSE)
```

**Parameters**

| Parameter | Meaning |
|-----------|---------|
| `dest` | cache directory (default: a per-user cache dir) |
| `url` | download URL of the tarball (default: `getOption("matilda.demo_url")`) |
| `force` | re-download even if the data is already cached |

**Returns:** path to the unpacked `matilda_teaseq_demo` directory (the native `.h5` / `.csv` files plus a `formats/` subfolder with `.h5ad`, 10x, `.rds`).

Downloads the small TEA-seq demo tarball used throughout the tutorial (the native Matilda `.h5` train/test files plus the same training data re-exported as `.h5ad`, 10x, and a Seurat `.rds`), caches it under the per-user cache directory, and returns the local directory. Re-running uses the cache instead of downloading again (pass `force = TRUE` to refresh).

---

## See also

- [R tutorial](tutorial-r.md): the complete workflow in R on real TEA-seq.
- [Python API](api-python.md): the matching function API for Python users.
