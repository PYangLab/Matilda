---
tags:
  - reference
  - api
---

# Python API

The Python interface is the **`matilda-sc`** package (import name `matilda`). The high-level
**object API** is `matilda.train()` plus one verb per task (`matilda.classify()` /
`matilda.reduce()` / `matilda.markers()` / `matilda.simulate()`), which take in-memory `AnnData`
(or `{"rna","adt","atac"}` dicts) and return result objects. Each maps **one-to-one** to a verb in
the [R API](api-r.md), and the two produce the same results.

```bash
pip install "git+https://github.com/PYangLab/Matilda.git"   # once on PyPI: pip install matilda-sc
```

```python
import matilda
from matilda import io                  # AnnData <-> Matilda format converters
```

Modes are auto-detected from which modalities you pass: supplying `adt` and `atac` selects
**TEA-seq** (RNA+ADT+ATAC), `adt` only selects **CITE-seq** (RNA+ADT), `atac` only selects
**SHARE-seq** (RNA+ATAC), and RNA alone selects **rna_only**. Pass `None` (or just omit) for any
modality you are not using.

---

## Training

### `matilda.train()`

```python
matilda.train(rna, adt=None, atac=None, labels=None, *, batch_size=64, epochs=30, lr=0.02,
              z_dim=100, hidden_rna=185, hidden_adt=30, hidden_atac=185, seed=1,
              augmentation=True, out_dir=None, device="auto")
```

Train a Matilda model from in-memory objects and return a [`TrainResult`](#results).

| Parameter | Meaning |
|-----------|---------|
| `rna` *(required)* | RNA modality: `AnnData` \| `ndarray` \| `scipy.sparse` \| path \| `None` |
| `adt`, `atac` | optional ADT / ATAC modalities (same accepted types); which are present selects the mode |
| `labels` *(required)* | cell-type labels: a vector, an `.obs` column name (resolved against `rna`), or a `.csv` path (strings or numbers; no missing values) |
| `epochs`, `lr`, `batch_size`, `z_dim`, `hidden_rna`, `hidden_adt`, `hidden_atac`, `seed` | training hyperparameters (defaults match upstream Matilda) |
| `augmentation` | if `True`, run the class-balancing VAE-augmentation stage before refitting |
| `out_dir` | persist the model here (relative paths resolve against the caller's CWD); else a temp dir kept for the session (`result.model_dir`) |
| `device` | `"auto"` (GPU if available), `"cuda"`, or `"cpu"` |

**Returns** a `TrainResult` carrying `.model_path`, `.mode`, `.classes`, and `.features` (the
per-modality ordered feature names, used by `classify` to reuse the model on a matching query).

---

## Tasks

After `train`, run **one verb per task**. Each takes the data (an `AnnData` or a
`{"rna","adt","atac"}` dict) plus the `TrainResult`, and returns a [`TaskResult`](#results) with only
the relevant fields populated.

| Function | Description |
|----------|-------------|
| [`classify()`](#classify) | predict cell types (automatic feature reconciliation) → `.predictions`, `.celltype_accuracy` |
| [`reduce()`](#reduce) | project into the integrated latent space → `.latent` |
| [`markers()`](#markers) | per-cell-type feature importance → `.markers` |
| [`simulate()`](#simulate) | synthetic cells from the VAE → `.simulated` |
| [`task()`](#task) | the combinable call the verbs wrap: run any mix of the four in one engine pass |

### `classify()`

```python
matilda.classify(query, model=None, reference=None, labels=None, query_labels=None, *,
                 epochs=30, seed=1, batch_size=64, z_dim=100, hidden_rna=185, hidden_adt=30,
                 hidden_atac=185, lr=0.02, augmentation=True, out_dir=None, device="auto")
```

Label `query` cells against a trained model. **It reconciles features automatically**, so the call
is the same whether or not the panels match: if the query carries every feature the model was
trained on (equal panel, or a superset) it slices the query to the model's features and **reuses**
the model (no retrain); if the query is missing some (the common cross-dataset case) it takes the
per-modality **intersection** of `reference` and `query` (reference order, real values, **no
zero-padding**), **retrains** on it, and classifies.

| Parameter | Meaning |
|-----------|---------|
| `query` | cells to label: `AnnData` or `{"rna","adt","atac"}` |
| `model` | a `TrainResult`; enables the no-retrain reuse path (omit for pure cross-dataset transfer) |
| `reference`, `labels` | the labelled reference + its labels, required only when a retrain is needed |
| `query_labels` | optional ground-truth labels for the query (adds the per-cell-type accuracy report) |

**Returns** a `TaskResult`: `.predictions` (DataFrame: `cell_id`, `real`, `predicted`,
`probability`), `.celltype_accuracy`, `.retrained` (which path ran), and `.common_features`
(per-modality feature counts used).

### `reduce()`

```python
matilda.reduce(data, model=None, *, labels=None, query=False, device="auto", ...)
```

Project `data` into the model's integrated latent space (the embedding that fuses all modalities).
`labels` is optional (only to annotate the coordinates). **Returns** `.latent` (cells × `z_dim`).

### `markers()`

```python
matilda.markers(data, model=None, *, method="IntegratedGradient", labels=None, query=False, device="auto", ...)
```

Per-cell-type feature importance via Captum. `method` is `"IntegratedGradient"` (default) or
`"Saliency"`. `labels` groups cells by type. **Returns** `.markers`, a tidy
`DataFrame(celltype, feature, importance)`.

### `simulate()`

```python
matilda.simulate(data, model=None, *, celltype=None, n=100, labels=None, include_real=False, device="auto", ...)
```

Generate `n` synthetic cells of `celltype` from the trained VAE (`celltype=None` reconstructs all
cells). **Returns** `.simulated`, the per-modality matrices + labels; add `include_real=True` to also
get the matched real reference cells (same feature space) for real-vs-simulated comparison.

### `task()`

```python
matilda.task(rna, adt=None, atac=None, labels=None, *, model=None, classification=False,
             dim_reduce=False, fs=False, fs_method="IntegratedGradient", simulation=False,
             simulation_ct=None, simulation_num=100, query=False, include_real=False, device="auto", ...)
```

The low-level **combinable** call the verbs wrap: enable any mix of `classification` / `dim_reduce` /
`fs` / `simulation` and they run in a single engine pass (the model loads once). `query=True` marks
the input as a held-out query. The verbs above are thin shortcuts over this.

---

## Results

| Class | Key attributes |
|-------|----------------|
| `TrainResult` | `model_path`, `model_dir`, `mode`, `classes`, `features` (per-modality ordered feature names) |
| `TaskResult` | `predictions`, `celltype_accuracy`, `latent`, `latent_labels`, `markers`, `simulated`, `retrained`, `common_features` (only the requested fields are populated) |

---

## Data / IO helpers (`matilda.io`)

Convert between standard single-cell objects and Matilda's on-disk format:

| Function | Description |
|----------|-------------|
| `io.read_matilda_h5(path)` | read a Matilda `.h5` matrix into an `AnnData` (cells × features) |
| `io.to_matilda_h5(obj, path)` | write an `AnnData` / array to Matilda's `.h5` layout |
| `io.to_matilda_cty(labels, path)` | write a label vector to Matilda's cty `.csv` |
| `io.from_10x(dir)` | read a 10x mtx directory into an `AnnData` (keeps ADT/ATAC features) |

---

## Lower-level function API (file paths)

The original engine functions remain importable for power users / validation. They take Matilda's
on-disk `.h5` + `.csv` **paths** and write outputs under `../trained_model/` and `../output/`; the
object API above wraps them and returns results in memory.

```python
from matilda import main_train, main_task, rna_train, rna_task
main_train("rna.h5", "adt.h5", "atac.h5", "cty.csv", seed=1)             # adt/atac: "NULL" to omit
main_task("rna.h5", "adt.h5", "atac.h5", "cty.csv", classification=True, query=True, seed=1)
```

`main_train` / `main_task` take the same hyperparameters as `train` / `task` (as keyword arguments)
plus the modality paths positionally; `rna_train` / `rna_task` are the RNA-only counterparts.
Each `main_matilda_*` module is also runnable directly as a script (e.g.
`python -m matilda.main_matilda_train`), reading the same options from the command line. Pass the
string `"NULL"` for any modality path you are not using.

---

## R ⇄ Python correspondence

Both languages share the same shape: **`train()`** plus **one verb per task**
(`classify` / `reduce` / `markers` / `simulate`), with a **combinable `task()`** underneath that
runs any mix of tasks in one call (the verbs wrap it). The engine code is identical.

| R | Python |
|---|--------|
| `matilda_train` / `matilda_train_files` | `matilda.train(...)` |
| `matilda_classify(query, reference, label, query_label)` | `matilda.classify(query, model=, reference=, labels=, query_labels=)` |
| `matilda_reduce` | `matilda.reduce(data, model=)` |
| `matilda_markers` | `matilda.markers(data, model=, method=)` |
| `matilda_simulate` | `matilda.simulate(data, model=, celltype=, n=)` |
| `matilda_task(..., classification=, dim_reduce=, fs=, simulation=)` | `matilda.task(..., classification=, dim_reduce=, fs=, simulation=)` |
| `matilda_task_files` / file-path inputs | engine `main_task(...)` / `rna_task(...)` (raw file paths; pre-trained model) |

**One idiomatic difference, kept on purpose:** R writes results back into the object and
returns it (`sce$matilda_pred`, `reducedDim(sce,"MATILDA")`, `metadata(sce)$matilda_markers`),
the Bioconductor pattern, while Python returns a `TaskResult` / `TrainResult`. The *structure*
(`train` + the task verbs over a combinable `task()`) is the same on both sides.

### Reference → query with mismatched features

When the query and reference don't share the same features (the query is missing some and/or
has extras, the usual case across datasets), **`classify` handles it automatically** with the same
call, no separate function. If the query carries every feature the model needs it reuses the
model; if it is missing some, `classify` takes the per-modality **feature intersection** (in
reference order), retrains on it, and applies it to the query (**real values only, never
zero-padding**):

```python
result = matilda.classify(query, model=fit,                    # query: AnnData or {"rna","adt","atac"}
                          reference={"rna": rna, "adt": adt, "atac": atac},
                          labels="cell_type", query_labels="cell_type")
result.predictions         # query cell-type predictions
result.retrained           # True if it reconciled features + retrained, False if it reused the model
result.common_features     # per-modality feature counts actually used
```
```r
out <- matilda_classify(query_sce, reference = reference_sce, label = "cell_type")
```
You pass the reference alongside the query so a retrain is possible when needed (a different
query → a different intersection → its own model). Only modalities present in both are used.

---

## See also

- [Python tutorial](tutorial-python.ipynb)
- [R API](api-r.md)
