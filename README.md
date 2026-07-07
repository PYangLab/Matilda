# Matilda: Multi-task learning from single-cell multimodal omics

<br />

<img src="https://raw.githubusercontent.com/PYangLab/Matilda/main/img/logo.png" align="right" width="100" />

Matilda is a multi-task framework for learning from single-cell multimodal omics data. Matilda leverages the information from the multi-modality of such data and trains a neural network model to simultaneously learn multiple tasks including data simulation, dimension reduction, visualization, classification, and feature selection.

<img width=100% src="https://raw.githubusercontent.com/PYangLab/Matilda/main/img/main.jpg"/>

For more details, please check out our [publication](https://academic.oup.com/nar/article/51/8/e45/7076464).

---

This repository packages Matilda as a **Python** package (`pip install matilda-sc`, import `matilda`) and an **R** package, wrapping the **unchanged** published engine, with documentation and runnable Colab tutorials.

- **Documentation:** https://pyanglab.github.io/Matilda/
- **Python tutorial (Colab):** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PYangLab/Matilda/blob/main/colab/tutorial-python.ipynb)
- **R tutorial (Colab):** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PYangLab/Matilda/blob/main/matilda-r/inst/colab/tutorial-r.ipynb)
- **R package:** https://github.com/PYangLab/Matilda/tree/main/matilda-r (SingleCellExperiment-based, Bioconductor-style)

## Install

**Python** (import name `matilda`; PyPI distribution `matilda-sc`):

```bash
pip install matilda-sc
```

**R** (Python is provisioned automatically by `basilisk`, so you never install or manage it):

```r
remotes::install_github("PYangLab/Matilda", subdir = "matilda-r")
```

## Two interfaces, one engine

Matilda ships as a Python package (`import matilda`) and an R package (`matilda`, operating on a
`SingleCellExperiment`). Both call the **same** engine, so on the same hardware they give the same
result. Pick the interface that matches your workflow; see the [documentation](https://pyanglab.github.io/Matilda/)
for both APIs and full tutorials.

## Quickstart (Python)

Work with in-memory `AnnData` (or arrays, or file paths) and get results back as objects. After
`train`, there is one verb per task: `classify` / `reduce` / `markers` / `simulate`.

```python
import matilda

# rna/adt/atac: AnnData | ndarray | scipy.sparse | path | None
# labels: a vector, an `.obs` column name, or a .csv path (string or numeric labels)
fit = matilda.train(rna, adt=adt, atac=atac, labels="cell_type")

res = matilda.classify({"rna": q_rna, "adt": q_adt, "atac": q_atac},
                       model=fit, query_labels=q_labels)
res.predictions        # DataFrame: cell_id, real, predicted, probability
res.celltype_accuracy  # DataFrame: celltype, accuracy, n

lat = matilda.reduce({"rna": rna, "adt": adt, "atac": atac}, model=fit)                        # lat.latent
mk  = matilda.markers({"rna": rna, "adt": adt, "atac": atac}, model=fit, labels="cell_type")   # mk.markers
sim = matilda.simulate({"rna": rna, "adt": adt, "atac": atac}, model=fit, celltype="B.Naive", n=200, labels="cell_type")  # sim.simulated
```

The modality combination is inferred automatically (RNA only, RNA+ADT for CITE-seq, RNA+ATAC for
SHARE-seq, all three for TEA-seq). `classify` reconciles features automatically: if the query is
missing some of the model's features it takes the per-modality reference intersection (real values,
no zero-padding), retrains, and classifies; `res.retrained` / `res.common_features` report what
happened. See the [Quickstart](https://pyanglab.github.io/Matilda/quickstart/) and
[API reference](https://pyanglab.github.io/Matilda/api-python/) for the full surface.

## Citation

If you use Matilda, please cite the Matilda paper 

Liu C, Huang H, Yang P. Multi-task learning from multimodal single-cell omics with Matilda. Nucleic Acids Research, 51(8), e45 (2023). https://doi.org/10.1093/nar/gkad157

(see the [Citation](https://pyanglab.github.io/Matilda/citation/) page).
