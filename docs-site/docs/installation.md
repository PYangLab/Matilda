---
tags:
  - getting-started
---

# Installation

Matilda has two interfaces to the **same** tool. The Python side is the `matilda-sc`
package with an object API (`matilda.train()` plus the task verbs `matilda.classify()` /
`reduce()` / `markers()` / `simulate()`). The R side is the `matilda` package with an object
API over a SingleCellExperiment. The two produce the same results, so pick the interface that
matches your workflow.

!!! tip "TL;DR"

    === "Python"

        ```bash
        pip install matilda-sc
        python -c "import matilda; print(matilda.__version__)"
        ```

    === "R"

        ```r
        remotes::install_github("PYangLab/Matilda", subdir = "matilda-r")
        # Python is provisioned automatically by basilisk on the first matilda_train()
        ```

---

## Install Matilda

=== "Python"

    Matilda is developed with **PyTorch** and is built to run on a **CUDA GPU**; CPU also
    works but is slower. We recommend an isolated environment so the `torch` /
    `scanpy` / `captum` dependencies don't pollute your base.

    ```bash title="terminal"
    pip install matilda-sc
    ```

    This installs the `matilda-sc` package (import name `matilda`) from GitHub with **all** its
    dependencies — `torch`, `h5py`, `numpy`, `pandas`, `captum`, `tqdm`, `scipy`, `anndata`, and
    `scanpy` (which in turn brings `umap-learn` / `matplotlib` / `scikit-learn`, so the tutorial
    runs out of the box). One command — nothing to install separately.

    Verify the install:

    ```bash title="terminal"
    python -c "import matilda; print(matilda.__version__)"
    ```

    If a version string prints, the object API (`matilda.train()` plus `matilda.classify()` /
    `reduce()` / `markers()` / `simulate()`) is reachable. See the Python
    [Quickstart](quickstart.md) next.

    !!! note "Prefer not to manage PyTorch / CUDA?"

        The [R interface](tutorial-r.md) provisions its own Python environment via
        `basilisk`, with no manual PyTorch or CUDA setup. Switch to the **R** tab above.

=== "R"

    The `matilda` R package (**v0.99.0**, a Bioconductor-style package) wraps Matilda's
    **unchanged** PyTorch code and exposes the object API on a SingleCellExperiment. The
    most direct route is to install from GitHub.

    ```r title="R console"
    # via remotes
    remotes::install_github("PYangLab/Matilda", subdir = "matilda-r")

    # or via devtools
    devtools::install_github("PYangLab/Matilda", subdir = "matilda-r")
    ```

    !!! tip "Python is provisioned automatically, you never install it"

        The first time you call `matilda_train()`, `basilisk` builds and manages the
        bundled Python environment for you (PyTorch, captum, scanpy, pandas, …) through
        `reticulate`. **You never install, activate, or manage Python or CUDA yourself.**
        The package's `SystemRequirements` lists Python (>= 3.7), but `basilisk`
        satisfies it, so there is no manual step.

    ### R dependencies

    These come from the package `DESCRIPTION` and are installed for you when you install
    `matilda`:

    - **Imports**: `methods`, `basilisk`, `reticulate`, `rhdf5`, `HDF5Array`,
      `S4Vectors`, `SummarizedExperiment`, `SingleCellExperiment`,
      `MultiAssayExperiment`, `utils`, `stats`
    - **Suggests**: `testthat`, `knitr`, `rmarkdown`, `Seurat`, `scater`, `uwot`,
      and `ggplot2` (Seurat for some data loaders; scater/uwot/ggplot2 for the
      tutorial UMAP and plots)

    The Bioconductor dependencies (`SingleCellExperiment`, `SummarizedExperiment`,
    `MultiAssayExperiment`, `S4Vectors`, `rhdf5`, `HDF5Array`, `basilisk`) install most
    smoothly with `BiocManager`. If `install_github` doesn't resolve them automatically,
    install them first, then the package:

    ```r title="R console"
    if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
    BiocManager::install(c("SingleCellExperiment", "SummarizedExperiment", "MultiAssayExperiment",
                           "S4Vectors", "rhdf5", "HDF5Array", "basilisk"))
    remotes::install_github("PYangLab/Matilda", subdir = "matilda-r")
    ```

    ### Verify the install

    ```r title="R console"
    library(matilda)
    data_dir <- matilda_download_example()   # downloads demo data (~75 MB)
    ```

    Then follow the [R tutorial](tutorial-r.md). The first `matilda_train()` call there
    provisions the `basilisk` Python environment (slower on first run; subsequent runs reuse it).

---

Next: the [Quickstart](quickstart.md).
