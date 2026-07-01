#!/usr/bin/env python3
"""Generate runnable Google Colab notebooks for the Python and R tutorials.

The notebooks shown on the docs site are the single source of truth and are NOT
modified by this script:
  - docs/tutorial-python.ipynb  (rendered Jupyter notebook)
  - docs/tutorial-r.md          (Markdown page; Colab can only open a notebook)

This script derives Colab-ready copies, each with a runnable install cell prepended
so "Open in Colab -> Run all" works on a fresh machine. They are written into this
repo (PYangLab/Matilda, public, which Colab can open notebooks from):
  - colab/tutorial-python.ipynb            = docs/tutorial-python.ipynb + pip-install cell
  - matilda-r/inst/colab/tutorial-r.ipynb  = docs/tutorial-r.md converted to an R-kernel
                                             notebook + install cell (inline result images,
                                             which Colab regenerates, are stripped)

The "Open in Colab" buttons on the site link to these files on GitHub (see
overrides/main.html). Re-run after editing either tutorial:

    python scripts/gen_colab_notebooks.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # docs-site/
DOCS = ROOT / "docs"
# Written into this same repo (colab/ and matilda-r/inst/colab/) so Colab can fetch them:
PY_OUT = ROOT.parent / "colab" / "tutorial-python.ipynb"
R_OUT = ROOT.parent / "matilda-r" / "inst" / "colab" / "tutorial-r.ipynb"

PY_INSTALL = (
    "# Colab setup: run this first on a fresh machine such as Google Colab.\n"
    "# (Skip if matilda-sc is already installed locally.)\n"
    "# captum==0.7.0 matches the R package's bundled stack so the feature-selection / marker\n"
    "# scores agree between the two interfaces (Colab ships a newer captum by default).\n"
    '%pip install -q "git+https://github.com/PYangLab/Matilda.git" "captum==0.7.0" anndata scanpy'
)

R_INSTALL = """\
# Colab setup: run this first. Choose an R runtime (this notebook requests it
# automatically). The first matilda_train() below builds the bundled Python env via basilisk.
# Install precompiled Linux binaries (Posit Public Package Manager) instead of compiling every
# CRAN dependency from source. This cuts the Seurat/CRAN build from ~20-30 min to a couple of
# minutes on Colab. (If a binary isn't available P3M falls back to source automatically.)
local({
  cn <- tryCatch(system("lsb_release -cs", intern = TRUE), error = function(e) "jammy")
  options(repos = c(CRAN = sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", cn)),
          HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
            paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))
})
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
# scater is a Bioconductor package (used for the latent-space UMAP in section 4), so it must
# be installed via BiocManager, NOT install.packages()/CRAN (CRAN reports it "not available").
BiocManager::install(
  c("SingleCellExperiment", "SummarizedExperiment", "MultiAssayExperiment",
    "S4Vectors", "rhdf5", "HDF5Array", "basilisk", "scater"),
  update = FALSE, ask = FALSE
)
install.packages(c("remotes", "Matrix", "Seurat", "ggplot2", "uwot"))
remotes::install_github("PYangLab/Matilda", subdir = "matilda-r")"""

FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
IMG_LINE = re.compile(r"^!\[[^\]]*\]\(assets/[^)]*\)\s*$", re.MULTILINE)


def _lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(src)}


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def _finalize(cells: list[dict], metadata: dict) -> dict:
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def gen_python() -> None:
    nb = json.loads((DOCS / "tutorial-python.ipynb").read_text(encoding="utf-8"))
    cells = [_code(PY_INSTALL)] + nb["cells"]
    out = _finalize(cells, nb.get("metadata", {}))
    PY_OUT.parent.mkdir(parents=True, exist_ok=True)
    PY_OUT.write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {PY_OUT} ({len(cells)} cells)")


def gen_r() -> None:
    text = IMG_LINE.sub("", (DOCS / "tutorial-r.md").read_text(encoding="utf-8"))
    cells: list[dict] = [_code(R_INSTALL)]
    pos = 0
    for m in FENCE.finditer(text):
        prose = text[pos : m.start()].strip("\n")
        if prose.strip():
            cells.append(_md(prose))
        lang, code = m.group(1), m.group(2).rstrip("\n")
        cells.append(_code(code) if lang == "r" else _md(m.group(0)))
        pos = m.end()
    tail = text[pos:].strip("\n")
    if tail.strip():
        cells.append(_md(tail))
    out = _finalize(
        cells,
        {
            "kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
            "language_info": {"name": "R"},
            # Request a GPU runtime by default: the bundled torch is the cu121 build, so on a
            # GPU it reproduces the reference numbers; on CPU the floating point differs slightly.
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
    )
    R_OUT.parent.mkdir(parents=True, exist_ok=True)
    R_OUT.write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {R_OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    gen_python()
    gen_r()
