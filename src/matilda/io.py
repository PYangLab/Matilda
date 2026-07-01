"""matilda.io — convert between standard single-cell objects and Matilda's on-disk format.

The Matilda engine reads a bespoke HDF5 layout and a quirky label CSV. These helpers
hide that layout so callers can work with ``AnnData`` / arrays instead of hand-writing
the files. The numeric content is exactly what the engine's ``read_h5_data`` /
``read_fs_label`` expect (verified by round-trip).

Matilda ``.h5`` layout (group ``matrix``):

* ``data``     : float matrix stored **genes x cells** (features x barcodes).
                 ``read_h5_data`` transposes it to cells x genes for the model.
* ``features`` : 1-D array of feature names, UTF-8 **bytes** (decoded with
                 ``str(x, encoding="utf-8")`` by the engine).
* ``barcodes`` : 1-D array of cell barcodes, UTF-8 bytes.

Label (cty) CSV: a ``pandas`` CSV whose **second column** holds the per-cell labels,
with the header row skipped and labels factorised alphabetically (``read_fs_label``).
Written as ``pd.DataFrame({"x": labels}).to_csv(path)``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import h5py

__all__ = [
    "to_matilda_h5",
    "read_matilda_h5",
    "to_matilda_cty",
    "from_10x",
]


def _densify(x):
    """Return a dense 2-D float64 ``ndarray`` from ndarray / scipy.sparse / AnnData.X."""
    if hasattr(x, "toarray"):          # scipy.sparse matrix/array
        x = x.toarray()
    arr = np.asarray(x, dtype="float64")
    if arr.ndim != 2:
        raise ValueError(
            "expected a 2-D cells x genes matrix, got shape %r" % (arr.shape,)
        )
    return arr


def _to_str(s):
    """Decode an HDF5 name (bytes or str) to a Python ``str``."""
    if isinstance(s, (bytes, bytearray)):
        return s.decode("utf-8")
    return str(s)


def to_matilda_h5(obj, path, *, n_cells=None, cells_axis=None):
    """Write ``obj`` to Matilda's ``.h5`` layout (genes x cells) and return ``path``.

    The engine stores ``matrix/data`` as **genes x cells**. Rather than blindly
    transposing, this resolves the input orientation from the dimensions and
    transposes only when needed:

    * ``AnnData`` — ``.X`` is cells x genes by the AnnData contract, so it is
      transposed to genes x cells. If ``n_cells`` is given and ``.X`` instead matches
      it on the *gene* axis, the object looks transposed and a ``ValueError`` is raised
      (an AnnData is never silently flipped).
    * bare ``ndarray`` / ``scipy.sparse`` — orientation is taken from ``cells_axis``
      (explicit), else inferred from ``n_cells`` (the axis whose length equals
      ``n_cells`` is the cell axis), else defaults to rows = cells. An array that is
      already genes x cells is written as-is; a cells x genes array is transposed.

    Parameters
    ----------
    obj : AnnData | numpy.ndarray | scipy.sparse
        Expression matrix (cells x genes for AnnData; either orientation for a bare
        array, disambiguated via ``cells_axis`` / ``n_cells``).
    path : str
        Destination ``.h5`` path; parent directories are created.
    n_cells : int, optional
        Known number of cells (e.g. ``len(labels)``). Used to infer a bare array's
        orientation and to validate an AnnData's orientation.
    cells_axis : {0, 1}, optional
        For a bare array, which axis indexes cells (0 = rows, 1 = columns). Takes
        precedence over ``n_cells`` inference; ignored for AnnData (fixed by contract).

    Notes
    -----
    The matrix is densified and written as float64 (the engine casts to float32 on
    read); densification mirrors the engine.
    """
    if hasattr(obj, "X"):                            # AnnData (duck-typed, no hard import)
        x = _densify(obj.X)                           # cells x genes by contract
        if n_cells is not None and x.shape[0] != n_cells:
            if x.shape[1] == n_cells:
                raise ValueError(
                    "AnnData looks transposed: X is %d x %d but n_cells=%d matches the "
                    "gene axis. AnnData.X must be cells x genes."
                    % (x.shape[0], x.shape[1], n_cells)
                )
            raise ValueError(
                "AnnData has %d cells (X rows) but n_cells=%d." % (x.shape[0], n_cells)
            )
        data = x.T                                    # genes x cells
        features = [str(s) for s in obj.var_names]
        barcodes = [str(s) for s in obj.obs_names]
    else:                                            # bare array / scipy.sparse
        m = _densify(obj)
        if cells_axis is not None:
            if cells_axis not in (0, 1):
                raise ValueError("cells_axis must be 0 or 1, got %r" % (cells_axis,))
            cells_on_rows = cells_axis == 0
        elif n_cells is not None:
            r, c = m.shape
            if r == n_cells and c != n_cells:
                cells_on_rows = True
            elif c == n_cells and r != n_cells:
                cells_on_rows = False
            elif r == n_cells and c == n_cells:
                cells_on_rows = True                  # square: default rows = cells
                warnings.warn(
                    "square %d x %d array with n_cells=%d is orientation-ambiguous; "
                    "assuming rows = cells. Pass cells_axis= to be explicit."
                    % (r, c, n_cells), RuntimeWarning,
                )
            else:
                raise ValueError(
                    "neither axis of the %d x %d array matches n_cells=%d; pass "
                    "cells_axis= to disambiguate." % (r, c, n_cells)
                )
        else:
            cells_on_rows = True                      # default convention: rows = cells
        data = m.T if cells_on_rows else m            # -> genes x cells
        n_c = m.shape[0] if cells_on_rows else m.shape[1]
        n_g = m.shape[1] if cells_on_rows else m.shape[0]
        features = ["feature_%d" % i for i in range(n_g)]
        barcodes = ["cell_%d" % i for i in range(n_c)]

    if data.shape != (len(features), len(barcodes)):
        raise ValueError(
            "internal orientation error: data %r vs (features=%d, barcodes=%d)"
            % (data.shape, len(features), len(barcodes))
        )

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=data)                           # genes x cells
        # UTF-8 bytes (dtype 'S' would ASCII-encode and crash on non-ASCII names);
        # the engine decodes with str(x, encoding="utf-8").
        g.create_dataset("features", data=np.array([s.encode("utf-8") for s in features]))
        g.create_dataset("barcodes", data=np.array([s.encode("utf-8") for s in barcodes]))
    return path


def read_matilda_h5(path):
    """Read a Matilda ``.h5`` (genes x cells) back into an ``AnnData`` (cells x genes)."""
    import anndata as ad

    with h5py.File(path, "r") as f:
        if "matrix/data" not in f:
            raise KeyError(
                "%r is not a Matilda .h5: missing 'matrix/data'. Expected group "
                "'matrix' with datasets data/features/barcodes." % path
            )
        data = np.asarray(f["matrix/data"][:]).T                      # cells x genes
        features = [_to_str(s) for s in f["matrix/features"][:]]
        barcodes = [_to_str(s) for s in f["matrix/barcodes"][:]]
    return ad.AnnData(
        X=data,
        obs=pd.DataFrame(index=pd.Index(barcodes, name=None)),
        var=pd.DataFrame(index=pd.Index(features, name=None)),
    )


def to_matilda_cty(labels, path):
    """Write per-cell ``labels`` to Matilda's cty CSV and return ``path``.

    The engine (``read_fs_label``) reads the **second column**, skips the header row,
    and factorises the labels alphabetically into integer codes.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    # Stringify so the CSV content matches the class order derived in api.train()
    # (`pd.Categorical([str(x) ...])`): the engine re-reads this column as text and
    # factorises it lexicographically, so labels must be the same string form on both sides.
    pd.DataFrame({"x": [str(v) for v in labels]}).to_csv(path)
    return path


def from_10x(directory, *, gex_only=False, var_names="gene_ids"):
    """Read a 10x ``mtx`` directory into an ``AnnData`` (cells x genes).

    ``gex_only`` defaults to ``False`` so non-RNA features (ADT "Antibody Capture",
    ATAC "Peaks") are **not** silently dropped — a common cause of a 0-feature modality.
    ``var_names="gene_ids"`` avoids duplicate-symbol collisions.
    """
    import scanpy as sc

    return sc.read_10x_mtx(directory, var_names=var_names, gex_only=gex_only)
