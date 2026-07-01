"""Round-trip check: matilda.io must produce exactly what the engine reads.

Run on a box with torch + the demo data:
    PYTHONPATH=<repo>/src MATILDA_DEMO=<demo_dir> python tests/verify_io.py
"""
import os
import tempfile

import numpy as np
import pandas as pd

from matilda.util import read_h5_data, read_fs_label
from matilda import io

DATA = os.environ.get(
    "MATILDA_DEMO",
    "",
)
rna_h5 = os.path.join(DATA, "train_rna.h5")
cty = os.path.join(DATA, "train_cty.csv")
tmp = tempfile.mkdtemp()

# 1. our read_matilda_h5 == engine read_h5_data (orientation, values, names)
T_engine = read_h5_data(rna_h5).cpu().numpy()          # cells x genes, float32
A = io.read_matilda_h5(rna_h5)                          # AnnData cells x genes
assert A.shape == T_engine.shape, (A.shape, T_engine.shape)
assert np.array_equal(A.X.astype("float32"), T_engine), "read values differ from engine"
print("1. read_matilda_h5 matches engine:", A.shape,
      "| var[:2]", list(A.var_names[:2]), "| obs[:2]", list(A.obs_names[:2]))

# 2. to_matilda_h5 -> engine read_h5_data round-trips bit-identical
p2 = io.to_matilda_h5(A, os.path.join(tmp, "rt.h5"))
T2 = read_h5_data(p2).cpu().numpy()
assert np.array_equal(T2, T_engine), "round-trip data differs"
print("2. to_matilda_h5 -> engine read: bit-identical round-trip OK")

# 2b. bare-array path (synthesised names) still readable by the engine
arr = T_engine[:50].astype("float64")
p3 = io.to_matilda_h5(arr, os.path.join(tmp, "arr.h5"))
T3 = read_h5_data(p3).cpu().numpy()
assert np.array_equal(T3, arr.astype("float32")), "bare-array round-trip differs"
print("2b. bare ndarray -> engine read OK", T3.shape)

# 2c. orientation: a genes x cells array must NOT be double-transposed (cells_axis=1)
gc = T_engine.T.astype("float64")                      # genes x cells
p4 = io.to_matilda_h5(gc, os.path.join(tmp, "gc.h5"), cells_axis=1)
assert np.array_equal(read_h5_data(p4).cpu().numpy(), T_engine), "cells_axis=1 differs"
print("2c. genes x cells input + cells_axis=1 -> correct orientation OK")

# 2d. orientation inferred from n_cells picks the right axis
p5 = io.to_matilda_h5(gc, os.path.join(tmp, "gc2.h5"), n_cells=T_engine.shape[0])
assert np.array_equal(read_h5_data(p5).cpu().numpy(), T_engine), "n_cells inference differs"
print("2d. genes x cells input + n_cells inference -> correct orientation OK")

# 2e. a transposed AnnData is rejected, not silently flipped
try:
    io.to_matilda_h5(A.T, os.path.join(tmp, "bad.h5"), n_cells=A.n_obs)
    raise AssertionError("expected ValueError for transposed AnnData")
except ValueError:
    print("2e. transposed AnnData rejected via n_cells check OK")

# 2f. non-ASCII feature/barcode names round-trip via UTF-8 (not ASCII 'S')
import anndata as ad
small = ad.AnnData(np.arange(6, dtype="float64").reshape(2, 3),
                   obs=pd.DataFrame(index=["célA", "cellβ"]),
                   var=pd.DataFrame(index=["CD8α", "β-actin", "gène"]))
back = io.read_matilda_h5(io.to_matilda_h5(small, os.path.join(tmp, "nonascii.h5")))
assert list(back.var_names) == ["CD8α", "β-actin", "gène"], list(back.var_names)
assert list(back.obs_names) == ["célA", "cellβ"], list(back.obs_names)
print("2f. non-ASCII UTF-8 feature/barcode names round-trip OK")

# 3. to_matilda_cty -> read_fs_label codes match the engine on the original CSV
codes_engine = read_fs_label(cty).cpu().numpy()
raw = pd.read_csv(cty, header=None, index_col=False)   # engine's own read shape
labels = list(raw.iloc[1:, 1])
pc = io.to_matilda_cty(labels, os.path.join(tmp, "cty.csv"))
codes2 = read_fs_label(pc).cpu().numpy()
assert np.array_equal(codes2, codes_engine), "cty codes differ"
print("3. to_matilda_cty -> read_fs_label: codes match, %d classes"
      % (int(codes_engine.max()) + 1))

# 4. from_10x reads non-RNA features (gex_only=False)
atac_dir = os.path.join(DATA, "formats", "10x", "train_atac")
if os.path.isdir(atac_dir):
    a10 = io.from_10x(atac_dir)
    print("4. from_10x ATAC:", a10.shape, "(gex_only=False, var=gene_ids)")
else:
    print("4. from_10x: skipped (no 10x dir at %s)" % atac_dir)

print("IO ROUNDTRIP ALL OK")
