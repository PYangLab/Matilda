"""Parity: the object API (train/task) must produce IDENTICAL predictions to the
path-based engine (main_train/main_task) on the same inputs, seed and device.

Uses few epochs — parity is about identity of the two code paths, not final accuracy.

    PYTHONPATH=<repo>/src MATILDA_DEMO=<demo_dir> python tests/verify_api.py
"""
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from matilda import io, train, task, main_train, main_task

DATA = os.environ.get(
    "MATILDA_DEMO", ""
)
EPOCHS = 2
SEED = 1


def parse_acc(path):
    real, pred = [], []
    for ln in open(path):
        a = re.search(r"real cell type:\s*(.+?)\s+predicted cell type:", ln)
        b = re.search(r"predicted cell type:\s*(.+?)\s+probability:", ln)
        if a and b:
            real.append(a.group(1).strip())
            pred.append(b.group(1).strip())
    return np.array(real), np.array(pred)


def labels_of(csv):
    return pd.read_csv(csv, header=None, index_col=False).iloc[1:, 1].tolist()


# ---- path-API baseline: canonical engine on the original demo .h5 files ----
base = tempfile.mkdtemp(prefix="paritybase_")
run = os.path.join(base, "main")
os.makedirs(run)
for m in ("TEAseq", "CITEseq", "SHAREseq", "rna_only", "RNAseq"):
    os.makedirs(os.path.join(base, "trained_model", m), exist_ok=True)
os.makedirs(os.path.join(base, "output"), exist_ok=True)
prev = os.getcwd()
os.chdir(run)
try:
    main_train(f"{DATA}/train_rna.h5", f"{DATA}/train_adt.h5", f"{DATA}/train_atac.h5",
               f"{DATA}/train_cty.csv", epochs=EPOCHS, seed=SEED)
    main_task(f"{DATA}/test_rna.h5", f"{DATA}/test_adt.h5", f"{DATA}/test_atac.h5",
              f"{DATA}/test_cty.csv", classification=True, query=True, seed=SEED)
    p_real, p_pred = parse_acc(
        os.path.join(base, "output", "classification", "TEAseq", "query",
                     "accuracy_each_cell.txt"))
finally:
    os.chdir(prev)
acc_path = float((p_real == p_pred).mean())
print("path-API baseline : acc=%.4f  n=%d" % (acc_path, len(p_real)))

# ---- object API on the same data (read .h5 -> AnnData, then train/task) ----
rna = io.read_matilda_h5(f"{DATA}/train_rna.h5")
adt = io.read_matilda_h5(f"{DATA}/train_adt.h5")
atac = io.read_matilda_h5(f"{DATA}/train_atac.h5")
tr = train(rna, adt, atac, labels=labels_of(f"{DATA}/train_cty.csv"), epochs=EPOCHS, seed=SEED)
print("train ->", tr)

te_rna = io.read_matilda_h5(f"{DATA}/test_rna.h5")
te_adt = io.read_matilda_h5(f"{DATA}/test_adt.h5")
te_atac = io.read_matilda_h5(f"{DATA}/test_atac.h5")
res = task(te_rna, te_adt, te_atac, labels=labels_of(f"{DATA}/test_cty.csv"),
           model=tr, classification=True, query=True, seed=SEED)
print("task  ->", res)
acc_obj = float((res.predictions["predicted"] == res.predictions["real"]).mean())
print("object-API        : acc=%.4f  n=%d" % (acc_obj, len(res.predictions)))

# ---- parity assertions ----
assert len(p_pred) == len(res.predictions), (len(p_pred), len(res.predictions))
assert np.array_equal(res.predictions["predicted"].values, p_pred), "predicted labels differ"
assert np.array_equal(res.predictions["real"].values, p_real), "real labels differ"
assert res.celltype_accuracy is not None and len(res.celltype_accuracy) > 0
print("celltype_accuracy rows:", len(res.celltype_accuracy))
print("PARITY OK: object API == path API (per-cell predictions identical), acc=%.4f" % acc_obj)
shutil.rmtree(base, ignore_errors=True)
