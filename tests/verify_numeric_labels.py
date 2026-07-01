"""Regression test for the numeric-label bug: with INTEGER labels the object API must
map predictions to the same class names as the path-based engine (which orders label
strings lexicographically). Integers 0..10 are chosen so numeric order [0,1,2,..,10]
differs from lexicographic order ['0','1','10','2',..,'9'] — the bug, if present, permutes
the names.

    PYTHONPATH=<repo>/src MATILDA_DEMO=<demo_dir> python tests/verify_numeric_labels.py
"""
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from matilda import io, train, task, main_train, main_task

DATA = os.environ.get("MATILDA_DEMO", "")
SEED = 1
EPOCHS = 2


def labels_of(csv):
    return pd.read_csv(csv, header=None, index_col=False).iloc[1:, 1].tolist()


def parse_pred(path):
    pred = []
    with open(path) as fh:
        for ln in fh:
            b = re.search(r"predicted cell type:\s*(.+?)\s+probability:", ln)
            if b:
                pred.append(b.group(1).strip())
    return np.array(pred)


# map the 11 string classes -> integers 0..10 (numeric vs lexicographic order differ)
trl = labels_of(f"{DATA}/train_cty.csv")
tel = labels_of(f"{DATA}/test_cty.csv")
mapping = {c: i for i, c in enumerate(sorted(set(trl)))}
tr_int = [mapping[c] for c in trl]
te_int = [mapping[c] for c in tel]

# ---- object API with integer labels ----
rna = io.read_matilda_h5(f"{DATA}/train_rna.h5")
adt = io.read_matilda_h5(f"{DATA}/train_adt.h5")
atac = io.read_matilda_h5(f"{DATA}/train_atac.h5")
m = train(rna, adt, atac, labels=tr_int, epochs=EPOCHS, seed=SEED)
print("classes:", m.classes)
assert m.classes == ["0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9"], \
    "class order is not lexicographic-of-strings: %r" % (m.classes,)

te_rna = io.read_matilda_h5(f"{DATA}/test_rna.h5")
te_adt = io.read_matilda_h5(f"{DATA}/test_adt.h5")
te_atac = io.read_matilda_h5(f"{DATA}/test_atac.h5")
res = task(te_rna, te_adt, te_atac, labels=te_int, model=m,
           classification=True, query=True, seed=SEED)
obj_pred = res.predictions["predicted"].astype(str).values

# ---- path API with the SAME integer labels (engine writes its own real_cty.csv) ----
base = tempfile.mkdtemp(prefix="numbase_")
run = os.path.join(base, "main"); os.makedirs(run)
for mo in ("TEAseq", "CITEseq", "SHAREseq", "rna_only", "RNAseq"):
    os.makedirs(os.path.join(base, "trained_model", mo), exist_ok=True)
os.makedirs(os.path.join(base, "output"), exist_ok=True)
tr_cty_p = io.to_matilda_cty(tr_int, os.path.join(base, "train_cty_int.csv"))
te_cty_p = io.to_matilda_cty(te_int, os.path.join(base, "test_cty_int.csv"))
prev = os.getcwd(); os.chdir(run)
try:
    main_train(f"{DATA}/train_rna.h5", f"{DATA}/train_adt.h5", f"{DATA}/train_atac.h5",
               tr_cty_p, epochs=EPOCHS, seed=SEED)
    main_task(f"{DATA}/test_rna.h5", f"{DATA}/test_adt.h5", f"{DATA}/test_atac.h5",
              te_cty_p, classification=True, query=True, seed=SEED)
    path_pred = parse_pred(os.path.join(base, "output", "classification", "TEAseq",
                                        "query", "accuracy_each_cell.txt"))
finally:
    os.chdir(prev); shutil.rmtree(base, ignore_errors=True)

assert len(obj_pred) == len(path_pred), (len(obj_pred), len(path_pred))
assert np.array_equal(obj_pred, path_pred), \
    "object vs path predicted NAMES differ with integer labels -> numeric-label bug present"
print("NUMERIC-LABEL FIX OK: object == path predicted names with integer labels (n=%d)"
      % len(obj_pred))
