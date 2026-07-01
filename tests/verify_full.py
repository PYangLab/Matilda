"""Full-settings check, per modality combination: the object API must produce the SAME
per-cell predictions as the path-based engine (object == path) for every mode. This
separates wrapper correctness (object==path) from environment effects (absolute number
vs the tutorial, which depends on the torch/CUDA version).

Uses the original demo .h5 / .csv paths directly (no anndata needed) so it runs on any env.

    PYTHONPATH=<repo>/src MATILDA_DEMO=<demo_dir> python tests/verify_full.py
"""
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

import torch
from matilda import train, task, main_train, main_task, rna_train, rna_task

DATA = os.environ.get("MATILDA_DEMO", "")
SEED = 1
EPOCHS = 30
print("torch %s | cuda %s | device %s"
      % (torch.__version__, torch.version.cuda, "cuda" if torch.cuda.is_available() else "cpu"))

RNA, ADT, ATAC = f"{DATA}/train_rna.h5", f"{DATA}/train_adt.h5", f"{DATA}/train_atac.h5"
TRNA, TADT, TATAC = f"{DATA}/test_rna.h5", f"{DATA}/test_adt.h5", f"{DATA}/test_atac.h5"
CTY, TCTY = f"{DATA}/train_cty.csv", f"{DATA}/test_cty.csv"

# mode -> (train adt/atac, test adt/atac) for the object API; path API uses NULL fillers
MODES = {
    "TEAseq":   dict(adt=ADT, atac=ATAC),
    "rna_only": dict(),
    "CITEseq":  dict(adt=ADT),
    "SHAREseq": dict(atac=ATAC),
}
TMODES = {
    "TEAseq":   dict(adt=TADT, atac=TATAC),
    "rna_only": dict(),
    "CITEseq":  dict(adt=TADT),
    "SHAREseq": dict(atac=TATAC),
}


def parse_pred(path):
    pred = []
    with open(path) as fh:
        for ln in fh:
            b = re.search(r"predicted cell type:\s*(.+?)\s+probability:", ln)
            if b:
                pred.append(b.group(1).strip())
    return np.array(pred)


def path_run(mode):
    """Run the raw engine for a mode and return its per-cell predicted labels."""
    base = tempfile.mkdtemp(prefix="full_path_")
    run = os.path.join(base, "main"); os.makedirs(run)
    for mo in ("TEAseq", "CITEseq", "SHAREseq", "rna_only", "RNAseq"):
        os.makedirs(os.path.join(base, "trained_model", mo), exist_ok=True)
    os.makedirs(os.path.join(base, "output"), exist_ok=True)
    # rna_train does NOT write real_cty.csv (engine gap); supply it the way main_train
    # would (lexicographic categories of the string labels) so rna_task can read it.
    trl = pd.read_csv(CTY, header=None, index_col=False).iloc[1:, 1].tolist()
    classes = list(pd.Categorical([str(x) for x in trl]).categories)
    prev = os.getcwd(); os.chdir(run)
    try:
        if mode == "rna_only":
            rna_train(RNA, CTY, epochs=EPOCHS, seed=SEED)
            pd.DataFrame(classes).to_csv("real_cty.csv", index=False, header=False)
            rna_task(TRNA, TCTY, classification=True, query=True, seed=SEED)
        else:
            adt = ADT if mode in ("TEAseq", "CITEseq") else "NULL"
            atac = ATAC if mode in ("TEAseq", "SHAREseq") else "NULL"
            tadt = TADT if mode in ("TEAseq", "CITEseq") else "NULL"
            tatac = TATAC if mode in ("TEAseq", "SHAREseq") else "NULL"
            main_train(RNA, adt, atac, CTY, epochs=EPOCHS, seed=SEED)
            main_task(TRNA, tadt, tatac, TCTY, classification=True, query=True, seed=SEED)
        pred = parse_pred(os.path.join(base, "output", "classification", mode, "query",
                                       "accuracy_each_cell.txt"))
    finally:
        os.chdir(prev); shutil.rmtree(base, ignore_errors=True)
    return pred


print("\n%-9s %9s %9s   %s" % ("mode", "object", "path", "object==path?"))
for mode in ("TEAseq", "rna_only", "CITEseq", "SHAREseq"):
    fit = train(RNA, labels=CTY, epochs=EPOCHS, seed=SEED, **MODES[mode])
    res = task(TRNA, labels=TCTY, model=fit, classification=True, query=True, seed=SEED, **TMODES[mode])
    obj = res.predictions["predicted"].astype(str).values
    pth = path_run(mode)
    same = (len(obj) == len(pth)) and np.array_equal(obj, pth)
    # accuracy vs the supplied test labels (read the engine's parsed real column)
    acc = float((res.predictions["predicted"] == res.predictions["real"]).mean())
    print("%-9s %9.4f %9.4f   %s  (acc=%.4f, n=%d)"
          % (mode, acc, float((pth == res.predictions["real"].astype(str).values).mean()),
             "IDENTICAL" if same else "*** DIFFER ***", acc, len(obj)))
    assert same, "object != path for mode %s — wrapper bug!" % mode

print("\nALL MODES: object API == path engine (per-cell predictions identical).")
print("DONE")
