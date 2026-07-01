"""Coverage: every task reader populated, plus the RNA-only dispatch path.

    PYTHONPATH=<repo>/src MATILDA_DEMO=<demo_dir> python tests/verify_api_tasks.py
"""
import os

import pandas as pd

from matilda import io, train, task

DATA = os.environ.get(
    "MATILDA_DEMO", ""
)
EPOCHS = 2
SEED = 1


def labels_of(csv):
    return pd.read_csv(csv, header=None, index_col=False).iloc[1:, 1].tolist()


rna = io.read_matilda_h5(f"{DATA}/train_rna.h5")
adt = io.read_matilda_h5(f"{DATA}/train_adt.h5")
atac = io.read_matilda_h5(f"{DATA}/train_atac.h5")
tr_labels = labels_of(f"{DATA}/train_cty.csv")
anchor = pd.Series(tr_labels).value_counts().index[0]   # a celltype guaranteed to exist
print("anchor celltype for simulation:", anchor)

# ---- TEAseq: all tasks in one call ----
tr = train(rna, adt, atac, labels=tr_labels, epochs=EPOCHS, seed=SEED)
res = task(rna, adt, atac, labels=tr_labels, model=tr,
           classification=True, dim_reduce=True, fs=True,
           simulation=True, simulation_ct=anchor, simulation_num=50,
           query=False, seed=SEED)
print("TEAseq task ->", res)
assert res.predictions is not None and len(res.predictions) > 0, "no predictions"
assert res.latent is not None and res.latent.shape[1] >= 1, "no latent"
assert res.markers is not None and {"celltype", "feature", "importance"}.issubset(res.markers.columns), "no markers"
assert res.simulated is not None and "rna" in res.simulated, "no simulation"
print("  latent:", res.latent.shape,
      "| markers:", res.markers.shape, "(%d celltypes)" % res.markers["celltype"].nunique(),
      "| sim rna:", res.simulated["rna"].shape,
      "| sim modalities:", sorted(res.simulated.keys()))
print("TEAseq all-task readers OK")

# ---- RNA-only dispatch (rna_train / rna_task; dir bug dodged by pre-created dirs) ----
rt = train(rna, labels=tr_labels, epochs=EPOCHS, seed=SEED)
assert rt.mode == "rna_only", rt.mode
rres = task(rna, labels=tr_labels, model=rt, classification=True, query=False, seed=SEED)
assert rres.mode == "rna_only"
acc = float((rres.predictions["predicted"] == rres.predictions["real"]).mean())
print("RNA-only -> train %r, task acc=%.4f n=%d" % (rt, acc, len(rres.predictions)))
print("RNA-only dispatch OK (rna_train/rna_task ran cleanly)")
print("ALL TASK COVERAGE OK")
