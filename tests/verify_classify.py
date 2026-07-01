"""matilda.classify (with transfer folded in) + the reduce/markers verbs.

(A)  reuse path  — query shares the model's panel  -> classify reuses the model (retrained False)
                   and gives the same predictions as a plain task(query, classification=True).
(A') superset    — query has extra features         -> classify slices to the model panel, still reuses.
(B)  retrain path — query misses features (+ extras) -> classify takes the reference∩query intersection,
                   retrains (retrained True), and equals an explicit train(intersection)+task.
(C)  verbs        — reduce/markers == the equivalent task() flags.

    PYTHONPATH=<repo>/src MATILDA_DEMO=<demo_dir> python tests/verify_classify.py
"""
import os

import numpy as np
import pandas as pd
import anndata as ad

from matilda import io, classify, reduce, markers, train, task

DATA = os.environ.get("MATILDA_DEMO", "")
SEED = 1
EPOCHS = 2


def labels_of(csv):
    return pd.read_csv(csv, header=None, index_col=False).iloc[1:, 1].tolist()


ref = {m: io.read_matilda_h5(f"{DATA}/train_{m}.h5") for m in ("rna", "adt", "atac")}
qry = {m: io.read_matilda_h5(f"{DATA}/test_{m}.h5") for m in ("rna", "adt", "atac")}
trl = labels_of(f"{DATA}/train_cty.csv")
tel = labels_of(f"{DATA}/test_cty.csv")
N = ref["rna"].n_vars

fit = train(ref["rna"], adt=ref["adt"], atac=ref["atac"], labels=trl, epochs=EPOCHS, seed=SEED)

# (A) reuse: query shares the panel -> reuse the model, no retrain, == plain task(query).
res_reuse = classify({"rna": qry["rna"], "adt": qry["adt"], "atac": qry["atac"]},
                     model=fit, query_labels=tel)
assert res_reuse.retrained is False, res_reuse.retrained
res_base = task(qry["rna"], adt=qry["adt"], atac=qry["atac"], labels=tel, model=fit,
                classification=True, query=True, seed=SEED)
assert np.array_equal(res_reuse.predictions["predicted"].values,
                      res_base.predictions["predicted"].values), "reuse classify != plain task"
acc = float((res_reuse.predictions["predicted"] == res_reuse.predictions["real"]).mean())
print("(A) reuse: retrained=%s  acc=%.4f  == plain task  OK" % (res_reuse.retrained, acc))

# (A') superset: query has 5 extra RNA features -> classify slices to the model panel, still reuses.
X_sup = np.hstack([np.asarray(qry["rna"].X), np.zeros((qry["rna"].n_obs, 5))])
q_sup_rna = ad.AnnData(X=X_sup, obs=qry["rna"].obs.copy(),
                       var=pd.DataFrame(index=list(qry["rna"].var_names) + [f"EXTRA_{i}" for i in range(5)]))
res_sup = classify({"rna": q_sup_rna, "adt": qry["adt"], "atac": qry["atac"]}, model=fit, query_labels=tel)
assert res_sup.retrained is False, res_sup.retrained
assert np.array_equal(res_sup.predictions["predicted"].values,
                      res_base.predictions["predicted"].values), "superset reuse changed predictions"
print("(A') superset reuse: sliced to model panel, retrained=%s, predictions unchanged  OK" % res_sup.retrained)

# (B) retrain: drop the first 500 RNA features from the query and rename 300 to fakes (extras).
q2 = {m: qry[m].copy() for m in qry}
keep = list(qry["rna"].var_names[500:])
q2["rna"] = qry["rna"][:, keep].copy()
vn = list(q2["rna"].var_names)
for i in range(300):
    vn[i] = "FAKE_%d" % i
q2["rna"].var_names = vn

res_re = classify(q2, model=fit, reference=ref, labels=trl, query_labels=tel, epochs=EPOCHS, seed=SEED)
assert res_re.retrained is True, res_re.retrained
exp = N - 800   # dropped 500 + renamed 300 of the rest to fakes -> 800 non-matching
assert res_re.common_features["rna"] == exp, (res_re.common_features["rna"], exp)
assert res_re.common_features["adt"] == ref["adt"].n_vars and res_re.common_features["atac"] == ref["atac"].n_vars
assert len(res_re.predictions) == q2["rna"].n_obs
acc_re = float((res_re.predictions["predicted"] == res_re.predictions["real"]).mean())
print("(B) retrain: retrained=%s common=%s acc=%.4f  OK" % (res_re.retrained, res_re.common_features, acc_re))

# (B') the retrain path equals an explicit train(intersection)+task(query-on-intersection).
qset = set(q2["rna"].var_names)
common_rna = [v for v in ref["rna"].var_names if v in qset]
fit_i = train(ref["rna"][:, common_rna].copy(), adt=ref["adt"], atac=ref["atac"],
              labels=trl, epochs=EPOCHS, seed=SEED)
res_i = task(q2["rna"][:, common_rna].copy(), adt=q2["adt"], atac=q2["atac"], labels=tel,
             model=fit_i, classification=True, query=True, seed=SEED)
assert np.array_equal(res_re.predictions["predicted"].values, res_i.predictions["predicted"].values), \
    "retrain classify != explicit train(intersection)+task"
print("(B') retrain == explicit train(intersection)+task  OK")

# (C) verbs == equivalent task() flags.
red_v = reduce({"rna": ref["rna"], "adt": ref["adt"], "atac": ref["atac"]}, model=fit, labels=trl)
red_t = task(ref["rna"], adt=ref["adt"], atac=ref["atac"], labels=trl, model=fit, dim_reduce=True, seed=SEED)
assert np.allclose(red_v.latent.values, red_t.latent.values), "reduce verb != task(dim_reduce)"
mk_v = markers({"rna": ref["rna"], "adt": ref["adt"], "atac": ref["atac"]}, model=fit, labels=trl)
mk_t = task(ref["rna"], adt=ref["adt"], atac=ref["atac"], labels=trl, model=fit, fs=True, seed=SEED)
assert mk_v.markers.shape == mk_t.markers.shape, "markers verb != task(fs)"
print("(C) reduce/markers verbs == task() flags  OK")

print("CLASSIFY OK")
