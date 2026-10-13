#!/usr/bin/env python3
"""
p300_baseline.py — real-data-only P300 classification, corrected vs uncorrected markers.

Training unit = (subject, site): the first usable session the subject recorded at that site.
Train on the first 20+5 trials per class (train+val pooled, as P300_pipeline's baseline does),
classify with P300_pipeline.train_p300_lda (SFS, 5 features, shrinkage LDA) on the
0.15–0.5 s window. Test sets, all in the SAME marker set as training:
    within        remaining trials of the training session
    cross_session every other usable session at the same site
    cross_site    every usable session at the other two sites

Classifiers: sfs5 (train_p300_lda) and full (shrinkage LDA on every feature), column `clf`.
Usage: python p300_baseline.py [--out FILE] [variant:thresh ...]   default: mine:100 gedai:100 gedai:50
Writes p300_baseline.csv (or FILE) + p300_sessions_<variant>_<thresh>.csv.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import p300_multisite as P                       # noqa: E402
from P300_pipeline import train_p300_lda         # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402

MARKERS = {"cor": "X_cor", "raw": "X_raw"}


def train_full_lda(X, y):
    """Secondary classifier: shrinkage LDA on every feature, no stepwise selection."""
    F = P.features(X)
    return LDA(solver="lsqr", shrinkage="auto").fit(F, y), list(range(F.shape[1]))


CLASSIFIERS = {"sfs5": lambda X, y: train_p300_lda(feat3(X), y), "full": train_full_lda}


def feat3(X):
    """(n, ch, 73) epochs -> (ch, 28, n), the layout train_p300_lda expects."""
    return X[:, :, P.feat_idx(X.shape[-1])].transpose(1, 2, 0)


def score(clf, sel, X, y):
    F = P.features(X)[:, sel]
    pred = clf.predict(F)
    d = clf.decision_function(F)
    return dict(n_test=len(y), n_tgt=int((y == 2).sum()),
                acc=float(np.mean(pred == y)),
                bacc=float(balanced_accuracy_score(y, pred)),
                auc=float(roc_auc_score(y == 2, d)) if len(np.unique(y)) == 2 else np.nan)


def load_all(variant, thresh):
    inv = P.inventory()
    inv = inv[inv.has_eeg].reset_index(drop=True)
    data, rows = {}, []
    for r in inv.itertuples():
        key = (r.subject, r.session)
        try:
            d = P.build(r.subject, r.session, variant, thresh)
            tt = P.trial_table(r.subject, r.session)
            d["err_med"] = float(tt.err_ms.median())
            data[key] = d
            rows.append(dict(subject=r.subject, session=r.session, site=r.site, visit=r.visit,
                             home=r.home, n_kept=len(d["y"]), n_tgt=int((d["y"] == 2).sum()),
                             n_rejected=d["n_rejected"], splittable=P.split_within(d["y"]) is not None,
                             err_med=d["err_med"], why=""))
        except Exception as ex:
            rows.append(dict(subject=r.subject, session=r.session, site=r.site, visit=r.visit,
                             home=r.home, splittable=False, why=str(ex)[:120]))
    return data, pd.DataFrame(rows)


def training_units(sess: pd.DataFrame):
    """(subject, site) -> first splittable session there, by visit order."""
    ok = sess[sess.splittable == True].sort_values("visit")  # noqa: E712
    return ok.groupby(["subject", "site"]).head(1)[["subject", "site", "session"]].reset_index(drop=True)


def run(variant, thresh):
    data, sess = load_all(variant, thresh)
    units = training_units(sess)
    usable = sess[sess.n_kept.fillna(0) > 0]
    out = []
    for u in units.itertuples():
        dtr = data[(u.subject, u.session)]
        tr, va, te = P.split_within(dtr["y"])
        pool = np.concatenate([tr, va])
        for (mk, xk), (cname, train) in [(m, c) for m in MARKERS.items() for c in CLASSIFIERS.items()]:
            clf, sel = train(dtr[xk][pool], dtr["y"][pool])
            if clf is None:
                continue
            base = dict(variant=variant, thresh=thresh, clf=cname, marker=mk, subject=u.subject,
                        train_session=u.session, train_site=u.site, err_train=dtr["err_med"])
            out.append({**base, "regime": "within", "test_session": u.session, "test_site": u.site,
                        "err_test": dtr["err_med"], **score(clf, sel, dtr[xk][te], dtr["y"][te])})
            for t in usable[(usable.subject == u.subject) & (usable.session != u.session)].itertuples():
                dte = data[(t.subject, t.session)]
                if len(np.unique(dte["y"])) < 2:
                    continue
                regime = "cross_session" if t.site == u.site else "cross_site"
                out.append({**base, "regime": regime, "test_session": t.session, "test_site": t.site,
                            "err_test": dte["err_med"], **score(clf, sel, dte[xk], dte["y"])})
        print(f"  {variant}@{thresh} {u.subject} {u.session} done", flush=True)
    sess.assign(variant=variant, thresh=thresh).to_csv(
        os.path.join(HERE, f"p300_sessions_{variant}_{int(thresh)}.csv"), index=False)
    return pd.DataFrame(out)


if __name__ == "__main__":
    args = sys.argv[1:]
    fname = "p300_baseline.csv"
    if "--out" in args:
        i = args.index("--out"); fname = args[i + 1]; del args[i:i + 2]
    configs = args or ["mine:100", "gedai:100", "gedai:50"]
    res = []
    for c in configs:
        v, t = c.split(":")
        res.append(run(v, float(t)))
    df = pd.concat(res, ignore_index=True)
    df["d_offset_ms"] = (df.err_test - df.err_train).abs()
    f = os.path.join(HERE, fname)
    if os.path.exists(f) and args:
        old = pd.read_csv(f)
        keep = ~old.set_index(["variant", "thresh"]).index.isin(df.set_index(["variant", "thresh"]).index)
        df = pd.concat([old[keep], df], ignore_index=True)
    df.to_csv(f, index=False)
    print(df.groupby(["variant", "thresh", "clf", "marker", "regime"])[["bacc", "auc"]].agg(["mean", "count"]).round(3))
