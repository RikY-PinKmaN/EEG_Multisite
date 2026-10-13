#!/usr/bin/env python3
"""
classify.py — ROADMAP step 3: does augmentation rescue decoding from bad markers?

Design (within-session, the clean one)
-------------------------------------
One session, split in two, only the timing differs:

    train    trials 1..N with photodiode-CORRECTED markers
    ceiling  held-out trials with CORRECTED markers   <- best achievable
    test     the SAME held-out trials with RAW markers <- the "no photodiode" case

Same subject, same session, same amplifier, same recording, disjoint trials. The only
difference between ceiling and test is whether the marker was corrected, so any gap is
caused by timing alone — no hardware or site confound. The question is how much of that
gap each augmentation recipe closes.

Trial count is swept because it is the independent variable the roadmap identifies: the
GAN's job is manufacturing data when there is too little, the slide's job is teaching the
classifier that timing varies. Only the second is tested here (GAN excluded by request).

Leakage control: shift samplers never draw from the session being evaluated
(leave-one-session-out), so no arm is fitted on the corruption it is scored against.

Also runs cross-session transfer (train on one session, test on another) covering both
same-site retest and true cross-site deployment.

Writes: classify_within.csv, classify_cross.csv, figures/c_*.png
"""
from __future__ import annotations
import os
import zlib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

from jitter_pipeline import ROOT
from raw_markers import build_trial_table, SITE_OF
import epochs as EP
import augment as AUG

OUT = os.path.join(ROOT, "analysis")
FIG = os.path.join(OUT, "figures")

TASKS = ["n170", "p3b"]
FEAT_LO, FEAT_HI = 0.0, 900.0     # wide enough to still contain a component displaced ~150 ms
BIN_MS = 40.0
N_GRID = [10, 20, 40, "full"]
N_REPEATS = 25
N_AUG_COPIES = 5
TEST_FRAC = 0.3
MIN_TEST_PER_CLASS = 8
SEED = 20260727

ARMS = ["none", "generic", "slide_broad", "slide_allsites", "slide_site", "slide_site+generic"]
SITE_COL = {"LDS": "#3b7dd8", "MAN": "#37a66b", "SHE": "#d1495b"}


# ------------------------------------------------------------------ features
def featurise(Xc):
    """Bin-average over time and flatten to (n_trials, n_chan * n_bins)."""
    n, ch, T = Xc.shape
    step = max(1, int(round(BIN_MS / (1000.0 / EP.TARGET_FS))))
    nb = T // step
    return Xc[:, :, :nb * step].reshape(n, ch, nb, step).mean(-1).reshape(n, ch * nb)


def fit_score(Xtr, ytr, tests):
    """Fit shrinkage LDA on standardised features; return AUC for each named test set."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LDA(solver="eigen", shrinkage="auto")
    clf.fit((Xtr - mu) / sd, ytr)
    out = {}
    for name, (Xte, yte) in tests.items():
        s = clf.decision_function((Xte - mu) / sd)
        out[name] = float(roc_auc_score(yte, s)) if len(np.unique(yte)) > 1 else np.nan
    return out


# ------------------------------------------------------------------ augmentation arms
def build_training(arm, Xtr_pad, ytr, t, samplers, rng):
    """Assemble the (featurised) training set for one arm."""
    base = AUG.shift_crop(Xtr_pad, t, FEAT_LO, FEAT_HI)
    Xs, ys = [base], [ytr]

    if arm in ("generic", "slide_site+generic"):
        Xa, ya = AUG.generic_augment(Xtr_pad, ytr, rng, n_copies=N_AUG_COPIES)
        Xs.append(AUG.shift_crop(Xa, t, FEAT_LO, FEAT_HI)); ys.append(ya)

    key = {"slide_broad": "broad", "slide_allsites": "allsites",
           "slide_site": "site", "slide_site+generic": "site"}.get(arm)
    if key is not None:
        draw = samplers[key]
        for _ in range(N_AUG_COPIES):
            Xs.append(AUG.shift_crop(Xtr_pad, t, FEAT_LO, FEAT_HI, draw(len(Xtr_pad))))
            ys.append(ytr)

    return featurise(np.concatenate(Xs)), np.concatenate(ys)


def make_samplers(err_pool_site, err_pool_all, within_sd, rng):
    return {
        "site": AUG.sampler_empirical(err_pool_site, rng),
        "allsites": AUG.sampler_empirical(err_pool_all, rng),
        "broad": AUG.sampler_broad(rng, 0.0, 150.0),
        "session": AUG.sampler_session(err_pool_site, within_sd, rng),
    }


# ------------------------------------------------------------------ experiments
def within_session(tt):
    rows = []
    runs = [(s, e, task) for (s, e) in tt.groupby(["subject", "session"]).groups for task in TASKS]
    for sub, ses, task in sorted(set(runs)):
        g = tt[(tt.subject == sub) & (tt.session == ses)]
        d = EP.build(sub, ses, task, g)
        if d is None:
            continue
        t, core = d["t"], d["core"]
        Xc, Xr, y = d["X_cor"], d["X_raw"], d["y"]
        site = SITE_OF[ses]

        # leave-one-session-out error pools
        other = tt[tt.session != ses]
        pool_site = other[other.site == site]["err_ms"].values
        pool_all = other["err_ms"].values
        if len(pool_site) == 0:
            pool_site = pool_all
        within_sd = float(np.mean([gg["err_ms"].std(ddof=1)
                                   for _, gg in other.groupby(["subject", "session", "task"])]))

        classes = np.unique(y)
        n_per = {c: int((y == c).sum()) for c in classes}
        n_test = max(MIN_TEST_PER_CLASS, int(min(n_per.values()) * TEST_FRAC))
        n_pool = min(n_per.values()) - n_test
        if n_pool < 10:
            continue

        for rep in range(N_REPEATS):
            # zlib.crc32, not hash(): str hashing is salted per process, which would make
            # the split assignment differ between runs of the same script
            run_key = zlib.crc32(f"{sub}|{ses}|{task}".encode()) % 10000
            rng = np.random.default_rng(SEED + rep + run_key)
            tr_idx, te_idx = [], []
            for c in classes:
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                te_idx.append(idx[:n_test]); tr_idx.append(idx[n_test:])
            te = np.concatenate(te_idx)
            trpool = np.concatenate(tr_idx)

            tests = {
                "ceiling": (featurise(AUG.shift_crop(Xc[te], t, FEAT_LO, FEAT_HI)), y[te]),
                "raw": (featurise(AUG.shift_crop(Xr[te], t, FEAT_LO, FEAT_HI)), y[te]),
            }
            samplers = make_samplers(pool_site, pool_all, within_sd, rng)

            for n_req in N_GRID:
                n_use = n_pool if n_req == "full" else min(int(n_req), n_pool)
                if n_req != "full" and n_use < int(n_req):
                    continue
                sel = np.concatenate([rng.choice(trpool[y[trpool] == c], n_use, replace=False)
                                      for c in classes])
                for arm in ARMS:
                    a_rng = np.random.default_rng(SEED + rep * 97 + ARMS.index(arm))
                    Xtr, ytr = build_training(arm, Xc[sel], y[sel], t,
                                              make_samplers(pool_site, pool_all, within_sd, a_rng),
                                              a_rng)
                    sc = fit_score(Xtr, ytr, tests)
                    rows.append(dict(subject=sub, session=ses, site=site, task=task,
                                     n_req=str(n_req), n_train_per_class=n_use, rep=rep, arm=arm,
                                     auc_ceiling=sc["ceiling"], auc_raw=sc["raw"],
                                     mean_err_ms=float(d["err_ms"].mean())))
        print(f"  {sub} {ses} {task}: done (pool {n_pool}/class, test {n_test}/class)", flush=True)
    return pd.DataFrame(rows)


def cross_session(tt):
    """Train on one session's corrected data, deploy on another session."""
    rows = []
    sessions = sorted(tt.session.unique())
    for sub in sorted(tt.subject.unique()):
        for task in TASKS:
            built = {}
            for ses in sessions:
                g = tt[(tt.subject == sub) & (tt.session == ses)]
                d = EP.build(sub, ses, task, g)
                if d is not None:
                    built[ses] = d
            for tr_ses, dtr in built.items():
                for te_ses, dte in built.items():
                    if tr_ses == te_ses:
                        continue
                    t = dtr["t"]
                    site_tr, site_te = SITE_OF[tr_ses], SITE_OF[te_ses]
                    other = tt[tt.session != te_ses]
                    pool_site = other[other.site == site_tr]["err_ms"].values
                    pool_all = other["err_ms"].values
                    within_sd = float(np.mean([gg["err_ms"].std(ddof=1) for _, gg
                                               in other.groupby(["subject", "session", "task"])]))
                    tests = {
                        "ceiling": (featurise(AUG.shift_crop(dte["X_cor"], t, FEAT_LO, FEAT_HI)),
                                    dte["y"]),
                        "raw": (featurise(AUG.shift_crop(dte["X_raw"], t, FEAT_LO, FEAT_HI)),
                                dte["y"]),
                    }
                    for rep in range(5):
                        for arm in ARMS:
                            rng = np.random.default_rng(SEED + rep * 31 + ARMS.index(arm))
                            Xtr, ytr = build_training(arm, dtr["X_cor"], dtr["y"], t,
                                                      make_samplers(pool_site, pool_all,
                                                                    within_sd, rng), rng)
                            sc = fit_score(Xtr, ytr, tests)
                            rows.append(dict(subject=sub, task=task, train_ses=tr_ses,
                                             test_ses=te_ses, train_site=site_tr,
                                             test_site=site_te,
                                             same_site=site_tr == site_te, rep=rep, arm=arm,
                                             auc_ceiling=sc["ceiling"], auc_raw=sc["raw"]))
            print(f"  cross-session {sub} {task}: done", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ reporting
def summarise(w):
    # group on the REQUESTED n, not the realised one: "full" is a different trial count
    # in every run, and grouping on it fragments the sweep into non-comparable rows
    g = (w.groupby(["task", "n_req", "arm"])
           .agg(auc_raw=("auc_raw", "mean"), auc_ceiling=("auc_ceiling", "mean"),
                sd=("auc_raw", "std"), n=("auc_raw", "size"),
                n_train_median=("n_train_per_class", "median")).reset_index())
    base = (w[w.arm == "none"].groupby(["task", "n_req"])
              .agg(base_raw=("auc_raw", "mean"), ceil=("auc_ceiling", "mean")).reset_index())
    g = g.merge(base, on=["task", "n_req"])
    # fraction of the timing-induced gap that the arm recovers
    g["gap_closed_pct"] = 100 * (g.auc_raw - g.base_raw) / (g.ceil - g.base_raw)
    return g.round(4)


def fig_sweep(w):
    for task in TASKS:
        s = w[w.task == task]
        if s.empty:
            continue
        ns = [n for n in ["10", "20", "40", "full"] if (s.n_req == n).any()]
        xs = np.arange(len(ns))
        labels = [f"{n}\n(n={int(s[s.n_req == n].n_train_per_class.median())})"
                  if n == "full" else n for n in ns]
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
        ax = axes[0]
        # the ceiling is the UNAUGMENTED model on the corrected test. Averaging
        # auc_ceiling over all arms would mix in the augmented models, which are
        # deliberately worse on correctly-timed data, and understate the reference.
        ceil = [s[(s.n_req == n) & (s.arm == "none")].auc_ceiling.mean() for n in ns]
        ax.plot(xs, ceil, "k--", lw=2, marker="s",
                label="ceiling (no augmentation, corrected test)")
        for arm in ARMS:
            v = [s[(s.arm == arm) & (s.n_req == n)].auc_raw.mean() for n in ns]
            ax.plot(xs, v, marker="o", lw=1.8, label=arm)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("training trials per class"); ax.set_ylabel("AUC on raw-marker test")
        ax.set_title(f"{task.upper()} — decoding with uncorrected markers", fontsize=11)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.legend(fontsize=8); ax.grid(alpha=.25)

        ax = axes[1]
        for site in ["LDS", "MAN", "SHE"]:
            ss = s[s.site == site]
            if ss.empty:
                continue
            v = [ss[ss.arm == a].auc_raw.mean() for a in ARMS]
            ax.plot(range(len(ARMS)), v, marker="o", color=SITE_COL[site], label=f"{site} raw")
            ax.axhline(ss[ss.arm == "none"].auc_ceiling.mean(),
                       color=SITE_COL[site], ls="--", lw=1, alpha=.6)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(ARMS, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("AUC"); ax.set_title(f"{task.upper()} — by site (dashed = that site's ceiling)",
                                           fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"c_sweep_{task}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def fig_cross(c):
    if c.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, task in zip(axes, TASKS):
        s = c[c.task == task]
        if s.empty:
            continue
        x = np.arange(len(ARMS))
        for k, (lab, sel) in enumerate([("same site", s[s.same_site]),
                                        ("cross site", s[~s.same_site])]):
            v = [sel[sel.arm == a].auc_raw.mean() for a in ARMS]
            ax.bar(x + (k - .5) * .38, v, .35, label=f"{lab} (raw test)")
            ax.axhline(sel.auc_ceiling.mean(), ls="--", lw=1.2,
                       color=f"C{k}", alpha=.8)
        ax.set_xticks(x); ax.set_xticklabels(ARMS, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("AUC"); ax.set_title(f"{task.upper()} — cross-session transfer", fontsize=11)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    fig.suptitle("Train on one session, deploy on another (dashed = corrected-test ceiling)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "c_cross_session.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    tt = build_trial_table()
    tt = tt[~tt.correction_failed]

    print("within-session sweep:")
    w = within_session(tt)
    w.to_csv(os.path.join(OUT, "classify_within.csv"), index=False)

    print("cross-session transfer:")
    c = cross_session(tt)
    c.to_csv(os.path.join(OUT, "classify_cross.csv"), index=False)

    s = summarise(w)
    s.to_csv(os.path.join(OUT, "classify_summary.csv"), index=False)
    fig_sweep(w); fig_cross(c)

    pd.set_option("display.width", 220)
    print("\n=== within-session: AUC on raw-marker test ===")
    print(s[["task", "n_train_per_class", "arm", "auc_raw", "auc_ceiling",
             "gap_closed_pct"]].to_string(index=False))
    print("\n=== by site (full training set) ===")
    full = w[w.n_req == "full"]
    print(full.pivot_table(index=["task", "site"], columns="arm",
                           values="auc_raw").round(3).to_string())
    print("\n=== cross-session transfer ===")
    print(c.groupby(["task", "same_site", "arm"])[["auc_raw", "auc_ceiling"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
