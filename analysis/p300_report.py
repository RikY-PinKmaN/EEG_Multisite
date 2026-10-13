#!/usr/bin/env python3
"""
p300_report.py — summarise p300_baseline.csv / p300_gan.csv into P300_GAN_RESULTS tables + figures.

Statistics: a test score is first averaged within (training unit, regime), so every unit
contributes one number per regime; paired contrasts are Wilcoxon signed-rank over units.
Units are (subject, site), so a subject contributes up to 3 units — subject-level means
are also reported as the conservative check.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
REGIMES = ["within", "cross_session", "cross_site"]


def unit_means(df, keys):
    return df.groupby(keys + ["subject", "train_session", "regime"], as_index=False)[["bacc", "auc", "acc"]].mean()


def paired(a: pd.Series, b: pd.Series):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(j) < 3:
        return dict(n=len(j), mean_a=np.nan, mean_b=np.nan, diff=np.nan, p=np.nan)
    d = j.b - j.a
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(n=len(j), mean_a=j.a.mean(), mean_b=j.b.mean(), diff=d.mean(),
                sem=d.std(ddof=1) / np.sqrt(len(d)), p=p)


def decide(base: pd.DataFrame):
    """The pre-registered preprocessing rule (GAN_CLASSIFY_PLAN.md, overnight section)."""
    u = unit_means(base[(base.marker == "cor") & (base.regime == "within")], ["variant", "thresh"])
    u["cfg"] = u.variant + "@" + u.thresh.astype(int).astype(str)
    w = u.pivot_table(index=["subject", "train_session"], columns="cfg", values="bacc")
    n_units = u.groupby("cfg").size()
    r = paired(w["mine@100"], w["gedai@100"])
    better = r["diff"] > 0 and r["p"] < 0.05
    worse = r["diff"] < 0 and r["p"] < 0.05
    more_units = n_units.get("gedai@100", 0) >= 1.2 * n_units.get("mine@100", 0)
    choice = "gedai" if (better or (not worse and more_units)) else "mine"
    txt = (f"within-session balanced accuracy, corrected markers, {r['n']} paired units: "
           f"mine {r['mean_a']:.3f} vs GEDAI {r['mean_b']:.3f} (GEDAI−mine {r['diff']:+.3f} "
           f"± {r.get('sem', np.nan):.3f} SEM, Wilcoxon p = {r['p']:.3g}); usable training units: "
           f"mine {n_units.get('mine@100', 0)}, GEDAI {n_units.get('gedai@100', 0)} "
           f"(GEDAI@50: {n_units.get('gedai@50', 0)}). Rule → **{choice}**.")
    return choice, txt, w


def table_markers(df, keys, arm_col=None):
    """Mean bacc per regime x marker (x arm), with raw-cor paired contrast."""
    u = unit_means(df, keys + ([arm_col] if arm_col else []) + ["marker"])
    rows = []
    groups = [None] if arm_col is None else sorted(u[arm_col].unique())
    for g in groups:
        ug = u if g is None else u[u[arm_col] == g]
        for reg in REGIMES:
            ur = ug[ug.regime == reg]
            w = ur.pivot_table(index=["subject", "train_session"], columns="marker", values="bacc")
            wa = ur.pivot_table(index=["subject", "train_session"], columns="marker", values="auc")
            if not {"cor", "raw"} <= set(w.columns):
                continue
            r = paired(w["cor"], w["raw"]); ra = paired(wa["cor"], wa["raw"])
            subj = ur.groupby(["subject", "marker"]).bacc.mean().unstack()
            rs = paired(subj["cor"], subj["raw"])
            rows.append(dict(arm=g or "baseline", regime=reg, units=r["n"],
                             bacc_cor=r["mean_a"], bacc_raw=r["mean_b"], raw_minus_cor=r["diff"],
                             sem=r.get("sem", np.nan), p_units=r["p"],
                             auc_cor=ra["mean_a"], auc_raw=ra["mean_b"],
                             subjects=rs["n"], p_subjects=rs["p"]))
    return pd.DataFrame(rows)


def table_arms(gan):
    """GAN arm minus baseline, per marker x regime."""
    u = unit_means(gan, ["marker", "arm"])
    rows = []
    for mk in ["cor", "raw"]:
        for reg in REGIMES:
            ur = u[(u.marker == mk) & (u.regime == reg)]
            w = ur.pivot_table(index=["subject", "train_session"], columns="arm", values="bacc")
            for arm in ["gan", "gan_synthonly"]:
                if arm in w and "baseline" in w:
                    r = paired(w["baseline"], w[arm])
                    rows.append(dict(marker=mk, regime=reg, arm=arm, units=r["n"],
                                     bacc_baseline=r["mean_a"], bacc_arm=r["mean_b"],
                                     arm_minus_baseline=r["diff"], sem=r.get("sem", np.nan), p=r["p"]))
    return pd.DataFrame(rows)


def fig_offset(df, title, fname, arm_col=None):
    """raw - cor balanced accuracy per test session vs |offset(train) - offset(test)|."""
    os.makedirs(FIG, exist_ok=True)
    keys = ["subject", "train_session", "test_session", "regime"] + ([arm_col] if arm_col else [])
    w = df.pivot_table(index=keys, columns="marker", values=["bacc", "d_offset_ms"]).dropna()
    w = w.reset_index()
    arms = [None] if arm_col is None else sorted(w[arm_col].unique())
    fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4), squeeze=False)
    col = {"within": "#555555", "cross_session": "#3b7dd8", "cross_site": "#d1495b"}
    for ax, a in zip(axes[0], arms):
        wa = w if a is None else w[w[arm_col] == a]
        for reg in REGIMES:
            s = wa[wa.regime == reg]
            ax.scatter(s[("d_offset_ms", "raw")], s[("bacc", "raw")] - s[("bacc", "cor")], s=14,
                       alpha=0.6, color=col[reg], label=reg.replace("_", " "))
        x = wa[("d_offset_ms", "raw")].values; yv = (wa[("bacc", "raw")] - wa[("bacc", "cor")]).values
        if len(x) > 3:
            b = np.polyfit(x, yv, 1); xx = np.linspace(0, x.max(), 50)
            r = np.corrcoef(x, yv)[0, 1]
            ax.plot(xx, np.polyval(b, xx), "k-", lw=1)
            ax.set_title(f"{a or title}: slope {b[0] * 1000:.2f} pp per 10 ms, r = {r:.2f}", fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("|marker offset, test − train session| (ms)")
        ax.set_ylabel("balanced acc: uncorrected − corrected")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    base = pd.read_csv(os.path.join(HERE, "p300_baseline.csv"))
    choice, txt, _ = decide(base)
    print("DECISION:", txt)
    for (v, t), g in base.groupby(["variant", "thresh"]):
        print(f"\n== baseline {v}@{int(t)} ==")
        print(table_markers(g, ["variant"]).round(3).to_string(index=False))
    b = base[(base.variant == choice) & (base.thresh == 100)]
    fig_offset(b, f"baseline ({choice})", "p300_offset_baseline.png")
    gp = os.path.join(HERE, "p300_gan.csv")
    if os.path.exists(gp):
        gan = pd.read_csv(gp)
        for (v, t), g in gan.groupby(["variant", "thresh"]):
            print(f"\n== GAN run {v}@{int(t)}, seeds {sorted(g.seed.unique())} ==")
            print(table_markers(g, ["variant"], arm_col="arm").round(3).to_string(index=False))
            print(table_arms(g).round(3).to_string(index=False))
            fig_offset(g, "", f"p300_offset_gan_{v}.png", arm_col="arm")
