"""Figures for the P300 baseline: grand-average ERPs, topographies, classification.

Run after `p300_baseline_v3.py preprocess` and `classify`. Rebuilds the session ERPs from
`_baseline_cache/` with exactly the epoching, rejection and session inclusion the classifier
used (p300_baseline_v3.epochs, >= 20 targets), and plots `baseline_scores.csv`.

    python analysis/baseline_figures.py

Writes figures/erp_*.png, figures/topo_*.png, figures/gedai_vs_nogedai_pz.png,
figures/classification.png and classification_summary.csv.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import mne                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p300_baseline_v3 as b             # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
T_MS = b.TS / b.FS * 1000
SITES = ["SHE", "LDS", "MAN"]
SITE_NAME = {"SHE": "Sheffield (Bitbrain, 256 Hz)", "LDS": "Leeds (BioSemi, 2048 Hz)",
             "MAN": "Manchester (ANT, 1000 Hz)"}
COL = {"SHE": "#C8102E", "LDS": "#1F6FB2", "MAN": "#2E8B57"}
PZ, FCZ = b.CH14.index("Pz"), b.CH14.index("FCz")

mne.set_log_level("ERROR")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------------------------------------
# session-level ERPs, same trials as the classifier
# ---------------------------------------------------------------------------------------------
def session_erps(variant: str) -> pd.DataFrame:
    """One row per session with >= MIN_TARGETS surviving: mean target / non-target epochs
    (14 x 73) for both marker sets, on the common clean-trial mask."""
    rows = []
    for r in b._ok_sessions().itertuples():
        try:
            ep, lab, n_all = b.epochs(r.subject, r.session, variant)
        except Exception as ex:                       # noqa: BLE001
            print(f"skip {r.subject} {r.session} {variant}: {ex}")
            continue
        nt = int((lab == 2).sum())
        if nt < b.MIN_TARGETS:
            continue
        row = dict(subject=r.subject, session=r.session, site=r.site, n_targets=nt,
                   n_nontargets=int((lab == 1).sum()), n_kept=len(lab), n_all=n_all)
        for mk, X in ep.items():
            row[f"{mk}_t"] = X[lab == 2].mean(0)
            row[f"{mk}_n"] = X[lab == 1].mean(0)
        rows.append(row)
    return pd.DataFrame(rows)


def grand(df: pd.DataFrame, site: str, key: str) -> np.ndarray:
    """Grand average over sessions (each session weighted equally): (14, 73)."""
    return np.mean(np.stack(df[df.site == site][key].values), axis=0)


def site_n(df, site):
    d = df[df.site == site]
    return len(d), d.subject.nunique()


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
def fig_erp_by_site(df: pd.DataFrame, variant: str, chan: int, name: str):
    """Row 1: target vs non-target at one channel per site (corrected solid, uncorrected dashed).
    Row 2: target - non-target difference wave, all sites overlaid, corrected markers."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), gridspec_kw=dict(height_ratios=[1.2, 1]))
    ymax = 0
    for j, s in enumerate(SITES):
        ax = axes[0, j]
        n_ses, n_sub = site_n(df, s)
        for mk, ls in (("corrected", "-"), ("uncorrected", "--")):
            t, n = grand(df, s, f"{mk}_t")[chan], grand(df, s, f"{mk}_n")[chan]
            ax.plot(T_MS, t, ls, color=COL[s], lw=2 if mk == "corrected" else 1.3,
                    label=f"target, {mk}")
            ax.plot(T_MS, n, ls, color="0.45", lw=2 if mk == "corrected" else 1.3,
                    label=f"non-target, {mk}")
            ymax = max(ymax, np.abs(t).max(), np.abs(n).max())
        ax.axvspan(150, 500, color="0.92", zorder=0)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{SITE_NAME[s]}\n{n_ses} sessions, {n_sub} subjects", fontsize=10)
        ax.set_xlabel("time from screen onset (ms)")
        if j == 0:
            ax.set_ylabel(f"{b.CH14[chan]} amplitude (µV), linked mastoids")
            ax.legend(fontsize=8, loc="upper left", frameon=False)
    for ax in axes[0]:
        ax.set_ylim(-ymax * 1.1, ymax * 1.1)
        ax.set_xlim(-100, 800)

    # difference waves
    ax = axes[1, 0]
    for s in SITES:
        d = grand(df, s, "corrected_t")[chan] - grand(df, s, "corrected_n")[chan]
        ax.plot(T_MS, d, color=COL[s], lw=2, label=f"{s} corrected")
        d2 = grand(df, s, "uncorrected_t")[chan] - grand(df, s, "uncorrected_n")[chan]
        ax.plot(T_MS, d2, "--", color=COL[s], lw=1.2, label=f"{s} uncorrected")
    ax.axvspan(150, 500, color="0.92", zorder=0)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-100, 800)
    ax.set_title(f"target − non-target at {b.CH14[chan]}", fontsize=10)
    ax.set_xlabel("time from screen onset (ms)"); ax.set_ylabel("µV")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")

    # per-session P3b amplitude and latency (corrected), 0.3-0.6 s mean of difference wave
    ax = axes[1, 1]
    amp = {s: [] for s in SITES}
    lat = {s: [] for s in SITES}
    for r in df.itertuples():
        d = (r.corrected_t - r.corrected_n)[chan]
        amp[r.site].append(d[b.P3].mean())
        lat[r.site].append(T_MS[b.P3][np.argmax(d[b.P3])])
    rng = np.random.default_rng(0)
    for i, s in enumerate(SITES):
        a = np.array(amp[s])
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a, color=COL[s], alpha=0.7, s=22)
        ax.hlines(np.median(a), i - 0.3, i + 0.3, color="k", lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(SITES)
    ax.set_title("per-session P3b (0.3–0.6 s mean), corrected", fontsize=10)
    ax.set_ylabel("µV")
    ax = axes[1, 2]
    for i, s in enumerate(SITES):
        a = np.array(lat[s])
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a, color=COL[s], alpha=0.7, s=22)
        ax.hlines(np.median(a), i - 0.3, i + 0.3, color="k", lw=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(SITES)
    ax.set_title("per-session peak latency (0.3–0.6 s), corrected", fontsize=10)
    ax.set_ylabel("ms")
    fig.suptitle(f"Grand-average P3b by site, {variant} variant, {b.CH14[chan]}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)
    return pd.DataFrame([dict(site=s, n_sessions=len(amp[s]), p3_median_uV=np.median(amp[s]),
                              p3_mean_uV=np.mean(amp[s]), lat_median_ms=np.median(lat[s]))
                         for s in SITES])


def fig_gedai_vs_nogedai(dg: pd.DataFrame, dn: pd.DataFrame, chan: int, name: str):
    """Pz difference wave with vs without GEDAI, on the sessions both variants classify."""
    key = ["subject", "session"]
    common = dg.merge(dn[key], on=key)[key]
    g = dg.merge(common, on=key); n = dn.merge(common, on=key)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for j, s in enumerate(SITES):
        ax = axes[j]
        n_ses, n_sub = site_n(g, s)
        if n_ses == 0:
            ax.set_title(f"{s}: no paired sessions"); continue
        for df, lab, ls in ((n, "no GEDAI", "-"), (g, "GEDAI", "--")):
            d = grand(df, s, "corrected_t")[chan] - grand(df, s, "corrected_n")[chan]
            ax.plot(T_MS, d, ls, color=COL[s], lw=2, label=f"{lab} ({d[b.P3].mean():.2f} µV)")
        ax.axvspan(300, 600, color="0.92", zorder=0)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-100, 800)
        ax.set_title(f"{SITE_NAME[s]}\n{n_ses} paired sessions, {n_sub} subjects", fontsize=10)
        ax.set_xlabel("time (ms)")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    axes[0].set_ylabel(f"target − non-target at {b.CH14[chan]} (µV)")
    fig.suptitle("Check 2: what GEDAI does to the P3b (same sessions, same trial rule, corrected markers)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


def fig_topo(df: pd.DataFrame, variant: str, name: str):
    """Target − non-target topographies per site in four 100 ms windows + the 0.3-0.6 s mean."""
    info = mne.create_info(b.CH14, b.FS, "eeg")
    info.set_montage("standard_1005")
    wins = [(200, 300), (300, 400), (400, 500), (500, 600)]
    cols = [f"{a}–{c} ms" for a, c in wins] + ["300–600 ms"]
    maps = {}
    for s in SITES:
        d = grand(df, s, "corrected_t") - grand(df, s, "corrected_n")
        for (a, c), lab in zip(wins, cols):
            m = (T_MS >= a) & (T_MS < c)
            maps[(s, lab)] = d[:, m].mean(1)
        maps[(s, cols[-1])] = d[:, b.P3].mean(1)
    vmax = max(np.abs(v).max() for v in maps.values())
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.5))
    for i, s in enumerate(SITES):
        for j, lab in enumerate(cols):
            ax = axes[i, j]
            im, _ = mne.viz.plot_topomap(maps[(s, lab)], info, axes=ax, show=False,
                                         vlim=(-vmax, vmax), cmap="RdBu_r", contours=4,
                                         sensors=True, names=b.CH14 if j == 4 else None)
            if i == 0:
                ax.set_title(lab, fontsize=10)
            if j == 0:
                n_ses, n_sub = site_n(df, s)
                ax.text(-0.25, 0.5, f"{s}\n{n_ses} ses", transform=ax.transAxes,
                        fontsize=11, color=COL[s], va="center", ha="right", weight="bold")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("target − non-target (µV)")
    fig.suptitle(f"P3b topography by site, {variant} variant, corrected markers, linked mastoids "
                 f"(14 channels)", fontsize=12)
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame([dict(site=s, window=lab, **{c: v for c, v in zip(b.CH14, maps[(s, lab)])})
                         for s in SITES for lab in cols])


def fig_classification(name: str):
    sc = pd.read_csv(os.path.join(HERE, "baseline_scores.csv"))
    ok = sc[sc.status == "ok"]
    s = ok.groupby(["config", "markers", "subject", "session", "site"])[["bal_acc", "auc"]].mean().reset_index()
    chance = s[s.config == "shuffled"][["bal_acc", "auc"]].median()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rng = np.random.default_rng(1)
    rows = []
    for i, (cfg, title) in enumerate((("primary", "GEDAI (primary)"), ("nogedai", "no GEDAI"))):
        for j, metric in enumerate(("bal_acc", "auc")):
            ax = axes[i, j]
            d = s[s.config == cfg].pivot_table(index=["subject", "session", "site"],
                                               columns="markers", values=metric).reset_index()
            for k, site in enumerate(SITES):
                g = d[d.site == site]
                x = k + rng.uniform(-0.12, 0.12, len(g))
                for xx, c, u in zip(x, g.corrected, g.uncorrected):
                    ax.plot([xx - 0.17, xx + 0.17], [c, u], color="0.75", lw=0.8, zorder=1)
                ax.scatter(x - 0.17, g.corrected, color=COL[site], s=28, zorder=3, label="corrected" if k == 0 else None)
                ax.scatter(x + 0.17, g.uncorrected, facecolor="white", edgecolor=COL[site], s=28,
                           zorder=3, label="uncorrected" if k == 0 else None)
                ax.hlines(g.corrected.median(), k - 0.32, k - 0.02, color="k", lw=2, zorder=4)
                ax.hlines(g.uncorrected.median(), k + 0.02, k + 0.32, color="k", lw=2, ls=":", zorder=4)
                rows.append(dict(config=cfg, metric=metric, site=site, n=len(g),
                                 corrected_median=g.corrected.median(), uncorrected_median=g.uncorrected.median(),
                                 diff_median=(g.corrected - g.uncorrected).median()))
            ax.axhline(chance[metric], color="r", lw=1, ls="--", label="shuffled labels")
            ax.set_xticks(range(3))
            ax.set_xticklabels([f"{st}\n(n={int((d.site == st).sum())})" for st in SITES])
            ax.set_ylim(0.4, 1.0)
            ax.set_ylabel({"bal_acc": "balanced accuracy", "auc": "ROC-AUC"}[metric])
            ax.set_title(f"{title}: {ax.get_ylabel()}", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Within-session classification, xDAWN(4) → shrinkage LDA, 5-fold × 10 repeats; "
                 "one point = one session (mean over repeats)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows), s


def main():
    dg = session_erps("gedai")
    dn = session_erps("nogedai")
    print(f"gedai sessions: {len(dg)} ({dg.groupby('site').size().to_dict()}), "
          f"nogedai: {len(dn)} ({dn.groupby('site').size().to_dict()})")
    t1 = fig_erp_by_site(dg, "GEDAI", PZ, "erp_pz_gedai.png")
    t2 = fig_erp_by_site(dn, "no-GEDAI", PZ, "erp_pz_nogedai.png")
    fig_erp_by_site(dg, "GEDAI", FCZ, "erp_fcz_gedai.png")
    fig_gedai_vs_nogedai(dg, dn, PZ, "gedai_vs_nogedai_pz.png")
    topo_g = fig_topo(dg, "GEDAI", "topo_gedai.png")
    topo_n = fig_topo(dn, "no-GEDAI", "topo_nogedai.png")
    cls, sess = fig_classification("classification.png")

    print("\n=== P3b at Pz, GEDAI variant (classified sessions) ===")
    print(t1.round(2).to_string(index=False))
    print("\n=== P3b at Pz, no-GEDAI variant ===")
    print(t2.round(2).to_string(index=False))
    print("\n=== topography, 300-600 ms, GEDAI ===")
    print(topo_g[topo_g.window == "300–600 ms"].set_index("site")[b.CH14].round(2).to_string())
    print("\n=== topography, 300-600 ms, no-GEDAI ===")
    print(topo_n[topo_n.window == "300–600 ms"].set_index("site")[b.CH14].round(2).to_string())
    print("\n=== classification medians by site ===")
    print(cls.round(3).to_string(index=False))
    cls.to_csv(os.path.join(HERE, "classification_summary.csv"), index=False)

    # paired corrected-vs-uncorrected test, primary config
    from scipy.stats import wilcoxon
    for cfg in ("primary", "nogedai"):
        for metric in ("bal_acc", "auc"):
            d = sess[sess.config == cfg].pivot_table(index=["subject", "session"], columns="markers", values=metric)
            diff = d.corrected - d.uncorrected
            print(f"{cfg:8s} {metric:8s} corrected−uncorrected: median {diff.median():+.3f}, "
                  f"mean {diff.mean():+.3f}, better in {(diff > 0).sum()}/{len(diff)}, "
                  f"wilcoxon p={wilcoxon(diff).pvalue:.3f}")


if __name__ == "__main__":
    main()
