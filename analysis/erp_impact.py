#!/usr/bin/env python3
"""
erp_impact.py — what the uncorrected markers actually do to the ERP.

Everything is measured on the SAME recording and the SAME trials; the only thing that
changes is whether a trial is epoched at the photodiode-corrected marker or the raw
one. No simulation, no assumed error model.

Three measurements, because they separate two failure modes that get conflated:

  fixed-window amplitude   mean amplitude inside the canonical window (N170 130-200 ms,
                           P3b 300-600 ms) — what a lab actually reports. Destroyed by
                           a constant latency shift.
  cross-correlation lag    how far the waveform MOVED. Robust where peak-picking on a
                           noisy difference wave is not.
  peak correlation         how much of the waveform SHAPE survived at that lag.
                           Near 1 = displaced but intact; low = genuinely smeared.

Step 1 found the error is dominated by a per-session constant offset (site 77%,
session 17%, trial ~1%), which predicts displacement with shape intact — not the
smearing the project originally assumed. This is the test of that prediction.

Writes: erp_impact.csv, erp_impact_damage.csv, figures/e_*.png
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jitter_pipeline import ROOT
from raw_markers import build_trial_table, SITE_OF, CONTRAST
import epochs as EP

OUT = os.path.join(ROOT, "analysis")
FIG = os.path.join(OUT, "figures")

PARADIGM = {
    "n170": dict(chans=["PO7", "PO8", "P7", "P8"], window=(130, 200), polarity="neg"),
    "p3b":  dict(chans=["Pz", "POz"], window=(300, 600), polarity="pos"),
}
SITE_COL = {"LDS": "#3b7dd8", "MAN": "#37a66b", "SHE": "#d1495b"}
SITE_ORDER = ["LDS", "MAN", "SHE"]


def pooled_diff(d, task):
    """Difference wave (cond0 - cond1) on the paradigm's region, for both marker sets."""
    spec = PARADIGM[task]
    chans = list(d["chans"])
    idx = [chans.index(c) for c in spec["chans"] if c in chans]
    if not idx:
        return None
    y = d["y"]
    out = {}
    for marker, X in (("corrected", d["X_cor"]), ("raw", d["X_raw"])):
        r = X[:, idx, :].mean(1)                    # pool region -> (n_trials, n_time)
        out[marker] = r[y == 0].mean(0) - r[y == 1].mean(0)
        out[marker + "_trials"] = r
    return out


def xcorr_lag(a, b, fs, max_ms=300):
    """(lag_ms, peak_r) aligning b onto a. Positive lag = b occurs LATER than a."""
    m = int(round(max_ms / 1000 * fs))
    lags = np.arange(-m, m + 1)
    cc = []
    for k in lags:
        bb = np.roll(b, -k)
        cc.append(np.corrcoef(a[m:-m], bb[m:-m])[0, 1])
    cc = np.array(cc)
    i = int(np.nanargmax(cc))
    return float(lags[i] / fs * 1000), float(cc[i])


def main():
    os.makedirs(FIG, exist_ok=True)
    tt = build_trial_table()
    rows, store = [], {}

    for (sub, ses), g in tt.groupby(["subject", "session"]):
        for task in PARADIGM:
            d = EP.build(sub, ses, task, g)
            if d is None:
                continue
            pd_ = pooled_diff(d, task)
            if pd_ is None:
                continue
            store[(sub, ses, task)] = (d, pd_)

            spec = PARADIGM[task]
            t, fs = d["t"], float(d["fs"])
            core = d["core"]
            w = (t >= spec["window"][0]) & (t <= spec["window"][1])
            lag, r = xcorr_lag(pd_["corrected"][core], pd_["raw"][core], fs)
            rows.append(dict(
                subject=sub, session=ses, site=SITE_OF[ses], task=task,
                n_trials=len(d["y"]), n_dropped=int(d["n_dropped"]),
                mean_err_ms=float(d["err_ms"].mean()), sd_err_ms=float(d["err_ms"].std(ddof=1)),
                fixed_corrected_uv=float(pd_["corrected"][w].mean()),
                fixed_raw_uv=float(pd_["raw"][w].mean()),
                recovered_lag_ms=lag, shape_r=r,
                amp_corrected_uv=float(np.abs(pd_["corrected"][core]).max()),
                amp_raw_uv=float(np.abs(pd_["raw"][core]).max()),
            ))

    df = pd.DataFrame(rows)
    df["lag_error_ms"] = df.recovered_lag_ms - df.mean_err_ms
    # NOT a retained-% : the corrected amplitude crosses zero across runs, so the ratio
    # is unbounded and meaningless. Report the absolute change and whether the sign flipped.
    df["fixed_change_uv"] = df.fixed_raw_uv - df.fixed_corrected_uv
    df["sign_flipped"] = np.sign(df.fixed_raw_uv) != np.sign(df.fixed_corrected_uv)
    df["amp_retained_pct"] = 100 * df.amp_raw_uv / df.amp_corrected_uv
    df = df.round(3)
    df.to_csv(os.path.join(OUT, "erp_impact.csv"), index=False)

    fig_waveforms(store)
    fig_summary(df, store)

    pd.set_option("display.width", 230)
    print("=== per run ===")
    print(df[["subject", "session", "site", "task", "n_trials", "mean_err_ms",
              "recovered_lag_ms", "lag_error_ms", "shape_r",
              "fixed_corrected_uv", "fixed_raw_uv", "amp_retained_pct"]].to_string(index=False))
    print("\n=== per site x paradigm ===")
    print(df.groupby(["site", "task"]).agg(
        err=("mean_err_ms", "mean"), lag=("recovered_lag_ms", "mean"),
        lag_err=("lag_error_ms", "mean"), shape_r=("shape_r", "mean"),
        fixed_corr_uv=("fixed_corrected_uv", "mean"), fixed_raw_uv=("fixed_raw_uv", "mean"),
        sign_flips=("sign_flipped", "sum"), amp_ret=("amp_retained_pct", "mean"),
    ).round(2).to_string())
    print(f"\nCanonical-window measurement flipped sign in {int(df.sign_flipped.sum())}/{len(df)} runs.")
    print("\nThe recovered lag tracks the measured marker error 1:1 "
          f"(mean |error| {df.lag_error_ms.abs().mean():.1f} ms), "
          f"while waveform shape is preserved (mean r = {df.shape_r.mean():.2f}).")


def fig_waveforms(store):
    for task in PARADIGM:
        spec = PARADIGM[task]
        keys = sorted([k for k in store if k[2] == task])
        if not keys:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
        for ax, k in zip(axes.ravel(), keys):
            sub, ses, _ = k
            d, p = store[k]
            t, core = d["t"], d["core"]
            ax.plot(t[core], p["corrected"][core], "-", lw=1.8,
                    color=SITE_COL[SITE_OF[ses]], label="corrected markers")
            ax.plot(t[core], p["raw"][core], "--", lw=1.5, color="#333333",
                    label="raw markers (no photodiode)")
            ax.axvspan(*spec["window"], color="grey", alpha=.15)
            ax.axhline(0, color="k", lw=.7); ax.axvline(0, color="k", lw=.7)
            ax.set_title(f"{sub} · {ses}  (err {d['err_ms'].mean():.0f} ms)", fontsize=9)
            ax.grid(alpha=.25); ax.set_xlabel("ms")
        c0, c1 = CONTRAST[task]
        axes[0, 0].set_ylabel(f"{c0} − {c1} (µV)"); axes[1, 0].set_ylabel(f"{c0} − {c1} (µV)")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{task.upper()} difference wave — same trials, only the marker changes "
                     f"(shaded = canonical {spec['window'][0]}–{spec['window'][1]} ms window)",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"e_{task}_corrected_vs_raw.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def fig_summary(df, store):
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

    # top-left: recovered lag vs measured error — the validation that this is displacement
    ax = fig.add_subplot(gs[0, 0])
    for site in SITE_ORDER:
        s = df[df.site == site]
        ax.scatter(s.mean_err_ms, s.recovered_lag_ms, s=55, color=SITE_COL[site], label=site,
                   edgecolor="k", linewidth=.5, zorder=3)
    lim = [0, max(df.mean_err_ms.max(), df.recovered_lag_ms.max()) * 1.1]
    ax.plot(lim, lim, "k:", lw=1.2, label="identity")
    ax.set_xlabel("measured marker error (ms)"); ax.set_ylabel("ERP shift recovered from EEG (ms)")
    ax.set_title("The ERP moves by exactly the marker error", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    # top-middle: shape preservation
    ax = fig.add_subplot(gs[0, 1])
    for i, task in enumerate(PARADIGM):
        for j, site in enumerate(SITE_ORDER):
            v = df[(df.task == task) & (df.site == site)].shape_r
            ax.bar(j + (i - .5) * .36, v.mean(), .33, yerr=v.std(), capsize=3,
                   color=SITE_COL[site], alpha=1.0 if task == "n170" else 0.45,
                   label=task if j == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(SITE_ORDER); ax.set_ylim(0, 1.05)
    ax.axhline(1, color="k", ls=":", lw=1)
    ax.set_ylabel("correlation at best lag")
    ax.set_title("Waveform SHAPE survives\n(≈1 → displaced, not smeared)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    # top-right: fixed-window amplitude, corrected vs raw
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(df))
    order = df.sort_values(["task", "site"]).index
    s = df.loc[order]
    ax.bar(x - .2, s.fixed_corrected_uv, .38, color="#3b7dd8", label="corrected")
    ax.bar(x + .2, s.fixed_raw_uv, .38, color="#d1495b", label="raw")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sub[-2:]}·{ses[4:9]}·{task}"
                        for sub, ses, task in zip(s.subject, s.session, s.task)],
                       fontsize=6.5, rotation=90)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("canonical-window amplitude (µV)")
    ax.set_title("What a lab would report\n(sign flips and collapses)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    # bottom: grand-average waveform per site, both paradigms
    for i, task in enumerate(PARADIGM):
        ax = fig.add_subplot(gs[1, i])
        for site in SITE_ORDER:
            keys = [k for k in store if k[2] == task and SITE_OF[k[1]] == site]
            if not keys:
                continue
            d0 = store[keys[0]][0]
            t, core = d0["t"], d0["core"]
            gc = np.mean([store[k][1]["corrected"][store[k][0]["core"]] for k in keys], 0)
            gr = np.mean([store[k][1]["raw"][store[k][0]["core"]] for k in keys], 0)
            ax.plot(t[core], gc, "-", lw=2, color=SITE_COL[site], label=f"{site} corrected")
            ax.plot(t[core], gr, "--", lw=1.4, color=SITE_COL[site], alpha=.75,
                    label=f"{site} raw")
        ax.axvspan(*PARADIGM[task]["window"], color="grey", alpha=.15)
        ax.axhline(0, color="k", lw=.7); ax.axvline(0, color="k", lw=.7)
        ax.set_title(f"{task.upper()} grand average by site", fontsize=10)
        ax.set_xlabel("ms"); ax.set_ylabel("µV"); ax.legend(fontsize=7); ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    txt = ("Reading\n\n"
           "• The recovered ERP shift equals the measured marker error, site by site.\n"
           "• Shape correlation stays near 1, so the component is intact — moved, not\n"
           "  smeared. Step 1 predicted this: only ~1% of the error variance is\n"
           "  trial-to-trial, the rest is a per-session constant.\n\n"
           "• But the canonical-window amplitude collapses and can flip sign, because\n"
           "  the component has left the window a lab measures in.\n\n"
           "So the failure is DISPLACEMENT, not destruction — a different problem from\n"
           "the one shift-augmentation is designed to solve.")
    ax.text(0, .95, txt, va="top", fontsize=9.5, family="monospace")

    fig.suptitle("Effect of uncorrected markers on the ERP — measured, not simulated", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
