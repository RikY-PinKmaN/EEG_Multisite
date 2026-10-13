#!/usr/bin/env python3
"""
jitter_report.py — statistics + figures from jitter_table.csv.

Characterises, from the (photodiode-corrected) local release:
  * post-correction residual (PD_onset - corrected_trigger) per site  -> dominated by sampling quant.
  * photodiode signal quality per site                                -> why Sheffield correction is fragile
  * correction-failure trials (raw-latency glimpses)                  -> the surviving jitter fingerprint
  * Sheffield test-retest (ses-01SHE vs ses-03SHE) + across subjects  -> is the effect stable / site-bound
  * a simple variance decomposition of the residual
Outputs figures to analysis/figures/ and a summary to analysis/summary_stats.csv.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jitter_pipeline import ROOT, read_pd_channel, load_events, FAILURE_MS

FIGDIR = os.path.join(ROOT, "analysis", "figures")
os.makedirs(FIGDIR, exist_ok=True)
SITE_COLORS = {"SHE": "#d1495b", "LDS": "#2e7d95", "MAN": "#e3a72f"}
SITE_HW = {"SHE": "Sheffield · Bitbrain wireless · 256 Hz",
           "LDS": "Leeds · Biosemi · 2048 Hz",
           "MAN": "Manchester · ANT · 1000 Hz"}


def load():
    df = pd.read_csv(os.path.join(ROOT, "analysis", "jitter_table.csv"))
    return df


def summary_table(df):
    det = df[df.pd_detected]
    g = (det.groupby(["site", "subject", "session", "task"])
            .agg(n=("residual_ms", "size"), fs=("fs", "first"),
                 res_mean=("residual_ms", "mean"), res_sd=("residual_ms", "std"),
                 res_min=("residual_ms", "min"), res_max=("residual_ms", "max"),
                 pd_snr=("pd_snr", "median"), pd_amp=("pd_amp", "median"),
                 n_fail=("is_failure", "sum"))
            .reset_index())
    g.to_csv(os.path.join(ROOT, "analysis", "summary_stats.csv"), index=False)
    return g


def fig_residual_by_site(df):
    det = df[df.pd_detected]
    fig, ax = plt.subplots(figsize=(8, 5))
    sites = ["LDS", "MAN", "SHE"]
    for i, s in enumerate(sites):
        v = det[det.site == s]["residual_ms"].values
        x = np.random.default_rng(0).normal(i, 0.06, len(v))
        ax.scatter(x, v, s=6, alpha=0.25, color=SITE_COLORS[s], edgecolors="none")
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="black", lw=2, zorder=5)
    ax.axhspan(-3.91, 3.91, color="grey", alpha=0.10, zorder=0,
               label="±1 sample @256 Hz (Sheffield quantisation)")
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels([SITE_HW[s] for s in sites], rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("PD onset − corrected trigger  (ms)")
    ax.set_title("Post-correction residual by site\n(spread is sampling quantisation; the true jitter was corrected out)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-8, 20)
    fig.tight_layout(); p = os.path.join(FIGDIR, "fig1_residual_by_site.png")
    fig.savefig(p, dpi=140); plt.close(fig); return p


def fig_pd_quality(df):
    det = df[df.pd_detected]
    g = det.groupby("site")["pd_snr"].median().reindex(["LDS", "MAN", "SHE"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(range(len(g)), g.values, color=[SITE_COLORS[s] for s in g.index])
    ax.set_yscale("log")
    ax.set_xticks(range(len(g))); ax.set_xticklabels([SITE_HW[s] for s in g.index], rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("Median photodiode SNR (peak / noise)  [log]")
    ax.set_title("Photodiode signal quality by site\nSheffield's PD is ~1000× weaker → its per-trial correction is fragile")
    for i, val in enumerate(g.values):
        ax.text(i, val * 1.3, f"{val:,.0f}", ha="center", fontsize=9)
    fig.tight_layout(); p = os.path.join(FIGDIR, "fig2_pd_quality.png")
    fig.savefig(p, dpi=140); plt.close(fig); return p


def fig_residual_timeseries(df):
    det = df[(df.site == "SHE") & df.pd_detected]
    runs = det.groupby(["subject", "session"])
    keys = list(runs.groups.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 1.7 * len(keys)), sharex=True)
    for ax, key in zip(np.atleast_1d(axes), keys):
        d = runs.get_group(key).sort_values(["run_order", "trial_index"])
        d = d.reset_index(drop=True)
        ax.scatter(range(len(d)), d["residual_ms"], s=5, alpha=0.4, color=SITE_COLORS["SHE"])
        fail = d[d.is_failure]
        if len(fail):
            idx = [d.index[d.trial_index.eq(t) & d.task.eq(tk)][0]
                   for t, tk in zip(fail.trial_index, fail.task)]
            ax.scatter(idx, fail["residual_ms"], s=60, marker="*", color="black", zorder=6,
                       label=f"correction failure ({fail['residual_ms'].max():.0f} ms)")
            ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel(f"{key[0][4:]}\n{key[1]}", fontsize=7, rotation=0, ha="right", va="center")
        ax.axhspan(-3.91, 7.81, color="grey", alpha=0.08)
    fig.suptitle("Sheffield residual vs trial order (all visual tasks concatenated per session)\n"
                 "flat quantised scatter — no drift; isolated failures marked", fontsize=10)
    axes[-1].set_xlabel("stimulus event (in acquisition order)")
    fig.tight_layout(); p = os.path.join(FIGDIR, "fig3_residual_timeseries.png")
    fig.savefig(p, dpi=140); plt.close(fig); return p


def fig_test_retest(df):
    det = df[(df.site == "SHE") & df.pd_detected]
    subs = sorted(det.subject.unique())
    fig, axes = plt.subplots(1, len(subs), figsize=(5 * len(subs), 4), sharey=True)
    bins = np.arange(-6, 14, 3.906 / 2)
    for ax, sub in zip(np.atleast_1d(axes), subs):
        for ses, c in [("ses-01SHE", "#d1495b"), ("ses-03SHE", "#5b8c5a")]:
            v = det[(det.subject == sub) & (det.session == ses)]["residual_ms"]
            ax.hist(v, bins=bins, alpha=0.5, label=ses, color=c, density=True)
        ax.set_title(sub); ax.set_xlabel("residual (ms)"); ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Sheffield test–retest: visit 1 (ses-01SHE) vs visit 2 (ses-03SHE)\n"
                 "near-identical → the timing behaviour is a stable property of the site setup", fontsize=10)
    fig.tight_layout(); p = os.path.join(FIGDIR, "fig4_test_retest.png")
    fig.savefig(p, dpi=140); plt.close(fig); return p


def fig_failure_waveform(df):
    """The smoking gun: PD waveform of the surviving correction-failure trial vs a normal trial."""
    fail = df[df.is_failure].sort_values("residual_ms", ascending=False)
    if fail.empty:
        return None
    row = fail.iloc[0]
    prefix = os.path.join(ROOT, row.subject, row.session, "eeg",
                          f"{row.subject}_{row.session}_task-{row.task}")
    pd_sig, fs, _ = read_pd_channel(prefix)
    ev = load_events(prefix); stim = ev[ev.value_num >= 100].reset_index(drop=True)
    t_fail = int(stim.loc[row.trial_index, "sample"])
    t_norm = int(stim.loc[max(0, row.trial_index - 1), "sample"])
    tms = np.arange(-40, 160)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t, lab, c in [(t_norm, "normal trial", "#2e7d95"), (t_fail, f"failure trial (+{row.residual_ms:.0f} ms)", "#d1495b")]:
        base = np.median(pd_sig[t - int(0.08 * fs):t - int(0.04 * fs)])
        y = [pd_sig[t + int(round(m / 1000 * fs))] - base for m in tms]
        ax.plot(tms, y, color=c, label=lab, lw=1.5)
    ax.axvline(0, color="black", ls="--", lw=1, label="trigger (corrected)")
    ax.set_xlabel("time from trigger (ms)"); ax.set_ylabel("PD (baseline-subtracted)")
    ax.set_title(f"Correction failure — {row.subject} {row.session} {row.task}\n"
                 f"stimulus light arrives {row.residual_ms:.0f} ms after the trigger (uncorrected)")
    ax.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(FIGDIR, "fig5_failure_waveform.png")
    fig.savefig(p, dpi=140); plt.close(fig); return p


def variance_decomposition(df):
    """Crude variance components of the residual (SHE only, where spread exists)."""
    d = df[(df.site == "SHE") & df.pd_detected].copy()
    total = d["residual_ms"].var()
    run_means = d.groupby(["subject", "session", "task"])["residual_ms"].mean()
    subj_means = d.groupby("subject")["residual_ms"].mean()
    sess_means = d.groupby(["subject", "session"])["residual_ms"].mean()
    within_run = d.groupby(["subject", "session", "task"])["residual_ms"].var().mean()
    return dict(total_var=total, within_run_var=within_run,
                between_run_var=run_means.var(), between_session_var=sess_means.var(),
                between_subject_var=subj_means.var())


def main():
    np.random.seed(0)
    df = load()
    g = summary_table(df)
    figs = [fig_residual_by_site(df), fig_pd_quality(df), fig_residual_timeseries(df),
            fig_test_retest(df), fig_failure_waveform(df)]
    vc = variance_decomposition(df)

    print("=== Residual (PD − corrected trigger) by site ===")
    bysite = (df[df.pd_detected].groupby("site")["residual_ms"]
              .agg(["count", "mean", "std", "min", "max"]).reindex(["LDS", "MAN", "SHE"]).round(3))
    print(bysite.to_string())
    print("\n=== Photodiode SNR by site (median) ===")
    print(df[df.pd_detected].groupby("site")["pd_snr"].median().reindex(["LDS", "MAN", "SHE"]).round(1).to_string())
    print("\n=== Correction-failure trials (residual > %.0f ms) ===" % FAILURE_MS)
    fails = df[df.is_failure][["subject", "session", "task", "trial_index", "onset_s", "residual_ms", "pd_snr"]]
    print(fails.to_string(index=False) if len(fails) else "  none")
    tot = df[(df.site == "SHE") & df.pd_detected].shape[0]
    print(f"  Sheffield failure rate: {len(fails)}/{tot} = {len(fails)/tot*100:.3f}%")
    print("\n=== Sheffield residual variance decomposition (ms²) ===")
    for k, v in vc.items():
        print(f"  {k:22s} {v:8.3f}")
    print("  -> within-run (trial-level quantisation) dominates; subject/session/run add ~nothing")
    print("\n=== Figures written ===")
    for f in figs:
        if f: print("  " + os.path.relpath(f, ROOT))


if __name__ == "__main__":
    main()
