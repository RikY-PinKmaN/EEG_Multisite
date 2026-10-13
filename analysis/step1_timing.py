#!/usr/bin/env python3
"""
step1_timing.py — ROADMAP step 1: measure the marker timing error at each site.

Consumes the per-trial measured error from `raw_markers.py` and answers, per site:
  * what the error distribution looks like (location, spread, shape),
  * whether it is independent trial-to-trial or DRIFTS / is autocorrelated,
  * how the variance splits across site / subject / session / run / trial.

The drift question is the one that matters downstream: if the error drifts within a
session, then generating augmented training data with independent random shifts is
mis-specified, and the shift-augmentation everyone does is wrong in a way nobody
checks. Outputs feed `augment.py` directly.

Writes: timing_per_run.csv, timing_per_site.csv, timing_variance.csv,
        timing_distfit.csv, figures/t1_*.png
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jitter_pipeline import ROOT
from raw_markers import build_trial_table

OUT = os.path.join(ROOT, "analysis")
FIG = os.path.join(OUT, "figures")
SITES = ["LDS", "MAN", "SHE"]
SITE_COL = {"LDS": "#3b7dd8", "MAN": "#37a66b", "SHE": "#d1495b"}
SITE_LABEL = {"LDS": "Leeds (Biosemi, 2048 Hz)",
              "MAN": "Manchester (ANT, 1000 Hz)",
              "SHE": "Sheffield (Bitbrain, 256 Hz)"}


# ----------------------------------------------------------------- helpers
def acf(x: np.ndarray, nlags: int = 30) -> np.ndarray:
    """Sample autocorrelation of a 1-D series, lags 0..nlags."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(nlags + 1)
    return np.array([np.dot(x[:n - k], x[k:]) / denom for k in range(nlags + 1)])


def ljung_box(x: np.ndarray, h: int = 20):
    """Ljung-Box Q on lags 1..h; returns (Q, p). H0: series is white noise."""
    n = len(x)
    r = acf(x, h)[1:]
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, h + 1)))
    return float(q), float(stats.chi2.sf(q, h))


def runs_test(x: np.ndarray):
    """Wald-Wolfowitz runs test about the median. H0: independent ordering."""
    s = np.sign(x - np.median(x))
    s = s[s != 0]
    n = len(s)
    if n < 20:
        return np.nan, np.nan
    n1, n2 = int((s > 0).sum()), int((s < 0).sum())
    runs = 1 + int((s[1:] != s[:-1]).sum())
    mu = 2 * n1 * n2 / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1)
    if var <= 0:
        return np.nan, np.nan
    z = (runs - mu) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def variance_components(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Hierarchical variance decomposition site > subject > session > run > trial.

    Method of moments on group means: the variance explained at each level is the
    variance of that level's means about its parent, and the trial component is the
    pooled within-run variance. Unbalanced-tolerant and transparent.

    ponytail: moment-based, not REML. n=2 subjects makes the subject component
    noisy regardless of estimator; upgrade to a mixed model if the full 36-subject
    release lands.
    """
    keys = [["site"], ["site", "subject"], ["site", "subject", "session"],
            ["site", "subject", "session", "task"]]
    names = ["site", "subject", "session", "run"]
    rows, prev = [], None
    for name, k in zip(names, keys):
        m = df.groupby(k)[col].mean()
        if prev is None:
            v = float(np.var(m.values, ddof=1)) if len(m) > 1 else 0.0
        else:
            # spread of these means about their own parent mean
            parent = df.groupby(k[:-1])[col].mean()
            dev = m - parent.reindex(m.index.droplevel(-1)).values
            v = float(np.var(dev.values, ddof=1)) if len(dev) > 1 else 0.0
        rows.append((name, max(v, 0.0)))
        prev = m
    within = df.groupby(["site", "subject", "session", "task"])[col].var(ddof=1)
    rows.append(("trial", float(np.nanmean(within.values))))
    out = pd.DataFrame(rows, columns=["level", "variance"])
    out["sd"] = np.sqrt(out["variance"])
    out["pct"] = 100 * out["variance"] / out["variance"].sum()
    return out.round(3)


# ----------------------------------------------------------------- analyses
def per_run_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sub, ses, task), g in df.groupby(["subject", "session", "task"]):
        g = g.sort_values("session_time_s")
        e = g["err_ms_rel"].values
        t = g["session_time_s"].values
        sl, ic, r, p, se = stats.linregress(t, e)
        resid = e - (ic + sl * t)
        q, pq = ljung_box(resid, h=min(20, len(resid) // 5))
        z, pz = runs_test(resid)
        rows.append(dict(
            subject=sub, session=ses, site=g["site"].iloc[0], visit=g["visit"].iloc[0], task=task,
            n=len(e), fs=g["fs"].iloc[0],
            mean_ms=e.mean(), sd_ms=e.std(ddof=1),
            min_ms=e.min(), max_ms=e.max(), range_ms=e.max() - e.min(),
            iqr_ms=np.subtract(*np.percentile(e, [75, 25])),
            drift_ms_per_min=sl * 60, drift_r=r, drift_p=p,
            sd_detrended_ms=resid.std(ddof=1),
            var_explained_by_drift_pct=100 * (1 - resid.var() / e.var()) if e.var() > 0 else 0.0,
            ar1=float(np.corrcoef(resid[:-1], resid[1:])[0, 1]),
            ljungbox_p=pq, runs_p=pz,
            sample_period_ms=1000.0 / g["fs"].iloc[0],
            quantisation_sd_ms=1000.0 / g["fs"].iloc[0] / np.sqrt(12),
        ))
    return pd.DataFrame(rows).round(4)


def per_site_stats(df: pd.DataFrame, per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, g in df.groupby("site"):
        r = per_run[per_run.site == site]
        e = g["err_ms_rel"].values
        rows.append(dict(
            site=site, n_trials=len(e), n_runs=len(r),
            mean_ms=e.mean(), sd_ms=e.std(ddof=1),
            median_ms=np.median(e), min_ms=e.min(), max_ms=e.max(),
            # the number that matters for augmentation: spread *within* a run
            within_run_sd_ms=r.sd_ms.mean(),
            within_run_sd_detrended_ms=r.sd_detrended_ms.mean(),
            between_run_sd_of_mean_ms=r.mean_ms.std(ddof=1),
            mean_abs_drift_ms_per_min=r.drift_ms_per_min.abs().mean(),
            max_abs_drift_ms_per_min=r.drift_ms_per_min.abs().max(),
            mean_ar1=r.ar1.mean(),
            frac_runs_drifting=float((r.drift_p < 0.01).mean()),
            frac_runs_autocorrelated=float((r.ljungbox_p < 0.01).mean()),
            sample_period_ms=1000.0 / g["fs"].iloc[0],
            # a marker can only land on a sample: uniform quantisation contributes
            # period/sqrt(12) of SD before any real timing variation
            quantisation_sd_ms=1000.0 / g["fs"].iloc[0] / np.sqrt(12),
        ))
    out = pd.DataFrame(rows)
    # how much of the within-run spread survives once quantisation is accounted for
    out["real_within_run_sd_ms"] = np.sqrt(
        np.maximum(out.within_run_sd_detrended_ms ** 2 - out.quantisation_sd_ms ** 2, 0.0))
    return out.round(3)


def dist_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Which parametric family best describes the within-run (detrended) error?"""
    rows = []
    for (site, sub, ses, task), g in df.groupby(["site", "subject", "session", "task"]):
        g = g.sort_values("session_time_s")
        e = g["err_ms_rel"].values
        t = g["session_time_s"].values
        sl, ic, *_ = stats.linregress(t, e)
        r = e - (ic + sl * t)
        if r.std() < 1e-9:
            continue
        z = (r - r.mean()) / r.std(ddof=1)
        fits = {
            "normal": stats.kstest(z, "norm").pvalue,
            "uniform": stats.kstest(r, "uniform", args=(r.min(), r.max() - r.min())).pvalue,
            "t": stats.kstest(z, "t", args=(stats.t.fit(z, floc=0, fscale=1)[0],)).pvalue,
        }
        rows.append(dict(site=site, subject=sub, session=ses, task=task,
                         skew=float(stats.skew(r)), kurtosis=float(stats.kurtosis(r)),
                         **{f"ks_p_{k}": v for k, v in fits.items()}))
    return pd.DataFrame(rows).round(4)


# ----------------------------------------------------------------- figures
def fig_drift(df: pd.DataFrame):
    subs = sorted(df.subject.unique())
    sess = ["ses-01SHE", "ses-02LDS", "ses-03SHE", "ses-04MAN"]
    fig, axes = plt.subplots(len(subs), len(sess), figsize=(17, 6.5), sharex=False)
    for i, sub in enumerate(subs):
        for j, ses in enumerate(sess):
            ax = axes[i, j]
            g = df[(df.subject == sub) & (df.session == ses)].sort_values("session_time_s")
            if g.empty:
                ax.axis("off"); continue
            site = g.site.iloc[0]
            tmin = g.session_time_s.min()
            x = (g.session_time_s - tmin) / 60
            ax.scatter(x, g.err_ms_rel, s=2.5, c=SITE_COL[site], alpha=.45, linewidths=0)
            sl, ic, r, p, _ = stats.linregress(g.session_time_s, g.err_ms_rel)
            ax.plot(x, ic + sl * g.session_time_s, "k-", lw=1.6)
            ax.set_title(f"{sub} · {ses} ({site})\n{sl*60:+.2f} ms/min, r={r:+.2f}", fontsize=9)
            if j == 0:
                ax.set_ylabel("marker error (ms)")
            if i == len(subs) - 1:
                ax.set_xlabel("time in session (min)")
            ax.grid(alpha=.25)
    fig.suptitle("Measured marker error over the session — the error DRIFTS, it is not i.i.d.",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "t1_drift_by_session.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_distributions(df: pd.DataFrame, per_run: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for s in SITES:
        e = df[df.site == s].err_ms_rel
        ax.hist(e, bins=60, alpha=.6, label=f"{s} (SD {e.std():.1f} ms)", color=SITE_COL[s])
    ax.set_xlabel("marker error (ms)"); ax.set_ylabel("trials")
    ax.set_title("Raw error, pooled over sessions"); ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[1]
    for s in SITES:
        vals = []
        for _, g in df[df.site == s].groupby(["subject", "session", "task"]):
            g = g.sort_values("session_time_s")
            sl, ic, *_ = stats.linregress(g.session_time_s, g.err_ms_rel)
            vals.append(g.err_ms_rel.values - (ic + sl * g.session_time_s.values))
        v = np.concatenate(vals)
        ax.hist(v, bins=60, alpha=.6, label=f"{s} (SD {v.std():.1f} ms)", color=SITE_COL[s])
    ax.set_xlabel("detrended error (ms)")
    ax.set_title("After removing within-run drift"); ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[2]
    for s in SITES:
        a = np.mean([acf(g.sort_values("session_time_s").err_ms_rel.values, 25)
                     for _, g in df[df.site == s].groupby(["subject", "session", "task"])], axis=0)
        ax.plot(range(26), a, "o-", ms=3, color=SITE_COL[s], label=s)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("lag (trials)"); ax.set_ylabel("autocorrelation")
    ax.set_title("Trial-to-trial dependence"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "t1_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def within_run_breakdown(per_run: pd.DataFrame) -> pd.DataFrame:
    """Split the within-run spread into drift, real residual jitter and quantisation."""
    rows = []
    for site, r in per_run.groupby("site"):
        tot = (r.sd_ms ** 2).mean()
        resid = (r.sd_detrended_ms ** 2).mean()
        quant = (r.quantisation_sd_ms ** 2).mean()
        rows.append(dict(site=site,
                         within_run_var=tot,
                         drift_var=max(tot - resid, 0.0),
                         quantisation_var=min(quant, resid),
                         real_jitter_var=max(resid - quant, 0.0)))
    out = pd.DataFrame(rows)
    for c in ["drift_var", "quantisation_var", "real_jitter_var"]:
        out[c.replace("_var", "_sd_ms")] = np.sqrt(out[c])
        out[c.replace("_var", "_pct")] = 100 * out[c] / out["within_run_var"]
    return out.round(3)


def fig_variance(vc: pd.DataFrame, wr: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.bar(vc["level"], vc["pct"], color="#d1495b")
    for i, (p, s) in enumerate(zip(vc["pct"], vc["sd"])):
        ax.text(i, p + 1.5, f"{p:.1f}%\nSD {s:.1f} ms", ha="center", fontsize=8)
    ax.set_ylabel("% of total variance"); ax.set_ylim(0, 100)
    ax.set_title("Where the timing error lives\n(site and session dominate; trial-to-trial is ~1%)",
                 fontsize=10)
    ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    bottom = np.zeros(len(wr))
    for comp, col in [("drift_pct", "#d1495b"), ("real_jitter_pct", "#3b7dd8"),
                      ("quantisation_pct", "#999999")]:
        ax.bar(wr["site"], wr[comp], bottom=bottom, color=col, label=comp.replace("_pct", ""))
        bottom += wr[comp].values
    ax.set_ylabel("% of within-run variance")
    ax.set_title("Within a single run, what is left\nonce the run's constant offset is removed", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "t1_variance_components.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = build_trial_table()
    df = df[~df.correction_failed].copy()          # 2 trials the correction never touched

    per_run = per_run_stats(df)
    per_site = per_site_stats(df, per_run)

    # detrended copy for the variance decomposition
    dt = []
    for _, g in df.groupby(["subject", "session", "task"]):
        g = g.sort_values("session_time_s").copy()
        sl, ic, *_ = stats.linregress(g.session_time_s, g.err_ms_rel)
        g["err_dt"] = g.err_ms_rel - (ic + sl * g.session_time_s)
        dt.append(g)
    dt = pd.concat(dt)

    vc = variance_components(df, "err_ms_rel")
    wr = within_run_breakdown(per_run)
    fits = dist_fits(df)

    per_run.to_csv(os.path.join(OUT, "timing_per_run.csv"), index=False)
    per_site.to_csv(os.path.join(OUT, "timing_per_site.csv"), index=False)
    vc.to_csv(os.path.join(OUT, "timing_variance.csv"), index=False)
    wr.to_csv(os.path.join(OUT, "timing_within_run_breakdown.csv"), index=False)
    fits.to_csv(os.path.join(OUT, "timing_distfit.csv"), index=False)

    fig_drift(df); fig_distributions(df, per_run); fig_variance(vc, wr)

    pd.set_option("display.width", 200)
    print("=== per site ===");  print(per_site.to_string(index=False))
    print("\n=== per run ===")
    print(per_run[["subject", "session", "site", "task", "n", "mean_ms", "sd_ms",
                   "drift_ms_per_min", "drift_r", "drift_p", "sd_detrended_ms",
                   "var_explained_by_drift_pct", "ar1", "ljungbox_p"]].to_string(index=False))
    print("\n=== variance components (hierarchical) ==="); print(vc.to_string(index=False))
    print("\n=== within-run breakdown ===")
    print(wr[["site", "within_run_var", "drift_sd_ms", "drift_pct", "real_jitter_sd_ms",
              "real_jitter_pct", "quantisation_sd_ms", "quantisation_pct"]].to_string(index=False))
    print("\n=== distribution fits (median KS p per site) ===")
    print(fits.groupby("site")[["skew", "kurtosis", "ks_p_normal", "ks_p_uniform", "ks_p_t"]]
          .median().round(3).to_string())


if __name__ == "__main__":
    main()
