#!/usr/bin/env python3
"""
rt_bias.py — how far the marker timing error propagates into REACTION TIME.

The photodiode correction moves the *stimulus* markers onto the true screen onset.
It cannot see a keyboard, so response markers are passed through untouched
(verified in §0: keypress err = 0.00 ms in 7/8 sessions, -3.00 ms at Manchester
where a global clock fix was applied to every marker). Reaction time is
stimulus-to-response, so the whole stimulus-side error lands directly in RT:

    RT_raw = kp_raw - stim_raw
    RT_cor = kp_cor - stim_cor = RT_raw + (kp_err - stim_err)
    bias   = RT_raw - RT_cor   = stim_err - kp_err      (>0: raw RT too LONG)

A lab without a photodiode reports RT_raw. Since stim_err is a site-level
property (77% of its variance, step 1), the bias is a site-level bias in a
dependent variable that nobody treats as timing-sensitive.

Because the response marker is identical in both arms, it cancels exactly and the
stimulus-side result needs no assumption about response-device latency. That is
the primary result (§1).

There are, however, THREE reaction times available per trial, not two, and they
disagree:

  rt_log  BIDS `response_rt` — the presentation software's own RT (flip request
          to OS key event). This is what almost every published study reports.
  rt_raw  amplifier keypress marker minus amplifier stimulus trigger.
  rt_cor  amplifier keypress marker minus photodiode screen onset.

rt_log and rt_cor track the same physical keypresses (within-run r = 0.92-0.98,
§0), but the response-marker path lags the software key event by a
session-dependent amount with a trial-level SD of 32-75 ms (§1b). So the two
channels behave oppositely: the stimulus side is a per-session DISPLACEMENT
(within-session SD 1.4-11.6 ms) and the response side is genuine JITTER, an
order of magnitude larger. A photodiode watches the screen and is blind to it.

Sections
  §0  pair each stimulus with its keypress; establish the three RT definitions
  §1  the stimulus-side bias, per run and per site (response side cancels)
  §1b the response channel — the jitter a photodiode cannot see
  §2  variance decomposition of both channels (site / subject / session / run / trial)
  §3  what a naive study concludes: between-site, test-retest, and the
      within-session condition contrast that is IMMUNE to the stimulus side
  §4  design-level false-positive rate for a 12/12/12 three-site study
  §5  shift vs distortion — displacement at session level is smearing when pooled

Writes: rt_bias_trials.csv, rt_bias_runs.csv, rt_bias_response.csv,
        rt_bias_inference.csv, rt_bias_fpr.csv, rt_bias_pooling.csv,
        rt_bias_variance.csv
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

from jitter_pipeline import ROOT
from raw_markers import (SUBJECTS, SESSIONS, TASKS, SITE_OF, VISIT_OF,
                         VISUAL_CODES, CONDITION, CONTRAST,
                         session_table, bids_prefix, _locate)
from step1_timing import variance_components

OUT = os.path.join(ROOT, "analysis")
FIG = os.path.join(OUT, "figures")
SITE_COL = {"LDS": "#3b7dd8", "MAN": "#37a66b", "SHE": "#d1495b"}
KEYPRESS = 55
RT_MIN_MS, RT_MAX_MS = 100.0, 1500.0   # plausible-response window, standard practice
SITES = ["LDS", "MAN", "SHE"]
N_PER_SITE = 12                        # the planned 36-subject release, 12/12/12
N_SIM = 4000
ALPHA = 0.05
SIGMA_B_GRID = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0]   # between-subject mean-RT SD


# ------------------------------------------------------------------ §0 pairing
def run_rt(subject: str, session: str, task: str) -> pd.DataFrame | None:
    """Stimulus->keypress pairs for one BIDS run, with raw and corrected RT.

    Each visual stimulus is paired with the FIRST keypress that follows it and
    precedes the next visual stimulus; stimuli with no such keypress are dropped
    (they are non-response trials, not failures). Rows come from the session-wide
    raw/corrected table via the same alignment raw_markers uses, so the BIDS
    columns (trial_type, response_correct) sit row-aligned with both onsets.
    """
    prefix = bids_prefix(subject, session, task)
    if not os.path.exists(prefix + "_events.tsv"):
        return None
    bids = pd.read_csv(prefix + "_events.tsv", sep="\t")
    bids_values = pd.to_numeric(bids["value"], errors="coerce").values

    ses = session_table(subject, session)
    start = _locate(ses["value"].values, bids_values)
    seg = ses.iloc[start:start + len(bids)].reset_index(drop=True)
    assert np.array_equal(seg["value"].values, bids_values.astype(int)), "row misalignment"

    val = seg["value"].values
    is_stim = np.isin(val, list(VISUAL_CODES))
    is_kp = val == KEYPRESS

    rows = []
    stim_idx = np.flatnonzero(is_stim)
    for k, i in enumerate(stim_idx):
        stop = stim_idx[k + 1] if k + 1 < len(stim_idx) else len(val)
        cand = np.flatnonzero(is_kp[i + 1:stop])
        if len(cand) == 0:
            continue
        j = i + 1 + cand[0]
        rt_raw = (seg["raw_onset"].iat[j] - seg["raw_onset"].iat[i]) * 1000.0
        rt_cor = (seg["cor_onset"].iat[j] - seg["cor_onset"].iat[i]) * 1000.0
        rows.append({
            "subject": subject, "session": session, "site": SITE_OF[session],
            "visit": VISIT_OF[session], "task": task,
            "value": int(val[i]),
            "condition": CONDITION[task].get(int(val[i])),
            "trial_index": k,
            "session_time_s": float(seg["raw_onset"].iat[i]),
            "stim_err_ms": float(seg["err_ms"].iat[i]),
            "kp_err_ms": float(seg["err_ms"].iat[j]),
            "rt_raw_ms": rt_raw,
            "rt_cor_ms": rt_cor,
            "n_extra_kp": int(len(cand) - 1),
            "bids_rt_ms": (float(bids["response_rt"].iat[i]) * 1000.0
                           if "response_rt" in bids
                           and pd.notna(pd.to_numeric(bids["response_rt"].iat[i], errors="coerce"))
                           else np.nan),
            "correct": (str(bids["response_correct"].iat[i])
                        if "response_correct" in bids else "n/a"),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["bias_ms"] = df["rt_raw_ms"] - df["rt_cor_ms"]
    # response-channel discrepancy: amplifier keypress marker vs software key event
    df["resp_delta_ms"] = df["rt_cor_ms"] - df["bids_rt_ms"]
    # identity the whole analysis rests on
    assert np.allclose(df["bias_ms"], df["stim_err_ms"] - df["kp_err_ms"], atol=1e-6), \
        "bias != stim_err - kp_err"
    return df


def build_rt_table() -> pd.DataFrame:
    frames = [run_rt(s, ses, t) for s in SUBJECTS for ses in SESSIONS for t in TASKS]
    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    # one row in sub-SHE02/ses-01SHE/p3b is a keypress coincident with an
    # uncorrected stimulus (rt_raw = 0, and the "keypress" itself carries a
    # photodiode correction) — a mismarked event, not a response. Degenerate
    # pairs are excluded explicitly rather than left to the RT window.
    df["degenerate"] = (df["rt_raw_ms"] <= 0) | (df["rt_cor_ms"] <= 0)
    df["usable"] = (~df["degenerate"]
                    & df["rt_cor_ms"].between(RT_MIN_MS, RT_MAX_MS)
                    & (df["correct"] == "True"))
    return df


# ------------------------------------------------------------- small utilities
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardised mean difference (a - b)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else np.nan


def _fmt(df: pd.DataFrame) -> str:
    return df.round(2).to_string(index=False)


# --------------------------------------------------------------------- §1 bias
def per_run(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["usable"]].groupby(["subject", "session", "site", "visit", "task"])
    out = g.agg(n=("rt_cor_ms", "size"),
                rt_cor=("rt_cor_ms", "mean"), rt_raw=("rt_raw_ms", "mean"),
                rt_cor_sd=("rt_cor_ms", "std"), rt_raw_sd=("rt_raw_ms", "std"),
                bias=("bias_ms", "mean"), bias_sd=("bias_ms", "std"),
                stim_err=("stim_err_ms", "mean"), kp_err=("kp_err_ms", "mean"),
                stim_err_sd=("stim_err_ms", "std"),
                resp_delta=("resp_delta_ms", "median"),
                resp_delta_sd=("resp_delta_ms", "std")).reset_index()
    out["bias_pct_of_rt"] = 100 * out["bias"] / out["rt_cor"]
    return out


# ------------------------------------------------------- §1b response channel
def response_channel(df: pd.DataFrame) -> pd.DataFrame:
    """The response side, which the photodiode is blind to.

    resp_delta = rt_cor - rt_log: how far the amplifier's keypress marker sits
    from the presentation software's key event for the SAME keypress. Reported
    against the stimulus side's within-session SD, because the contrast is the
    point: the stimulus error is a per-session constant, the response error is
    trial-level jitter an order of magnitude larger.
    """
    d = df[df["usable"] & df["resp_delta_ms"].notna()]
    rows = []
    for (sub, ses), g in d.groupby(["subject", "session"]):
        r = np.array([np.corrcoef(x["bids_rt_ms"], x["rt_cor_ms"])[0, 1]
                      for _, x in g.groupby("task") if len(x) > 10])
        rows.append(dict(
            subject=sub, session=ses, site=SITE_OF[ses], n=len(g),
            resp_delta_med=float(np.median(g["resp_delta_ms"])),
            resp_delta_sd=float(g["resp_delta_ms"].std()),
            resp_delta_iqr=float(np.subtract(*np.percentile(g["resp_delta_ms"], [75, 25]))),
            stim_err_med=float(np.median(g["stim_err_ms"])),
            stim_err_sd=float(g["stim_err_ms"].std()),
            r_log_vs_marker=float(np.mean(r))))
    out = pd.DataFrame(rows)
    out["jitter_ratio"] = out["resp_delta_sd"] / out["stim_err_sd"]
    return out


# ----------------------------------------------------------------- §3 naive study
def naive_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Three contrasts, each run on corrected and on raw RT, at trial level.

    (a) between-site  SHE vs LDS, within subject (the traveling design: the
        strongest possible site comparison, and it is still fooled)
    (b) test-retest   SHE visit 1 vs visit 2, same subject, same site
    (c) within-session condition contrast — the control that must be immune,
        because both conditions share one session offset
    """
    d = df[df["usable"] & df["bids_rt_ms"].notna()]
    rows = []
    for arm, col in [("corrected", "rt_cor_ms"), ("raw", "rt_raw_ms"),
                     ("log", "bids_rt_ms")]:
        for sub in sorted(d["subject"].unique()):
            s = d[d["subject"] == sub]
            # (a) between-site
            a = s.loc[s["site"] == "SHE", col].values
            b = s.loc[s["site"] == "LDS", col].values
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append(dict(contrast="site SHE-LDS", subject=sub, arm=arm,
                             n_a=len(a), n_b=len(b),
                             diff_ms=np.mean(a) - np.mean(b), d=cohens_d(a, b),
                             t=float(t), p=float(p)))
            # (b) test-retest at Sheffield
            a = s.loc[(s["site"] == "SHE") & (s["visit"] == 2), col].values
            b = s.loc[(s["site"] == "SHE") & (s["visit"] == 1), col].values
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rows.append(dict(contrast="SHE retest v2-v1", subject=sub, arm=arm,
                             n_a=len(a), n_b=len(b),
                             diff_ms=np.mean(a) - np.mean(b), d=cohens_d(a, b),
                             t=float(t), p=float(p)))
            # (c) within-session condition contrast, pooled over sessions
            for task in sorted(s["task"].unique()):
                c1, c2 = CONTRAST[task]
                st = s[s["task"] == task]
                a = st.loc[st["condition"] == c1, col].values
                b = st.loc[st["condition"] == c2, col].values
                if len(a) < 10 or len(b) < 10:
                    continue
                t, p = stats.ttest_ind(a, b, equal_var=False)
                rows.append(dict(contrast=f"{task} {c1}-{c2}", subject=sub, arm=arm,
                                 n_a=len(a), n_b=len(b),
                                 diff_ms=np.mean(a) - np.mean(b), d=cohens_d(a, b),
                                 t=float(t), p=float(p)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- §4 the FPR
def fpr_simulation(runs: pd.DataFrame, df: pd.DataFrame, rng=None) -> pd.DataFrame:
    """False-positive rate for a site main effect on RT, when no real one exists.

    Design simulated: the ordinary multi-site study — independent subjects at each
    site (NOT traveling), n=12 per site, one-way ANOVA on subject mean RT.
    Truth: no site effect on true RT. Every rejection is therefore a false
    positive, and the raw arm's rejections are caused by hardware alone.

    Each simulated subject draws a session offset by bootstrap from the offsets
    actually MEASURED at that site (SHE pool of 4, LDS 2, MAN 2 — a small pool,
    which is the honest limit of n=2 traveling subjects). Between-subject SD of
    true mean RT is unknown here and is swept, not assumed.
    """
    rng = rng or np.random.default_rng(0)
    # measured session-level offsets, one value per subject-session
    pool = {s: g.groupby(["subject", "session"])["stim_err"].mean().values
            for s, g in runs.groupby("site")}
    # within-subject SE of a mean RT, from the real trial-level scatter and trial count
    d = df[df["usable"]]
    sigma_w = float(d.groupby(["subject", "session", "task"])["rt_cor_ms"].std().mean())
    n_trials = float(d.groupby(["subject", "session", "task"]).size().mean())
    se_within = sigma_w / np.sqrt(n_trials)

    rows = []
    for sigma_b in SIGMA_B_GRID:
        rej = {"corrected": 0, "raw": 0}
        eff = []
        for _ in range(N_SIM):
            groups = {"corrected": [], "raw": []}
            for site in SITES:
                true_mean = rng.normal(0.0, sigma_b, N_PER_SITE)
                noise = rng.normal(0.0, se_within, N_PER_SITE)
                off = rng.choice(pool[site], size=N_PER_SITE, replace=True)
                groups["corrected"].append(true_mean + noise)
                groups["raw"].append(true_mean + noise + off)
            for arm in ("corrected", "raw"):
                f, p = stats.f_oneway(*groups[arm])
                if p < ALPHA:
                    rej[arm] += 1
            m = [np.mean(g) for g in groups["raw"]]
            eff.append(max(m) - min(m))
        rows.append(dict(sigma_b_ms=sigma_b, se_within_ms=se_within,
                         fpr_corrected=rej["corrected"] / N_SIM,
                         fpr_raw=rej["raw"] / N_SIM,
                         mean_max_site_gap_ms=float(np.mean(eff))))
    return pd.DataFrame(rows)


# -------------------------------------------------------- §5 shift vs distortion
def pooling_effect(df: pd.DataFrame) -> pd.DataFrame:
    """A per-session constant displaces RT within a session and SMEARS it once
    sessions are pooled — but the net effect on the pooled SD is not signed.

    Exactly:  var(rt_raw) = var(rt_cor) + var(offset) + 2*cov(rt_cor, offset)

    The smearing term var(offset) is always positive and is the part that
    generalises. The covariance term is an accident of the sample: if the site
    with the long display latency also has the fast responders, it is negative
    and the pooled SD can SHRINK even though timing error was added. Both terms
    are reported rather than only the net.
    """
    d = df[df["usable"]]
    rows = []
    for level, keys in [("within run", ["subject", "session", "task"]),
                        ("across sessions (within subject)", ["subject", "task"]),
                        ("across sites (all)", ["task"])]:
        sd_c, sd_r, sd_o, cov, sk_c, sk_r = [], [], [], [], [], []
        for _, g in d.groupby(keys):
            c, o = g["rt_cor_ms"].values, g["bias_ms"].values
            sd_c.append(np.std(c, ddof=1))
            sd_r.append(np.std(c + o, ddof=1))
            sd_o.append(np.std(o, ddof=1))
            cov.append(np.cov(c, o, ddof=1)[0, 1])
            sk_c.append(stats.skew(c))
            sk_r.append(stats.skew(c + o))
        rows.append(dict(pooling=level,
                         sd_corrected=float(np.mean(sd_c)),
                         sd_raw=float(np.mean(sd_r)),
                         smear_sd_of_offset=float(np.mean(sd_o)),
                         cov_term=float(2 * np.mean(cov)),
                         net_var_change=float(np.mean(np.array(sd_r) ** 2
                                                     - np.array(sd_c) ** 2)),
                         skew_corrected=float(np.mean(sk_c)),
                         skew_raw=float(np.mean(sk_r))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- figure
def make_figure(df, runs, fpr):
    """figures/rt1_bias.png — the bias by site, the three RT distributions, the FPR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    d = df[df["usable"] & df["bids_rt_ms"].notna()]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (a) per-run stimulus-side bias, by site
    for i, site in enumerate(SITES):
        r = runs[runs["site"] == site]
        ax[0].scatter(np.full(len(r), i) + np.linspace(-.12, .12, len(r)),
                      r["bias"], s=34, color=SITE_COL[site], zorder=3,
                      edgecolor="white", linewidth=.6)
        ax[0].hlines(r["bias"].mean(), i - .28, i + .28, color=SITE_COL[site], lw=2)
    ax[0].set_xticks(range(len(SITES)))
    ax[0].set_xticklabels(SITES)
    ax[0].set_ylabel("RT bias (ms)  reported - true")
    ax[0].set_title("a  display latency lands in RT\n(one point per run)", loc="left", fontsize=10)
    ax[0].axhline(0, color="0.5", lw=.8, ls=":")
    ax[0].grid(axis="y", alpha=.25)

    # (b) the three RTs, pooled across everything
    bins = np.linspace(100, 1300, 70)
    for col, lab, c in [("rt_cor_ms", "rt_cor  (photodiode)", "#222222"),
                        ("rt_raw_ms", "rt_raw  (markers)", "#d1495b"),
                        ("bids_rt_ms", "rt_log  (software)", "#3b7dd8")]:
        ax[1].hist(d[col], bins=bins, histtype="step", lw=1.8, color=c,
                   label=f"{lab}   {d[col].mean():.0f} ms")
    ax[1].set_xlabel("reaction time (ms)")
    ax[1].set_ylabel("trials")
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_title("b  three RTs, same keypresses", loc="left", fontsize=10)

    # (c) false-positive rate for a site main effect that does not exist
    ax[2].plot(fpr["sigma_b_ms"], fpr["fpr_raw"], "o-", color="#d1495b",
               label="uncorrected markers")
    ax[2].plot(fpr["sigma_b_ms"], fpr["fpr_corrected"], "o-", color="#222222",
               label="photodiode-corrected")
    ax[2].axhline(ALPHA, color="0.5", lw=.8, ls=":")
    ax[2].set_ylim(0, 1.05)
    ax[2].set_xlabel("between-subject SD of true mean RT (ms)")
    ax[2].set_ylabel(f"P(site effect, p<{ALPHA})")
    ax[2].legend(fontsize=8, frameon=False, loc="upper right")
    ax[2].set_title(f"c  false positives, {N_PER_SITE}/{N_PER_SITE}/{N_PER_SITE} sites,\nno real effect",
                    loc="left", fontsize=10)
    ax[2].grid(alpha=.25)

    fig.tight_layout()
    out = os.path.join(FIG, "rt1_bias.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


# ------------------------------------------------------------------- selftest
def _selftest():
    df = build_rt_table()
    assert df.groupby(["subject", "session", "task"]).ngroups == 24, "expected 24 runs"
    assert np.allclose(df["bias_ms"], df["stim_err_ms"] - df["kp_err_ms"], atol=1e-6)
    # RT must be positive on every pair the analysis actually uses
    assert (df.loc[df["usable"], "rt_raw_ms"] > 0).all(), "non-positive raw RT"
    assert (df.loc[df["usable"], "rt_raw_ms"] >= df.loc[df["usable"], "rt_cor_ms"] - 4.0).all(), \
        "raw RT shorter than corrected — sign error"
    assert df["degenerate"].sum() <= 5, f"unexpected degenerate pairs: {df['degenerate'].sum()}"
    # rt_log and the marker RT must be the SAME keypresses (high within-run r),
    # while being allowed to differ by a large session-dependent lag
    rc = response_channel(df)
    assert (rc["r_log_vs_marker"] > 0.85).all(), \
        f"log and marker RT are not the same events (min r {rc['r_log_vs_marker'].min():.3f})"
    assert (rc["jitter_ratio"] > 2.0).all(), \
        "response channel not noisier than the stimulus channel — check the pairing"
    # the stimulus side must be a near-constant within a session
    assert (rc["stim_err_sd"] < 20.0).all(), "stimulus error not a per-session constant"
    # a constant per-session offset must cancel in a within-session contrast
    d = df[df["usable"] & (df["task"] == "n170")]
    for (sub, ses), g in d.groupby(["subject", "session"]):
        a = g[g["condition"] == "face"]; b = g[g["condition"] == "car"]
        if len(a) < 10 or len(b) < 10:
            continue
        dc = a["rt_cor_ms"].mean() - b["rt_cor_ms"].mean()
        dr = a["rt_raw_ms"].mean() - b["rt_raw_ms"].mean()
        assert abs(dc - dr) < 6.0, f"{sub} {ses}: contrast moved {dc - dr:.2f} ms"
    # the simulation must be calibrated on corrected data
    fpr = fpr_simulation(per_run(df), df, rng=np.random.default_rng(1))
    assert (fpr["fpr_corrected"] < 0.10).all(), "corrected-arm FPR not near nominal"
    assert (fpr["fpr_raw"] >= fpr["fpr_corrected"]).all(), "raw arm not worse"
    print(f"selftest OK — {len(df)} pairs, 24 runs, "
          f"log-vs-marker r {rc['r_log_vs_marker'].min():.3f}-{rc['r_log_vs_marker'].max():.3f}, "
          f"jitter ratio {rc['jitter_ratio'].min():.1f}-{rc['jitter_ratio'].max():.1f}x, "
          f"corrected-arm FPR {fpr['fpr_corrected'].min():.3f}-{fpr['fpr_corrected'].max():.3f}")


# ----------------------------------------------------------------------- main
def main():
    df = build_rt_table()
    runs = per_run(df)

    print("=" * 78)
    print("§0  PAIRING")
    print("=" * 78)
    n_stim_total = len(df)
    u = df["usable"].sum()
    m = df["bids_rt_ms"].notna()
    print(f"stimulus->keypress pairs : {n_stim_total}  ({df.groupby(['subject','session','task']).ngroups} runs)")
    print(f"usable (correct, {RT_MIN_MS:.0f}-{RT_MAX_MS:.0f} ms) : {u} ({100*u/n_stim_total:.1f}%)")
    print(f"trials with an extra keypress before the next stimulus : "
          f"{(df['n_extra_kp'] > 0).sum()}")
    print(f"degenerate pairs excluded (mismarked coincident events) : "
          f"{int(df['degenerate'].sum())}")
    print(f"trials with a software RT (BIDS response_rt) : {int(m.sum())}")
    print("\nthree RTs per trial, all present:")
    print(f"  rt_log (software)          mean {df.loc[df['usable'],'bids_rt_ms'].mean():7.1f} ms")
    print(f"  rt_raw (markers, raw)      mean {df.loc[df['usable'],'rt_raw_ms'].mean():7.1f} ms")
    print(f"  rt_cor (markers, PD-corr)  mean {df.loc[df['usable'],'rt_cor_ms'].mean():7.1f} ms")
    print("  -> rt_raw and rt_cor share the same response marker, so it cancels and\n"
          "     their difference is the stimulus-side error exactly (§1). rt_log uses a\n"
          "     different response timestamp and is treated separately (§1b).")

    print()
    print("=" * 78)
    print("§1  THE BIAS  (bias = raw RT - true RT, >0 means reported RT too long)")
    print("=" * 78)
    print(_fmt(runs[["subject", "session", "site", "task", "n", "rt_cor", "rt_raw",
                     "bias", "bias_sd", "stim_err", "kp_err", "bias_pct_of_rt"]]))
    site = runs.groupby("site").agg(
        n_runs=("n", "size"), trials=("n", "sum"),
        rt_true=("rt_cor", "mean"), rt_reported=("rt_raw", "mean"),
        bias=("bias", "mean"), bias_min=("bias", "min"), bias_max=("bias", "max"),
        stim_side=("stim_err", "mean"), resp_side=("kp_err", "mean"),
        pct_of_rt=("bias_pct_of_rt", "mean")).reset_index()
    print("\nby site:")
    print(_fmt(site))
    b = site.set_index("site")["bias"]
    print(f"\nspurious between-site RT differences (hardware only, no real effect):")
    for x, y in [("SHE", "LDS"), ("MAN", "LDS"), ("SHE", "MAN")]:
        print(f"  {x} - {y} : {b[x] - b[y]:+7.1f} ms")
    print(f"\nresponse MARKER contributes {site['resp_side'].abs().mean():.2f} ms on average and "
          f"is identical in both arms,\n  so it cancels: the numbers above are the "
          f"stimulus side alone, with no assumption\n  about response-device latency.")

    print()
    print("=" * 78)
    print("§1b  THE RESPONSE CHANNEL — the jitter a photodiode cannot see")
    print("=" * 78)
    rc = response_channel(df)
    print(_fmt(rc[["subject", "session", "site", "n", "r_log_vs_marker",
                   "resp_delta_med", "resp_delta_sd", "resp_delta_iqr",
                   "stim_err_med", "stim_err_sd", "jitter_ratio"]]))
    print(f"\n  r = {rc['r_log_vs_marker'].min():.2f}-{rc['r_log_vs_marker'].max():.2f} within run, so these are the "
          f"same keypresses timestamped twice.\n"
          f"  The amplifier keypress marker lags the software key event by "
          f"{rc['resp_delta_med'].min():.0f} to {rc['resp_delta_med'].max():.0f} ms,\n"
          f"  session-dependent, with a TRIAL-LEVEL SD of {rc['resp_delta_sd'].min():.0f}-"
          f"{rc['resp_delta_sd'].max():.0f} ms — "
          f"{rc['jitter_ratio'].min():.0f}-{rc['jitter_ratio'].max():.0f}x the\n"
          f"  stimulus side's within-session SD ({rc['stim_err_sd'].min():.1f}-"
          f"{rc['stim_err_sd'].max():.1f} ms).\n"
          f"  The two channels behave oppositely: stimulus = per-session displacement,\n"
          f"  response = trial-level jitter. Which of the two response timestamps is\n"
          f"  closer to the finger is NOT resolvable from this data — that is the honest\n"
          f"  limit, and it is why §1 is built to cancel the response side rather than\n"
          f"  correct it.")

    print()
    print("=" * 78)
    print("§2  WHERE THE BIAS LIVES  (variance decomposition, usable trials)")
    print("=" * 78)
    u = df[df["usable"]].copy()
    vc = variance_components(u.rename(columns={"bias_ms": "bias"}), "bias")
    vc["channel"] = "stimulus (rt_raw - rt_cor)"
    vr = variance_components(u[u["resp_delta_ms"].notna()].rename(
        columns={"resp_delta_ms": "bias"}), "bias")
    vr["channel"] = "response (rt_cor - rt_log)"
    vc = pd.concat([vc, vr], ignore_index=True)
    print(_fmt(vc[["channel", "level", "variance", "sd", "pct"]]))
    print("  -> the stimulus channel is site-dominated (inherited from the marker error,\n"
          "     so the RT claim carries the same level structure as the EEG claim). The\n"
          "     response channel is trial-dominated. Same dependent variable, two error\n"
          "     sources living at opposite ends of the hierarchy.")

    print()
    print("=" * 78)
    print("§3  WHAT A NAIVE STUDY CONCLUDES")
    print("=" * 78)
    inf = naive_inference(df)
    piv = inf.pivot_table(index=["contrast", "subject"], columns="arm",
                          values=["diff_ms", "d"]).reset_index()
    piv.columns = ["_".join(c).rstrip("_") for c in piv.columns]
    piv["diff_shift_ms"] = piv["diff_ms_raw"] - piv["diff_ms_corrected"]
    piv["diff_shift_log_ms"] = piv["diff_ms_log"] - piv["diff_ms_corrected"]
    print(_fmt(piv[["contrast", "subject", "diff_ms_corrected", "diff_ms_raw",
                    "diff_ms_log", "diff_shift_ms", "diff_shift_log_ms",
                    "d_corrected", "d_raw", "d_log"]]))
    print("\n  Between-site and test-retest contrasts move by 17-80 ms. The within-session\n"
          "  condition contrasts move by <3 ms — a per-session constant cancels inside a\n"
          "  session and only bites across sessions or sites. That is the precise\n"
          "  statement of the hazard.\n"
          "\n  Note the DIRECTION in this sample: Sheffield is genuinely faster than Leeds\n"
          "  (corrected d = -0.86 / -1.12), and the display bias very nearly erases it\n"
          "  (raw d = -0.04 / -0.59). Here the error destroys a real effect rather than\n"
          "  manufacturing one. Both failures are available — which one you get depends on\n"
          "  whether the hardware ordering happens to align with the biological ordering,\n"
          "  which no one can know without measuring. §4 quantifies the manufacturing\n"
          "  case, where there is no real effect to destroy.\n"
          "\n  The `log` arm is what a published study would actually have reported, and it\n"
          "  is the worst of the three: SHE-LDS goes from d = -0.86 / -1.12 (corrected) to\n"
          "  d = +0.31 / +0.02 (log). For sub-SHE01 the site effect SIGN-REVERSES. A paper\n"
          "  using the released RT column would report Sheffield slower than Leeds when it\n"
          "  is in fact substantially faster.")

    print()
    print("=" * 78)
    print("§4  FALSE-POSITIVE RATE, 12/12/12 THREE-SITE STUDY, NO REAL SITE EFFECT")
    print("=" * 78)
    fpr = fpr_simulation(runs, df)
    print(_fmt(fpr))
    print(f"\n  {N_SIM} simulated studies per row, one-way ANOVA on subject mean RT at "
          f"alpha={ALPHA}.\n  Corrected arm sits at nominal; the raw arm's rejections are "
          f"caused by display\n  latency alone. Between-subject SD is swept because n=2 "
          f"cannot estimate it.")

    print()
    print("=" * 78)
    print("§5  SHIFT OR DISTORTION — displacement becomes smearing one level up")
    print("=" * 78)
    pe = pooling_effect(df)
    print(_fmt(pe))
    print("\n  Within a run the offset is a pure shift: SD of the offset ~0, SD unchanged,\n"
          "  skew unchanged. Pool sessions and then sites and the between-session\n"
          "  constants become genuine spread — the smearing term smear_sd_of_offset grows\n"
          "  from 3.8 ms to 34.1 ms, which is the displacement-becomes-smearing effect and\n"
          "  is the part that generalises.\n"
          "\n  The NET change is negative here, because in this sample the site with the\n"
          "  largest display latency (Sheffield) also has the fastest responders, so the\n"
          "  covariance term is strongly negative and over-cancels the smearing. The\n"
          "  pooled SD therefore SHRINKS while timing error is being added — a pooled\n"
          "  distribution that looks tighter, not looser, for a bad reason. Skew still\n"
          "  rises, which is the distortion showing through.")

    make_figure(df, runs, fpr)

    for name, obj in [("rt_bias_trials.csv", df), ("rt_bias_runs.csv", runs),
                      ("rt_bias_response.csv", rc),
                      ("rt_bias_inference.csv", inf), ("rt_bias_fpr.csv", fpr),
                      ("rt_bias_pooling.csv", pe), ("rt_bias_variance.csv", vc)]:
        obj.to_csv(os.path.join(OUT, name), index=False)
    print(f"\nwrote rt_bias_{{trials,runs,response,inference,fpr,pooling,variance}}.csv "
          f"to {OUT}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
