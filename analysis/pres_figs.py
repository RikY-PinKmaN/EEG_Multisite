#!/usr/bin/env python3
"""Single-panel slide figures for the timing presentation (no GAN content)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SITE_COL = {"LDS": "#3b7dd8", "MAN": "#37a66b", "SHE": "#d1495b"}
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.titlesize": 17})


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---------- P1: every measured trial, as recorded ----------
tj = pd.read_csv(os.path.join(HERE, "trial_jitter.csv"))
tj = tj[~tj.correction_failed]
subs = sorted(tj.subject.unique())
sess = ["ses-01SHE", "ses-02LDS", "ses-03SHE", "ses-04MAN"]
fig, axes = plt.subplots(len(subs), len(sess), figsize=(15.5, 6.4), sharey=True)
for i, sub in enumerate(subs):
    for j, se in enumerate(sess):
        ax = axes[i, j]
        g = tj[(tj.subject == sub) & (tj.session == se)]
        if not len(g):
            ax.axis("off"); continue
        ax.scatter(g.session_time_s / 60.0, g.err_ms, s=3.5,
                   color=SITE_COL[g.site.iloc[0]], alpha=.55, linewidths=0)
        ax.set_ylim(0, 155)
        ax.grid(alpha=.25)
        if i == 0:
            ax.set_title(f"{se[4:]}", fontsize=15)
        if j == 0:
            ax.set_ylabel(f"{sub[-5:]}\nmarker error (ms)", fontsize=13)
        if i == len(subs) - 1:
            ax.set_xlabel("time into session (min)", fontsize=12)
        ax.text(.97, .93, f"{g.err_ms.mean():.0f} ms", transform=ax.transAxes,
                ha="right", va="top", fontsize=13, weight="bold",
                color=SITE_COL[g.site.iloc[0]],
                bbox=dict(fc="white", ec="none", alpha=.85, pad=1.6))
fig.suptitle("Every trial, as recorded: the error is steady within a run and differs by site",
             fontsize=18, y=1.02)
_save(fig, "p1_raw_trials.png")


# ---------- P2: where the error lives ----------
v = pd.read_csv(os.path.join(HERE, "timing_variance.csv"))
order = ["site", "session", "subject", "trial", "run"]
v = v.set_index("level").loc[order].reset_index()
fig, ax = plt.subplots(figsize=(9.4, 4.6))
cols = ["#d1495b", "#e08a4f", "#8f9bb3", "#b9c0cc", "#cdd3dc"]
b = ax.barh(range(len(v))[::-1], v.pct, color=cols, edgecolor="k", lw=.6, height=.68)
for k, (lv, pc) in enumerate(zip(v.level, v.pct)):
    ax.text(pc + 1.5, len(v) - 1 - k, f"{pc:.1f}%", va="center", fontsize=15, weight="bold")
ax.set_yticks(range(len(v))[::-1])
ax.set_yticklabels([l.upper() for l in v.level], fontsize=14)
ax.set_xlim(0, 92); ax.set_xlabel("share of total variance in marker error (%)")
ax.set_title("The error belongs to the SITE, not to the trial", pad=12)
ax.spines["left"].set_visible(False); ax.tick_params(left=False)
_save(fig, "p2_variance.png")


# ---------- P3: the ERP moves by exactly the measured error ----------
e = pd.read_csv(os.path.join(HERE, "erp_impact.csv"))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.6),
                              gridspec_kw={"width_ratios": [1.15, 1]})
ax.plot([0, 155], [0, 155], ls=":", color="0.35", lw=1.6, label="identity")
for site, g in e.groupby("site"):
    ax.scatter(g.mean_err_ms, g.recovered_lag_ms, s=170, color=SITE_COL[site],
               edgecolor="k", lw=.9, label=site, zorder=3)
ax.set_xlabel("marker error measured by photodiode (ms)")
ax.set_ylabel("ERP shift recovered from the EEG (ms)")
ax.set_title("The ERP moves by exactly the marker error", pad=10)
ax.legend(fontsize=13); ax.grid(alpha=.25)
ax.text(.04, .93, f"mean |residual| = {e.lag_error_ms.abs().mean():.2f} ms",
        transform=ax.transAxes, fontsize=14, weight="bold")

ax2.bar(range(len(e)), e.shape_r, color=[SITE_COL[s] for s in e.site],
        edgecolor="k", lw=.5)
ax2.axhline(1.0, ls=":", color="0.35", lw=1.6)
ax2.set_ylim(0.9, 1.02); ax2.set_xticks([])
ax2.set_ylabel("waveform correlation at best lag")
ax2.set_xlabel("the 16 session × task comparisons")
ax2.set_title("...and keeps its shape: displaced, not smeared", pad=10)
_save(fig, "p3_erp_shift.png")


# ---------- P4: what a lab would actually report ----------
fig, ax = plt.subplots(figsize=(13.0, 5.0))
x = np.arange(len(e))
lab = [f"{r.session[4:]}\n{r.task}" for _, r in e.iterrows()]
ax.bar(x - .2, e.fixed_corrected_uv, .4, label="with photodiode correction",
       color="#2f6fb8", edgecolor="k", lw=.5)
ax.bar(x + .2, e.fixed_raw_uv, .4, label="without (uncorrected markers)",
       color="#c94f5e", edgecolor="k", lw=.5)
for xi, r in zip(x, e.itertuples()):
    if r.sign_flipped:
        ax.plot(xi, max(r.fixed_corrected_uv, r.fixed_raw_uv) + .55, marker="v",
                color="#a8203c", ms=9)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9.5)
ax.set_ylabel("mean amplitude in the fixed\nmeasurement window (µV)")
ax.set_title("The number a lab reports, measured in the standard window "
             "(▼ = sign flipped)", pad=10)
ax.legend(fontsize=13, loc="upper center")
n_flip = int(e.sign_flipped.sum())
ax.text(.012, .05, f"sign flips in {n_flip}/{len(e)} comparisons", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=14, weight="bold", color="#a8203c",
        bbox=dict(fc="white", ec="none", alpha=.85, pad=2))
_save(fig, "p4_lab_reports.png")
print("done")
