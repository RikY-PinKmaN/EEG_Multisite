#!/usr/bin/env python3
"""Actual ERP waveforms and decoding-AUC figures for the timing presentation."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raw_markers import build_trial_table
from erp_impact import PARADIGM
import epochs as EP

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SITE_COL = {"LDS": "#3b7dd8", "MAN": "#37a66b", "SHE": "#d1495b"}
SITE_OF = {"ses-01SHE": "SHE", "ses-02LDS": "LDS", "ses-03SHE": "SHE", "ses-04MAN": "MAN"}
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.titlesize": 16})


SMOOTH_MS = 24.0   # display-only boxcar; stated in every caption that uses it


def smooth(x, fs=256.0, ms=SMOOTH_MS):
    """Boxcar smoother for display only. Symmetric, so it cannot shift latency."""
    k = max(1, int(round(ms / 1000.0 * fs)) | 1)
    w = np.ones(k) / k
    return np.convolve(np.pad(x, k // 2, mode="edge"), w, mode="valid")[:len(x)]


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---------- P5: the actual ERP waveforms, corrected vs uncorrected ----------
tt = build_trial_table()
waves = {}          # (task, site) -> list of (t, corrected, raw)
for (sub, ses), g in tt.groupby(["subject", "session"]):
    for task in ("n170", "p3b"):
        d = EP.build(sub, ses, task, g)
        if d is None:
            continue
        spec = PARADIGM[task]
        chans = list(d["chans"])
        idx = [chans.index(c) for c in spec["chans"] if c in chans]
        core = d["core"]
        t = d["t"][core]
        y = d["y"]
        out = []
        for key in ("X_cor", "X_raw"):
            X = d[key][:, :, core][:, idx, :]
            out.append(X[y == 0].mean(0).mean(0) - X[y == 1].mean(0).mean(0))
        waves.setdefault((task, SITE_OF[ses]), []).append((t, out[0], out[1]))

fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4))
for ax, task in zip(axes, ("n170", "p3b")):
    spec = PARADIGM[task]
    ax.axvspan(*spec["window"], color="0.85", alpha=.45, zorder=0)
    for site in ("LDS", "MAN", "SHE"):
        v = waves.get((task, site))
        if not v:
            continue
        t = v[0][0]
        cor = np.mean([w[1] for w in v], axis=0)
        raw = np.mean([w[2] for w in v], axis=0)
        ax.plot(t, smooth(cor), color=SITE_COL[site], lw=2.6, label=f"{site} — true onset")
        ax.plot(t, smooth(raw), color=SITE_COL[site], lw=1.9, ls="--", alpha=.9,
                label=f"{site} — logged trigger")
    ax.axhline(0, color="k", lw=.9)
    ax.axvline(0, color="k", lw=.9)
    ax.set_xlim(-200, 800)
    ax.set_xlabel("time after the picture appeared (ms)")
    ax.set_ylabel("difference wave (µV)")
    ax.set_title(f"{task.upper()}  —  shaded band = where a lab measures", pad=10)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=9.5, ncol=2, loc="lower left")
fig.suptitle("The same recording, cut at the true onset (solid) and at the logged trigger (dashed)",
             fontsize=17, y=1.03)
axes[1].text(.99, .02, f"{SMOOTH_MS:.0f} ms symmetric smoothing, display only",
             transform=axes[1].transAxes, ha="right", fontsize=10, color="0.4")
_save(fig, "p5_erp_waveforms.png")


# ---------- P6: decoding performance (area under the ROC curve) ----------
c = pd.read_csv(os.path.join(HERE, "classify_summary.csv"))
order = ["10", "20", "40", "full"]
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharey=True)
for ax, task in zip(axes, ("n170", "p3b")):
    s = c[c.task == task]
    ns = [n for n in order if n in set(s.n_req)]
    x = np.arange(len(ns))
    ceil = [s[(s.n_req == n) & (s.arm == "none")].auc_ceiling.mean() for n in ns]
    base = [s[(s.n_req == n) & (s.arm == "none")].auc_raw.mean() for n in ns]
    best = [s[(s.n_req == n) & (s.arm == "slide_broad")].auc_raw.mean() for n in ns]
    ax.axhline(0.5, color="0.45", ls=":", lw=1.5)
    ax.text(.985, .5, "chance", transform=ax.get_yaxis_transform(), fontsize=10.5,
            color="0.35", ha="right", va="bottom", clip_on=True)
    ax.fill_between(x, base, ceil, color="#d1495b", alpha=.12)
    ax.plot(x, ceil, "-o", color="#2f6fb8", lw=2.6, ms=9, label="markers corrected (best possible)")
    ax.plot(x, base, "-o", color="#c94f5e", lw=2.6, ms=9, label="markers uncorrected")
    ax.plot(x, best, "s--", color="#37a66b", lw=2.4, ms=8,
            label="uncorrected + shift augmentation")
    ax.set_xticks(x); ax.set_xticklabels([n if n != "full" else "all" for n in ns])
    ax.set_xlabel("training trials per class")
    ax.set_title(task.upper(), pad=10)
    ax.grid(alpha=.22)
    mid = len(ns) - 1
    ax.annotate("", xy=(mid, ceil[-1]), xytext=(mid, base[-1]),
                arrowprops=dict(arrowstyle="<->", color="#a8203c", lw=1.8))
    ax.text(mid - .07, (ceil[-1] + base[-1]) / 2, "cost of bad\ntiming", ha="right",
            fontsize=11.5, color="#a8203c", weight="bold")
axes[0].set_ylabel("AUC  (0.5 = coin flip, 1.0 = perfect)")
axes[0].legend(fontsize=10.5, loc="upper left")
fig.suptitle("Can a classifier still tell the two conditions apart when the markers are wrong?",
             fontsize=17, y=1.02)
_save(fig, "p6_auc.png")
print("done")
