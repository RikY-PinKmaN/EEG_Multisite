"""Four-panel mastoid evidence figure (figures/mastoid_evidence.png).

Run after mastoid_qc.py, mastoid_position_check.py and mastoid_reference_check.py.

A  mastoid noise: max(M1, M2) robust SD / median scalp channel, per session, by site (mastoid_qc.csv)
B  M1-M2 windowed correlation after average reference, per session, by site (mastoid_qc.csv)
C  fingerprint assignment of Sheffield "M1"/"M2" vs the positive control (mastoid_position_check.csv)
D  linked-mastoid / 14-channel-average P3b ratio at Pz, raw data (mastoid_reference_check.csv)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = ["SHE", "LDS", "MAN"]
COL = {"SHE": "#C8102E", "LDS": "#1F6FB2", "MAN": "#2E8B57"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

qc = pd.read_csv(os.path.join(HERE, "mastoid_qc.csv"))
qc["sd_max"] = qc[["sd_ratio_M1", "sd_ratio_M2"]].max(axis=1)
qc["dropped"] = qc.status.eq("drop_bad_mastoid")
fp = pd.read_csv(os.path.join(HERE, "mastoid_position_check.csv"))
pc = pd.read_csv(os.path.join(HERE, "mastoid_reference_check.csv"))

fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
rng = np.random.default_rng(0)


def strip(ax, df, col, thr, label, logy=False):
    for i, s in enumerate(SITES):
        g = df[df.site == s]
        x = i + rng.uniform(-0.18, 0.18, len(g))
        ax.scatter(x[~g.dropped.values], g[col][~g.dropped], color=COL[s], s=30, label="kept" if i == 0 else None)
        ax.scatter(x[g.dropped.values], g[col][g.dropped], facecolor="white", edgecolor=COL[s], s=36, lw=1.6,
                   label="dropped (bad mastoid)" if i == 0 else None)
        ax.hlines(g[col].median(), i - 0.3, i + 0.3, color="k", lw=2)
    ax.axhline(thr, color="0.3", ls="--", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{s}\n{int(df[df.site == s].dropped.sum())}/{int((df.site == s).sum())} dropped" for s in SITES])
    ax.set_ylabel(label)
    if logy:
        ax.set_yscale("log")


ax = axes[0, 0]
strip(ax, qc, "sd_max", 2.0, "mastoid noise (worse of M1/M2) ÷ median scalp channel", logy=True)
ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
ax.text(2.45, 2.05, "noisy rule: > 2× (and z > 5)", fontsize=8, color="0.3", ha="right", va="bottom")
ax.set_title("A  Mastoid amplitude relative to scalp (0.1–15 Hz, average ref.)", loc="left", fontsize=10)
ax.set_ylabel("mastoid noise ÷ median scalp channel")
ax.legend(fontsize=8, frameon=False, loc="upper right")

ax = axes[0, 1]
strip(ax, qc, "r_pair_car", 0.2, "median 2 s-window corr(M1, M2), average ref.")
ax.text(2.45, 0.21, "dead-pair rule: r < 0.2", fontsize=8, color="0.3", ha="right", va="bottom")
ax.set_title("B  Do the two mastoids agree with each other?", loc="left", fontsize=10)
ax.set_ylim(-0.4, 1.0)

# C: fingerprint assignments. SHE probes vs LDS/MAN candidates; positive control = LDS probes vs MAN candidates (and v.v.)
ax = axes[1, 0]
def frac(src, probe, feat):
    d = fp[(fp.src == src) & (fp.probe == probe) & (fp.feat == feat)]
    side = ["M1", "T7", "TP7"] if probe.endswith("1") or probe in ("T7", "TP7") else ["M2", "T8", "TP8"]
    return [(d.nearest == c).mean() for c in side], len(d)
bars = [("SHE 'M1'", "SHE", "M1"), ("SHE 'M2'", "SHE", "M2"),
        ("LDS M1 (control)", "LDS", "M1"), ("LDS M2 (control)", "LDS", "M2"),
        ("MAN M1 (control)", "MAN", "M1"), ("MAN M2 (control)", "MAN", "M2")]
cols3 = ["#444444", "#bbbbbb", "#888888"]
names3 = ["mastoid", "T7/T8 (temporal)", "TP7/TP8 (in between)"]
ytick = []
for k, (lab, src, probe) in enumerate(bars):
    f_corr, n1 = frac(src, probe, "corr")
    f_erp, n2 = frac(src, probe, "erp")
    f = (np.array(f_corr) + np.array(f_erp)) / 2
    left = 0
    for j in range(3):
        ax.barh(k, f[j], left=left, color=cols3[j], label=names3[j] if k == 0 else None, height=0.65)
        if f[j] > 0.08:
            ax.text(left + f[j] / 2, k, f"{f[j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if j == 0 else "black")
        left += f[j]
    ytick.append(f"{lab}  (n={n1 + n2})")
ax.set_yticks(range(len(bars))); ax.set_yticklabels(ytick, fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel("share of sessions assigned to each candidate position (signal + ERP fingerprints)", fontsize=9)
ax.set_title("C  Where does each electrode's signal say it sits? (within subject)", loc="left", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.22))

# D: lm / avg14 ratio, raw data
ax = axes[1, 1]
u = pc[pc.usable].copy()
for i, s in enumerate(SITES):
    g = u[u.site == s]
    x = i + rng.uniform(-0.18, 0.18, len(g))
    ax.scatter(x, g.ratio, color=COL[s], s=30)
    ax.hlines(g.ratio.median(), i - 0.3, i + 0.3, color="k", lw=2)
ax.axhline(0.25, color="0.3", ls="--", lw=1)
ax.axhline(1.0, color="0.7", lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{s}\nmedian {u[u.site == s].ratio.median():.2f}\ncollapsed {int((u[u.site == s].ratio < 0.25).sum())}/{len(u[u.site == s])}" for s in SITES], fontsize=9)
ax.set_ylim(-0.3, 4.2)
ax.set_ylabel("P3b at Pz: linked mastoids ÷ 14-ch average")
ax.set_title("D  Linked mastoids vs 14-ch average: P3b ratio at Pz (raw data)", loc="left", fontsize=10)
ax.text(2.45, 0.27, "collapsed: ratio < 0.25", fontsize=8, color="0.3", ha="right", va="bottom")

fig.tight_layout(h_pad=2.6)
fig.savefig(os.path.join(HERE, "figures", "mastoid_evidence.png"), dpi=150)
print("written")
