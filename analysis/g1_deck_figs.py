#!/usr/bin/env python3
"""Figures for the G1 presentation deck. Reads the run-2 CSVs, writes to figures/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SITE_COL = {"LDS": "#3b7dd8", "MAN": "#37a66b", "SHE": "#d1495b"}
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

d = pd.read_csv(os.path.join(HERE, "gan_fidelity.csv"))
s = pd.read_csv(os.path.join(HERE, "gan_fidelity_stats.csv"))
d = d.merge(s[["subject", "session", "task", "p_corrected", "p_inherited"]],
            on=["subject", "session", "task"])
d["ctrl_ok"] = (d.synth_lag_corrected_ms.abs() <= 15) & \
               (d.split_disagreement_raw_ms <= 10) & (~d.saturated)


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---- FIG A: the deliverable — raw-trained synthetic lag vs measured marker error ----
fig, ax = plt.subplots(figsize=(7.2, 6.2))
lim = 200
ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=1.4, ls="--", zorder=1,
        label="identity (perfect inheritance)")
ax.axhline(0, color="0.85", lw=1, zorder=0)
bad = d[~d.ctrl_ok]
ax.scatter(bad.measured_err_ms, np.clip(bad.synth_lag_raw_ms, -lim, lim),
           s=90, facecolor="none", edgecolor="0.6", lw=1.6, zorder=3,
           label=f"instrument invalid, excluded (n={len(bad)})")
for _, r in d[d.ctrl_ok].iterrows():
    ax.scatter(r.measured_err_ms, np.clip(r.synth_lag_raw_ms, -lim, lim),
               s=150, color=SITE_COL[r.site], edgecolor="k", lw=0.8, zorder=4)
ok = d[d.ctrl_ok & (d.p_inherited < 0.05)]
b = np.sum(ok.measured_err_ms * ok.synth_lag_raw_ms) / np.sum(ok.measured_err_ms ** 2)
rr = np.corrcoef(ok.measured_err_ms, ok.synth_lag_raw_ms)[0, 1]
xs = np.array([0, 150])
ax.plot(xs, b * xs, color="#111", lw=2.2, zorder=5,
        label=f"fit, n={len(ok)}: slope {b:.2f}, r = {rr:.3f}")
out = d[d.ctrl_ok & (d.p_inherited >= 0.05)]
for _, r in out.iterrows():
    ax.annotate("counterexample\n(survives every check)",
                (r.measured_err_ms, max(r.synth_lag_raw_ms, -lim)),
                textcoords="offset points", xytext=(-40, 44), fontsize=11, color="#b3002d",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#b3002d", lw=1.4))
for k, v in SITE_COL.items():
    ax.scatter([], [], s=150, color=v, edgecolor="k", lw=0.8, label=f"site {k}")
ax.set_xlabel("measured marker error (ms)  —  photodiode ground truth")
ax.set_ylabel("synthetic ERP lag, raw-trained generator (ms)")
ax.set_title("The generator reproduces its training set's timing error", pad=12)
ax.set_xlim(0, 155); ax.set_ylim(-lim - 10, lim)
ax.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
_save(fig, "d1_inheritance_scatter.png")


# ---- FIG B: the control arm — how we know when the instrument works ----
fig, ax = plt.subplots(figsize=(8.4, 4.6))
v = d.sort_values("synth_lag_corrected_ms").reset_index(drop=True)
cols = ["#2a9d5c" if abs(x) <= 15 else "#c44536" for x in v.synth_lag_corrected_ms]
ax.bar(range(len(v)), v.synth_lag_corrected_ms, color=cols, edgecolor="k", lw=0.6)
ax.axhspan(-15, 15, color="#2a9d5c", alpha=0.13, zorder=0)
ax.axhline(0, color="k", lw=1)
ax.text(0.4, 22, "valid zone: control generator returns true latency (±15 ms)",
        fontsize=11, color="#1c6b3f")
ax.set_xlabel("the 16 units, sorted")
ax.set_ylabel("corrected-trained lag (ms)")
ax.set_title("Control arm: ground truth is 0 ms. Half the generators fail it.", pad=10)
ax.set_xticks([])
n_ok = int((v.synth_lag_corrected_ms.abs() <= 15).sum())
ax.text(0.98, 0.06, f"{n_ok}/16 pass", transform=ax.transAxes, ha="right",
        fontsize=13, weight="bold")
_save(fig, "d2_control_arm.png")


# ---- FIG C: split-half stability does NOT imply correctness ----
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.scatter(d.split_disagreement_corrected_ms.clip(upper=120),
           d.synth_lag_corrected_ms.abs(), s=110,
           color=["#2a9d5c" if a <= 15 else "#c44536" for a in d.synth_lag_corrected_ms.abs()],
           edgecolor="k", lw=0.8)
ax.axvline(10, color="0.3", ls="--", lw=1.5)
ax.axhline(15, color="0.3", ls=":", lw=1.5)
ax.text(10.8, 285, "split-half gate\n(passes to the left)", fontsize=11)
ax.annotate("stable AND wrong:\n+301 ms lag, 0.00 ms test-retest",
            (0.5, 300.8), textcoords="offset points", xytext=(60, -46), fontsize=11,
            color="#b3002d", arrowprops=dict(arrowstyle="->", color="#b3002d", lw=1.4))
ax.set_xlabel("split-half test–retest disagreement (ms)")
ax.set_ylabel("|corrected-trained lag| (ms)   —   truth is 0")
ax.set_title("Why reliability ≠ validity", pad=10)
_save(fig, "d3_stability_vs_validity.png")


# ---- FIG D: seed instability, run 1 vs run 2 ----
r1 = pd.read_csv(os.path.join(HERE, "_gan_run1_20260808", "gan_fidelity.csv"))
m = r1.merge(d, on=["subject", "session", "task"], suffixes=("_r1", "_r2"))
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.axhline(0, color="0.85", lw=1); ax.axvline(0, color="0.85", lw=1)
ax.plot([-5, 8], [-5, 8], color="0.5", ls="--", lw=1.3, label="reproducible")
ax.scatter(m.inherited_fraction_r1, m.inherited_fraction_r2, s=130,
           color="#4a4e69", edgecolor="k", lw=0.8, zorder=3)
for _, r in m.iterrows():
    if abs(r.inherited_fraction_r1 - r.inherited_fraction_r2) > 2.0:
        ax.annotate(f"{r.session[4:]} {r.task}",
                    (r.inherited_fraction_r1, r.inherited_fraction_r2),
                    textcoords="offset points", xytext=(9, 5), fontsize=10)
ax.set_xlabel("inherited fraction — run 1 (seed A)")
ax.set_ylabel("inherited fraction — run 2 (seed B)")
ax.set_title("Same data, same code, new seed: per-unit values move", pad=10)
ax.legend(fontsize=11)
_save(fig, "d4_seed_instability.png")


# ---- FIG E: the 2x2 that is the actual result ----
fig, ax = plt.subplots(figsize=(6.4, 4.8))
tab = np.array([[int(((d.ctrl_ok) & (d.p_inherited < 0.05)).sum()),
                 int(((d.ctrl_ok) & (d.p_inherited >= 0.05)).sum())],
                [int(((~d.ctrl_ok) & (d.p_inherited < 0.05)).sum()),
                 int(((~d.ctrl_ok) & (d.p_inherited >= 0.05)).sum())]])
ax.imshow(tab, cmap="Blues", vmin=0, vmax=9)
for i in range(2):
    for j in range(2):
        ax.text(j, i, tab[i, j], ha="center", va="center", fontsize=34,
                color="white" if tab[i, j] > 4 else "#123", weight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["significant\n(pre-registered test)", "n.s."])
ax.set_yticks([0, 1]); ax.set_yticklabels(["instrument\nvalid", "instrument\ninvalid"])
ax.set_title("Two independent criteria agree on 14/16", pad=12)
_save(fig, "d5_agreement.png")
print("done")
