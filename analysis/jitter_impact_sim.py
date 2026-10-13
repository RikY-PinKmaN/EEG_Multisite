#!/usr/bin/env python3
"""
jitter_impact_sim.py — why the jitter *type* matters more than its size.

Uses a REAL N170 (Leeds, clean 2048 Hz, PO8, face trials) as the template, then simulates
what averaging over different trigger-jitter distributions does to the averaged ERP. Maps the
three hypotheses to their measurable consequence:

  * constant offset  (per-session / per-participant constant, hyp. a/b) -> pure latency shift, NO amplitude loss
  * trial-to-trial random jitter (Sheffield-like erratic, hyp. c)       -> amplitude ATTENUATION + smearing

So distinguishing the hypotheses is not academic: (a)/(b) are benign and fully correctable, (c) is
the one that actually degrades the neural measures. This is the payoff argument for the whole QC.
"""
from __future__ import annotations
import os, csv
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jitter_pipeline import ROOT, parse_vhdr

FIGDIR = os.path.join(ROOT, "analysis", "figures")


def leeds_n170_template(elec="PO8"):
    pfx = os.path.join(ROOT, "sub-SHE01/ses-02LDS/eeg/sub-SHE01_ses-02LDS_task-n170")
    m = parse_vhdr(pfx + "_eeg.vhdr"); fs = m["fs"]
    idx = {c.strip(): i for i, c in enumerate(m["chans"])}
    mm = np.memmap(pfx + "_eeg.eeg", dtype="<f4", mode="r").reshape(-1, m["nchan"])
    sig = np.asarray(mm[:, idx[elec]], dtype=float); del mm
    b, a = butter(4, 30 / (fs / 2), "low"); sig = filtfilt(b, a, sig)
    rows = list(csv.DictReader(open(pfx + "_events.tsv"), delimiter="\t"))
    face = [int(r["sample"]) for r in rows if r["value"] == "100"]
    pre, post = int(0.2 * fs), int(0.5 * fs)
    t = np.arange(-pre, post) / fs * 1000
    ep = []
    for s in face:
        if s - pre >= 0 and s + post < len(sig):
            e = sig[s - pre:s + post]; ep.append(e - e[:pre].mean())
    return t, np.mean(ep, 0), fs


def jittered_average(t, erp, fs, draw, n=20000, rng=None):
    """Average the template over n trials, each shifted by a jitter sample from draw()."""
    rng = rng or np.random.default_rng(0)
    deltas = draw(rng, n)                       # ms
    acc = np.zeros_like(erp)
    for d in deltas:
        acc += np.interp(t, t + d, erp, left=erp[0], right=erp[-1])
    return acc / n, np.std(deltas)


def measure(t, erp, win=(100, 240)):
    w = (t >= win[0]) & (t <= win[1])
    i = np.argmin(erp[w])
    return erp[w][i], t[w][i]                   # peak amplitude, latency


REGIMES = [
    ("no jitter",                        lambda r, n: np.zeros(n),                       "#444444"),
    ("constant 50 ms  (hyp a/b)",        lambda r, n: np.full(n, 50.0),                  "#2e7d95"),
    ("uniform 40-60 ms  (other sites)",  lambda r, n: r.uniform(40, 60, n),              "#4c9f70"),
    ("uniform 10-90 ms  (Sheffield)",    lambda r, n: r.uniform(10, 90, n),              "#d1495b"),
    ("gaussian SD 20 ms",                lambda r, n: r.normal(50, 20, n),               "#e3a72f"),
]


def main():
    t, erp, fs = leeds_n170_template()
    base_amp, base_lat = measure(t, erp)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    print(f"{'jitter regime':34s} {'jit_SD':>7} {'peak_uV':>8} {'retain%':>8} {'latency_ms':>10} {'shift_ms':>9}")
    rows = []
    for name, draw, color in REGIMES:
        avg, jsd = jittered_average(t, erp, fs, draw)
        amp, lat = measure(t, avg)
        retain = 100 * amp / base_amp
        ax.plot(t, avg, color=color, lw=2, label=f"{name}  ({retain:.0f}% amp)")
        print(f"{name:34s} {jsd:7.1f} {amp:8.2f} {retain:8.0f} {lat:10.0f} {lat-base_lat:9.0f}")
        rows.append((name, jsd, amp, retain, lat, lat - base_lat))
    ax.axvline(base_lat, color="grey", ls=":", lw=1)
    ax.set_xlim(-100, 400); ax.set_xlabel("time (ms)"); ax.set_ylabel("PO8 amplitude (µV)")
    ax.set_title("What each jitter type does to a real N170 (Leeds template)\n"
                 "constant offset → only a latency shift; erratic jitter → amplitude loss + smearing")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig6_jitter_impact.png"); fig.savefig(p, dpi=140); plt.close(fig)
    print("\nfigure:", os.path.relpath(p, ROOT))
    print(f"\nTemplate N170: {base_amp:.1f} uV at {base_lat:.0f} ms (80 face trials, Leeds PO8)")


if __name__ == "__main__":
    main()
