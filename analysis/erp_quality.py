#!/usr/bin/env python3
"""
erp_quality.py — cross-site ERP quality for the same traveling subjects (analysis #1).

For each subject at each site, extract two robust ERP CORE components on the harmonised
(linked-mastoid, 256 Hz, common-montage) data and compare amplitude / latency / cleanliness:
  * N170  — face trials, mean(PO7,PO8), negative peak 110-200 ms
  * P3b    — target-minus-standard, Pz, positive peak 300-600 ms
Metric of cleanliness = SME (bootstrap SD of the peak); lower = better. If Sheffield's timing/
hardware degrades the data, its amplitude drops and SME rises versus Leeds/Manchester.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from erp_common import (load_reref_resampled, event_samples, epoch, peak, sme_bootstrap,
                        SESSIONS, SITE_OF, prefix_for, STIM)
from jitter_pipeline import ROOT

FIGDIR = os.path.join(ROOT, "analysis", "figures")
SUBJECTS = ["sub-SHE01", "sub-SHE02"]
SITE_COL = {"SHE": "#d1495b", "LDS": "#2e7d95", "MAN": "#e3a72f"}


def collapse(chdict, chans):
    present = [chdict[c] for c in chans if c in chdict]
    return np.mean(present, axis=0) if present else None


def n170_run(prefix):
    ch, fs, ofs = load_reref_resampled(prefix)
    sig = collapse(ch, ["PO7", "PO8"])
    on = event_samples(prefix, STIM["n170"]["face"], ofs)
    t, ep = epoch(sig, fs, on)
    if len(ep) < 10:
        return None
    avg = ep.mean(0)
    amp, lat = peak(avg, t, (110, 200), "neg")
    sme = sme_bootstrap(ep, t, (110, 200), "neg")
    return dict(t=t, avg=avg, n=len(ep), amp=amp, lat=lat, sme=sme)


def p3b_run(prefix):
    ch, fs, ofs = load_reref_resampled(prefix)
    sig = ch.get("Pz")
    if sig is None:
        return None
    tt, ep_t = epoch(sig, fs, event_samples(prefix, STIM["p3b"]["target"], ofs), tmax=0.8)
    ts, ep_s = epoch(sig, fs, event_samples(prefix, STIM["p3b"]["standard"], ofs), tmax=0.8)
    if len(ep_t) < 5 or len(ep_s) < 5:
        return None
    diff = ep_t.mean(0) - ep_s.mean(0)
    amp, lat = peak(diff, tt, (300, 600), "pos")
    # SME of the difference peak: bootstrap both conditions
    rng = np.random.default_rng(0); w = (tt >= 300) & (tt <= 600); peaks = []
    for _ in range(2000):
        d = ep_t[rng.integers(0, len(ep_t), len(ep_t))].mean(0) - ep_s[rng.integers(0, len(ep_s), len(ep_s))].mean(0)
        peaks.append(d[w].max())
    return dict(t=tt, avg=diff, n=len(ep_t), amp=amp, lat=lat, sme=float(np.std(peaks)))


def main():
    comps = {"N170": (n170_run, "neg"), "P3b": (p3b_run, "pos")}
    results = []
    figs = {}
    for cname, (fn, pol) in comps.items():
        fig, axes = plt.subplots(1, len(SUBJECTS), figsize=(6 * len(SUBJECTS), 4.2), sharey=True)
        for ax, sub in zip(np.atleast_1d(axes), SUBJECTS):
            for ses in SESSIONS:
                site = SITE_OF[ses[4:]]
                prefix = prefix_for(sub, ses, "n170" if cname == "N170" else "p3b")
                if not os.path.exists(prefix + "_eeg.vhdr"):
                    continue
                r = fn(prefix)
                if r is None:
                    continue
                results.append(dict(subject=sub, session=ses, site=site, component=cname, **{k: r[k] for k in ("n", "amp", "lat", "sme")}))
                ax.plot(r["t"], r["avg"], color=SITE_COL[site], lw=1.6, alpha=0.9,
                        label=f"{site} ({ses[4:6]})  {r['amp']:.1f}µV/{r['sme']:.2f}SME")
            ax.axvline(0, color="grey", lw=0.6); ax.axhline(0, color="grey", lw=0.6)
            ax.set_title(f"{sub}  —  {cname}"); ax.set_xlabel("time (ms)"); ax.legend(fontsize=7)
            ax.set_xlim(-100, 500 if cname == "N170" else 700)
        axes[0].set_ylabel("µV (linked-mastoid)")
        fig.suptitle(f"Cross-site {cname}: same subjects, 3 sites (amplitude / SME in legend)", fontsize=11)
        fig.tight_layout()
        p = os.path.join(FIGDIR, f"fig7_{cname.lower()}_crosssite.png"); fig.savefig(p, dpi=140); plt.close(fig)
        figs[cname] = p

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(ROOT, "analysis", "erp_quality.csv"), index=False)
    print(df.round(2).to_string(index=False))
    print("\n=== By component × site (mean over subjects/visits) ===")
    print(df.groupby(["component", "site"]).agg(
        amp=("amp", "mean"), lat=("lat", "mean"), sme=("sme", "mean"), n=("n", "mean")
    ).round(2).to_string())
    for c, p in figs.items():
        print("figure:", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
