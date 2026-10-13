#!/usr/bin/env python3
"""
timing_extras.py — analyses #2 and #3.

#2 Effective jitter from the EEG (Woody-style single-trial latency).
   For P3b target trials (largest single-trial SNR) estimate each trial's latency by
   cross-correlating it with the run's average, in 250-600 ms. The SD of single-trial lags is
   the *net* timing jitter surviving into analysis (stimulus timing + neural + noise). A pre-
   stimulus baseline run gives the pure-noise floor. Effective stimulus jitter shows up as
   stimulus-window lag SD being *below* the noise floor (i.e. real locking). Same subject across
   sites controls neural variability, so site differences ~ residual timing.

#3 Photodiode timing audit. Corrected triggers == true light onsets, so their inter-onset
   intervals (IOIs) are what the participant actually saw. The PsychoPy design is identical
   across sites, so any site difference in IOI variability = site presentation fidelity.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from erp_common import (load_reref_resampled, event_samples, epoch, SESSIONS, SITE_OF,
                        prefix_for, STIM)
from jitter_pipeline import ROOT, parse_vhdr, load_events

SUBJECTS = ["sub-SHE01", "sub-SHE02"]


# ---------------------------------------------------------------- #2 effective jitter
def single_trial_lags(ep, t, win, ref=None, maxlag_ms=80):
    w = (t >= win[0]) & (t <= win[1])
    tw = t[w]
    template = (ref if ref is not None else ep.mean(0))[w]
    template = template - template.mean()
    maxlag = int(maxlag_ms / 1000 * 256)
    lags = []
    for e in ep:
        x = e[w] - e[w].mean()
        cc = np.correlate(x, template, mode="full")
        mid = len(x) - 1
        seg = cc[mid - maxlag: mid + maxlag + 1]
        lags.append((np.argmax(seg) - maxlag) / 256 * 1000)  # ms
    return np.array(lags)


def effective_jitter():
    rows = []
    for sub in SUBJECTS:
        for ses in SESSIONS:
            pfx = prefix_for(sub, ses, "p3b")
            if not os.path.exists(pfx + "_eeg.vhdr"):
                continue
            ch, fs, ofs = load_reref_resampled(pfx)
            sig = ch.get("Pz")
            on = event_samples(pfx, STIM["p3b"]["target"], ofs)
            t, ep = epoch(sig, fs, on, tmax=0.8)
            if len(ep) < 15:
                rows.append(dict(subject=sub, session=ses, site=SITE_OF[ses[4:]], n=len(ep),
                                 stim_lag_sd=np.nan, noise_lag_sd=np.nan)); continue
            stim = single_trial_lags(ep, t, (250, 600))
            noise = single_trial_lags(ep, t, (-200, -50))   # pre-stim = noise floor
            rows.append(dict(subject=sub, session=ses, site=SITE_OF[ses[4:]], n=len(ep),
                             stim_lag_sd=float(np.std(stim)), noise_lag_sd=float(np.std(noise))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- #3 photodiode IOI audit
def pd_ioi_audit():
    rows = []
    for sub in SUBJECTS:
        for ses in SESSIONS:
            for task in ("n170", "n2pc", "p3b"):
                pfx = prefix_for(sub, ses, task)
                if not os.path.exists(pfx + "_eeg.vhdr"):
                    continue
                fs = parse_vhdr(pfx + "_eeg.vhdr")["fs"]
                ev = load_events(pfx)
                stim = ev[ev["value_num"] >= 100].sort_values("sample")
                ioi = np.diff(stim["onset"].values) * 1000.0   # ms (onset already in s)
                ioi = ioi[(ioi > 0) & (ioi < 5000)]            # drop block gaps
                rows.append(dict(subject=sub, session=ses, site=SITE_OF[ses[4:]], task=task,
                                 n=len(ioi), ioi_median=np.median(ioi), ioi_iqr=np.subtract(*np.percentile(ioi, [75, 25])),
                                 ioi_sd=np.std(ioi)))
    return pd.DataFrame(rows)


def main():
    ej = effective_jitter()
    print("=== #2 Effective jitter from EEG (P3b single-trial lag SD, ms) ===")
    print(ej.round(2).to_string(index=False))
    print("\n  by site (mean):")
    print(ej.groupby("site")[["stim_lag_sd", "noise_lag_sd"]].mean().round(2).to_string())
    print("  (stim_lag_sd << noise_lag_sd => real locking; site diffs ~ residual timing. "
          "Note: n=2, low single-trial SNR — interpret cautiously.)")

    io = pd_ioi_audit()
    io.to_csv(os.path.join(ROOT, "analysis", "pd_ioi_audit.csv"), index=False)
    print("\n=== #3 Photodiode inter-onset-interval audit (ms) ===")
    print(io.round(1).to_string(index=False))
    print("\n  by site (mean over runs):")
    print(io.groupby("site")[["ioi_median", "ioi_iqr", "ioi_sd"]].mean().round(1).to_string())
    print("  (design is identical across sites, so site differences in IOI spread = presentation fidelity)")


if __name__ == "__main__":
    main()
