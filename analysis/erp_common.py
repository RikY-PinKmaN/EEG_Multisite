#!/usr/bin/env python3
"""
erp_common.py — shared EEG epoch extraction harmonised across the three sites.

Sites differ in amplifier, montage, sampling rate and reference (Sheffield=right ear,
Leeds=CMS/DRL, Manchester=Cz). To compare fairly we:
  * keep the channel subset common to all sites (the 16-ch Sheffield montage),
  * re-reference every site to LINKED MASTOIDS (mean of M1, M2),
  * band-pass 0.1-30 Hz and resample to a common 256 Hz,
  * epoch with identical timing/baseline.
"""
from __future__ import annotations
import csv, os, re, math
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from jitter_pipeline import parse_vhdr, ROOT

COMMON = ["F7", "F3", "Fz", "F4", "F8", "FCz", "C3", "C4", "Pz", "POz", "PO7", "PO8", "P7", "P8"]
MASTOIDS = ["M1", "M2"]
TARGET_FS = 256.0

STIM = {  # condition -> trigger value(s)
    "n170": {"face": [100], "car": [200], "face_scr": [300], "car_scr": [400]},
    "p3b":  {"target": [111], "standard": [222]},
    "n2pc": {"target_l": [111], "target_r": [222]},
}


def _channel_resolutions(vhdr_path):
    """Map channel name -> resolution (µV per stored unit); BrainVision Ch<n>=name,ref,res,unit."""
    res = {}
    for line in open(vhdr_path):
        mm = re.match(r"Ch\d+=([^,]+),[^,]*,([^,]*),", line)
        if mm:
            res[mm.group(1).strip()] = float(mm.group(2)) if mm.group(2).strip() else 1.0
    return res


def _read_named_channels(prefix, names):
    m = parse_vhdr(prefix + "_eeg.vhdr"); fs = m["fs"]
    idx = {c.strip(): i for i, c in enumerate(m["chans"])}
    res = _channel_resolutions(prefix + "_eeg.vhdr")
    mm = np.memmap(prefix + "_eeg.eeg", dtype="<f4", mode="r").reshape(-1, m["nchan"])
    out = {}
    for nm in names:
        if nm in idx:
            out[nm] = np.asarray(mm[:, idx[nm]], dtype=float) * res.get(nm, 1.0)  # -> µV
    del mm
    return out, fs


def _resample(x, fs):
    if abs(fs - TARGET_FS) < 1:
        return x
    g = math.gcd(int(TARGET_FS), int(round(fs)))
    return resample_poly(x, int(TARGET_FS) // g, int(round(fs)) // g)


def load_reref_resampled(prefix, channels=COMMON, hp=0.5, lp=30.0):
    """Return {chan: signal@256Hz, linked-mastoid ref, band-passed}, fs=256, orig_fs.

    Order: read (µV-scaled) -> resample to 256 Hz -> linked-mastoid reref -> sos band-pass
    (resample-first + second-order sections avoids the numerical blow-up of a 0.1-0.5 Hz
    Butterworth designed at 2048 Hz)."""
    raw, fs = _read_named_channels(prefix, channels + MASTOIDS)
    raw = {k: _resample(v, fs) for k, v in raw.items()}
    mast = np.mean([raw[m] for m in MASTOIDS if m in raw], axis=0)
    sos = butter(4, [hp / (TARGET_FS / 2), lp / (TARGET_FS / 2)], "band", output="sos")
    out = {ch: sosfiltfilt(sos, raw[ch] - mast) for ch in channels if ch in raw}
    return out, TARGET_FS, fs


def event_samples(prefix, values, orig_fs, target_fs=TARGET_FS):
    """Stimulus samples for given trigger values, rescaled to the resampled clock."""
    rows = list(csv.DictReader(open(prefix + "_events.tsv"), delimiter="\t"))
    vals = set(str(v) for v in values)
    s = [int(r["sample"]) for r in rows if r["value"] in vals]
    return (np.array(s) * target_fs / orig_fs).round().astype(int)


def epoch(sig, fs, onsets, tmin=-0.2, tmax=0.5, baseline=(-0.2, 0.0), reject_p2p=120.0):
    pre, post = int(round(-tmin * fs)), int(round(tmax * fs))
    t = np.arange(-pre, post) / fs * 1000.0
    bmask = (t >= baseline[0] * 1000) & (t < baseline[1] * 1000)
    ep = []
    for s in onsets:
        if s - pre >= 0 and s + post < len(sig):
            e = sig[s - pre:s + post] - sig[s - pre:s + post][bmask].mean()
            if np.ptp(e) <= reject_p2p:
                ep.append(e)
    return t, np.array(ep)


def sme_bootstrap(ep, t, win, polarity="neg", n_boot=2000, rng=None):
    """Standardised measurement error of the peak amplitude (bootstrap SD). Lower = cleaner."""
    rng = rng or np.random.default_rng(0)
    w = (t >= win[0]) & (t <= win[1])
    peaks = []
    N = len(ep)
    for _ in range(n_boot):
        avg = ep[rng.integers(0, N, N)].mean(0)[w]
        peaks.append(avg.min() if polarity == "neg" else avg.max())
    return float(np.std(peaks))


def peak(ep_avg, t, win, polarity="neg"):
    w = (t >= win[0]) & (t <= win[1])
    seg = ep_avg[w]
    i = np.argmin(seg) if polarity == "neg" else np.argmax(seg)
    return float(seg[i]), float(t[w][i])


SITE_OF = {"01SHE": "SHE", "02LDS": "LDS", "03SHE": "SHE", "04MAN": "MAN"}
SESSIONS = ["ses-01SHE", "ses-02LDS", "ses-03SHE", "ses-04MAN"]


def prefix_for(subject, session, task):
    return os.path.join(ROOT, subject, session, "eeg", f"{subject}_{session}_task-{task}")
