"""Are Sheffield's (Bitbrain) "M1"/"M2" really on the mastoids?

The same subjects were recorded at Leeds and Manchester, where M1/M2 are real mastoid
electrodes and T7/T8, TP7/TP8 exist too. Each channel gets two fingerprints, both under a
common reference (average of the 14 non-mastoid channels shared by every cap):

  corr  correlation of its continuous 0.5-15 Hz signal with each of the 14 shared channels
  erp   its target ERP waveform, 0-0.8 s

A Sheffield channel is assigned to whichever candidate (M1/T7/TP7 on the left, M2/T8/TP8 on
the right) of the same subject at LDS/MAN it is nearest to. Positive control: LDS channels
against MAN candidates and vice versa, where the true positions are known.
Writes mastoid_position_check.csv.

    python analysis/mastoid_position_check.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import multisite_data as md                      # noqa: E402

SHARED = ["F7", "F3", "Fz", "F4", "F8", "FCz", "C3", "C4", "Pz", "POz", "PO7", "PO8", "P7", "P8"]
SIDES = {"L": ["M1", "T7", "TP7"], "R": ["M2", "T8", "TP8"]}
CAND = SIDES["L"] + SIDES["R"]
FS = 80
WIN = 2 * FS
PTP = 150.0
ERP_T = np.arange(0, round(0.8 * FS) + 1)
BL_T = np.arange(-round(0.1 * FS), 1)


def load(sub, ses):
    names = SHARED + (["M1", "M2"] if md.site_of(ses) == "SHE" else CAND)
    ch, fs = md.read_channels(md.eeg_prefix(sub, ses), names)
    names = [n for n in names if n in ch]
    x = np.vstack([ch[n] for n in names])
    sos = butter(4, [0.5, 15], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    g = np.gcd(FS, int(fs))
    x = resample_poly(x, FS // g, int(fs) // g, axis=-1)
    shared = x[[names.index(c) for c in SHARED]]
    x = x - shared.mean(axis=0, keepdims=True)
    return dict(zip(names, x)), fs


def fingerprints(sub, ses):
    sig, fs = load(sub, ses)
    sh = np.vstack([sig[c] for c in SHARED])
    # clean 2 s windows only, so blinks/movement don't dominate the correlations
    n = sh.shape[1] // WIN
    keep = [k for k in range(n) if np.ptp(sh[:, k * WIN:(k + 1) * WIN], axis=1).max() < PTP]
    idx = np.concatenate([np.arange(k * WIN, (k + 1) * WIN) for k in keep])
    tt = md.trial_table(sub, ses)
    s80 = np.round(tt.cor_sample.values[tt.label.values == 2] * FS / fs).astype(int)
    out = {}
    for c, v in sig.items():
        if c in SHARED:
            continue
        corr = np.array([np.corrcoef(v[idx], sh[j, idx])[0, 1] for j in range(len(SHARED))])
        eps = []
        for s in s80:
            if s + BL_T[0] < 0 or s + ERP_T[-1] >= v.size:
                continue
            seg = np.vstack([sh[:, s + BL_T[0]:s + ERP_T[-1] + 1], v[s + BL_T[0]:s + ERP_T[-1] + 1]])
            if np.ptp(seg, axis=1).max() >= PTP:
                continue
            ep = v[s + ERP_T]
            eps.append(ep - v[s + BL_T].mean())
        out[c] = dict(corr=corr, erp=np.mean(eps, axis=0) if eps else None, n_ep=len(eps),
                      sd=v[idx].std(), sd_shared=np.median(sh[:, idx].std(axis=1)))
    return out


def nearest(fp, cands, feat):
    d = {}
    for name, f in cands.items():
        if fp[feat] is None or f[feat] is None:
            continue
        d[name] = np.sqrt(np.mean((fp[feat] - f[feat]) ** 2))
    return d


def main():
    inv = md.inventory()
    inv = inv[inv.has_eeg]
    fps = {}
    for _, r in inv.iterrows():
        try:
            fps[(r.subject, r.session)] = fingerprints(r.subject, r.session)
        except Exception as e:                       # noqa: BLE001
            print(f"skip {r.subject} {r.session}: {e}")
    rows = []
    for sub in sorted({k[0] for k in fps}):
        by_site = {}
        for (s, ses), fp in fps.items():
            if s == sub:
                by_site.setdefault(md.site_of(ses), []).append((ses, fp))
        for src in ("SHE", "LDS", "MAN"):
            ref_sites = [t for t in ("LDS", "MAN") if t != src]
            for ses, fp in by_site.get(src, []):
                for side, names in SIDES.items():
                    probes = [names[0]] if src == "SHE" else names
                    for probe in probes:
                        if probe not in fp:
                            continue
                        for t in ref_sites:
                            for rses, rfp in by_site.get(t, []):
                                cands = {n: rfp[n] for n in names if n in rfp}
                                for feat in ("corr", "erp"):
                                    d = nearest(fp[probe], cands, feat)
                                    if not d:
                                        continue
                                    best = min(d, key=d.get)
                                    rows.append(dict(subject=sub, src=src, session=ses, probe=probe,
                                                     ref_session=rses, feat=feat, nearest=best,
                                                     **{f"d_{k}": v for k, v in d.items()}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "mastoid_position_check.csv"), index=False)

    # amplitude: mastoids sit near the reference region -> lower SD than the shared channels
    amp = [dict(subject=s, session=ses, site=md.site_of(ses), channel=c, sd_ratio=f["sd"] / f["sd_shared"])
           for (s, ses), fp in fps.items() for c, f in fp.items()]
    amp = pd.DataFrame(amp)
    print("\nSD relative to median shared channel (median over sessions):")
    print(amp.groupby(["site", "channel"]).sd_ratio.median().unstack(0).round(2))

    print("\nPositive control (true positions known): fraction assigned to the same-named channel")
    ctl = df[df.src != "SHE"]
    print(ctl.assign(hit=ctl.probe == ctl.nearest).groupby(["feat", "probe"]).hit.mean().unstack(0).round(2))

    print("\nSheffield channels: which candidate are they nearest to? (counts over session pairs)")
    she = df[df.src == "SHE"]
    print(she.groupby(["feat", "probe", "nearest"]).size().unstack(fill_value=0))
    print("\nper subject (erp + corr pooled):")
    print(she.groupby(["subject", "probe", "nearest"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
