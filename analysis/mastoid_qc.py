"""Mastoid QC numbers for every session (the evidence behind the drop rule in p300_baseline_v3).

Same loading, filter and 256 Hz resample as p300_baseline_v3.preprocess, then per session:
  sd_ratio_M1, sd_ratio_M2   robust SD of each mastoid / median robust SD of the 14 scalp channels,
                             after average reference (what the 'noisy' rule looks at)
  r_pair_rec                 median 2 s-window corr(M1, M2) on the recording reference
  r_pair_car                 the same after average reference (what the drop rule uses)
  status                     from baseline_sessions.csv
Run after p300_baseline_v3.py preprocess (needs baseline_sessions.csv). Writes mastoid_qc.csv.

    python analysis/mastoid_qc.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import multisite_data as md               # noqa: E402
import p300_baseline_v3 as b              # noqa: E402


def _pair_corr(m1, m2):
    w = 2 * b.FS_CLEAN
    return float(np.median([np.corrcoef(m1[k * w:(k + 1) * w], m2[k * w:(k + 1) * w])[0, 1]
                            for k in range(len(m1) // w)]))


def one(sub, ses):
    row = dict(subject=sub, session=ses, site=md.site_of(ses))
    p = md.eeg_prefix(sub, ses)
    if p is None:
        return {**row, "error": "no_eeg"}
    ch, fs0 = md.read_channels(p, b.CH16)
    x = np.vstack([ch[c] for c in b.CH16])
    raw = b._raw(x, fs0, b.CH16)
    raw.filter(b.L_FREQ, b.H_FREQ, method="fir", phase="zero", fir_design="firwin", verbose=False)
    x = b._rs(raw.get_data() * 1e6, fs0, b.FS_CLEAN)
    i1, i2 = b.CH16.index("M1"), b.CH16.index("M2")
    car = x - x.mean(axis=0, keepdims=True)
    sd = b._robust_sd(car)
    med14 = np.median(sd[:14])
    row.update(sd_ratio_M1=sd[i1] / med14, sd_ratio_M2=sd[i2] / med14,
               r_pair_rec=_pair_corr(x[i1], x[i2]), r_pair_car=_pair_corr(car[i1], car[i2]),
               sd_med14_uV=med14)
    return row


if __name__ == "__main__":
    inv = md.inventory()
    pairs = inv[inv.has_eeg][["subject", "session"]].itertuples(index=False, name=None)
    rows = Parallel(n_jobs=8)(delayed(one)(s, e) for s, e in pairs)
    df = pd.DataFrame(rows)
    st = pd.read_csv(os.path.join(HERE, "baseline_sessions.csv"))[["subject", "session", "status"]]
    df = df.merge(st, on=["subject", "session"], how="left")
    df.to_csv(os.path.join(HERE, "mastoid_qc.csv"), index=False)
    print(df.round(2).to_string(index=False))
    print(df.groupby("site")[["sd_ratio_M1", "sd_ratio_M2", "r_pair_rec", "r_pair_car"]].median().round(2))
