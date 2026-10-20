"""Does a linked-mastoid reference behave the same at Sheffield as at Leeds/Manchester?

Every session: the 14 shared channels + M1/M2, 0.1-15 Hz zero-phase, 80 Hz, corrected markers,
epochs -0.1..0.8 s, trials rejected at 100 uV peak-to-peak on Pz. The target-minus-non-target
P300 at Pz (mean 0.3-0.6 s) is measured under two references:

  lm     (M1+M2)/2
  avg14  average of the 14 shared channels (no mastoids involved)

ratio = lm / avg14 depends only on the reference geometry (amplifier gain cancels), so if
Sheffield's "M1"/"M2" sit where Leeds/Manchester's do, its ratio matches theirs within subject.
Writes mastoid_reference_check.csv.

    python analysis/mastoid_reference_check.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import multisite_data as md                      # noqa: E402
from mastoid_position_check import SHARED        # noqa: E402

FS = 80
TS = np.arange(-round(0.1 * FS), round(0.8 * FS) + 1)
BL = TS <= 0
P3 = (TS >= round(0.3 * FS)) & (TS <= round(0.6 * FS))
PTP = 100.0


def session(sub, ses):
    names = SHARED + ["M1", "M2"]
    ch, fs = md.read_channels(md.eeg_prefix(sub, ses), names)
    x = np.vstack([ch[n] for n in names])
    x = sosfiltfilt(butter(4, [0.1, 15], btype="band", fs=fs, output="sos"), x, axis=-1)
    g = np.gcd(FS, int(fs))
    x = resample_poly(x, FS // g, int(fs) // g, axis=-1)
    pz = x[names.index("Pz")]
    refs = {"lm": pz - x[[names.index("M1"), names.index("M2")]].mean(axis=0),
            "avg14": pz - x[:len(SHARED)].mean(axis=0)}
    tt = md.trial_table(sub, ses)
    s80 = np.round(tt.cor_sample.values * FS / fs).astype(int)
    ok = (s80 + TS[0] >= 0) & (s80 + TS[-1] < x.shape[1])
    s80, lab = s80[ok], tt.label.values[ok]
    ep = {r: v[s80[:, None] + TS] for r, v in refs.items()}
    ep = {r: e - e[:, BL].mean(axis=1, keepdims=True) for r, e in ep.items()}
    good = np.ones(len(s80), bool)
    for e in ep.values():                              # same trials under both references
        good &= np.ptp(e, axis=1) < PTP
    row = dict(subject=sub, session=ses, site=md.site_of(ses),
               n_t=int((good & (lab == 2)).sum()), n_nt=int((good & (lab == 1)).sum()))
    for r, e in ep.items():
        d = e[good & (lab == 2)].mean(axis=0) - e[good & (lab == 1)].mean(axis=0)
        row[f"p3_{r}"] = d[P3].mean()
        row[f"lat_{r}"] = TS[P3][np.argmax(d[P3])] / FS * 1000
    row["ratio"] = row["p3_lm"] / row["p3_avg14"]
    return row


def main():
    inv = md.inventory()
    inv = inv[inv.has_eeg]
    rows = []
    for _, r in inv.iterrows():
        try:
            rows.append(session(r.subject, r.session))
        except Exception as e:                       # noqa: BLE001
            print(f"skip {r.subject} {r.session}: {e}")
    df = pd.DataFrame(rows)
    # at least 15 targets, and a P300 that is clearly there under avg14 (ratio is unstable near 0)
    df["usable"] = (df.n_t >= 15) & (df.p3_avg14 > 1.0)
    df.to_csv(os.path.join(HERE, "mastoid_reference_check.csv"), index=False)
    print(df.round(2).to_string(index=False))

    u = df[df.usable]
    print(f"\nusable sessions: {len(u)}/{len(df)}")
    print("\nper site, median over sessions:")
    print(u.groupby("site")[["p3_lm", "p3_avg14", "ratio", "lat_lm", "lat_avg14", "n_t"]].median().round(2))

    subj = u.groupby(["subject", "site"])[["p3_lm", "p3_avg14", "ratio"]].mean().unstack("site")
    print("\nper subject (mean over sessions):")
    print(subj.round(2).to_string())
    print("\nwithin-subject paired differences (median, Wilcoxon p, n subjects):")
    for m in ("ratio", "p3_lm", "p3_avg14"):
        for a, b in (("SHE", "LDS"), ("SHE", "MAN"), ("LDS", "MAN")):
            d = (subj[(m, a)] - subj[(m, b)]).dropna()
            p = wilcoxon(d).pvalue if len(d) >= 5 else np.nan
            print(f"  {m:9s} {a}-{b}: {d.median():+.2f}  p={p:.3f}  n={len(d)}")


if __name__ == "__main__":
    main()
