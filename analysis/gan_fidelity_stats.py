#!/usr/bin/env python3
"""G1 statistics — per-unit significance for the inherited-lag result.

WHY THIS FILE EXISTS. `gan_fidelity.py` records a `usable` flag based on the synthetic
wave's correlation with the real one at its best lag (`shape_r`). That flag is **not a valid
quality gate**, discovered after the first unit completed:

    peak-r null, spectrum-matched surrogates : median 0.560, p95 0.701
    unit 1 observed                          : 0.625 (p = 0.24), 0.566 (p = 0.48)

Maximising correlation over ~150 candidate lags lets any smooth, low-frequency waveform
reach r ~= 0.56 by chance, so a peak r of 0.6 is unremarkable and a 0.50 threshold sits
*below* the chance median. Ignore `usable`; it is retained in the CSV only so the record of
what was originally computed stays intact.

WHAT REPLACES IT. The lag, not the correlation, is the claim — and the lag has a well-behaved
null. Under the same surrogates the recovered lag is nearly uniform across the +/-300 ms
search range (sd ~187 ms), so a lag landing where the marker error predicts is a specific and
unlikely coincidence rather than a generic alignment. For each unit this computes:

    p_corrected  P(a null lag falls at least as close to zero as the corrected-trained
                 generator's did) — the generator trained on good markers should produce an
                 ERP at true latency.
    p_inherited  P(a null lag-difference falls at least as close to the measured marker
                 error as the observed difference did) — the actual G1 prediction.

Both nulls are built from each unit's own anchor waveform, which is reproducible from the
data, so this runs post hoc on `gan_fidelity.csv` with no retraining and no GPU.

    python3 gan_fidelity_stats.py                    # needs no TensorFlow
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

from jitter_pipeline import ROOT
from raw_markers import build_trial_table
from erp_impact import PARADIGM
import epochs as EP
from gan_lag_check import best_lag, phase_randomise
from gan_fidelity import normalise, diff_wave

CSV = os.path.join(ROOT, "analysis", "gan_fidelity.csv")
OUT = os.path.join(ROOT, "analysis", "gan_fidelity_stats.csv")
N_SURROGATE = 4000


def anchor_for(sub, ses, task, group):
    """The real corrected-marker difference wave — deterministic, so reproducible here."""
    d = EP.build(sub, ses, task, group)
    if d is None:
        return None, None
    chans = list(d["chans"])
    idx = [chans.index(c) for c in PARADIGM[task]["chans"] if c in chans]
    if not idx:
        return None, None
    Xc, _ = normalise(d["X_cor"][:, :, d["core"]])
    return diff_wave(Xc, d["y"].astype(int), idx), float(d["fs"])


def null_lags(anchor, fs, n, rng):
    return np.array([best_lag(anchor, phase_randomise(anchor, rng), fs) for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-surrogate", type=int, default=N_SURROGATE)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    tt = build_trial_table()
    rng = np.random.default_rng(11)
    rows = []

    for _, r in df.iterrows():
        g = tt[(tt.subject == r.subject) & (tt.session == r.session)]
        anchor, fs = anchor_for(r.subject, r.session, r.task, g)
        if anchor is None:
            continue
        nl = null_lags(anchor, fs, args.n_surrogate, rng)
        half = args.n_surrogate // 2
        nd = nl[:half] - nl[half:2 * half]        # null for a difference of two lags

        p_cor = float((np.abs(nl) <= abs(r.synth_lag_corrected_ms)).mean())
        resid = abs(r.synth_lag_diff_ms - r.measured_err_ms)
        p_inh = float((np.abs(nd - r.measured_err_ms) <= resid).mean())

        rows.append(dict(
            subject=r.subject, session=r.session, site=r.site, task=r.task,
            measured_err_ms=r.measured_err_ms,
            synth_lag_corrected_ms=r.synth_lag_corrected_ms,
            synth_lag_raw_ms=r.synth_lag_raw_ms,
            synth_lag_diff_ms=r.synth_lag_diff_ms,
            inherited_fraction=r.inherited_fraction,
            residual_ms=r.synth_lag_diff_ms - r.measured_err_ms,
            shape_r_min=min(r.shape_r_corrected, r.shape_r_raw),
            null_lag_sd=float(nl.std()),
            p_corrected=p_cor, p_inherited=p_inh,
        ))
        print(f"{r.subject} {r.session} {r.task:5s} | measured {r.measured_err_ms:7.2f} "
              f"| synth diff {r.synth_lag_diff_ms:7.2f} ({r.inherited_fraction:+.2f}x) "
              f"| p_cor {p_cor:.4f} p_inh {p_inh:.4f}")

    if not rows:
        print("nothing to analyse")
        return
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(out)} units)")

    print(f"\ninherited fraction : median {out.inherited_fraction.median():+.3f} "
          f"[{out.inherited_fraction.quantile(.25):+.3f}, "
          f"{out.inherited_fraction.quantile(.75):+.3f}] IQR")
    print(f"residual           : mean abs {out.residual_ms.abs().mean():.2f} ms, "
          f"worst {out.residual_ms.abs().max():.2f} ms")
    print(f"null lag sd        : ~{out.null_lag_sd.mean():.0f} ms "
          f"(the lag is near-uniform under the null, which is what makes this a test)")

    # Combine across units. Fisher needs independence; units are 2 subjects x 4 sessions x
    # 2 tasks, so tasks within a session share a recording — report the binomial count too,
    # which makes no independence claim beyond the unit level.
    def combine(sub, col):
        p = sub[col].clip(lower=1.0 / args.n_surrogate).values
        chi2, pf = sps.combine_pvalues(p, method="fisher")
        k = int((sub[col] < 0.05).sum())
        pb = sps.binomtest(k, len(sub), 0.05, alternative="greater").pvalue
        return k, len(sub), pb, chi2, pf

    for name, col in (("corrected-at-true-latency", "p_corrected"),
                      ("inherited the marker error", "p_inherited")):
        k, n, pb, chi2, pf = combine(out, col)
        print(f"\n{name}:")
        print(f"  {k}/{n} units at p < 0.05 (expected {0.05*n:.1f} by chance) "
              f"-> binomial p = {pb:.2e}")
        print(f"  Fisher combined: chi2 = {chi2:.1f}, p = {pf:.2e}")
        if len(out) < 16:
            print("  NOTE: run incomplete — these will change as more units land")

    # Split by paradigm. The N170 is narrow and the P3b broad and lower-SNR, so if the
    # inherited-error effect holds for one and not the other that is a coherent finding
    # about component width rather than a mixed result. Stated as a planned comparison
    # because it was flagged before the units that would decide it had run.
    print("\nBy paradigm:")
    for task, sub in out.groupby("task"):
        line = f"  {task:5s} n={len(sub):2d}"
        for label, col in (("p_cor", "p_corrected"), ("p_inh", "p_inherited")):
            k, n, pb, _, pf = combine(sub, col)
            line += f" | {label} {k}/{n} sig, Fisher p={pf:.2e}"
        line += (f" | inherited median {sub.inherited_fraction.median():+.2f}x"
                 f" | resid mean abs {sub.residual_ms.abs().mean():.1f} ms")
        print(line)

    if "reliable" in df.columns:
        rel = out.merge(df[["subject", "session", "task", "reliable", "worst_split_ms"]],
                        on=["subject", "session", "task"], how="left")
        good = rel[rel.reliable.fillna(False)]
        print(f"\nSplit-half reliable units: {len(good)}/{len(rel)}")
        if len(good):
            k, n, pb, chi2, pf = combine(good, "p_inherited")
            print(f"  inherited, reliable only: {k}/{n} sig, Fisher p = {pf:.2e}, "
                  f"median {good.inherited_fraction.median():+.3f}x")
    else:
        print("\nNo split-half columns in this CSV — it predates the reliability gate "
              "(GAN_ROADMAP.md, 8 Aug outcome note). Interpret per-unit results via "
              "p_inherited only.")


if __name__ == "__main__":
    main()
