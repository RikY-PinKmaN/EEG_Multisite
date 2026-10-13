#!/usr/bin/env python3
"""
offset_recovery.py — can the marker offset be recovered from the EEG WITHOUT labels
and WITHOUT a photodiode on the rig being corrected?

VERDICT: NO. See section 5 below and OFFSET_RECOVERY.md. Kept as the archived negative
result and the control that kills it — a constant guess beats the estimator.

Why the question was worth asking
---------------------------------
Section 2 of RESULTS.md recovers the marker error from brain signal to within 1.2 ms —
but from a *difference wave* on the SAME session, which needs both condition labels and
a photodiode-corrected copy of the very run being corrected. A lab with no photodiode
has neither. If a label-free, cross-session estimate worked, the markers could simply be
repaired and the clean classifier used (0.866 AUC) instead of buying shift-invariance
with augmentation (0.690 AUC, and -0.15 AUC on correctly-timed data).

The estimator
-------------
Every trial has a visual onset, so the average over ALL trials of a run — ignoring
condition entirely — is a visual evoked response. Cross-correlate it against a label-free
template from a DIFFERENT session whose timing is known good; the lag is the offset.

    target : epoched at RAW markers   (the no-photodiode condition; what a lab has)
    donor  : epoched at CORRECTED markers, different session (the published template)
    lag(donor, target) ~= target's marker error

Leakage control
---------------
The donor is always a different session, so no run is ever aligned against its own
timing. `rel_site` splits pairs into the cases a real lab faces: a different site
entirely (the deployment case) versus a different session at the same site.

The controls that decide it
--------------------------
Two baselines, because an estimator this weak must be shown to beat something:

  naive         assume zero offset — what a lab does today. mean |err| = 70.3 ms.
  prior_const   guess the SAME constant for every session, the midpoint of the same
                Uniform(0, 150) ms prior that section 3 already justified for
                `slide_broad`. Uses no EEG at all.

`prior_const` is the one that matters. If the EEG-based estimator cannot beat a fixed
number, it is extracting nothing from the brain signal.

What is measured
----------------
  1. grid            region x smoothing x lag-bound, so the negative result is not an
                     artefact of one arbitrary configuration
  2. labelled arm    same pairs via the difference wave — prices the labels
  3. trial sweep     how few trials the estimate survives on
  4. end-to-end      apply the recovered lag by re-cutting inside the pad, re-measure
                     the canonical-window amplitude section 2 showed collapsing
  5. baselines       the estimator against naive and prior_const

Writes: offset_recovery.csv, offset_recovery_sweep.csv, offset_recovery_repair.csv,
        offset_recovery_baselines.csv
"""
from __future__ import annotations
import os
import itertools
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from jitter_pipeline import ROOT
from raw_markers import build_trial_table, SITE_OF
import epochs as EP
from augment import shift_crop

OUT = os.path.join(ROOT, "analysis")

# Label-free alignment rides on the visual onset response, which is posterior. Both
# regions are reported: POSTERIOR was the pre-specified choice, ALL14 is the whole
# harmonised montage. Reporting both is the point — a result that only survives in one
# configuration is not a result.
POSTERIOR = ["PO7", "PO8", "P7", "P8", "POz", "Pz"]
SEARCH_MS = 300.0                       # widest lag searched; caps the epoch trim
BOUNDS = {"unbounded": (-300.0, 300.0),  # no prior knowledge of sign or size
          "prior_0_150": (0.0, 150.0)}   # section 3's Uniform(0,150), pre-justified
SMOOTH_GRID = [None, 20.0, 12.0, 8.0]    # None = the cached 0.5-30 Hz band
PRIOR_RANGE = (0.0, 150.0)
TASKS = ["n170", "p3b"]
WINDOW = {"n170": (130, 200), "p3b": (300, 600)}
SWEEP_N = [10, 20, 40, 80, 160, None]
N_BOOT = 25
FS = 256.0


def _znorm(x):
    """Zero-mean, unit-norm along the last axis; flat rows are left at zero."""
    x = x - x.mean(-1, keepdims=True)
    nrm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(nrm == 0.0, 1.0, nrm)


def _smooth(X, hz, fs=FS):
    if hz is None:
        return X
    return sosfiltfilt(butter(4, hz / (fs / 2), "low", output="sos"), X, axis=-1)


def multichan_lag(A, B, fs, lo_ms=-SEARCH_MS, hi_ms=SEARCH_MS):
    """
    (lag_ms, mean_r) for the lag in [lo_ms, hi_ms] that best aligns B onto A.

    Positive lag = the response appears LATER in B than in A, matching the convention in
    `erp_impact.xcorr_lag` and `augment.shift_crop`. Shifting is done by SLICING, not
    np.roll, so no sample ever wraps from one end of the epoch to the other. The trim is
    fixed at SEARCH_MS regardless of the bound, so every cell of the grid compares the
    same number of samples.
    """
    assert A.shape == B.shape, f"template/target shape mismatch {A.shape} vs {B.shape}"
    n = A.shape[-1]
    m = int(round(SEARCH_MS / 1000.0 * fs))
    assert n > 3 * m, f"epoch too short ({n} samples) for a +-{SEARCH_MS} ms search"
    assert lo_ms >= -SEARCH_MS and hi_ms <= SEARCH_MS, "bound exceeds the trimmed margin"
    a = _znorm(A[:, m:n - m])
    ks = np.arange(int(round(lo_ms / 1000.0 * fs)), int(round(hi_ms / 1000.0 * fs)) + 1)
    score = np.array([float(np.mean(np.sum(a * _znorm(B[:, m + k:n - m + k]), axis=-1)))
                      for k in ks])
    j = int(np.argmax(score))
    return float(ks[j] / fs * 1000.0), float(score[j])


def label_free_evoked(d, marker, ch_idx, n=None, rng=None):
    """
    Average over ALL trials of a run, ignoring `d['y']` entirely -> (n_chan, n_time).

    `n` takes a random subset without replacement, to ask how many trials the estimate
    needs. Returns None if the run has fewer than n trials.
    """
    X = d["X_raw"] if marker == "raw" else d["X_cor"]
    if n is not None:
        if len(X) < n:
            return None
        X = X[rng.choice(len(X), n, replace=False)]
    return X[:, ch_idx, :].mean(0)


def diff_wave(d, marker, ch_idx):
    """Condition-contrast wave -> (n_chan, n_time). Needs labels; the thing to beat."""
    X = (d["X_raw"] if marker == "raw" else d["X_cor"])[:, ch_idx, :]
    y = d["y"]
    if len(np.unique(y)) < 2:
        return None
    return X[y == 0].mean(0) - X[y == 1].mean(0)


def _common_idx(d_a, d_b, region):
    """Row indices of the channels present in BOTH runs, in a fixed shared order."""
    ca, cb = list(d_a["chans"]), list(d_b["chans"])
    names = [c for c in region if c in ca and c in cb]
    return names, [ca.index(c) for c in names], [cb.index(c) for c in names]


def load_runs():
    """{(subject, session, task): epoch dict} for every run with a cached epoch set."""
    tt = build_trial_table()
    runs = {}
    for (sub, ses), g in tt.groupby(["subject", "session"]):
        for task in TASKS:
            d = EP.build(sub, ses, task, g)
            if d is not None:
                runs[(sub, ses, task)] = d
    return runs


def _pairs(runs, same_task_only=False):
    """Every (target, donor) ordered pair with donor session != target session."""
    for tgt, don in itertools.permutations(runs, 2):
        if tgt[1] == don[1]:
            continue
        if same_task_only and tgt[2] != don[2]:
            continue
        yield tgt, don


def experiment_grid(runs):
    """Label-free recovery across region x smoothing x lag-bound."""
    rows = []
    for region, rlabel in ((POSTERIOR, "posterior"), (list(EP.COMMON), "all14")):
        for hz in SMOOTH_GRID:
            for bname, (lo, hi) in BOUNDS.items():
                for tgt, don in _pairs(runs):
                    d_t, d_d = runs[tgt], runs[don]
                    names, it, idd = _common_idx(d_t, d_d, region)
                    if len(names) < 2:
                        continue
                    A = _smooth(label_free_evoked(d_d, "cor", idd), hz)
                    B = _smooth(label_free_evoked(d_t, "raw", it), hz)
                    lag, r = multichan_lag(A, B, float(d_t["fs"]), lo, hi)
                    measured = float(d_t["err_ms"].mean())
                    rows.append(dict(
                        region=rlabel, smooth_hz=hz or 30.0, bound=bname,
                        target_subject=tgt[0], target_session=tgt[1], target_task=tgt[2],
                        target_site=SITE_OF[tgt[1]],
                        donor_subject=don[0], donor_session=don[1], donor_task=don[2],
                        donor_site=SITE_OF[don[1]],
                        rel_site="same_site" if SITE_OF[tgt[1]] == SITE_OF[don[1]] else "cross_site",
                        rel_subject="same_subj" if tgt[0] == don[0] else "cross_subj",
                        rel_task="same_task" if tgt[2] == don[2] else "cross_task",
                        n_chan=len(names), measured_err_ms=measured,
                        recovered_ms=lag, peak_r=r, abs_error_ms=abs(lag - measured)))
    return pd.DataFrame(rows)


def experiment_labelled(runs, region=POSTERIOR):
    """The same pairs via the label-dependent difference wave. Prices the labels."""
    rows = []
    for tgt, don in _pairs(runs, same_task_only=True):
        d_t, d_d = runs[tgt], runs[don]
        names, it, idd = _common_idx(d_t, d_d, region)
        if len(names) < 2:
            continue
        A, B = diff_wave(d_d, "cor", idd), diff_wave(d_t, "raw", it)
        if A is None or B is None:
            continue
        for bname, (lo, hi) in BOUNDS.items():
            lag, r = multichan_lag(A, B, float(d_t["fs"]), lo, hi)
            measured = float(d_t["err_ms"].mean())
            rows.append(dict(estimator="difference_wave(labelled)", bound=bname,
                             target_session=tgt[1], target_task=tgt[2], donor_session=don[1],
                             measured_err_ms=measured, recovered_ms=lag, peak_r=r,
                             abs_error_ms=abs(lag - measured)))
    return pd.DataFrame(rows)


def experiment_within_session(runs, region=POSTERIOR):
    """
    The upper bound: align a run against its OWN corrected epochs.

    Not deployable (it needs the photodiode-corrected copy of the run being corrected),
    but it isolates where the cross-session estimator loses its accuracy — if this is
    near-perfect then the label-free average is fine and the template transfer is not.
    """
    rows = []
    for (sub, ses, task), d in runs.items():
        names = [c for c in region if c in list(d["chans"])]
        idx = [list(d["chans"]).index(c) for c in names]
        lag, r = multichan_lag(label_free_evoked(d, "cor", idx),
                               label_free_evoked(d, "raw", idx), float(d["fs"]))
        measured = float(d["err_ms"].mean())
        rows.append(dict(subject=sub, session=ses, task=task, site=SITE_OF[ses],
                         measured_err_ms=measured, recovered_ms=lag, peak_r=r,
                         abs_error_ms=abs(lag - measured)))
    return pd.DataFrame(rows)


def experiment_sweep(runs, region=POSTERIOR, bound="prior_0_150", seed=0):
    """How many target trials the label-free estimate needs. Donor always uses full N."""
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[bound]
    rows = []
    for tgt, don in _pairs(runs):
        d_t, d_d = runs[tgt], runs[don]
        names, it, idd = _common_idx(d_t, d_d, region)
        if len(names) < 2:
            continue
        A = label_free_evoked(d_d, "cor", idd)
        measured = float(d_t["err_ms"].mean())
        for n in SWEEP_N:
            for rep in range(1 if n is None else N_BOOT):
                B = label_free_evoked(d_t, "raw", it, n=n, rng=rng)
                if B is None:
                    break
                lag, r = multichan_lag(A, B, float(d_t["fs"]), lo, hi)
                rows.append(dict(
                    requested_n=-1 if n is None else n, rep=rep,
                    target_session=tgt[1], target_task=tgt[2], target_site=SITE_OF[tgt[1]],
                    rel_site="same_site" if SITE_OF[tgt[1]] == SITE_OF[don[1]] else "cross_site",
                    measured_err_ms=measured, recovered_ms=lag, peak_r=r,
                    abs_error_ms=abs(lag - measured)))
    return pd.DataFrame(rows)


def experiment_baselines(runs, grid):
    """
    The estimator against two EEG-free baselines.

    prior_const guesses the midpoint of Uniform(0, 150) for EVERY session and touches no
    brain signal. If it wins, the estimator is extracting nothing.
    """
    meas = np.array([float(runs[k]["err_ms"].mean()) for k in sorted(runs)])
    mid = float(np.mean(PRIOR_RANGE))
    rows = [dict(estimator="naive (assume 0 ms)", uses_eeg=False,
                 mean_abs_err=float(np.abs(meas).mean()),
                 median_abs_err=float(np.median(np.abs(meas))),
                 p90_abs_err=float(np.quantile(np.abs(meas), .9)),
                 within_5ms=float(100 * (np.abs(meas) <= 5).mean()),
                 within_10ms=float(100 * (np.abs(meas) <= 10).mean())),
            dict(estimator=f"prior_const (guess {mid:.0f} ms)", uses_eeg=False,
                 mean_abs_err=float(np.abs(meas - mid).mean()),
                 median_abs_err=float(np.median(np.abs(meas - mid))),
                 p90_abs_err=float(np.quantile(np.abs(meas - mid), .9)),
                 within_5ms=float(100 * (np.abs(meas - mid) <= 5).mean()),
                 within_10ms=float(100 * (np.abs(meas - mid) <= 10).mean()))]
    # the estimator's best grid cell, chosen on mean |error| — generous to the estimator,
    # since picking the winner post hoc can only flatter it
    cells = grid.groupby(["region", "smooth_hz", "bound"]).abs_error_ms.mean()
    r_, s_, b_ = cells.idxmin()
    best = grid[(grid.region == r_) & (grid.smooth_hz == s_) & (grid.bound == b_)]
    rows.append(dict(estimator=f"label-free EEG (best cell: {r_}, {s_:.0f} Hz, {b_})",
                     uses_eeg=True,
                     mean_abs_err=float(best.abs_error_ms.mean()),
                     median_abs_err=float(best.abs_error_ms.median()),
                     p90_abs_err=float(best.abs_error_ms.quantile(.9)),
                     within_5ms=float(100 * (best.abs_error_ms <= 5).mean()),
                     within_10ms=float(100 * (best.abs_error_ms <= 10).mean())))
    return pd.DataFrame(rows), (r_, s_, b_)


def experiment_repair(runs, grid, cell, region=POSTERIOR):
    """
    End-to-end: repair the raw markers with the recovered lag and re-measure the damage.

    Section 2's damage metric is the canonical-window mean amplitude, which collapses and
    flips sign under raw markers. The epochs carry PAD_MS of real signal, so applying
    `-lag` via `shift_crop` re-cuts genuine data rather than zero-padding an edge. The lag
    used is the MEDIAN over cross-site donors in the estimator's best cell — the
    deployment case, where no template from the target's own site exists.

    Both repairs are applied to every run: the EEG-derived lag, and the EEG-free constant
    from `PRIOR_RANGE`. Without the constant arm this section would read as a partial
    success for the estimator when the prior is doing the work.
    """
    r_, s_, b_ = cell
    cs = grid[(grid.region == r_) & (grid.smooth_hz == s_) & (grid.bound == b_)
              & (grid.rel_site == "cross_site")]
    rows = []
    for (sub, ses, task), d in runs.items():
        sel = cs[(cs.target_subject == sub) & (cs.target_session == ses)
                 & (cs.target_task == task)]
        if sel.empty:
            continue
        lag = float(np.median(sel.recovered_ms))
        names = [c for c in region if c in list(d["chans"])]
        idx = [list(d["chans"]).index(c) for c in names]
        t, y = d["t"], d["y"]
        lo, hi = -200.0, 950.0
        tw = t[(t >= lo) & (t < hi)]
        win = (tw >= WINDOW[task][0]) & (tw <= WINDOW[task][1])

        def fixed_amp(X, shift_ms):
            c = shift_crop(X[:, idx, :], t, lo, hi, shift_ms=np.full(len(X), shift_ms))
            r = c.mean(1)                                    # pool region -> (trials, time)
            return float((r[y == 0].mean(0) - r[y == 1].mean(0))[win].mean())

        a_cor, a_raw = fixed_amp(d["X_cor"], 0.0), fixed_amp(d["X_raw"], 0.0)
        measured = float(d["err_ms"].mean())
        for arm, shift in (("eeg_lag", lag), ("prior_const", float(np.mean(PRIOR_RANGE)))):
            a_fix = fixed_amp(d["X_raw"], -shift)            # undo the estimated offset
            rows.append(dict(
                repair=arm, subject=sub, session=ses, task=task, site=SITE_OF[ses],
                measured_err_ms=measured, applied_ms=shift,
                residual_ms=abs(shift - measured),
                amp_corrected_uv=a_cor, amp_raw_uv=a_raw, amp_repaired_uv=a_fix,
                raw_sign_flipped=bool(np.sign(a_raw) != np.sign(a_cor)),
                repaired_sign_flipped=bool(np.sign(a_fix) != np.sign(a_cor)),
                gap_closed_pct=100.0 * (1.0 - abs(a_fix - a_cor) / max(abs(a_raw - a_cor), 1e-9))))
    return pd.DataFrame(rows)


def _agg(g):
    return g.agg(n=("abs_error_ms", "size"),
                 median_abs_err=("abs_error_ms", "median"),
                 mean_abs_err=("abs_error_ms", "mean"),
                 p90_abs_err=("abs_error_ms", lambda s: s.quantile(.9)),
                 within_5ms=("abs_error_ms", lambda s: 100.0 * (s <= 5).mean()),
                 within_10ms=("abs_error_ms", lambda s: 100.0 * (s <= 10).mean()),
                 peak_r=("peak_r", "mean")).round(2)


def main():
    runs = load_runs()
    print(f"loaded {len(runs)} runs: {sorted(set(k[1] for k in runs))} "
          f"x {sorted(set(k[2] for k in runs))}\n")

    wi = experiment_within_session(runs)
    print("=== 0. upper bound: align each run to its OWN corrected epochs (label-free) ===")
    print(f"median |error| {wi.abs_error_ms.median():.2f} ms, mean peak r {wi.peak_r.mean():.3f}, "
          f"{100 * (wi.abs_error_ms <= 5).mean():.0f}% within 5 ms  (n={len(wi)} runs)")
    print("-> the LABEL-FREE AVERAGE IS NOT THE PROBLEM. Whatever fails below is the "
          "cross-session template.\n")

    grid = experiment_grid(runs)
    grid.round(3).to_csv(os.path.join(OUT, "offset_recovery.csv"), index=False)
    print("=== 1. cross-session grid (all donors) ===")
    print(_agg(grid.groupby(["region", "bound", "smooth_hz"])).to_string())

    print("\n--- the deployment case only (cross-site donors, best-by-mean cell) ---")
    print(_agg(grid[grid.bound == "prior_0_150"].groupby(["region", "rel_site"])).to_string())

    lab = experiment_labelled(runs)
    print("\n=== 2. were the labels the problem? ===")
    st = grid[(grid.rel_task == "same_task") & (grid.region == "posterior")]
    for b in BOUNDS:
        a = st[st.bound == b].abs_error_ms
        c = lab[lab.bound == b].abs_error_ms
        print(f"  {b:12s}  label-free median |err| {a.median():6.2f} ms   "
              f"difference-wave (labelled) {c.median():6.2f} ms")
    print("-> the labelled estimator is no better. Dropping labels is not what breaks it.")

    sweep = experiment_sweep(runs)
    sweep.round(3).to_csv(os.path.join(OUT, "offset_recovery_sweep.csv"), index=False)
    print("\n=== 3. trial-count sweep (cross-site donors, prior_0_150) ===")
    print(_agg(sweep[sweep.rel_site == "cross_site"].groupby("requested_n")).to_string())

    base, cell = experiment_baselines(runs, grid)
    base.round(2).to_csv(os.path.join(OUT, "offset_recovery_baselines.csv"), index=False)
    print("\n=== 4. THE CONTROL THAT DECIDES IT ===")
    print(base.round(2).to_string(index=False))

    rep = experiment_repair(runs, grid, cell)
    rep.round(3).to_csv(os.path.join(OUT, "offset_recovery_repair.csv"), index=False)
    print("\n=== 5. end-to-end: repair the markers, re-measure the reported amplitude ===")
    n_runs = rep.repair.value_counts().iloc[0]
    print(f"raw-marker sign flips before any repair: "
          f"{int(rep[rep.repair == 'eeg_lag'].raw_sign_flipped.sum())}/{n_runs}")
    print(rep.groupby("repair").agg(
        median_residual_ms=("residual_ms", "median"), mean_residual_ms=("residual_ms", "mean"),
        sign_flips_after=("repaired_sign_flipped", "sum"),
        median_gap_closed_pct=("gap_closed_pct", "median"),
        mean_gap_closed_pct=("gap_closed_pct", "mean")).round(1).to_string())
    print("(gap_closed negative = the repair made the reported amplitude worse than raw)")
    print("-> split decision: the EEG lag avoids catastrophic failures (no sign flips, "
          "positive\n   mean gap) while the constant is better typically (lower median "
          "residual). Neither\n   is accurate enough to substitute for a photodiode.")

    eeg = base[base.uses_eeg].iloc[0]
    pri = base[base.estimator.str.startswith("prior_const")].iloc[0]
    print("\n=== VERDICT ===")
    print(f"Best EEG estimator : mean {eeg.mean_abs_err:.1f} ms, median "
          f"{eeg.median_abs_err:.1f} ms, {eeg.within_10ms:.0f}% within 10 ms")
    print(f"Constant guess     : mean {pri.mean_abs_err:.1f} ms, median "
          f"{pri.median_abs_err:.1f} ms, {pri.within_10ms:.0f}% within 10 ms")
    print("The EEG-free constant guess wins on every metric. Label-free cross-session\n"
          "offset recovery does not work on this data, and the reason is not the labels\n"
          "(section 2) nor the trial count (section 3) but the absence of a transferable\n"
          "cross-session template. NOT a route to repairing markers without a photodiode.")


def _selftest():
    """Sign and sanity checks — a sign error here would invert the whole conclusion."""
    runs = load_runs()
    rng = np.random.default_rng(0)
    base = rng.standard_normal((4, 500))
    for true_ms in (-40.0, 0.0, 90.0):
        k = int(round(true_ms / 1000.0 * FS))
        A, B = base[:, 150:400], base[:, 150 - k:400 - k]   # B[t] = A[t-k] -> B is later
        lag, r = multichan_lag(A, B, FS, -150.0, 150.0)
        assert abs(lag - true_ms) < 4.0, f"synthetic lag {lag} != {true_ms}"
        assert r > 0.99, f"synthetic peak r too low: {r}"
    print("synthetic lag recovery OK (sign and magnitude)")

    d = runs[("sub-SHE01", "ses-01SHE", "n170")]
    idx = [list(d["chans"]).index(c) for c in POSTERIOR if c in list(d["chans"])]
    lag, r = multichan_lag(label_free_evoked(d, "cor", idx),
                           label_free_evoked(d, "raw", idx), float(d["fs"]))
    err = float(d["err_ms"].mean())
    assert lag > 0, f"raw epoch must lag the corrected one, got {lag:+.1f} ms"
    assert abs(lag - err) < 8.0, f"within-session label-free lag {lag:.1f} vs measured {err:.1f}"
    print(f"within-session upper bound OK — recovered {lag:+.1f} ms vs measured {err:.1f} ms (r={r:.2f})")

    # the bounded search must never return a lag outside its bound
    lo, hi = BOUNDS["prior_0_150"]
    l2, _ = multichan_lag(label_free_evoked(d, "cor", idx),
                          label_free_evoked(d, "raw", idx), float(d["fs"]), lo, hi)
    assert lo - 4 <= l2 <= hi + 4, f"bounded search escaped its bound: {l2}"
    print("bounded search respects its bound")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
