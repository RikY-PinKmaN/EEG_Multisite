#!/usr/bin/env python3
"""G0.6 — the measuring instrument for the GAN fidelity arm (GAN_ROADMAP.md §G1).

G1 claims that a generator trained on raw-marker epochs reproduces its training set's
marker error ~1:1. Measuring that needs a lag estimator whose own error is well under the
effect being measured, and `epochs.py::_best_lag` is not it: it argmaxes over integer
sample shifts, so at 256 Hz it can only ever return multiples of 3.906 ms. Reporting a
1:1 identity relation with a 3.9 ms-quantised ruler would put the quantisation into the
residuals and invite exactly the reviewer question we cannot answer.

`best_lag` here fits a parabola through the correlation peak and its two neighbours, which
recovers the sub-sample maximum. Run this file to verify the instrument against known
injected lags before trusting any number G1 produces.

    python3 gan_lag_check.py

No TensorFlow needed — this validates the ruler, not the generator.
"""
from __future__ import annotations

import numpy as np

from epochs import _best_lag as _best_lag_integer

FS = 256.0


def best_lag(a, b, fs=FS, max_ms=300.0, return_r=False):
    """Lag in ms that best aligns `b` onto a reference `a`, to sub-sample precision.

    Positive means `b` is later than `a`. Both inputs are z-scored, so the estimate is
    invariant to amplitude and offset — required here because synthetic ERPs are not
    amplitude-matched to real ones.

    With `return_r`, also returns the correlation at that best lag. Use it: a lag is only
    meaningful if something actually aligned, and correlation *at the best lag* is the right
    quality measure rather than correlation at zero lag — a waveform displaced by 90 ms is
    a perfect match at its own lag and near-uncorrelated at zero. This matches what
    `erp_impact.py` reports as `shape_r`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.ndim == b.ndim == 1, "best_lag takes 1-D waveforms"
    assert a.size == b.size, f"length mismatch: {a.size} vs {b.size}"

    a = (a - a.mean()) / (a.std() or 1.0)
    b = (b - b.mean()) / (b.std() or 1.0)

    m = int(round(max_ms / 1000.0 * fs))
    assert a.size > 3 * m, f"waveform too short ({a.size}) for max_ms={max_ms}"

    lags = np.arange(-m, m + 1)
    core = slice(m, -m)
    cc = np.array([np.corrcoef(a[core], np.roll(b, -k)[core])[0, 1] for k in lags])

    k = int(np.argmax(cc))
    lag_samples = float(lags[k])

    # Parabolic refinement. Skipped at the search edges, where there is no neighbour pair.
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            delta = 0.5 * (y0 - y2) / denom
            # A well-formed peak puts the vertex inside the neighbouring half-samples.
            if abs(delta) <= 0.5:
                lag_samples += delta

    lag_ms = lag_samples / fs * 1000.0
    if return_r:
        return lag_ms, float(cc[k])
    return lag_ms


def phase_randomise(x, rng):
    """Surrogate with the same power spectrum but randomised phase.

    The right null for "did this waveform align by chance?" — it preserves the spectral
    character that makes smooth signals correlate, and destroys the timing structure.
    """
    x = np.asarray(x, dtype=float)
    F = np.fft.rfft(x)
    ph = rng.uniform(0.0, 2.0 * np.pi, F.size)
    ph[0] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=x.size)


_phase_randomise = phase_randomise   # used by the self-test below


def synthetic_erp(n=295, fs=FS, lag_ms=0.0, amp=1.0, noise=0.0, rng=None):
    """An N170-ish deflection: a Gaussian-windowed negative lobe on a slow drift.

    Built in continuous time and sampled at `lag_ms`, so a fractional lag is genuinely
    present in the waveform rather than being a rolled integer shift.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    t = (np.arange(n) / fs - 0.2) * 1000.0 - lag_ms      # ms, relative to the deflection
    peak = -np.exp(-0.5 * ((t - 165.0) / 22.0) ** 2)      # N170 at ~165 ms, ~50 ms wide
    later = 0.45 * np.exp(-0.5 * ((t - 320.0) / 60.0) ** 2)
    drift = 0.12 * np.sin(2 * np.pi * (t / 1000.0) * 2.0)
    sig = amp * (peak + later + drift)
    if noise:
        sig = sig + rng.normal(0.0, noise * amp, n)
    return sig


def _selftest():
    rng = np.random.default_rng(7)
    ref = synthetic_erp()

    # 1. Fractional lags, clean. The instrument must beat the sample period it replaces.
    truths = [0.0, 3.9, 12.5, 37.3, 74.2, 91.6, 135.6, -22.7]
    err_sub, err_int = [], []
    for true_ms in truths:
        moved = synthetic_erp(lag_ms=true_ms)
        err_sub.append(abs(best_lag(ref, moved) - true_ms))
        err_int.append(abs(_best_lag_integer(ref, moved, FS) - true_ms))
    err_sub, err_int = np.array(err_sub), np.array(err_int)
    assert err_sub.max() < 1.0, f"sub-sample recovery worse than 1 ms: {err_sub.max():.2f}"
    assert err_sub.mean() < err_int.mean(), "no improvement over the integer estimator"

    # 2. Amplitude invariance — synthetic ERPs are not amplitude-matched to real ones.
    for amp in (0.25, 4.0):
        got = best_lag(ref, synthetic_erp(lag_ms=74.2, amp=amp))
        assert abs(got - 74.2) < 1.0, f"amp={amp} broke recovery: {got:.2f} ms"

    # 3. Noise robustness at the SNR a modest trial-average plausibly has.
    #
    # Thresholds are derived from the effect, not chosen for roundness. The smallest
    # marker error G1 must resolve is Leeds' 23.9 ms; Sheffield's is 91.6 ms. Requiring a
    # median error under SMALLEST_EFFECT/8 (2.99 ms) keeps the ruler's error under 13% of
    # the smallest effect and under 4% of the largest. Error scales smoothly with noise
    # (median 0.7 / 1.3 / 2.0 / 3.4 ms at noise 0.1 / 0.2 / 0.3 / 0.5), so if real
    # synthetic ERPs turn out noisier than 0.3, re-measure here before trusting G1.
    # The tail criterion guards against a single grossly wrong point in G1's 16-point
    # scatter, so it is set at SMALLEST_EFFECT/3 (7.97 ms) — an error that large would
    # visibly move a point off the identity line. A 5 ms bar was tried and rejected: at
    # this noise level ~91% of draws clear it, so the test passed or failed on the draw
    # rather than on the estimator.
    SMALLEST_EFFECT_MS = 23.9
    median_budget = SMALLEST_EFFECT_MS / 8.0
    tail_budget = SMALLEST_EFFECT_MS / 3.0
    noisy_err = []
    for _ in range(400):
        true_ms = float(rng.uniform(-140, 140))
        got = best_lag(ref, synthetic_erp(lag_ms=true_ms, noise=0.30, rng=rng))
        noisy_err.append(abs(got - true_ms))
    noisy_err = np.array(noisy_err)
    assert np.median(noisy_err) < median_budget, \
        f"noisy median {np.median(noisy_err):.2f} ms exceeds budget {median_budget:.2f} ms"
    within_tail = (noisy_err < tail_budget).mean()
    assert within_tail > 0.90, \
        f"only {within_tail:.0%} of noisy draws within {tail_budget:.2f} ms"

    # 4. Sign convention, asserted rather than assumed: a later signal reports positive.
    assert best_lag(ref, synthetic_erp(lag_ms=50.0)) > 0, "sign convention inverted"

    # 4b. return_r: a matched pair correlates ~1 at its own lag.
    lag_r, r_good = best_lag(ref, synthetic_erp(lag_ms=50.0), return_r=True)
    assert abs(lag_r - 50.0) < 1.0 and r_good > 0.99, f"lag {lag_r:.2f}, r {r_good:.3f}"

    # 4c. Peak correlation is NOT a usable quality gate, and this test exists to say so.
    #
    # Against WHITE noise, peak r stays low (p99 ~0.29) — which is what an earlier version
    # of this test checked, with a loose < 0.75 bound that made 0.75 look like a meaningful
    # ceiling. It is not. The honest null for a smooth ERP-shaped waveform is a
    # SPECTRUM-MATCHED surrogate: same power spectrum, randomised phase, no timing
    # structure. Maximising correlation over ~150 candidate lags then reaches a median of
    # ~0.56 and a p95 of ~0.70 by chance, because low-frequency waveforms align well
    # somewhere. So an observed peak r of 0.6 is unremarkable, and gating on it would admit
    # generators that learned nothing. Use the LAG against a surrogate null instead — the
    # null lag is nearly uniform over the search range, so a lag landing where predicted is
    # a specific and unlikely event (see gan_fidelity_stats.py).
    white = np.array([best_lag(ref, rng.normal(size=ref.size), return_r=True)[1]
                      for _ in range(300)])
    surr = np.array([best_lag(ref, _phase_randomise(ref, rng), return_r=True)[1]
                     for _ in range(300)])
    assert np.percentile(white, 99) < 0.40, f"white-noise peak r p99 {np.percentile(white,99):.3f}"
    assert np.median(surr) > 0.40, (
        "spectrum-matched surrogates no longer reach a high peak r — if this fails, "
        "re-derive whether peak r can be used as a quality gate")

    # 5. UNANCHORED configuration — comparing two independently noisy waveforms.
    #
    # This is what "compare the G_raw synthetic ERP against the G_corrected synthetic ERP"
    # would do, and it is measurably worse: both sides carry noise, so the errors add.
    # Kept as a test because the number is the reason G1 must NOT be designed this way.
    unanchored = np.array([abs(best_lag(synthetic_erp(noise=0.30, rng=rng),
                                        synthetic_erp(noise=0.30, rng=rng)))
                           for _ in range(200)])
    assert np.median(unanchored) < 12.0, \
        f"unanchored spurious lag regressed: {np.median(unanchored):.2f} ms"

    # 6. ANCHORED configuration — what G1 must actually do.
    #
    # Measure each synthetic ERP against a common clean reference (the real
    # corrected-marker grand average, which is averaged over every trial of the run and so
    # is far less noisy than any single generator sample), then take the difference. The
    # shared anchor cancels and only one noisy side enters each estimate.
    anchored_err = []
    for _ in range(300):
        true_diff = float(rng.uniform(20, 140))
        a = synthetic_erp(lag_ms=0.0, noise=0.30, rng=rng)          # G_corrected sample
        b = synthetic_erp(lag_ms=true_diff, noise=0.30, rng=rng)     # G_raw sample
        est = best_lag(ref, b) - best_lag(ref, a)                    # ref = clean anchor
        anchored_err.append(abs(est - true_diff))
    anchored_err = np.array(anchored_err)
    assert np.median(anchored_err) < np.median(unanchored), \
        "anchoring failed to beat the unanchored comparison"
    assert np.median(anchored_err) < SMALLEST_EFFECT_MS / 4.0, \
        f"anchored median {np.median(anchored_err):.2f} ms too coarse for G1"

    print("selftest OK")
    print(f"  clean recovery  : sub-sample {err_sub.mean():.3f} ms mean "
          f"({err_sub.max():.3f} worst) vs integer {err_int.mean():.3f} ms mean "
          f"({err_int.max():.3f} worst)")
    print(f"  sample period   : {1000.0 / FS:.3f} ms — the integer estimator's floor")
    print(f"  noisy (SNR~3.3) : median {np.median(noisy_err):.2f} ms "
          f"(budget {median_budget:.2f}), {within_tail:.0%} within {tail_budget:.2f} ms, "
          f"worst {noisy_err.max():.2f} ms")
    print(f"  error vs effect : {np.median(noisy_err) / SMALLEST_EFFECT_MS:.1%} of the "
          f"smallest marker error to resolve (Leeds, {SMALLEST_EFFECT_MS} ms), "
          f"{np.median(noisy_err) / 91.6:.1%} of Sheffield's")
    print(f"  unanchored      : median {np.median(unanchored):.2f} ms spurious lag "
          f"(two noisy waveforms) — do NOT design G1 this way")
    print(f"  anchored        : median {np.median(anchored_err):.2f} ms on the "
          f"synthetic-minus-synthetic difference, via a common clean reference")
    print("  amplitude-invariant, sign convention positive = later. Instrument is fit for G1,")
    print("  provided G1 uses the anchored comparison (GAN_ROADMAP.md §G1).")


if __name__ == "__main__":
    _selftest()
