#!/usr/bin/env python3
"""
augment.py — the non-GAN augmentation arms.

Time-shift augmentation
-----------------------
Epochs from `epochs.py` are time-locked to the photodiode-CORRECTED marker, so t=0 is
the true stimulus onset. A raw marker sits `e` ms EARLIER than the true onset, so in a
raw-marker epoch the whole response appears `e` ms LATER.

To synthesise a mis-timed training trial from a correctly-timed one, the new epoch must
satisfy  new[tau] = padded[tau - e], i.e. the source window is read `e` ms earlier. The
epochs carry PAD_MS of real signal on both sides, so this is a re-cut of genuine data —
never a zero-padded edge, which would hand the classifier an artificial step to learn.

Where the shifts come from
--------------------------
Step 1 found the error is ~77% site and ~17% session, with only ~1% trial-to-trial. So
the choice of sampling distribution is the experiment, not a detail:

  site        pooled measured errors from OTHER sessions at the same site
              — "I have a photodiode somewhere in my lab"
  allsites    pooled measured errors from every other session, any site
              — "I have published error distributions, but not for my rig"
  broad       Uniform(0, 150) ms — "I know nothing, cover the range"
  session     draws a per-copy constant offset plus within-run jitter, mirroring the
              structure Step 1 actually measured

Draws are always leave-one-session-out: a run never sees its own measured errors, so no
augmenter is fitted on the thing it is evaluated against.

Generic (timing-unaware) augmentation
-------------------------------------
Gaussian noise, amplitude scaling and same-class interpolation (SMOTE-style) — what gets
used by default when nobody suspects a timing problem. Included to show that adding data
without addressing timing does not help.
"""
from __future__ import annotations
import numpy as np


def shift_crop(X, t, lo_ms, hi_ms, shift_ms=None):
    """
    Cut [lo_ms, hi_ms] out of padded epochs, optionally displacing each trial in time.

    shift_ms[i] > 0 makes trial i look like it was marked EARLY by that many ms, i.e. its
    response appears that much later — the same corruption the raw markers produce.
    """
    n = len(X)
    if shift_ms is None:
        shift_ms = np.zeros(n)
    shift_ms = np.broadcast_to(np.asarray(shift_ms, float), (n,))
    width = int(np.sum((t >= lo_ms) & (t < hi_ms)))
    start0 = int(np.searchsorted(t, lo_ms))
    step = t[1] - t[0]

    out = np.empty((n, X.shape[1], width), dtype=X.dtype)
    for i in range(n):
        s = start0 - int(round(shift_ms[i] / step))
        s = min(max(s, 0), X.shape[2] - width)      # clamp; pad makes this a no-op in practice
        out[i] = X[i, :, s:s + width]
    return out


# ------------------------------------------------------------------ shift samplers
def sampler_empirical(pool, rng):
    """i.i.d. draws from a pool of measured errors."""
    pool = np.asarray(pool, float)
    return lambda n: rng.choice(pool, size=n, replace=True)


def sampler_broad(rng, lo=0.0, hi=150.0):
    return lambda n: rng.uniform(lo, hi, size=n)


def sampler_session(pool_means, within_sd, rng):
    """
    Structure-aware: one constant offset per augmented copy, plus small per-trial jitter.

    This is the generator Step 1's variance decomposition implies (a session-level
    constant dominating a small trial-level term), as opposed to the i.i.d. draw that
    shift-augmentation normally assumes.
    """
    pool_means = np.asarray(pool_means, float)
    def draw(n):
        offset = rng.choice(pool_means)
        return offset + rng.normal(0, within_sd, size=n)
    return draw


# ------------------------------------------------------------------ generic augmentation
def generic_augment(X, y, rng, n_copies=1, noise_frac=0.2, scale_sd=0.1, smote_frac=0.5):
    """Timing-unaware augmentation: noise + amplitude scaling + same-class interpolation."""
    Xs, ys = [], []
    for _ in range(n_copies):
        Z = X.copy()
        sd = Z.std(axis=(1, 2), keepdims=True)
        Z = Z + rng.normal(0, 1, Z.shape) * sd * noise_frac
        Z = Z * (1 + rng.normal(0, scale_sd, (len(Z), 1, 1)))
        # interpolate a random fraction of trials toward another trial of the same class
        pick = rng.random(len(Z)) < smote_frac
        for c in np.unique(y):
            idx = np.where((y == c) & pick)[0]
            if len(idx) < 2:
                continue
            partner = rng.choice(np.where(y == c)[0], size=len(idx))
            lam = rng.uniform(0, 0.5, (len(idx), 1, 1))
            Z[idx] = (1 - lam) * Z[idx] + lam * X[partner]
        Xs.append(Z); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def _selftest():
    rng = np.random.default_rng(0)
    fs = 256.0
    t = np.arange(-450, 1200, 1000 / fs)
    n, ch = 8, 3
    # an impulse at t=100 ms in every trial
    X = np.zeros((n, ch, len(t)))
    i100 = int(np.argmin(np.abs(t - 100)))
    X[:, :, i100] = 1.0

    # no shift: impulse must land at 100 ms in the crop
    c = shift_crop(X, t, 0, 500)
    tc = t[(t >= 0) & (t < 500)]
    assert abs(tc[np.argmax(c[0, 0])] - 100) < 5, "unshifted crop moved the impulse"

    # shift of +80 ms must move the impulse LATER, to ~180 ms
    c = shift_crop(X, t, 0, 500, np.full(n, 80.0))
    got = tc[np.argmax(c[0, 0])]
    assert abs(got - 180) < 5, f"expected impulse at ~180 ms, got {got:.1f}"

    # per-trial shifts must be applied per trial, not collapsed to a scalar
    sh = np.linspace(0, 140, n)
    c = shift_crop(X, t, 0, 500, sh)
    pos = np.array([tc[np.argmax(c[i, 0])] for i in range(n)])
    assert np.all(np.diff(pos) > 0), f"shifts not applied per-trial: {pos}"
    assert abs(pos[-1] - (100 + 140)) < 5, "largest shift wrong"

    # generic augmentation preserves shape and labels
    y = np.array([0, 1] * 4)
    Xa, ya = generic_augment(X, y, rng, n_copies=2)
    assert Xa.shape == (2 * n, ch, len(t)) and len(ya) == 2 * n, "generic augment shape"
    assert (ya == np.concatenate([y, y])).all(), "generic augment scrambled labels"

    # samplers return the right length
    assert len(sampler_empirical([1, 2, 3], rng)(5)) == 5
    assert len(sampler_broad(rng)(7)) == 7
    assert len(sampler_session([70, 90], 3.0, rng)(9)) == 9
    print("selftest OK — shifts are per-trial and move responses later, as raw markers do")


if __name__ == "__main__":
    _selftest()
