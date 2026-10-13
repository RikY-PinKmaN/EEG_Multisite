#!/usr/bin/env python3
"""
epochs.py — build matched corrected-marker / raw-marker epoch sets for a run.

Shared by `erp_impact.py` and `classify.py` so both see exactly the same trials.

Two things here are not optional for a valid comparison:

  common trial mask   a trial is kept only if it survives artifact rejection under
                      BOTH marker sets. Rejecting independently would let the two
                      conditions be built from different trials, and any difference
                      could then be a trial-selection effect rather than a timing one.

  shared padding      epochs are cut with `PAD_MS` of extra signal on each side so the
                      augmenter in `augment.py` can slide a trial in time by re-cutting
                      from real data instead of zero-padding an edge.

Epochs are cached to .npz — reading BrainVision + filtering the whole run is the slow
part and every experiment re-uses the same cuts.
"""
from __future__ import annotations
import os
import numpy as np

from jitter_pipeline import ROOT
from erp_common import load_reref_resampled, COMMON, TARGET_FS
from raw_markers import bids_prefix, CONTRAST

CACHE = os.path.join(ROOT, "analysis", "_epoch_cache")

TMIN, TMAX = -0.2, 0.95        # analysis span
PAD_MS = 250.0                 # extra signal each side, for time-shift augmentation
# µV peak-to-peak on the worst of 14 channels, analysis span only. Deliberately
# permissive: the criterion is the max over all channels, so frontal blinks would
# otherwise discard ~70% of trials while barely touching the posterior components
# these paradigms live on. Retains 86-98% uniformly across the three sites, and the
# corrected-vs-raw contrast is within-trial so residual artifact loads both sides equally.
REJECT_P2P = 250.0


def _cut(x, fs, samples, pre, post):
    """Cut epochs; returns (array, valid_mask) without applying artifact rejection."""
    n = len(x)
    keep = (samples - pre >= 0) & (samples + post < n)
    out = np.zeros((len(samples), pre + post), dtype=np.float32)
    for i, s in enumerate(samples):
        if keep[i]:
            out[i] = x[s - pre:s + post]
    return out, keep


def build(subject, session, task, trials, chans=COMMON, use_cache=True):
    """
    Returns dict with:
      X_cor, X_raw : (n_trials, n_chan, n_time) epoched at corrected / raw markers,
                     baseline-corrected, INCLUDING the pad on both sides
      t            : time vector (ms) covering the padded span
      core         : boolean mask selecting the TMIN..TMAX analysis span within t
      y            : 0/1 class labels for the paradigm's two-class contrast
      err_ms       : measured marker error per kept trial
      chans        : channel names in row order
    """
    key = f"{subject}_{session}_{task}.npz"
    path = os.path.join(CACHE, key)
    if use_cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files} | {"chans": list(z["chans"])}

    prefix = bids_prefix(subject, session, task)
    if not os.path.exists(prefix + "_eeg.vhdr"):
        return None
    sig, fs, orig_fs = load_reref_resampled(prefix, channels=chans)
    names = [c for c in chans if c in sig]
    if not names:
        return None
    X = np.stack([sig[c] for c in names])          # (n_chan, n_samples)

    c0, c1 = CONTRAST[task]
    g = trials[(trials.task == task) & (trials.condition.isin([c0, c1]))
               & (~trials.correction_failed)].sort_values("trial_index")
    if g.empty:
        return None

    scale = TARGET_FS / orig_fs
    cor = np.round(g["corr_sample"].values * scale).astype(int)
    raw = np.round(g["raw_sample"].values * scale).astype(int)

    pad = int(round(PAD_MS / 1000 * fs))
    pre = int(round(-TMIN * fs)) + pad
    post = int(round(TMAX * fs)) + pad
    t = (np.arange(-pre, post) / fs) * 1000.0
    core = (t >= TMIN * 1000) & (t <= TMAX * 1000)
    base = (t >= -200) & (t < 0)

    ec, er, valid = [], [], None
    for ch in range(X.shape[0]):
        a, ka = _cut(X[ch], fs, cor, pre, post)
        b, kb = _cut(X[ch], fs, raw, pre, post)
        ec.append(a); er.append(b)
        valid = (ka & kb) if valid is None else (valid & ka & kb)
    Xc = np.stack(ec, 1)                            # (n_trials, n_chan, n_time)
    Xr = np.stack(er, 1)

    # baseline correct each epoch on its own pre-stimulus window
    Xc -= Xc[:, :, base].mean(-1, keepdims=True)
    Xr -= Xr[:, :, base].mean(-1, keepdims=True)

    # artifact rejection on the analysis span, required to pass under BOTH marker sets
    ptp_c = np.ptp(Xc[:, :, core], axis=-1).max(1)
    ptp_r = np.ptp(Xr[:, :, core], axis=-1).max(1)
    keep = valid & (ptp_c <= REJECT_P2P) & (ptp_r <= REJECT_P2P)

    y = (g["condition"].values == c1).astype(int)
    out = dict(X_cor=Xc[keep], X_raw=Xr[keep], t=t, core=core, y=y[keep],
               err_ms=g["err_ms"].values[keep], chans=np.array(names),
               fs=np.array(TARGET_FS), n_dropped=np.array(int((~keep).sum())))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **out)
    out["chans"] = names
    return out


def _selftest():
    from raw_markers import build_trial_table
    tt = build_trial_table()
    g = tt[(tt.subject == "sub-SHE01") & (tt.session == "ses-01SHE")]
    d = build("sub-SHE01", "ses-01SHE", "n170", g, use_cache=False)
    assert d["X_cor"].shape == d["X_raw"].shape, "corrected/raw shape mismatch"
    assert len(d["y"]) == len(d["X_cor"]) == len(d["err_ms"]), "label/epoch length mismatch"
    assert set(np.unique(d["y"])) == {0, 1}, "contrast is not two-class"
    # the raw marker is earlier, so the same signal must appear LATER in the raw epoch
    lag = _best_lag(d["X_cor"].mean(0).mean(0), d["X_raw"].mean(0).mean(0), d["fs"])
    assert lag > 0, f"raw epoch should lag the corrected one, got {lag:.1f} ms"
    print(f"selftest OK — {len(d['y'])} trials kept, {int(d['n_dropped'])} dropped, "
          f"recovered lag {lag:+.1f} ms")


def _best_lag(a, b, fs, max_ms=300):
    """Lag (ms) that best aligns b onto a, by cross-correlation."""
    a = (a - a.mean()) / (a.std() or 1)
    b = (b - b.mean()) / (b.std() or 1)
    m = int(round(max_ms / 1000 * fs))
    lags = np.arange(-m, m + 1)
    cc = [np.corrcoef(a[m:-m], np.roll(b, -k)[m:-m])[0, 1] for k in lags]
    return float(lags[int(np.argmax(cc))] / fs * 1000)


if __name__ == "__main__":
    _selftest()
