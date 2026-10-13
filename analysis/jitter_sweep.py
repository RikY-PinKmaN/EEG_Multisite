#!/usr/bin/env python3
"""
jitter_sweep.py — corruption x augmentation MISMATCH sweep.

The premise of the augmentation plan is "most datasets have no photodiode, so their
timing is wrong and uncorrectable — train the classifier not to care". But the site's
jitter distribution is measured *with* the photodiode. Anyone who actually needs the
method has no photodiode, so they cannot measure that distribution: they must GUESS it.

    Central question: how wrong can the guess be and still help?

    Flat optimum  -> a generous, roughly-right guess works -> deployable recipe.
    Peaked optimum-> you must already know the answer      -> method fails where needed.

This needs NO raw pre-correction triggers. The released onsets are already photodiode-
corrected (see FINDINGS.md), i.e. they are ground-truth visual onsets, so we impose a
KNOWN corruption on KNOWN-good timing and sweep it. When the raw triggers arrive they
are simply one more cell in this grid: swap `recorded = true + draw_jitter(...)` for the
real recorded onsets. Same code path.

Design (per session, per fold):
    true_onset            ground truth (PD-corrected release)
    recorded = true + d   d ~ P_corrupt  (what the experimenter's marker file says)
    train:  epochs cut at recorded + e,  e ~ P_augment, K copies per trial (slide aug)
    test:   epochs cut at recorded       (their best guess — no photodiode)
    ceiling:epochs cut at true           (corrupt_sd = 0)

Sign convention: a marker recorded LATE by d means epoching at `recorded` puts the
component d ms EARLIER in the epoch. Only the VARIABLE part of the error is modelled —
a constant offset is trivially correctable and harmless (FINDINGS F1) — so every
distribution here is zero-mean and parameterised by its SD in ms.

NOTE on the deployment scenario: Mahnaz's draft trains on corrected markers and tests on
raw ones. That presumes a photodiode at training time. If the premise is "no photodiode",
BOTH train and test are corrupted — which is the default here (`train_corrupt=True`).
Kept as a flag because it is a real design fork worth showing.

Usage:  python3 jitter_sweep.py            # full grid -> jitter_sweep.csv
        python3 jitter_sweep.py --quick    # small grid, one session, for smoke-testing
"""
from __future__ import annotations
import argparse
import csv
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from erp_common import (COMMON, MASTOIDS, TARGET_FS, STIM, SITE_OF, SESSIONS,
                        load_reref_resampled, prefix_for)
import erp_common

# ---------------------------------------------------------------------------
# jitter distributions  (delegated to `neon exec`, verified against spec)
# ---------------------------------------------------------------------------

def draw_jitter(family, sd_ms, n, rng):
    """Zero-mean per-trial marker timing error in ms, with SD == sd_ms."""
    if sd_ms == 0:
        return np.zeros(n)
    if family == 'uniform':
        w = sd_ms * np.sqrt(12.0)
        return rng.uniform(-w / 2, w / 2, n)
    elif family == 'gaussian':
        return rng.normal(0.0, sd_ms, n)
    elif family == 'drift':
        # AR(1): same marginal SD as the others but AUTOCORRELATED across trials.
        # This is the corruption family an i.i.d. slide-augmenter structurally cannot
        # match — the roadmap flags it as the open question the raw triggers settle.
        phi = 0.95
        x = np.zeros(n)
        x[0] = rng.normal(0.0, sd_ms)
        innov = rng.normal(0.0, sd_ms * np.sqrt(1.0 - phi ** 2), n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + innov[i]
        return x
    elif family == 'heavy':
        df = 3.0
        return rng.standard_t(df, n) / np.sqrt(df / (df - 2.0)) * sd_ms
    else:
        raise ValueError(family)


# ---------------------------------------------------------------------------
# fractional-onset epoching
# ---------------------------------------------------------------------------

def epoch_frac(data, fs, onsets, tmin=-0.2, tmax=0.5):
    """Cut epochs at FRACTIONAL sample positions via linear interpolation.

    At 256 Hz one sample is 3.9 ms; integer-sample shifts would quantise the jitter
    we are trying to study, so onsets stay float and the signal is interpolated.

    data   : (n_chan, n_samples)
    onsets : float array of fractional sample positions
    returns: t_ms (L,), epochs (n_keep, n_chan, L), keep (bool, len(onsets))
    """
    data = np.asarray(data, dtype=float)
    onsets = np.asarray(onsets, dtype=float)
    pre, post = int(round(-tmin * fs)), int(round(tmax * fs))
    offsets = np.arange(-pre, post)
    t_ms = offsets / fs * 1000.0
    n_chan, n_samp = data.shape
    xs = np.arange(n_samp)

    keep = (onsets + offsets[0] >= 0) & (onsets + offsets[-1] <= n_samp - 1)
    kept = onsets[keep]
    if kept.size == 0:
        return t_ms, np.zeros((0, n_chan, len(offsets))), keep

    pos = kept[:, None] + offsets[None, :]          # (n_keep, L)
    flat = pos.ravel()
    epochs = np.stack([np.interp(flat, xs, data[c]).reshape(pos.shape)
                       for c in range(n_chan)], axis=1)   # (n_keep, n_chan, L)
    return t_ms, epochs, keep


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def _true_onsets(prefix, values, orig_fs, target_fs=TARGET_FS):
    """Photodiode-corrected stimulus onsets, on the resampled clock, kept FRACTIONAL."""
    rows = list(csv.DictReader(open(prefix + "_events.tsv"), delimiter="\t"))
    vals = {str(v) for v in values}
    s = np.array([int(r["sample"]) for r in rows if r["value"] in vals], dtype=float)
    return s * target_fs / orig_fs


def load_session(subject, session, task="n170", channels=("PO7", "PO8", "P7", "P8")):
    """Return (data (n_chan,n_samp), fs, {cond: true_onsets_frac}, channels)."""
    prefix = prefix_for(subject, session, task)
    sig, fs, orig_fs = load_reref_resampled(prefix, channels=list(channels))
    chans = [c for c in channels if c in sig]
    if not chans:
        raise RuntimeError(f"no requested channels present for {subject}/{session}")
    data = np.stack([sig[c] for c in chans])
    onsets = {cond: _true_onsets(prefix, vals, orig_fs)
              for cond, vals in STIM[task].items()}
    return data, fs, onsets, chans


# ---------------------------------------------------------------------------
# feature extraction / classification
# ---------------------------------------------------------------------------

FEAT_TMIN, FEAT_TMAX = -0.1, 0.4      # window handed to the classifier
DECIM = 4                              # 256 Hz -> 64 Hz effective (keeps p sane)


def _features(epochs, t_ms):
    """(n,chan,L) -> (n, chan*L') restricted to the feature window and decimated."""
    w = (t_ms >= FEAT_TMIN * 1000) & (t_ms <= FEAT_TMAX * 1000)
    return epochs[:, :, w][:, :, ::DECIM].reshape(len(epochs), -1)


def _fit_auc(Xtr, ytr, Xte, yte):
    clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.decision_function(Xte))


# ---------------------------------------------------------------------------
# one cell of the grid
# ---------------------------------------------------------------------------

def run_cell(data, fs, true_onsets, labels, reject_mask, corrupt_family, corrupt_sd,
             augment_family, augment_sd, rng, n_aug=8, n_folds=5, train_corrupt=True):
    """Return mean AUC across folds for one (corruption, augmentation) combination."""
    # what the experimenter's marker file claims
    d = draw_jitter(corrupt_family, corrupt_sd, len(true_onsets), rng)
    recorded = true_onsets + d * fs / 1000.0
    train_onsets = recorded if train_corrupt else true_onsets

    aucs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for tr, te in skf.split(np.zeros(len(labels)), labels):
        tr = tr[reject_mask[tr]]
        te = te[reject_mask[te]]
        if len(np.unique(labels[tr])) < 2 or len(np.unique(labels[te])) < 2:
            continue

        # ---- training set: K slid copies per trial (augment_sd == 0 -> no aug) ----
        Xtr, ytr = [], []
        reps = n_aug if augment_sd > 0 else 1
        for _ in range(reps):
            e = draw_jitter(augment_family, augment_sd, len(tr), rng)
            t_ms, ep, keep = epoch_frac(data, fs, train_onsets[tr] + e * fs / 1000.0,
                                        FEAT_TMIN - 0.1, FEAT_TMAX + 0.1)
            if len(ep) == 0:
                continue
            Xtr.append(_features(ep, t_ms))
            ytr.append(labels[tr][keep])
        if not Xtr:
            continue
        Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)

        # ---- test set: cut at the recorded marker, no augmentation ----
        t_ms, ep_te, keep_te = epoch_frac(data, fs, recorded[te],
                                          FEAT_TMIN - 0.1, FEAT_TMAX + 0.1)
        if len(ep_te) == 0:
            continue
        Xte, yte = _features(ep_te, t_ms), labels[te][keep_te]
        if len(np.unique(yte)) < 2:
            continue
        aucs.append(_fit_auc(Xtr, ytr, Xte, yte))

    return float(np.mean(aucs)) if aucs else np.nan


# ---------------------------------------------------------------------------
# session prep (trial set fixed ONCE on true onsets — see note)
# ---------------------------------------------------------------------------

def prepare(subject, session, task="n170", cond_a="face", cond_b="car", reject_p2p=120.0):
    """Build the trial set and a rejection mask.

    The mask is computed ONCE from the TRUE onsets and reused for every cell. If we
    re-rejected per condition, the corruption would change which trials survive and the
    trial set would confound the manipulation.
    """
    data, fs, onsets, chans = load_session(subject, session, task)
    oa, ob = onsets[cond_a], onsets[cond_b]
    true_onsets = np.concatenate([oa, ob])
    labels = np.concatenate([np.zeros(len(oa), int), np.ones(len(ob), int)])

    t_ms, ep, keep = epoch_frac(data, fs, true_onsets, FEAT_TMIN - 0.1, FEAT_TMAX + 0.1)
    true_onsets, labels = true_onsets[keep], labels[keep]
    ok = np.ptp(ep, axis=2).max(axis=1) <= reject_p2p
    return data, fs, true_onsets, labels, ok, chans


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

SUBJECTS = ["sub-SHE01", "sub-SHE02"]
CORRUPT_FAMILIES = ["uniform", "gaussian", "drift", "heavy"]
CORRUPT_SDS = [0, 10, 20, 40]
AUGMENT_SDS = [0, 5, 10, 20, 40, 80, 160]
AUGMENT_FAMILY = "gaussian"          # the naive default a practitioner would reach for
N_REPS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grid, one session")
    ap.add_argument("--out", default="jitter_sweep.csv")
    ap.add_argument("--task", default="n170")
    args = ap.parse_args()

    subjects, sessions = SUBJECTS, SESSIONS
    corrupt_families, corrupt_sds, augment_sds, n_reps = (
        CORRUPT_FAMILIES, CORRUPT_SDS, AUGMENT_SDS, N_REPS)
    if args.quick:
        subjects, sessions = SUBJECTS[:1], SESSIONS[:1]
        corrupt_families, corrupt_sds = ["uniform"], [0, 20]
        augment_sds, n_reps = [0, 20, 80], 2

    rows = []
    for subject in subjects:
        for session in sessions:
            try:
                data, fs, true_onsets, labels, ok, chans = prepare(
                    subject, session, task=args.task)
            except (FileNotFoundError, RuntimeError, KeyError) as e:
                print(f"  skip {subject}/{session}: {e}", file=sys.stderr)
                continue
            site = SITE_OF[session.split("-")[1]]
            print(f"{subject} {session} [{site}] chans={chans} "
                  f"trials={len(true_onsets)} kept={int(ok.sum())}", flush=True)

            for cf in corrupt_families:
                for csd in corrupt_sds:
                    if csd == 0 and cf != corrupt_families[0]:
                        continue          # sd=0 is the same ceiling for every family
                    for asd in augment_sds:
                        for rep in range(n_reps):
                            rng = np.random.default_rng(hash((subject, session, cf, csd,
                                                              asd, rep)) % 2**32)
                            auc = run_cell(data, fs, true_onsets, labels, ok,
                                           cf, csd, AUGMENT_FAMILY, asd, rng)
                            rows.append(dict(subject=subject, session=session, site=site,
                                             task=args.task, corrupt_family=cf,
                                             corrupt_sd=csd, augment_family=AUGMENT_FAMILY,
                                             augment_sd=asd, rep=rep, auc=auc))
                    print(f"   {cf:9s} sd={csd:3d} -> "
                          + " ".join(f"{a:3d}:{np.nanmean([r['auc'] for r in rows if r['corrupt_family']==cf and r['corrupt_sd']==csd and r['augment_sd']==a and r['session']==session and r['subject']==subject]):.3f}"
                                     for a in augment_sds), flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
