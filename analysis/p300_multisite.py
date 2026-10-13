#!/usr/bin/env python3
"""
p300_multisite.py — data layer for the P300 corrected-vs-uncorrected GAN study (36 subjects).

Preprocessing follows ~/…/EEG_scripts/scripts/P300_pipeline.py (copied to repo root) exactly,
except where this dataset forces a change. Every forced change is listed here:

  * channels: 8 shared by all three sites' caps — Fz FCz C3 C4 Pz POz PO7 PO8
    (Cz/P3/P4/Oz do not exist on the Sheffield 16-ch cap). Average reference over those 8.
  * epochs are cut and baselined on the SAME stimulus marker (the speller code event that
    precedes each flash does not exist in p3b).
  * single trials, no 15-trial averaging (only 40 targets per session).
  * the 50 µV peak-to-peak rejection is applied over the whole -0.1..0.8 s ERP epoch and the
    trial is dropped under BOTH marker sets if it fails under EITHER (common trial mask), so the
    corrected and uncorrected arms classify the same trials.

Marker sets: 'cor' = photodiode-corrected (BIDS events.tsv), 'raw' = uncorrected marker, from
the row-aligned `*_events-raw.tsv` / `*_events-pdcorrected.tsv` pair in
derivatives/data-quality-metrics/pd_latency. raw_sample = cor_sample - round(err_s * fs).

Preprocessing variants: 'mine' (the P300 pipeline) and 'gedai' (same filters, then GEDAI on the
16 channels common to all caps, before channel pick / resample / reref / epoching).
"""
from __future__ import annotations
import os
import sys
import glob
import math
import numpy as np
import pandas as pd
from scipy.signal import buttord, butter, sosfiltfilt, resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jitter_pipeline import parse_vhdr          # noqa: E402
from erp_common import _read_named_channels     # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "Multisite-EEG", "EEGMultiSite_Raw")
PDL = os.path.join(DATA, "derivatives", "data-quality-metrics", "pd_latency")
CACHE = os.path.join(HERE, "_p300_cache")

CH8 = ["Fz", "FCz", "C3", "C4", "Pz", "POz", "PO7", "PO8"]
CH16 = ["F7", "F3", "Fz", "F4", "F8", "FCz", "C3", "C4", "Pz", "POz",
        "PO7", "PO8", "P7", "P8", "M1", "M2"]

# P300_pipeline.py constants
TARGET_FS = 80
ERP_TMIN, ERP_TMAX = -0.1, 0.8
FEAT_TMIN, FEAT_TMAX = 0.15, 0.5
BL_TMIN = -0.1
ARTIF_THRESH = 50.0      # P300_pipeline value; this dataset needs 100 (see GAN_CLASSIFY_PLAN.md)

TS = np.arange(round(ERP_TMIN * TARGET_FS), round(ERP_TMAX * TARGET_FS) + 1)      # -8..64 (73)
BL = np.arange(round(BL_TMIN * TARGET_FS), 0 + 1)                                  # -8..0
FEAT_IDX = np.searchsorted(TS, np.arange(round(FEAT_TMIN * TARGET_FS),
                                         round(FEAT_TMAX * TARGET_FS)))            # 28 samples
assert len(TS) == 73 and len(FEAT_IDX) == 28

# v2 ("-w" variants, 29 Sep 2026): epoch -0.1..1.0 s so a late-shifted uncorrected P3b still
# fits, and features on 0.2-0.8 s (P3b peaks ~450-500 ms here; 0.15-0.5 s cut it off).
# Reference stays the average of the 8 channels: a linked-mastoid reference triples the P3b
# amplitude but gave no paired accuracy gain (LDA is ~invariant to it) and loses most trials to
# Fz blinks at 100 uV (p300_diagnostic_ablation.csv, 9 paired sessions).
TS2 = np.arange(round(-0.1 * TARGET_FS), round(1.0 * TARGET_FS) + 1)             # -8..80 (89)
FEAT2_IDX = np.searchsorted(TS2, np.arange(round(0.2 * TARGET_FS), round(0.8 * TARGET_FS)))  # 48


def is_v2(variant: str) -> bool:
    return variant.endswith("-w")


def grid(variant: str):
    return TS2 if is_v2(variant) else TS


def feat_idx(n_times: int):
    return FEAT2_IDX if n_times == len(TS2) else FEAT_IDX

VISUAL = {100, 200, 300, 400, 111, 222}
LABEL = {111: 2, 222: 1}   # P300_pipeline convention: 2 = target, 1 = non-target


# ---------------------------------------------------------------------------------------------
# inventory + markers
# ---------------------------------------------------------------------------------------------
def site_of(session: str) -> str:
    return session[-3:]


def prefix(sub: str, ses: str, task: str = "p3b") -> str:
    return os.path.join(DATA, sub, ses, "eeg", f"{sub}_{ses}_task-{task}")


LOCAL = os.path.dirname(HERE)   # repo root: sub-SHE01/02 recorded data (identical to the release)


def eeg_prefix(sub: str, ses: str, task: str = "p3b") -> str | None:
    """Prefix whose _eeg.eeg holds real signal data. Most release files are git-annex
    placeholders (content still on GIN); fall back to the repo-root copy for SHE01/02."""
    for base in (DATA, LOCAL):
        p = os.path.join(base, sub, ses, "eeg", f"{sub}_{ses}_task-{task}")
        if os.path.exists(p + "_eeg.eeg") and os.path.getsize(p + "_eeg.eeg") > 10_000:
            return p
    return None


def inventory() -> pd.DataFrame:
    rows = []
    for vhdr in sorted(glob.glob(os.path.join(DATA, "sub-*", "ses-*", "eeg", "*task-p3b_eeg.vhdr"))):
        sub, ses = vhdr.split(os.sep)[-4:-2]
        rows.append(dict(subject=sub, session=ses, site=site_of(ses), home=sub[4:7],
                         visit=int(ses[4:6]), has_eeg=eeg_prefix(sub, ses) is not None))
    return pd.DataFrame(rows)


def _visual_rows(t: pd.DataFrame) -> pd.DataFrame:
    v = pd.to_numeric(t["value"], errors="coerce")
    return t[v.isin(VISUAL)].assign(value=v[v.isin(VISUAL)].astype(int)).reset_index(drop=True)


def trial_table(sub: str, ses: str) -> pd.DataFrame:
    """p3b stimulus trials with corrected + uncorrected sample (run clock, original fs).

    Alignment is done on stimulus rows only: in several sessions a fast keypress shares a
    timestamp with the stimulus in the raw table and the correction reorders the two, which
    breaks whole-table row alignment but not the stimulus sequence. Raises if the stimulus
    sequences differ or the p3b run cannot be located uniquely.
    """
    stem = os.path.join(PDL, f"{sub}_{ses}_task-all_events")
    raw = _visual_rows(pd.read_csv(stem + "-raw.tsv", sep="\t"))
    cor = _visual_rows(pd.read_csv(stem + "-pdcorrected.tsv", sep="\t"))
    if len(raw) == 0 or len(raw) != len(cor) or not np.array_equal(raw.value.values, cor.value.values):
        raise ValueError(f"{sub} {ses}: stimulus rows differ (raw {len(raw)}, cor {len(cor)})")
    err_s = (cor.onset.values - raw.onset.values).astype(float)

    p = prefix(sub, ses)
    bids = pd.read_csv(p + "_events.tsv", sep="\t")
    bv = pd.to_numeric(bids["value"], errors="coerce")
    stim = bids[bv.isin(LABEL.keys())].copy()
    stim["value"] = bv[stim.index].astype(int)
    vals = stim.value.values
    n = len(vals)
    sv = cor.value.values
    hits = [i for i in range(len(sv) - n + 1) if np.array_equal(sv[i:i + n], vals)]
    # disambiguate (n2pc also uses 111/222) by inter-onset intervals
    good = [i for i in hits
            if np.allclose(np.diff(cor.onset.values[i:i + n]), np.diff(stim.onset.values), atol=0.002)]
    if len(good) != 1:
        raise ValueError(f"{sub} {ses}: p3b run located {len(good)} times ({len(hits)} value hits)")
    i0 = good[0]

    fs = parse_vhdr(p + "_eeg.vhdr")["fs"]
    e = err_s[i0:i0 + n]
    cs = stim["sample"].values.astype(int)
    out = pd.DataFrame(dict(
        subject=sub, session=ses, site=site_of(ses), fs=fs,
        trial=np.arange(n), value=vals, label=[LABEL[v] for v in vals],
        cor_sample=cs, raw_sample=cs - np.round(e * fs).astype(int),
        err_ms=e * 1000.0,
    ))
    out["correction_failed"] = out.err_ms.abs() < 1.0
    return out


# ---------------------------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------------------------
def design_filters(orig_fs):
    """P300_pipeline.design_filters_at_orig_fs, verbatim maths."""
    nyq = orig_fs / 2.0
    N_hp, Wn_hp = buttord(0.5 / nyq, 0.4 / nyq, 1, 40)
    N_lp, Wn_lp = buttord(30 / nyq, 40 / nyq, 1, 80)
    return butter(N_hp, Wn_hp, btype="high", output="sos"), butter(N_lp, Wn_lp, btype="low", output="sos")


def _resample(x, orig_fs):
    g = math.gcd(int(TARGET_FS), int(orig_fs))
    return resample_poly(x, int(TARGET_FS) // g, int(orig_fs) // g, axis=-1)


def _gedai_clean(data_uv: np.ndarray, names: list, fs: float) -> np.ndarray:
    import mne
    from gedai import Gedai
    info = mne.create_info(names, fs, "eeg")
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)
    raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    g = Gedai()
    g.fit_raw(raw, verbose=False)
    clean = g.transform_raw(raw, verbose=False)
    return clean.get_data() * 1e6


def continuous(sub: str, ses: str, variant: str = "mine"):
    """(8, n) average-referenced µV at 80 Hz, plus the original fs (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"cont_{sub}_{ses}_{variant}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["x"], float(z["fs"])
    x, fs = _continuous(sub, ses, variant)
    np.savez(f, x=x.astype(np.float32), fs=fs)
    return x, fs


def _continuous(sub: str, ses: str, variant: str):
    p = eeg_prefix(sub, ses)
    if p is None:
        raise FileNotFoundError(f"{sub} {ses}: EEG content not present locally (git-annex)")
    base = variant[:-2] if is_v2(variant) else variant
    names = CH16 if base == "gedai" else CH8
    chans, fs = _read_named_channels(p, names)
    missing = [c for c in names if c not in chans]
    if missing:
        raise ValueError(f"{sub} {ses}: missing channels {missing}")
    x = np.vstack([chans[c] for c in names])
    sos_hp, sos_lp = design_filters(fs)
    x = sosfiltfilt(sos_hp, x, axis=-1)
    x = sosfiltfilt(sos_lp, x, axis=-1)
    if base == "gedai":
        x = _gedai_clean(x, names, fs)
    elif base != "mine":
        raise ValueError(variant)
    x = _resample(x, fs)
    x = x[[names.index(c) for c in CH8]]
    x = x - x.mean(axis=0, keepdims=True)
    return x, fs


# ---------------------------------------------------------------------------------------------
# epochs
# ---------------------------------------------------------------------------------------------
def _cut(x, samples80, ts=TS):
    """(n_trials, 8, len(ts)) baselined epochs; NaN-filled where the window leaves the recording."""
    out = np.full((len(samples80), x.shape[0], len(ts)), np.nan)
    for k, s in enumerate(samples80):
        idx = s + ts
        if idx[0] < 0 or idx[-1] >= x.shape[1]:
            continue
        ep = x[:, idx]
        out[k] = ep - ep[:, :len(BL)].mean(axis=1, keepdims=True)
    return out


def build(sub: str, ses: str, variant: str = "mine", thresh: float = 100.0, use_cache: bool = True):
    """Epochs for both marker sets over the common trial mask.

    Returns dict(X_cor, X_raw: (n, 8, 73) µV; y: (n,) 1/2; trials: table rows kept;
                 n_total, n_rejected).
    """
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{sub}_{ses}_{variant}_{int(thresh)}uv.npz")
    if use_cache and os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        tt = pd.DataFrame(z["trials"].tolist(), columns=list(z["cols"]))
        return dict(X_cor=z["X_cor"], X_raw=z["X_raw"], y=z["y"], trials=tt,
                    n_total=int(z["n_total"]), n_rejected=int(z["n_rejected"]))
    tt = trial_table(sub, ses)
    x, fs = continuous(sub, ses, variant)
    to80 = lambda s: np.round(s * TARGET_FS / fs).astype(int)
    Xc = _cut(x, to80(tt.cor_sample.values), grid(variant))
    Xr = _cut(x, to80(tt.raw_sample.values), grid(variant))
    ok = np.isfinite(Xc).all(axis=(1, 2)) & np.isfinite(Xr).all(axis=(1, 2))
    p2p_ok = lambda X: ~np.any(np.nan_to_num(X.max(axis=2) - X.min(axis=2), nan=np.inf) > thresh, axis=1)
    keep = ok & p2p_ok(Xc) & p2p_ok(Xr)
    kept = tt[keep].reset_index(drop=True)
    res = dict(X_cor=Xc[keep].astype(np.float32), X_raw=Xr[keep].astype(np.float32),
               y=kept.label.values.astype(int), trials=kept,
               n_total=len(tt), n_rejected=int((~keep).sum()))
    np.savez_compressed(f, X_cor=res["X_cor"], X_raw=res["X_raw"], y=res["y"],
                        trials=np.array(kept.values.tolist(), dtype=object),
                        cols=np.array(kept.columns), n_total=res["n_total"],
                        n_rejected=res["n_rejected"])
    return res


def features(X):
    """P300_pipeline classification features: 0.15-0.5 s window, flattened (trials, T*ch)
    in the pipeline's transpose(2,1,0) order (time-major, channel-minor)."""
    F = X[:, :, feat_idx(X.shape[-1])]          # (n, ch, 28) v1 / (n, ch, 48) v2
    return F.transpose(0, 2, 1).reshape(len(F), -1)


# ---------------------------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------------------------
N_TRAIN, N_VAL = 20, 5   # per class, chronological


def split_within(y: np.ndarray):
    """First N_TRAIN per class -> train, next N_VAL per class -> val, all others -> test.
    Returns index arrays, or None if a class has too few surviving trials."""
    tr, va = [], []
    for c in (1, 2):
        idx = np.where(y == c)[0]
        if len(idx) < N_TRAIN + N_VAL + 3:
            return None
        tr += list(idx[:N_TRAIN]); va += list(idx[N_TRAIN:N_TRAIN + N_VAL])
    tr, va = np.sort(tr), np.sort(va)
    te = np.setdiff1d(np.arange(len(y)), np.concatenate([tr, va]))
    return tr, va, te


if __name__ == "__main__":
    inv = inventory()
    print(inv.groupby("site").size())
    rows = []
    for r in inv.itertuples():
        try:
            t = trial_table(r.subject, r.session)
            rows.append(dict(subject=r.subject, session=r.session, site=r.site, ok=True,
                             n=len(t), n_tgt=int((t.label == 2).sum()),
                             err_med=float(t.err_ms.median()), err_sd=float(t.err_ms.std()),
                             n_failed=int(t.correction_failed.sum()), why=""))
        except Exception as ex:
            rows.append(dict(subject=r.subject, session=r.session, site=r.site, ok=False, why=str(ex)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(HERE, "p300_marker_inventory.csv"), index=False)
    print(d[~d.ok][["subject", "session", "why"]].to_string())
    print(d[d.ok].groupby("site")[["n", "n_tgt", "err_med", "err_sd", "n_failed"]].median())
    print("usable sessions:", int(d.ok.sum()), "/", len(d))
