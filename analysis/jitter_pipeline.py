#!/usr/bin/env python3
"""
jitter_pipeline.py — trigger/photodiode timing analysis for the EEG multisite dataset.

Background
----------
Each stimulus event has a digital trigger (in events.tsv / .vmrk) and a physical
photodiode (PD) light-onset recorded on the `PD` MISC channel. The "jitter" of
interest is the delay between the trigger and the true visual onset.

The RELEASED data is already photodiode-CORRECTED: each trigger was moved onto its
PD onset, so `PD_onset - corrected_trigger ~= 0` for successfully corrected trials.
This is not invertible (the raw trigger position is gone). What survives is:
  * the residual (PD_onset - corrected_trigger), dominated by sampling quantization, and
  * correction-FAILURE trials, where the trigger kept its raw position and the PD
    fires tens of ms later -- a direct (sparse) glimpse of the original jitter.

This module reads the BrainVision files directly (no MNE needed), detects the PD
onset for every visual stimulus event, and builds a tidy per-event table.

Track A hook: if a table of RAW (pre-correction) trigger samples becomes available,
`onset_relative_to()` computes the true jitter = PD_onset - raw_trigger unchanged.
"""
from __future__ import annotations
import json, re, glob, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISUAL_TASKS = ("n170", "n2pc", "p3b")   # tasks with per-trial visual stimuli + PD response
STIM_MIN_CODE = 100                        # stimulus trigger codes are >=100; keypress=55, blocks<100
FAILURE_MS = 15.0                          # residual above this => correction failed (>> 1-sample quant)

# ------------------------------------------------------------------ file readers
def parse_vhdr(vhdr_path: str) -> dict:
    """Return sampling rate, channel names and PD channel index from a BrainVision header."""
    txt = open(vhdr_path).read()
    nchan = int(re.search(r"NumberOfChannels=(\d+)", txt).group(1))
    fs = 1e6 / float(re.search(r"SamplingInterval=([\d.]+)", txt).group(1))  # us -> Hz
    chans = re.findall(r"Ch\d+=([^,]+),", txt)
    pd_idx = next((i for i, c in enumerate(chans) if c.strip().upper() == "PD"), None)
    return dict(nchan=nchan, fs=fs, chans=chans, pd_idx=pd_idx)


def read_pd_channel(prefix: str):
    """Memory-map the .eeg binary and return (pd_array, fs, meta). PD read lazily as one column."""
    meta = parse_vhdr(prefix + "_eeg.vhdr")
    if meta["pd_idx"] is None:
        return None, meta["fs"], meta
    mm = np.memmap(prefix + "_eeg.eeg", dtype="<f4", mode="r").reshape(-1, meta["nchan"])
    pd_sig = np.asarray(mm[:, meta["pd_idx"]], dtype=np.float64)   # copy only the PD column
    del mm
    return pd_sig, meta["fs"], meta


def read_sidecar(prefix: str) -> dict:
    with open(prefix + "_eeg.json") as f:
        return json.load(f)


def load_events(prefix: str) -> pd.DataFrame:
    ev = pd.read_csv(prefix + "_events.tsv", sep="\t", dtype={"value": str})
    ev["value_num"] = pd.to_numeric(ev["value"], errors="coerce")
    return ev


# ------------------------------------------------------------------ PD detection
def detect_pd_onset(pd_sig: np.ndarray, fs: float, trig: int,
                    base_pre_ms=(80, 40), search_pre_ms=40, search_post_ms=180,
                    gate_sigma=6.0, onset_sigma=5.0, sustain_ms=20):
    """
    Detect the photodiode light-onset near a trigger; return its time RELATIVE to the trigger.

    Robust leading-edge method (handles clean steps, weak Bitbrain steps, ANT onset-transients,
    and late correction-failure steps with one rule):
      1. baseline + robust noise (MAD) from a pre-trigger window;
      2. require a real deflection: |largest signed deviation| in the search window > gate_sigma*noise;
      3. onset = first sample that crosses a LOW threshold (onset_sigma*noise) in the deflection's
         direction AND stays beyond it for `sustain_ms` (rejects lone noise spikes).

    Returns dict(detected, onset_ms, snr, amp). onset_ms>0 => PD fires AFTER the trigger
    (a residual/uncorrected latency). Thresholding on the leading edge, not half of the peak,
    keeps the ANT overshoot transient from breaking the sustain test.
    """
    n = len(pd_sig)
    b0 = trig - int(base_pre_ms[0] / 1000 * fs)
    b1 = trig - int(base_pre_ms[1] / 1000 * fs)
    s0 = trig - int(search_pre_ms / 1000 * fs)
    s1 = trig + int(search_post_ms / 1000 * fs)
    if b0 < 0 or s1 >= n or b1 <= b0:
        return dict(detected=False, onset_ms=np.nan, snr=np.nan, amp=np.nan)
    base = np.median(pd_sig[b0:b1])
    noise = 1.4826 * np.median(np.abs(pd_sig[b0:b1] - base)) or np.std(pd_sig[b0:b1])
    noise = noise or 1e-9
    seg = pd_sig[s0:s1] - base
    trig_off = trig - s0                       # index of the trigger within seg
    pk = seg[int(np.argmax(np.abs(seg)))]      # signed peak deflection (sets direction)
    snr = abs(pk) / noise
    if abs(pk) < gate_sigma * noise:
        return dict(detected=False, onset_ms=np.nan, snr=snr, amp=abs(pk))
    sign = 1.0 if pk > 0 else -1.0
    thr = sign * onset_sigma * noise
    above = (seg >= thr) if sign > 0 else (seg <= thr)
    sustain = max(1, int(sustain_ms / 1000 * fs))
    onset = None
    for i in np.where(above)[0]:
        if i + sustain <= len(seg):
            blk = seg[i:i + sustain]
            if (np.median(blk) >= thr) if sign > 0 else (np.median(blk) <= thr):
                onset = i
                break
    if onset is None:
        return dict(detected=False, onset_ms=np.nan, snr=snr, amp=abs(pk))
    onset_ms = (onset - trig_off) / fs * 1000.0
    return dict(detected=True, onset_ms=onset_ms, snr=snr, amp=abs(pk))


def onset_relative_to(pd_sig, fs, ref_sample, **kw):
    """Track A hook: same detector, for a RAW trigger sample -> returns true jitter in ms."""
    return detect_pd_onset(pd_sig, fs, ref_sample, **kw)


# ------------------------------------------------------------------ table builder
def site_of(session: str) -> str:
    return re.sub(r"^ses-\d+", "", session)          # 'ses-02LDS' -> 'LDS'


def scans_order(sub_dir: str, session: str) -> dict:
    """Map task -> acquisition order index within a session (from scans.tsv acq_time)."""
    p = os.path.join(sub_dir, session, f"{os.path.basename(sub_dir)}_{session}_scans.tsv")
    if not os.path.exists(p):
        return {}
    s = pd.read_csv(p, sep="\t")
    s["task"] = s["filename"].str.extract(r"task-([a-z0-9]+)_")
    s = s.dropna(subset=["acq_time"]).sort_values("acq_time").reset_index(drop=True)
    return {row["task"]: i for i, row in s.iterrows()}


def build_run(prefix, subject, session, task) -> pd.DataFrame | None:
    pd_sig, fs, meta = read_pd_channel(prefix)
    if pd_sig is None:
        return None
    side = read_sidecar(prefix)
    ev = load_events(prefix)
    stim = ev[ev["value_num"] >= STIM_MIN_CODE].reset_index(drop=True)
    rows = []
    for k, r in stim.iterrows():
        d = detect_pd_onset(pd_sig, fs, int(r["sample"]))
        rows.append(dict(
            subject=subject, session=session, site=site_of(session), task=task,
            trial_index=k, onset_s=r["onset"], trigger_value=int(r["value_num"]),
            trial_type=r["trial_type"], residual_ms=d["onset_ms"], pd_snr=d["snr"],
            pd_amp=d["amp"], pd_detected=d["detected"], fs=fs,
            manufacturer=side.get("Manufacturer"), institution=side.get("InstitutionName"),
        ))
    df = pd.DataFrame(rows)
    df["is_failure"] = df["pd_detected"] & (df["residual_ms"].abs() > FAILURE_MS)
    return df


def build_jitter_table(subjects=None, tasks=VISUAL_TASKS) -> pd.DataFrame:
    subjects = subjects or [os.path.basename(p) for p in sorted(glob.glob(os.path.join(ROOT, "sub-*")))]
    frames = []
    for sub in subjects:
        sub_dir = os.path.join(ROOT, sub)
        for ses_path in sorted(glob.glob(os.path.join(sub_dir, "ses-*"))):
            session = os.path.basename(ses_path)
            order = scans_order(sub_dir, session)
            for task in tasks:
                prefix = os.path.join(ses_path, "eeg", f"{sub}_{session}_task-{task}")
                if not os.path.exists(prefix + "_eeg.vhdr"):
                    continue
                run = build_run(prefix, sub, session, task)
                if run is not None:
                    run["run_order"] = order.get(task, np.nan)
                    frames.append(run)
    return pd.concat(frames, ignore_index=True)


def main():
    out_csv = os.path.join(ROOT, "analysis", "jitter_table.csv")
    df = build_jitter_table()
    df.to_csv(out_csv, index=False)
    print(f"[jitter_pipeline] wrote {out_csv}  ({len(df)} events, "
          f"{df['subject'].nunique()} subjects, {df['session'].nunique()} sessions)")
    # quick per-run summary
    g = (df[df.pd_detected]
         .groupby(["subject", "session", "site", "task"])
         .agg(n=("residual_ms", "size"),
              fs=("fs", "first"),
              res_mean=("residual_ms", "mean"),
              res_sd=("residual_ms", "std"),
              res_max=("residual_ms", "max"),
              pd_snr=("pd_snr", "median"),
              n_fail=("is_failure", "sum"))
         .round(2).reset_index())
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()
