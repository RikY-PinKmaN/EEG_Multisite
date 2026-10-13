#!/usr/bin/env python3
"""
raw_markers.py — recover the PRE-CORRECTION trigger position for every BIDS trial.

The `Uncorrected/` release ships, per session, two event tables over the *same*
recording timeline:

    <sub>_<ses>_task-all_events-raw.tsv          trigger where the amplifier logged it
    <sub>_<ses>_task-all_events-pdcorrected.tsv  trigger moved onto the photodiode onset

Row counts and `value` sequences are identical between the two, so the per-trial
marker error is a straight subtraction (ROADMAP step 1):

    err_ms = (corrected_onset - raw_onset) * 1000

The released BIDS, however, is split per task (n170 / n2pc / p3b) with its own
sample origin. This module maps each per-task BIDS event table into the session
table by locating the unique contiguous run with an identical `value` sequence,
then carries the error back onto the BIDS sample index:

    raw_sample_bids = bids_sample - (ta_corrected_sample - ta_raw_sample)

The offset between the two sample columns is exactly constant across every
visual stimulus event of a run (verified: 0 variance, 24/24 runs), so epoching a
BIDS recording at `raw_sample` reproduces the "no photodiode" condition on the
identical signal — no need to touch the multi-GB source recordings.

Caveat carried into the stats: at Manchester the correction also applied a
global -3 ms shift to *every* marker (keypresses included). `err_ms` is reported
as-is (it is what a photodiode-less lab would actually be wrong by);
`err_ms_rel` removes that global shift for latency interpretation.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from jitter_pipeline import parse_vhdr, ROOT

RAW_DIR = os.path.join(ROOT, "Uncorrected", "raw")
SUBJECTS = ["sub-SHE01", "sub-SHE02"]
SESSIONS = ["ses-01SHE", "ses-02LDS", "ses-03SHE", "ses-04MAN"]
TASKS = ["n170", "n2pc", "p3b"]
SITE_OF = {"ses-01SHE": "SHE", "ses-02LDS": "LDS", "ses-03SHE": "SHE", "ses-04MAN": "MAN"}
VISIT_OF = {"ses-01SHE": 1, "ses-02LDS": 1, "ses-03SHE": 2, "ses-04MAN": 1}

# stimulus trigger codes that carry a visual onset (these are the PD-corrected ones)
VISUAL_CODES = {100, 200, 300, 400, 111, 222}
# markers with no visual event: any residual shift on these is a global clock fix
NONVISUAL_CODES = {55, 10, 11}

CONDITION = {
    "n170": {100: "face", 200: "car", 300: "face_scr", 400: "car_scr"},
    "p3b": {111: "target", 222: "standard"},
    "n2pc": {111: "target_left", 222: "target_right"},
}
# the two-class contrast used for decoding, per paradigm
CONTRAST = {"n170": ("face", "car"), "p3b": ("target", "standard"), "n2pc": ("target_left", "target_right")}


def session_table(subject: str, session: str) -> pd.DataFrame:
    """Session-wide event table with raw and corrected onsets/samples side by side."""
    stem = os.path.join(RAW_DIR, f"{subject}_{session}_task-all_events")
    raw = pd.read_csv(stem + "-raw.tsv", sep="\t")
    cor = pd.read_csv(stem + "-pdcorrected.tsv", sep="\t")
    if len(raw) != len(cor) or not np.array_equal(raw["value"].values, cor["value"].values):
        raise ValueError(f"{subject} {session}: raw/corrected event tables are not row-aligned")
    t = pd.DataFrame({
        "value": raw["value"].astype(int),
        "raw_onset": raw["onset"].astype(float),
        "cor_onset": cor["onset"].astype(float),
        "raw_sample": raw["sample"].astype(int),
        "cor_sample": cor["sample"].astype(int),
    })
    t["err_ms"] = (t["cor_onset"] - t["raw_onset"]) * 1000.0
    nonvis = t.loc[t["value"].isin(NONVISUAL_CODES), "err_ms"]
    t.attrs["global_shift_ms"] = float(nonvis.median()) if len(nonvis) else 0.0
    return t


def _locate(session_values: np.ndarray, bids_values: np.ndarray) -> int:
    """Unique start index of the contiguous run in the session table matching a BIDS run."""
    n = len(bids_values)
    hits = [i for i in range(len(session_values) - n + 1)
            if np.array_equal(session_values[i:i + n], bids_values)]
    if len(hits) != 1:
        raise ValueError(f"expected exactly 1 alignment, found {len(hits)}")
    return hits[0]


def bids_prefix(subject: str, session: str, task: str) -> str:
    return os.path.join(ROOT, subject, session, "eeg", f"{subject}_{session}_task-{task}")


def run_trials(subject: str, session: str, task: str) -> pd.DataFrame | None:
    """Per-trial table for one BIDS run: corrected sample, raw sample, measured error."""
    prefix = bids_prefix(subject, session, task)
    if not os.path.exists(prefix + "_events.tsv"):
        return None
    bids = pd.read_csv(prefix + "_events.tsv", sep="\t")
    bids_values = pd.to_numeric(bids["value"], errors="coerce").values

    ses = session_table(subject, session)
    start = _locate(ses["value"].values, bids_values)
    seg = ses.iloc[start:start + len(bids)].reset_index(drop=True)

    fs = parse_vhdr(prefix + "_eeg.vhdr")["fs"]
    vis = seg["value"].isin(VISUAL_CODES).values
    if not vis.any():
        return None

    # sample-space offset between session table and this BIDS run (exact on visual events)
    delta_sample = (seg["cor_sample"].values - seg["raw_sample"].values)
    corr_sample = bids["sample"].values.astype(int)

    out = pd.DataFrame({
        "subject": subject,
        "session": session,
        "site": SITE_OF[session],
        "visit": VISIT_OF[session],
        "task": task,
        "value": seg["value"].values,
        "trial_type": bids["trial_type"].values,
        "fs": fs,
        "session_time_s": seg["raw_onset"].values,   # position within the whole session
        "run_time_s": bids["onset"].values.astype(float),
        "corr_sample": corr_sample,
        "raw_sample": corr_sample - delta_sample,
        "err_ms": seg["err_ms"].values,
    })[vis]
    out["condition"] = out["value"].map(CONDITION[task])
    out["err_ms_rel"] = out["err_ms"] - ses.attrs["global_shift_ms"]
    out["global_shift_ms"] = ses.attrs["global_shift_ms"]
    out["trial_index"] = np.arange(len(out))
    # a handful of trials kept their raw position (the correction failed there)
    out["correction_failed"] = out["err_ms"].abs() < 1.0
    return out.reset_index(drop=True)


def build_trial_table() -> pd.DataFrame:
    frames = []
    for sub in SUBJECTS:
        for ses in SESSIONS:
            for task in TASKS:
                r = run_trials(sub, ses, task)
                if r is not None:
                    frames.append(r)
    return pd.concat(frames, ignore_index=True)


def load_trial_table(cache=os.path.join(ROOT, "analysis", "trial_jitter.csv")) -> pd.DataFrame:
    if os.path.exists(cache):
        return pd.read_csv(cache)
    df = build_trial_table()
    df.to_csv(cache, index=False)
    return df


def _selftest():
    df = build_trial_table()
    assert len(df) == 24 * 0 + len(df) and len(df) > 6000, f"too few trials: {len(df)}"
    assert df.groupby(["subject", "session", "task"]).ngroups == 24, "expected 24 runs"
    # the raw marker must sit at or before the corrected one, and inside the recording
    good = ~df["correction_failed"]
    assert (df.loc[good, "raw_sample"] < df.loc[good, "corr_sample"]).all(), "raw marker not earlier"
    assert (df["raw_sample"] > 0).all(), "negative raw sample"
    # error must equal the sample-space shift it was derived from, to within one sample
    shift_ms = (df["corr_sample"] - df["raw_sample"]) / df["fs"] * 1000.0
    assert np.allclose(shift_ms, df["err_ms"], atol=1.01 / df["fs"].min() * 1000), "sample/onset disagree"
    # conditions fully resolved
    assert df["condition"].notna().all(), "unmapped trigger value"
    print(f"selftest OK — {len(df)} visual trials, 24 runs, "
          f"{df['correction_failed'].sum()} uncorrected trials")


if __name__ == "__main__":
    _selftest()
    df = build_trial_table()
    out = os.path.join(ROOT, "analysis", "trial_jitter.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} trials)")
    print(df.groupby(["site", "task"])["err_ms"].agg(["size", "mean", "std", "min", "max"]).round(2).to_string())
