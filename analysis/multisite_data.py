"""Data access for the multi-site P300 baseline.

Reads the BIDS release directly (BrainVision .vhdr/.eeg, events.tsv) and builds the per-session
trial table with photodiode-corrected and uncorrected (raw) stimulus samples. No MNE needed here.

Data roots are searched in order. Set EEG_MULTISITE_DATA to your copy of the release
(several roots can be given, separated by ':'). Default: <repo>/Multisite-EEG/EEGMultiSite_Raw,
then the repo root itself (for subject folders placed next to the code).

Marker sets: 'cor' = photodiode-corrected (BIDS events.tsv); 'raw' = the uncorrected marker,
taken from the row-aligned *_events-raw.tsv / *_events-pdcorrected.tsv pair in
derivatives/data-quality-metrics/pd_latency. raw_sample = cor_sample - round(err_s * fs).
"""
from __future__ import annotations
import glob
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
_env = os.environ.get("EEG_MULTISITE_DATA")
DATA_ROOTS = (_env.split(os.pathsep) if _env
              else [os.path.join(REPO, "Multisite-EEG", "EEGMultiSite_Raw"), REPO])
PDL_REL = os.path.join("derivatives", "data-quality-metrics", "pd_latency")

CH16 = ["F7", "F3", "Fz", "F4", "F8", "FCz", "C3", "C4", "Pz", "POz",
        "PO7", "PO8", "P7", "P8", "M1", "M2"]
VISUAL = {100, 200, 300, 400, 111, 222}      # visual stimulus codes shared by n170/p3b/n2pc
LABEL = {111: 2, 222: 1}                     # p3b: 2 = target, 1 = non-target


# ---------------------------------------------------------------------------------------------
# BrainVision readers
# ---------------------------------------------------------------------------------------------
def parse_vhdr(vhdr_path: str) -> dict:
    """Sampling rate, channel names and per-channel resolution (µV per stored unit)."""
    txt = open(vhdr_path).read()
    nchan = int(re.search(r"NumberOfChannels=(\d+)", txt).group(1))
    fs = 1e6 / float(re.search(r"SamplingInterval=([\d.]+)", txt).group(1))   # µs -> Hz
    chans = re.findall(r"Ch\d+=([^,]+),", txt)
    res = {}
    for line in txt.splitlines():
        mm = re.match(r"Ch\d+=([^,]+),[^,]*,([^,]*),", line)
        if mm:
            res[mm.group(1).strip()] = float(mm.group(2)) if mm.group(2).strip() else 1.0
    return dict(nchan=nchan, fs=fs, chans=chans, resolution=res)


def read_channels(prefix: str, names: list[str]):
    """{name: signal in µV} for the requested channels that exist, plus the sampling rate."""
    m = parse_vhdr(prefix + "_eeg.vhdr")
    idx = {c.strip(): i for i, c in enumerate(m["chans"])}
    mm = np.memmap(prefix + "_eeg.eeg", dtype="<f4", mode="r").reshape(-1, m["nchan"])
    out = {}
    for nm in names:
        if nm in idx:
            out[nm] = np.asarray(mm[:, idx[nm]], dtype=float) * m["resolution"].get(nm, 1.0)
    del mm
    return out, m["fs"]


# ---------------------------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------------------------
def site_of(session: str) -> str:
    return session[-3:]


def _prefix(root: str, sub: str, ses: str, task: str) -> str:
    return os.path.join(root, sub, ses, "eeg", f"{sub}_{ses}_task-{task}")


def prefix(sub: str, ses: str, task: str = "p3b") -> str | None:
    """First root holding the session's header and events (small text files, never annexed)."""
    for root in DATA_ROOTS:
        p = _prefix(root, sub, ses, task)
        if os.path.exists(p + "_eeg.vhdr"):
            return p
    return None


def eeg_prefix(sub: str, ses: str, task: str = "p3b") -> str | None:
    """First root whose _eeg.eeg holds real signal data. A fresh GIN clone has git-annex
    placeholders instead (a few bytes) until the content is fetched."""
    for root in DATA_ROOTS:
        p = _prefix(root, sub, ses, task)
        if os.path.exists(p + "_eeg.eeg") and os.path.getsize(p + "_eeg.eeg") > 10_000:
            return p
    return None


def inventory() -> pd.DataFrame:
    """One row per p3b session found in any data root; has_eeg = signal content is present."""
    seen, rows = set(), []
    for root in DATA_ROOTS:
        for vhdr in sorted(glob.glob(os.path.join(root, "sub-*", "ses-*", "eeg", "*task-p3b_eeg.vhdr"))):
            sub, ses = vhdr.split(os.sep)[-4:-2]
            if (sub, ses) in seen:
                continue
            seen.add((sub, ses))
            rows.append(dict(subject=sub, session=ses, site=site_of(ses), home=sub[4:7],
                             visit=int(ses[4:6]), has_eeg=eeg_prefix(sub, ses) is not None))
    return pd.DataFrame(rows).sort_values(["subject", "session"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# trial table
# ---------------------------------------------------------------------------------------------
def _visual_rows(t: pd.DataFrame) -> pd.DataFrame:
    v = pd.to_numeric(t["value"], errors="coerce")
    return t[v.isin(VISUAL)].assign(value=v[v.isin(VISUAL)].astype(int)).reset_index(drop=True)


def _pd_latency_stem(sub: str, ses: str) -> str:
    for root in DATA_ROOTS:
        stem = os.path.join(root, PDL_REL, f"{sub}_{ses}_task-all_events")
        if os.path.exists(stem + "-raw.tsv") and os.path.exists(stem + "-pdcorrected.tsv"):
            return stem
    raise FileNotFoundError(f"{sub} {ses}: no events-raw / events-pdcorrected pair under "
                            f"{PDL_REL} in {DATA_ROOTS}")


def trial_table(sub: str, ses: str) -> pd.DataFrame:
    """p3b stimulus trials with corrected + uncorrected sample (run clock, original fs).

    Alignment is done on stimulus rows only: in several sessions a fast keypress shares a
    timestamp with the stimulus in the raw table and the correction reorders the two, which
    breaks whole-table row alignment but not the stimulus sequence. Raises if the stimulus
    sequences differ or the p3b run cannot be located uniquely.
    """
    stem = _pd_latency_stem(sub, ses)
    raw = _visual_rows(pd.read_csv(stem + "-raw.tsv", sep="\t"))
    cor = _visual_rows(pd.read_csv(stem + "-pdcorrected.tsv", sep="\t"))
    if len(raw) == 0 or len(raw) != len(cor) or not np.array_equal(raw.value.values, cor.value.values):
        raise ValueError(f"{sub} {ses}: stimulus rows differ (raw {len(raw)}, cor {len(cor)})")
    err_s = (cor.onset.values - raw.onset.values).astype(float)

    p = prefix(sub, ses)
    if p is None:
        raise FileNotFoundError(f"{sub} {ses}: no p3b header in {DATA_ROOTS}")
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


if __name__ == "__main__":
    inv = inventory()
    print(f"data roots: {DATA_ROOTS}")
    print(inv.groupby("site").has_eeg.agg(sessions="size", with_signal="sum"))
