# EEG_Multisite: P300 baseline pipeline

Preprocessing and within-session classification of the p3b task in the traveling-subjects
multi-site EEG dataset (36 subjects, each recorded at Sheffield/Bitbrain, Leeds/BioSemi and
Manchester/ANT). Every session has a photodiode, so each stimulus marker exists in two versions:
the uncorrected trigger and the marker corrected to true screen onset. The pipeline epochs on
both and classifies target vs non-target with the same trials, folds and model, so the two
marker sets can be compared directly.

This repository is code only. The EEG data are the BIDS release on GIN (not included).

## Files

| File | Role |
|---|---|
| `analysis/p300_baseline_v3.py` | the pipeline: `preprocess`, `checks`, `classify` stages |
| `analysis/multisite_data.py` | data access: BrainVision reader, session inventory, trial table with corrected + uncorrected markers |
| `analysis/meeting_figs.py` | figures from the pipeline outputs: grand-average ERPs, topographies, classification strip plots |
| `requirements.txt` | pinned versions the results were produced with |

## Setup

```bash
git clone https://github.com/RikY-PinKmaN/EEG_Multisite.git
cd EEG_Multisite
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The pipeline reads the BIDS release `EEGMultiSite_Raw` (GIN, `dominik.welke/EEGMultiSite_Raw`;
ask the dataset owner for access). A fresh `gin get` gives git-annex placeholders for the
`.eeg` binaries, so fetch the p3b content afterwards:

```bash
gin get dominik.welke/EEGMultiSite_Raw
cd EEGMultiSite_Raw
gin get-content sub-*/ses-*/eeg/*task-p3b_eeg.eeg
```

What the pipeline needs from the release:

- `sub-*/ses-*/eeg/*_task-p3b_eeg.{vhdr,eeg}` and `*_task-p3b_events.tsv` (corrected markers)
- `derivatives/data-quality-metrics/pd_latency/*_task-all_events-raw.tsv` and
  `*_events-pdcorrected.tsv` (row-aligned pair that gives the uncorrected marker per trial)

Point the code at your copy:

```bash
export EEG_MULTISITE_DATA=/path/to/EEGMultiSite_Raw
```

Without the variable it looks in `Multisite-EEG/EEGMultiSite_Raw` inside the repo, then in the
repo root. Check what it sees:

```bash
python analysis/multisite_data.py        # sessions per site, and how many have signal content
```

## Run

```bash
# one session first (writes _baseline_cache/<sub>_<ses>_{gedai,nogedai}.npz next to the script)
python analysis/p300_baseline_v3.py preprocess sub-SHE01 ses-01SHE

# all sessions with signal content; EEG_MULTISITE_JOBS sets the parallel workers (default 8)
EEG_MULTISITE_JOBS=4 python analysis/p300_baseline_v3.py preprocess

python analysis/p300_baseline_v3.py checks
python analysis/p300_baseline_v3.py classify
```

Outputs, all written to `analysis/`:

| Stage | Writes |
|---|---|
| `preprocess` | `_baseline_cache/*.npz` (cleaned 16-channel continuous data at 80 Hz), `baseline_sessions.csv` (status per session), `baseline_bad_channels.csv` |
| `checks` | `baseline_checks.csv` (P300 at Pz with/without GEDAI, trial yield), `baseline_mastoid_ratio.csv` |
| `classify` | `baseline_scores.csv` (one row per session, config, marker set and CV repeat) |

`preprocess` can be re-run on a subset: it replaces those rows in `baseline_sessions.csv` and
keeps the rest. `checks` and `classify` only use sessions whose status is `ok`.

## Figures

After `preprocess` and `classify` have run:

```bash
python analysis/meeting_figs.py
```

It rebuilds the session ERPs from `_baseline_cache/` with the same epoching, rejection and
session inclusion the classifier used, and reads `baseline_scores.csv`. Writes to
`analysis/figures/`:

| File | Content |
|---|---|
| `meeting_erp_pz_gedai.png`, `meeting_erp_pz_nogedai.png`, `meeting_erp_fcz_gedai.png` | grand-average target / non-target ERPs per site (corrected solid, uncorrected dashed), difference waves, per-session P3b amplitude and latency |
| `meeting_gedai_vs_nogedai_pz.png` | Pz difference wave with and without GEDAI on the sessions both variants classify |
| `meeting_topo_gedai.png`, `meeting_topo_nogedai.png` | target − non-target topographies per site in 100 ms windows and the 0.3–0.6 s mean |
| `meeting_classification.png` | per-session balanced accuracy and ROC-AUC, corrected vs uncorrected markers, with the shuffled-label chance line |

It also writes `analysis/meeting_summary.csv` (classification medians per site) and prints the
paired Wilcoxon test of corrected vs uncorrected markers.

## Method

Fixed before any classification result was looked at. Identical at all three sites.

Per session:

1. Load the 16 channels common to every cap, in µV at the original rate (256 / 2048 / 1000 Hz):
   F7 F3 Fz F4 F8 FCz C3 C4 Pz POz PO7 PO8 P7 P8 M1 M2.
2. Band-pass 0.1–15 Hz, zero-phase FIR (MNE `firwin`), at the original rate.
3. Resample to 256 Hz.
4. Detect bad channels on the average-referenced data (detection only):
   flat = robust SD < 1 µV; noisy = robust z of channel SD > 5 and SD > 2× the median channel;
   dead mastoid pair = median 2-s-window corr(M1, M2) < 0.2.
5. Drop the session if M1 or M2 is bad (the reference cannot be interpolated) or if more than
   one other channel is bad.
6. Two variants from the same filtered data: `gedai` (average reference, `Gedai()` with library
   defaults, bad channel excluded from the fit) and `nogedai`.
7. Interpolate the bad channel, if any (spherical spline).
8. Resample to 80 Hz and cache all 16 channels. The reference is applied at epoch time.
9. Re-reference to linked mastoids (M1+M2)/2 and keep the 14 scalp channels.
10. Epoch on both marker sets: −0.1 to 0.8 s, baseline −0.1 to 0 s.
11. Reject trials over 100 µV peak-to-peak on any channel under either marker set. The surviving
    common trial mask is used for both marker sets.
12. Classify only sessions with at least 20 surviving targets.

Classifier, one sklearn pipeline so xDAWN is fit inside each training fold:
crop 0.15–0.5 s → pyriemann `Xdawn(nfilter=4)` → flatten → `LDA(solver="lsqr", shrinkage="auto")`.
Stratified 5-fold CV, 10 repeats, seed 0, same folds for both marker sets. Metrics: balanced
accuracy and ROC-AUC. `classify` also runs the sensitivity configs (no GEDAI, 0.2–0.6 s window,
nfilter 2) and a shuffled-label control.

`checks` prints the filter response, the Pz P300 (target − non-target, 0.3–0.6 s) with and
without GEDAI, the stability of the linked-mastoid reference versus a 14-channel average, and
trial yield per site.
