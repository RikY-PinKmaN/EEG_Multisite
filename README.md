# EEG_Multisite

Analysis code for a traveling-subjects multi-site EEG study: 36 subjects, each recorded at
Sheffield (Bitbrain), Leeds (BioSemi) and Manchester (ANT). Each session has a photodiode, so
stimulus markers can be corrected to true screen onset, and the per-trial marker timing error
is measured directly.

This repository contains code only. The EEG data are in the BIDS release (GIN) and are not
included here.

## Layout

- `analysis/step1_timing.py`, `raw_markers.py`, `timing_extras.py`: marker timing error from
  raw vs photodiode-corrected events.
- `analysis/rt_bias.py`: the same error measured in reaction times.
- `analysis/p300_multisite.py`: shared loading, marker alignment and preprocessing helpers.
- `analysis/p300_baseline_v3.py`: current P300 baseline. 16 channels, 0.1–15 Hz zero-phase
  FIR, GEDAI, linked mastoids, epochs −0.1–0.8 s, 100 µV rejection, xDAWN + shrinkage LDA,
  5-fold CV. Stages: `preprocess`, `checks`, `classify`.
- `analysis/mastoid_check.py`, `mastoid_p300_check.py`: validation of the mastoid reference
  across sites.
- `P300_pipeline.py`: original P300 pipeline the analyses were ported from.

## Requirements

Python 3.10, mne, numpy, scipy, pandas, scikit-learn, pyriemann, gedai, joblib.
