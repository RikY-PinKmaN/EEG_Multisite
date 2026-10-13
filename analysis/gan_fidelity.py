#!/usr/bin/env python3
"""G1 — does a generator inherit the marker timing error of its training data?

See GAN_ROADMAP.md §G1. The prediction, registered before any generator saw this data:
the synthetic ERP latency from a generator trained on raw-marker epochs, minus that from a
generator trained on corrected-marker epochs, equals the session's measured marker error
~1:1. Falsified if the difference regresses toward zero.

Per subject x session x task, two generators are trained on the SAME trials, differing only
in which marker each trial was cut at. Everything else is held fixed, including the
normalisation statistics — those are computed once on the corrected epochs and applied to
both, so the two generators live in the same amplitude space and the only difference between
them is timing.

The measurement is ANCHORED (G0.6 established why): each synthetic difference wave is
compared against a common clean reference — the real corrected-marker difference wave,
averaged over every trial of the run — and the two estimates are subtracted. Comparing the
two synthetic waves directly would put noise on both sides and roughly double the error.

    <venv>/bin/python gan_fidelity.py --limit 1          # pilot, one unit
    <venv>/bin/python gan_fidelity.py                    # full run (~12 h)

Needs TensorFlow: use the EEG_scripts venv interpreter (see gan_model.py).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from jitter_pipeline import ROOT
from raw_markers import build_trial_table, SITE_OF
from erp_impact import PARADIGM          # single source of truth for chans/window
import epochs as EP
import gan_model as G
from gan_lag_check import best_lag

OUT_CSV = os.path.join(ROOT, "analysis", "gan_fidelity.csv")
CLIP_SIGMA = 3.0
N_SYNTH_PER_CLASS = 200
LAG_SEARCH_MS = 300.0
# DEPRECATED — do not gate on this. Peak correlation against spectrum-matched surrogates has
# a chance median of 0.560, so this 0.50 threshold sits BELOW chance and admits generators
# that learned nothing. Kept only so the `usable` column in existing CSVs stays reproducible.
# See gan_fidelity_stats.py and GAN_ROADMAP.md's 8 Aug outcome note.
MIN_SHAPE_R = 0.50
# The replacement reliability gate: two independent synthetic batches from the same generator
# must recover lags within this many ms of each other. Derived from the effect — the smallest
# marker error to resolve is Leeds' 23.9 ms, so a test-retest spread beyond ~10 ms (under
# half of it) means the lag cannot support attributing a displacement. Reported per unit
# alongside the raw disagreement, so the threshold can be revisited against real data.
SPLIT_MAX_MS = 10.0


def normalise(X, stats=None):
    """Per-channel z-score -> clip at CLIP_SIGMA -> scale to [-1, 1].

    Follows `normalize_per_channel` in the source P300 pipeline, so the generator sees the
    amplitude range its tanh output layer was designed for. Pass `stats` to reuse another
    array's statistics — required here so corrected and raw share one space.
    """
    X = np.asarray(X, dtype=np.float32)
    if stats is None:
        mu = X.mean(axis=(0, 2), keepdims=True)
        sd = X.std(axis=(0, 2), keepdims=True) + 1e-8
        stats = (mu, sd)
    mu, sd = stats
    return np.clip((X - mu) / sd / CLIP_SIGMA, -1.0, 1.0).astype(np.float32), stats


def diff_wave(X, y, chan_idx):
    """Class-difference wave pooled over a paradigm's channels -> (time,).

    Sign convention copied from erp_impact.pooled_diff: class 0 minus class 1.
    """
    r = X[:, chan_idx, :].mean(1)
    return r[y == 0].mean(0) - r[y == 1].mean(0)


def synth_diff_wave(gen, chan_idx, n_per_class, time_points):
    """Difference wave of generated data, averaged over many samples to damp generator noise."""
    a = G.generate_synthetic(gen, n_per_class, 0)
    b = G.generate_synthetic(gen, n_per_class, 1)
    assert a.shape == (n_per_class, G.NCHAN, time_points), a.shape
    return a[:, chan_idx, :].mean(1).mean(0) - b[:, chan_idx, :].mean(1).mean(0)


def split_half_lags(gen, anchor, chan_idx, n_per_class, time_points, fs):
    """Lag measured on two INDEPENDENT synthetic batches from the same generator.

    The reliability check that peak correlation could not provide. If a generator has really
    learned a component at some latency, two independent draws agree; if the recovered lag is
    an arbitrary alignment of a wave that never learned the component, they do not. Unlike a
    correlation threshold this needs no null model — it is a direct test-retest.

    Motivated by unit 2 of the first full run: the raw-trained P3b generator returned
    -131.6 ms where +73.7 was predicted, and a lag-ambiguity explanation was tested and
    ruled out (a displaced real P3b wave plus noise recovers cleanly, no competing peaks).
    The remaining explanation is that the generator produced a wave with no recoverable
    latency, which a split-half check detects directly.
    """
    la = best_lag(anchor, synth_diff_wave(gen, chan_idx, n_per_class, time_points), fs)
    lb = best_lag(anchor, synth_diff_wave(gen, chan_idx, n_per_class, time_points), fs)
    return la, lb, abs(la - lb)


def run_unit(sub, ses, task, group, epochs, n_synth, verbose=True, wave_dir=None):
    """Train both generators for one subject x session x task and measure the lags."""
    d = EP.build(sub, ses, task, group)
    if d is None:
        return None
    spec = PARADIGM[task]
    chans = list(d["chans"])
    chan_idx = [chans.index(c) for c in spec["chans"] if c in chans]
    if not chan_idx:
        return None

    core = d["core"]
    y = d["y"].astype(np.int32)
    Xc = d["X_cor"][:, :, core].astype(np.float32)
    Xr = d["X_raw"][:, :, core].astype(np.float32)
    assert Xc.shape == Xr.shape, "corrected and raw epochs must be row-aligned"
    assert Xc.shape[2] == G.TIME_POINTS, f"{Xc.shape[2]} samples, expected {G.TIME_POINTS}"

    # Shared normalisation: stats from corrected, applied to both.
    Xc_n, stats = normalise(Xc)
    Xr_n, _ = normalise(Xr, stats)

    fs = float(d["fs"])
    w_start, w_end = G.component_window(task)
    anchor = diff_wave(Xc_n, y, chan_idx)          # real corrected wave = the clean anchor

    # Control: reproduce the known real-data displacement with this same instrument.
    real_raw_lag = best_lag(anchor, diff_wave(Xr_n, y, chan_idx), fs)

    out, waves = {}, {"anchor": anchor}
    for marker, Xn in (("corrected", Xc_n), ("raw", Xr_n)):
        tmpl = G.class_templates(Xn, y, w_start, w_end)
        G.assert_template_provenance(tmpl, Xn, y, w_start, w_end)
        t0 = time.time()
        gen, hist = G.train_cgan(Xn, y, G.TIME_POINTS, epochs=epochs, batch_size=16,
                                 templates=tmpl, window=(w_start, w_end), verbose=verbose)
        sw = synth_diff_wave(gen, chan_idx, n_synth, G.TIME_POINTS)
        lag, peak_r = best_lag(anchor, sw, fs, return_r=True)
        la, lb, split = split_half_lags(gen, anchor, chan_idx, n_synth, G.TIME_POINTS, fs)
        waves[marker] = sw            # persisted so any later diagnosis needs no retraining
        out[marker] = dict(
            lag=lag,
            split_lag_a=la, split_lag_b=lb, split_disagreement_ms=split,
            # Correlation AT the best lag, as erp_impact reports it. Zero-lag correlation
            # would be wrong here by construction: the raw-trained generator's wave is
            # displaced ~90 ms, so it *should* be uncorrelated at zero lag.
            shape_r=peak_r,
            saturated=bool(abs(lag) >= LAG_SEARCH_MS - 1000.0 / fs),
            epochs_run=len(hist),
            gen_total=float(hist[-1]["gen_total"]),
            train_s=time.time() - t0,
        )

    if wave_dir:
        os.makedirs(wave_dir, exist_ok=True)
        np.savez(os.path.join(wave_dir, f"{sub}_{ses}_{task}.npz"), fs=fs, **waves)

    err = float(np.mean(d["err_ms"]))
    est = out["raw"]["lag"] - out["corrected"]["lag"]
    # A lag only means something if the synthetic wave actually resembles the component at
    # its own latency. `usable` is a sanity gate on the generator, not on the hypothesis:
    # both gated and ungated summaries are reported so the gate cannot hide a result.
    usable = (min(out["corrected"]["shape_r"], out["raw"]["shape_r"]) >= MIN_SHAPE_R
              and not out["corrected"]["saturated"] and not out["raw"]["saturated"])
    worst_split = max(out["corrected"]["split_disagreement_ms"],
                      out["raw"]["split_disagreement_ms"])
    return dict(
        split_disagreement_corrected_ms=out["corrected"]["split_disagreement_ms"],
        split_disagreement_raw_ms=out["raw"]["split_disagreement_ms"],
        worst_split_ms=worst_split,
        reliable=bool(worst_split <= SPLIT_MAX_MS and not out["corrected"]["saturated"]
                      and not out["raw"]["saturated"]),
        subject=sub, session=ses, site=SITE_OF[ses], task=task,
        n_trials=int(Xc.shape[0]),
        measured_err_ms=err,
        real_raw_lag_ms=real_raw_lag,
        usable=usable,
        saturated=out["corrected"]["saturated"] or out["raw"]["saturated"],
        synth_lag_corrected_ms=out["corrected"]["lag"],
        synth_lag_raw_ms=out["raw"]["lag"],
        synth_lag_diff_ms=est,
        inherited_fraction=est / err if err else np.nan,
        residual_ms=est - err,
        shape_r_corrected=out["corrected"]["shape_r"],
        shape_r_raw=out["raw"]["shape_r"],
        epochs_corrected=out["corrected"]["epochs_run"],
        epochs_raw=out["raw"]["epochs_run"],
        train_s=out["corrected"]["train_s"] + out["raw"]["train_s"],
    )


def figure(df, path):
    """The G1 deliverable: synthetic displacement against measured marker error.

    Deliberately the same form as the real-data panel in erp_impact.py's summary figure, so
    the two can sit side by side: the EEG carries the error, and so does everything generated
    from it. Open markers are the real raw-marker waves (the control), filled markers are the
    generators.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from erp_impact import SITE_COL

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    lim = max(10.0, float(np.nanmax([df.measured_err_ms.max(),
                                     df.synth_lag_diff_ms.abs().max()])) * 1.15)
    ax.plot([0, lim], [0, lim], ls=":", c="0.35", lw=1.2, zorder=1, label="identity")

    # Reliability comes from the split-half gate where the CSV has it; older CSVs predate it
    # and fall back to "show everything", never to the discredited correlation gate.
    ok = df["reliable"] if "reliable" in df.columns else pd.Series(True, index=df.index)
    seen_task = set()
    for site, g in df.groupby("site"):
        c = SITE_COL.get(site, "0.4")
        ax.scatter(g.measured_err_ms, g.real_raw_lag_ms, s=52, facecolors="none",
                   edgecolors=c, lw=1.4, zorder=2)
        for task, gg in g.groupby("task"):
            mk = {"n170": "o", "p3b": "s"}.get(task, "D")
            good = ok.reindex(gg.index).fillna(True).astype(bool)
            lbl = None if task in seen_task else f"{task} (filled = split-half stable)"
            seen_task.add(task)
            ax.scatter(gg.measured_err_ms[good], gg.synth_lag_diff_ms[good], s=80, c=c,
                       marker=mk, edgecolors="white", lw=0.8, zorder=4, label=lbl)
            ax.scatter(gg.measured_err_ms[~good], gg.synth_lag_diff_ms[~good], s=80,
                       facecolors="none", edgecolors=c, marker=mk, lw=1.2,
                       alpha=0.6, zorder=3)

    ax.set_xlabel("measured marker error (ms)")
    ax.set_ylabel("displacement of the SYNTHETIC ERP (ms)")
    ax.set_title("Does the generator inherit its training set's timing error?\n"
                 "open = real raw-marker wave (control)\n"
                 "filled = generator · X = failed the shape gate", fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(min(0, float(df.synth_lag_diff_ms.min()) * 1.15), lim)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=G.GAN_EPOCHS)
    ap.add_argument("--n-synth", type=int, default=N_SYNTH_PER_CLASS)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N units — use 1 for a pilot")
    ap.add_argument("--tasks", default=",".join(PARADIGM))
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--figure-only", action="store_true",
                    help="replot from an existing CSV without retraining")
    ap.add_argument("--wave-dir", default=os.path.join(ROOT, "analysis", "_gan_waves"),
                    help="where to persist synthetic difference waves for later diagnosis")
    args = ap.parse_args()
    tasks = [t for t in args.tasks.split(",") if t]

    fig_path = os.path.join(ROOT, "analysis", "figures", "g1_gan_fidelity.png")
    if args.figure_only:
        figure(pd.read_csv(args.out), fig_path)
        return

    tt = build_trial_table()
    units = [(sub, ses, task, g)
             for (sub, ses), g in tt.groupby(["subject", "session"])
             for task in tasks]
    if args.limit:
        units = units[:args.limit]

    print(f"G1 — {len(units)} unit(s), {args.epochs} epochs, "
          f"{args.n_synth} synthetic/class, 2 generators each")
    rows, t0 = [], time.time()
    for i, (sub, ses, task, g) in enumerate(units, 1):
        print(f"\n[{i}/{len(units)}] {sub} {ses} {task}")
        r = run_unit(sub, ses, task, g, args.epochs, args.n_synth,
                     wave_dir=args.wave_dir)
        if r is None:
            print("  skipped — no epochs or channels for this unit")
            continue
        rows.append(r)
        print(f"  measured error {r['measured_err_ms']:7.2f} ms | "
              f"real-raw lag {r['real_raw_lag_ms']:7.2f} | "
              f"synth diff {r['synth_lag_diff_ms']:7.2f} "
              f"({r['inherited_fraction']:+.2f}x) | "
              f"split {r['worst_split_ms']:6.2f} ms "
              f"{'RELIABLE' if r['reliable'] else 'unreliable'} | "
              f"{r['train_s']/60:.1f} min")
        pd.DataFrame(rows).to_csv(args.out, index=False)   # checkpoint every unit

    if not rows:
        print("no units completed")
        return
    df = pd.DataFrame(rows)
    print(f"\nwrote {args.out} ({len(df)} rows) in {(time.time()-t0)/60:.1f} min")
    print(f"control — real raw-marker lag vs measured error: mean abs residual "
          f"{(df.real_raw_lag_ms - df.measured_err_ms).abs().mean():.2f} ms "
          f"(this reproduces erp_impact.py and validates the pipeline)")
    ok = df[df.usable]
    print(f"usable units: {len(ok)}/{len(df)} "
          f"(shape r >= {MIN_SHAPE_R} at best lag, no search saturation)")
    for label, sub in (("ALL   ", df), ("USABLE", ok)):
        if not len(sub):
            print(f"  {label}: none")
            continue
        print(f"  {label}: inherited fraction median {sub.inherited_fraction.median():+.3f} "
              f"| residual mean abs {sub.residual_ms.abs().mean():6.2f} ms "
              f"| shape r min {sub[['shape_r_corrected','shape_r_raw']].min().min():.3f}")
    figure(df, fig_path)


if __name__ == "__main__":
    main()
