#!/usr/bin/env python3
"""
p300_gan.py — GAN-augmented P300 classification, corrected vs uncorrected markers.

Follows P300_pipeline.py's GAN evaluation flow step for step, on single trials with the
P300_pipeline_single_trial.py settings (500 epochs, 2 runs, 20 batches x 200 synthetic
targets, mix stride n_tgt//5). The architecture, losses, training loop, gates, quality
score and strategy score are imported unchanged from P300_pipeline.py.

TWO deliberate fixes. (1) Class balancing uses `balance` (minority-class duplication), because
PP.explicit_upsample truncates targets when they outnumber non-targets, which silently drops
every synthetic trial in a balanced split. (2) synthetic trials are generated in the GAN's normalised space
(per-channel z/3, clipped to [-1, 1]) and P300_pipeline.py mixes them with µV features
without inverting that. Here they are de-normalised (x = s*3*std + mean, using the
training set's own stats) before every feature-level step: feature-distance filter,
Fisher, classification utility, strategies, and the final classifier. Steps the original
runs in normalised space (ERP-correlation gate, PSD / amplitude / ERP-corr quality) stay there.

Per training unit and marker set (all corrected or all uncorrected), reports on
within / cross_session / cross_site tests:
    baseline      real train+val only (as P300_pipeline's baseline)
    gan           real train+val + best synthetic mix (P300_pipeline's "best_acc")
    gan_synthonly synthetic targets + real non-targets (P300_pipeline's "so_acc")

Usage: python p300_gan.py VARIANT THRESH SEED [SEED ...]    (resumable; appends to p300_gan.csv)
"""
from __future__ import annotations
import os
import sys
import time
import zlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import p300_multisite as P                              # noqa: E402
import p300_baseline as B                               # noqa: E402
import P300_pipeline as PP                              # noqa: E402
import tensorflow as tf                                 # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402

GAN_EPOCHS_ST, NUM_RUNS_ST, NUM_BATCHES_ST, SYNTH_PER_BATCH_ST = 500, 2, 20, 200
OUT = os.path.join(HERE, "p300_gan.csv")
WAVES = os.path.join(HERE, "_p300_gan_waves")
WS, WE = int(P.FEAT_IDX[0]), int(P.FEAT_IDX[-1]) + 1


def flat(X, sel):
    return P.features(X)[:, sel]


def balance(X, y):
    """Duplicate the MINORITY class up to the majority count. Identical to
    PP.explicit_upsample when targets are the minority (the speller case); unlike it, never
    truncates targets when they outnumber non-targets — which here would silently drop every
    synthetic target added by a Mix strategy (the 20/20 training split is balanced)."""
    idx = {c: np.where(y == c)[0] for c in (1, 2)}
    lo, hi = sorted((1, 2), key=lambda c: len(idx[c]))
    n = len(idx[hi])
    reps, rem = divmod(n, len(idx[lo]))
    chosen = np.concatenate([np.tile(idx[lo], reps), idx[lo][:rem]])
    keep = np.concatenate([chosen, idx[hi]])
    return X[keep], y[keep]


def fit_lda(X, y):
    Xu, yu = balance(X, y)
    clf = LDA(solver="lsqr", shrinkage="auto")
    clf.fit(Xu, yu)
    return clf, Xu, yu


def gan_unit(X, y, tr, va, sel_train, seed, odir, tag):
    """Everything from 'Step 4: GAN training' to the chosen strategy. Returns the
    synthetic target trials (µV) for the best augmentation and for synth-only, or None."""
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    norm, (mu, sd) = PP.normalize_per_channel(Xtr.transpose(1, 2, 0))       # (ch, T, n)
    x_gan = norm.transpose(2, 0, 1)                                           # (n, ch, T)
    y_gan = ytr.copy()
    mu_b, sd_b = mu.reshape(1, -1, 1), sd.reshape(1, -1, 1)
    denorm = lambda s: s * 3.0 * sd_b + mu_b                                  # the fix
    erp_template = x_gan[y_gan == 2][:, :, WS:WE].mean(axis=0)
    n_tgt = int((ytr == 2).sum())

    real_train_X = flat(Xtr, sel_train)
    real_tgt_sel = real_train_X[ytr == 2]
    val_X = flat(Xva, sel_train)
    ntgt_idx = np.where(ytr == 1)[0]

    batches = []
    for run_idx in range(NUM_RUNS_ST):
        np.random.seed(seed * 100 + run_idx)
        tf.random.set_seed(seed * 100 + run_idx)
        gen = PP.train_cgan_p300(x_gan, y_gan, epochs=GAN_EPOCHS_ST,
                                 batch_size=min(64, x_gan.shape[0]), num_channels=len(P.CH8),
                                 output_dir=odir, subject_id=tag, run_number=run_idx + 1,
                                 time_points=x_gan.shape[2], erp_template=erp_template,
                                 p300_win_start=WS, p300_win_end=WE)
        if gen is None:
            continue
        for _ in range(NUM_BATCHES_ST):
            sc = PP.generate_synthetic(gen, SYNTH_PER_BATCH_ST, 1)
            if sc.size == 0:
                continue
            sy = np.ones(sc.shape[0]) * 2
            keep = PP.filter_erp_correlation_gate(sc, sy, erp_template, WS, WE, target_corr_thresh=0.1)
            sc, sy = sc[keep], sy[keep]
            if sc.shape[0] < 10:
                continue
            sc_uv = denorm(sc)
            sf = flat(sc_uv, sel_train)
            z = np.mean(np.abs((sf - real_tgt_sel.mean(0)) / (real_tgt_sel.std(0) + 1e-8)), axis=1)
            n_keep = max(10, min(int(0.4 * sc.shape[0]), sc.shape[0]))
            order = np.argsort(z)[:n_keep]
            f_norm, f_uv, f_sy, f_feat = sc[order], sc_uv[order], sy[order], sf[order]

            real_tgt_norm = x_gan[y_gan == 2]
            psd_c = PP.compute_psd_correlation(real_tgt_norm, f_norm, fs=P.TARGET_FS)
            amp_s = PP.compute_amplitude_similarity(real_tgt_norm, f_norm)
            erp_c = PP.compute_erp_correlation(x_gan, f_norm, y_gan, f_sy)
            fisher = PP.compute_fisher_separability(np.vstack([real_train_X, f_feat]),
                                                    np.concatenate([ytr, f_sy]))
            qX = np.vstack([f_feat, real_train_X[ntgt_idx]])
            qy = np.concatenate([f_sy, ytr[ntgt_idx]])
            clf_q, _, _ = fit_lda(qX, qy)
            cls_u = float(np.mean(clf_q.predict(val_X) == yva))
            combined = 0.20 * psd_c + 0.20 * amp_s + 0.20 * erp_c + 0.10 * fisher + 0.30 * cls_u
            batches.append(dict(score=combined, uv=f_uv, feat=f_feat, sy=f_sy))
        del gen
        tf.keras.backend.clear_session()

    if not batches:
        return None
    batches.sort(key=lambda b: b["score"], reverse=True)
    top = batches[:10]
    stride = max(2, n_tgt // 5)
    steps = list(range(stride, n_tgt + 1, stride))
    if n_tgt not in steps:
        steps.append(n_tgt)

    tgt_idx = np.where(ytr == 2)[0]
    strats = []
    for b in top:
        na = min(n_tgt, b["uv"].shape[0])
        Xso = np.vstack([b["feat"][:na], real_train_X[ntgt_idx]])
        yso = np.concatenate([b["sy"][:na], ytr[ntgt_idx]])
        clf, Xu, yu = fit_lda(Xso, yso)
        va_acc = float(np.mean(clf.predict(val_X) == yva)); tr_acc = float(np.mean(clf.predict(Xu) == yu))
        strats.append(dict(name="SynthOnly", so=True, b=b, n=na,
                           s=0.6 * tr_acc + 0.3 * va_acc + 0.1 * b["score"]))
        for n_syn in steps:
            na = min(n_syn, b["uv"].shape[0])
            Xm = np.vstack([real_train_X[tgt_idx], b["feat"][:na], real_train_X[ntgt_idx]])
            ym = np.concatenate([ytr[tgt_idx], b["sy"][:na], ytr[ntgt_idx]])
            clf, Xu, yu = fit_lda(Xm, ym)
            va_acc = float(np.mean(clf.predict(val_X) == yva)); tr_acc = float(np.mean(clf.predict(Xu) == yu))
            strats.append(dict(name=f"Mix+{n_syn}", so=False, b=b, n=na,
                               s=0.6 * tr_acc + 0.3 * va_acc + 0.1 * b["score"]))
    aug = sorted([s for s in strats if not s["so"]], key=lambda s: s["s"], reverse=True)
    so = max([s for s in strats if s["so"]], key=lambda s: s["s"])
    best = aug[0]
    return dict(best_name=best["name"], best_uv=best["b"]["uv"][:best["n"]],
                so_uv=so["b"]["uv"][:so["n"]], n_batches=len(batches))


def main(variant, thresh, seeds, clf_mode="sfs"):
    label = variant if clf_mode == "sfs" else f"{variant}_fullLDA"
    out = OUT if clf_mode == "sfs" else OUT.replace(".csv", "_fullLDA.csv")
    data, sess = B.load_all(variant, thresh)
    units = B.training_units(sess)
    usable = sess[sess.n_kept.fillna(0) > 0]
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out)
        done = set(zip(d.variant, d.thresh, d.seed, d.marker, d.subject, d.train_session))
    os.makedirs(WAVES, exist_ok=True)
    for seed in seeds:
        for u in units.itertuples():
            dtr = data[(u.subject, u.session)]
            tr, va, te = P.split_within(dtr["y"])
            pool = np.concatenate([tr, va])
            for mk, xk in B.MARKERS.items():
                if (label, thresh, seed, mk, u.subject, u.session) in done:
                    continue
                t0 = time.time()
                X, y = dtr[xk], dtr["y"]
                if clf_mode == "sfs":
                    clf_tr, sel_train = PP.train_p300_lda(B.feat3(X[tr]), y[tr])
                    clf_bl, sel_bl = PP.train_p300_lda(B.feat3(X[pool]), y[pool])
                else:   # secondary: shrinkage LDA on all 224 features, no stepwise selection
                    sel_train = sel_bl = list(range(len(P.FEAT_IDX) * len(P.CH8)))
                    clf_bl = fit_lda(flat(X[pool], sel_bl), y[pool])[0]
                s0 = zlib.crc32(f"{u.subject}{u.session}{mk}".encode()) % 1000 + 1000 * seed
                odir = os.path.join(WAVES, f"{label}{int(thresh)}_s{seed}_{u.subject}_{u.session}_{mk}")
                os.makedirs(odir, exist_ok=True)
                g = gan_unit(X, y, tr, va, sel_train, s0, odir, f"{u.subject}_{u.session}_{mk}")
                models = {"baseline": (clf_bl, sel_bl)}
                best_name = ""
                if g is not None:
                    tgt, ntgt = pool[y[pool] == 2], pool[y[pool] == 1]
                    for arm, synth in (("gan", g["best_uv"]), ("gan_synthonly", g["so_uv"])):
                        real_t = [] if arm == "gan_synthonly" else [flat(X[tgt], sel_train)]
                        Xf = np.vstack(real_t + [flat(synth, sel_train), flat(X[ntgt], sel_train)])
                        yf = np.concatenate(([] if arm == "gan_synthonly" else [y[tgt]]) +
                                            [np.full(len(synth), 2), y[ntgt]])
                        models[arm] = (fit_lda(Xf, yf)[0], sel_train)
                    best_name = g["best_name"]
                    np.savez_compressed(os.path.join(odir, "waves.npz"),
                                        real_tgt=X[tr][y[tr] == 2].mean(0), real_nt=X[tr][y[tr] == 1].mean(0),
                                        synth_best=g["best_uv"].mean(0), synth_so=g["so_uv"].mean(0))
                tests = [("within", u.session, u.site, dtr, te)]
                for t in usable[(usable.subject == u.subject) & (usable.session != u.session)].itertuples():
                    dte = data[(t.subject, t.session)]
                    if len(np.unique(dte["y"])) == 2:
                        tests.append(("cross_session" if t.site == u.site else "cross_site",
                                      t.session, t.site, dte, np.arange(len(dte["y"]))))
                rows = []
                for arm, (clf, sel) in models.items():
                    for regime, tses, tsite, dte, idx in tests:
                        Xt, yt = dte[xk][idx], dte["y"][idx]
                        # score() featurises with P.features and selects `sel`
                        sc = B.score(clf, sel, Xt, yt)
                        rows.append(dict(variant=label, thresh=thresh, seed=seed, marker=mk,
                                         subject=u.subject, train_session=u.session, train_site=u.site,
                                         arm=arm, best_strategy=best_name if arm == "gan" else "",
                                         regime=regime, test_session=tses, test_site=tsite,
                                         err_train=dtr["err_med"], err_test=dte["err_med"],
                                         d_offset_ms=abs(dte["err_med"] - dtr["err_med"]), **sc))
                pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                print(f"[{time.strftime('%H:%M')}] {label}@{thresh} s{seed} {u.subject} {u.session} {mk}: "
                      f"{time.time() - t0:.0f}s best={best_name} "
                      f"within bacc base={rows[0]['bacc']:.3f}"
                      + (f" gan={[r for r in rows if r['arm'] == 'gan'][0]['bacc']:.3f}" if g else ""),
                      flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = "full" if "--full-lda" in sys.argv else "sfs"
    main(args[0], float(args[1]), [int(s) for s in args[2:]] or [1], mode)
