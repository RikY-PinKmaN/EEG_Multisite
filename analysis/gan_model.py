#!/usr/bin/env python3
"""G0.1–G0.5 — the cWGAN-GP, vendored for this dataset (GAN_ROADMAP.md §G0).

PROVENANCE
    Source: ~/Documents/University/PhD/Python_Codes/EEG_scripts/scripts/P300_pipeline.py
            (reachable from here as Python_Codes/EEG_Multisite/../EEG_scripts/...)
            functions build_generator, build_critic, frequency_domain_loss,
            p300_feature_loss, gradient_penalty, train_cgan_p300, generate_synthetic.
    Copied 8 August 2026 rather than imported: the source file is 1,779 lines fused to its
    own speller preprocessing, and importing it would drag all of that in.

    `frequency_domain_loss` and `gradient_penalty` are byte-identical to the source.
    `build_generator` and `build_critic` differ by exactly one added line each — the
    `_require_tf()` guard — and are otherwise identical, so results here stay comparable to
    the P300 work. Verified by diffing the function bodies, not by inspection; re-run that
    diff if the source file ever changes. What changed, and why:

    G0.2  Shape only — num_channels=14, time_points=295 are arguments, not edits. The
          generator's existing crop path handles 296 -> 295; the critic's three stride-2
          blocks leave 37 samples. Asserted in _selftest, not assumed.
    G0.4  `p300_feature_loss` -> `component_feature_loss`. The original template-matches
          the target class against one grand-average ERP inside the P300 window. Here each
          class gets its own template, because for N170 face-vs-car neither class is a
          "non-target". The class-contrast term is unchanged, including its weight of 3.0.
    G0.5  Generation is per-class. The original synthesises targets only and keeps
          non-targets real; that asymmetry does not apply to a face/car contrast.

    NOT changed: layer stacks, filter counts, kernel sizes, latent/embedding dims, Adam
    settings, 5 critic steps, the gradient-penalty weight of 10.0, early-stopping logic.

RUNTIME
    Needs TensorFlow, which lives in the EEG_scripts venv, not in this project:
        /home/pinkman/Documents/University/PhD/Python_Codes/EEG_scripts/.venv/bin/python
    That interpreter also has numpy/scipy/sklearn/pandas, so it runs the rest of this
    directory too. The data-unit builders below are pure numpy and import without TF, so
    their self-tests run under either interpreter.

    python3 gan_model.py          # data-unit tests only, TF skipped
    <venv>/bin/python gan_model.py  # full tests including the model

WARNING — the quality gate enforces the training set's latency.
    `filter_erp_correlation_gate` in the source scores synthetic trials against an ERP
    template. If that template comes from corrected markers while the generator trained on
    raw ones, the gate rejects correctly-generated trials and silently repairs the
    displacement G1 exists to measure. `class_templates` must always be built from the same
    epochs the generator was trained on. `assert_template_provenance` enforces it.
"""
from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAVE_TF = True
except ImportError:  # pragma: no cover - depends on interpreter
    HAVE_TF = False

# Vendored verbatim from the source file's constant block.
LATENT_DIM = 50
EMBEDDING_DIM = 25
GAN_EPOCHS = 2000
N_CRITIC_STEPS = 5

# This dataset's epoch geometry (erp_common.TARGET_FS, epochs.TMIN/TMAX).
NCHAN = 14
TIME_POINTS = 295
FS = 256.0
TMIN_MS = -200.0

# Component windows for the feature loss (ms). N170 from the canonical measurement window
# used throughout erp_impact.py; P3b from the same source's P3b window.
COMPONENT_WINDOWS_MS = {"n170": (130.0, 200.0), "p3b": (300.0, 600.0)}


# =============================================================================
# G0.3  Training units
# =============================================================================

def component_window(task, fs=FS, tmin_ms=TMIN_MS, time_points=TIME_POINTS):
    """Sample indices of a task's component window within the epoch."""
    assert task in COMPONENT_WINDOWS_MS, f"unknown task {task!r}"
    lo_ms, hi_ms = COMPONENT_WINDOWS_MS[task]
    lo = int(round((lo_ms - tmin_ms) / 1000.0 * fs))
    hi = int(round((hi_ms - tmin_ms) / 1000.0 * fs))
    assert 0 <= lo < hi <= time_points, f"{task} window {lo}:{hi} outside 0:{time_points}"
    return lo, hi


def class_templates(X, y, w_start, w_end, num_classes=2):
    """Per-class grand-average ERP inside the window -> (num_classes, ch, win_T).

    Built from whatever epochs are passed in. Pass the generator's OWN training epochs —
    see the module warning about the quality gate.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int).ravel()
    assert X.ndim == 3, f"expected (trials, ch, T), got {X.shape}"
    assert X.shape[0] == y.size, f"{X.shape[0]} trials vs {y.size} labels"
    out = np.empty((num_classes, X.shape[1], w_end - w_start), dtype=np.float32)
    for c in range(num_classes):
        sel = X[y == c]
        assert sel.shape[0] > 0, f"class {c} absent — cannot build a template"
        out[c] = sel[:, :, w_start:w_end].mean(axis=0)
    return out


def assert_template_provenance(templates, X_train, y_train, w_start, w_end):
    """Fail if `templates` was not built from these exact training epochs.

    Cheap insurance against the failure mode that would invalidate G1 silently: templates
    from corrected epochs used to gate a generator trained on raw ones.
    """
    expected = class_templates(X_train, y_train, w_start, w_end, len(templates))
    assert np.allclose(templates, expected, atol=1e-5), (
        "template does not match the training epochs it will gate — this silently repairs "
        "the timing displacement G1 measures (see module docstring)")


def single_trials(X, y):
    """The `single` training unit: trials as they are. Matches classify.py's regime."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.int32).ravel()
    assert X.ndim == 3 and X.shape[0] == y.size
    return X, y, {"unit": "single", "n_exemplars": int(X.shape[0]),
                  "n_input_trials": int(X.shape[0]), "k": 1,
                  "n_independent": int(X.shape[0])}


def subaverage(X, y, k, m, rng=None, num_classes=2):
    """The `subavg` unit: `m` averages of `k` random trials, per class.

    Raises the SNR the generator sees at the cost of correlated exemplars, because the
    draws overlap. The returned provenance carries `n_independent` — the disjoint-subset
    count, floor(n_class / k) — which is the number to quote alongside `m` in any writeup.
    On this dataset that is ~11 per class at k=5 from 55 trials, which is why G1 uses every
    trial of a run and G2 is treated as exploratory.

    Only ever pass epochs from a single fold. Averaging across a train/test boundary would
    leak test trials into training, and this function cannot detect that for you.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int).ravel()
    assert X.ndim == 3 and X.shape[0] == y.size
    assert k >= 1 and m >= 1, "k and m must be positive"
    rng = np.random.default_rng() if rng is None else rng

    out_X, out_y, independent = [], [], {}
    for c in range(num_classes):
        idx = np.flatnonzero(y == c)
        assert idx.size >= k, f"class {c} has {idx.size} trials, need >= k={k}"
        independent[c] = int(idx.size // k)
        for _ in range(m):
            pick = rng.choice(idx, size=k, replace=False)
            out_X.append(X[pick].mean(axis=0))
            out_y.append(c)

    Xa = np.stack(out_X).astype(np.float32)
    ya = np.asarray(out_y, dtype=np.int32)
    return Xa, ya, {"unit": f"subavg(k={k},m={m})", "n_exemplars": int(Xa.shape[0]),
                    "n_input_trials": int(X.shape[0]), "k": int(k),
                    "n_independent": int(min(independent.values())),
                    "n_independent_per_class": independent}


# =============================================================================
# G0.1  Model — vendored verbatim
# =============================================================================

def _require_tf():
    assert HAVE_TF, ("TensorFlow not importable. Use the EEG_scripts venv:\n"
                     "  /home/pinkman/Documents/University/PhD/Python_Codes/"
                     "EEG_scripts/.venv/bin/python")


def build_generator(num_channels, time_points, latent_dim, num_classes, embedding_dim):
    _require_tf()
    noise_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(1,))
    merged = layers.Concatenate()([
        noise_input,
        layers.Flatten()(layers.Embedding(num_classes, embedding_dim)(label_input))
    ])

    x = layers.Dense(((time_points + 7) // 8) * 64)(merged)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape(((time_points + 7) // 8, 64))(x)

    for filters, k in [(128, 5), (64, 5)]:
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(filters, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(num_channels, 7, padding="same", activation="tanh")(x)

    tp = x.shape[1]
    if tp > time_points:
        crop_l = (tp - time_points) // 2
        crop_r = (tp - time_points) - crop_l
        x = layers.Cropping1D((crop_l, crop_r))(x)
    elif tp < time_points:
        pad_l = (time_points - tp) // 2
        pad_r = (time_points - tp) - pad_l
        x = layers.ZeroPadding1D((pad_l, pad_r))(x)

    return models.Model([noise_input, label_input], layers.Permute((2, 1))(x))


def build_critic(num_channels, time_points, num_classes, embedding_dim):
    _require_tf()
    data_input = layers.Input(shape=(num_channels, time_points))
    label_input = layers.Input(shape=(1,))
    x = layers.Permute((2, 1))(data_input)

    for filters in [64, 128, 256]:
        x = layers.Conv1D(filters, 5, strides=2, padding="same")(x)
        if filters > 64:
            x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    merged = layers.Concatenate()([
        layers.Flatten()(x),
        layers.Flatten()(layers.Embedding(num_classes, embedding_dim)(label_input))
    ])
    return models.Model([data_input, label_input], layers.Dense(1)(merged))


def frequency_domain_loss(real_data, generated_data):
    real_fft = tf.abs(tf.signal.rfft(tf.transpose(real_data, perm=[0, 2, 1])))
    gen_fft = tf.abs(tf.signal.rfft(tf.transpose(generated_data, perm=[0, 2, 1])))
    return tf.reduce_mean(tf.square(real_fft - gen_fft))


def gradient_penalty(critic, real_samples, fake_samples, real_labels):
    alpha = tf.random.uniform([tf.shape(real_samples)[0], 1, 1], 0., 1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic([interpolated, real_labels], training=True)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    return 10.0 * tf.reduce_mean(
        (tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]) + 1e-10) - 1.0) ** 2)


# =============================================================================
# G0.4  Feature loss — generalised to per-class templates
# =============================================================================

def component_feature_loss(real_data, fake_data, labels, templates,
                           window_start, window_end, num_classes=2,
                           contrast_weight=3.0):
    """Template matching per class + class-contrast matching.

    Generalises the source's `p300_feature_loss`. Component 1 there matched the target
    class against one grand-average template; here every class is matched against its own,
    since a face/car contrast has no "non-target" side. Component 2 — matching the
    between-class amplitude contrast — and its weight of 3.0 are unchanged.

    real/fake: (batch, channels, time).  labels: (batch, 1).
    templates: (num_classes, channels, window_T).
    """
    lab_flat = tf.cast(tf.reshape(labels, [-1]), tf.int32)

    # Component 1: per-class template matching. Classes absent from the batch contribute 0.
    template_loss = tf.constant(0.0)
    for c in range(num_classes):
        mask = tf.equal(lab_flat, c)
        fake_c = tf.boolean_mask(fake_data, mask)
        template_loss += tf.cond(
            tf.shape(fake_c)[0] > 0,
            lambda fc=fake_c, cc=c: tf.reduce_mean(tf.square(
                fc[:, :, window_start:window_end] - templates[cc][tf.newaxis, :, :])),
            lambda: tf.constant(0.0)
        )
    template_loss /= float(num_classes)

    # Component 2: between-class amplitude contrast, computed as in the source.
    hi_mask = tf.equal(lab_flat, num_classes - 1)
    lo_mask = tf.equal(lab_flat, 0)
    n_hi = tf.maximum(tf.reduce_sum(tf.cast(hi_mask, tf.float32)), 1.0)
    n_lo = tf.maximum(tf.reduce_sum(tf.cast(lo_mask, tf.float32)), 1.0)

    def _amp(data, mask, n):
        win = data[:, :, window_start:window_end]
        return tf.reduce_sum(tf.reduce_mean(tf.boolean_mask(win, mask), axis=2), axis=0) / n

    real_contrast = _amp(real_data, hi_mask, n_hi) - _amp(real_data, lo_mask, n_lo)
    fake_contrast = _amp(fake_data, hi_mask, n_hi) - _amp(fake_data, lo_mask, n_lo)
    contrast_loss = tf.reduce_mean(tf.square(real_contrast - fake_contrast))

    return template_loss + contrast_weight * contrast_loss


# =============================================================================
# G0.1  Training loop — vendored, retargeted
# =============================================================================

def train_cgan(eeg_data, eeg_labels, time_points, num_channels=NCHAN,
               num_classes=2, epochs=GAN_EPOCHS, batch_size=16,
               latent_dim=LATENT_DIM, embedding_dim=EMBEDDING_DIM,
               n_critic_steps=N_CRITIC_STEPS, freq_loss_weight=1.0,
               feat_loss_weight=3.5, templates=None, window=None,
               patience=500, log_every=200, verbose=True):
    """Train the cWGAN-GP. Structure follows `train_cgan_p300`.

    eeg_data: (n_exemplars, channels, time), float32, per-channel normalised.
    eeg_labels: (n_exemplars,) with values 0..num_classes-1 — note the source used 1/2 and
        converted internally; here labels are already 0-indexed.
    templates: (num_classes, channels, window_T) from `class_templates`, built on THIS data.
    window: (start, end) sample indices from `component_window`.
    """
    _require_tf()
    X = np.asarray(eeg_data, dtype=np.float32)
    y = np.asarray(eeg_labels).astype(np.int32).ravel()
    assert X.ndim == 3 and X.shape[0] == y.size, f"bad shapes {X.shape} / {y.shape}"
    assert X.shape[1] == num_channels, f"{X.shape[1]} channels, expected {num_channels}"
    assert X.shape[2] == time_points, f"{X.shape[2]} samples, expected {time_points}"
    assert y.min() >= 0 and y.max() < num_classes, f"labels out of range: {np.unique(y)}"
    labels_col = y[:, np.newaxis]

    gen = build_generator(num_channels, time_points, latent_dim, num_classes, embedding_dim)
    crit = build_critic(num_channels, time_points, num_classes, embedding_dim)
    opt_g = tf.keras.optimizers.Adam(1e-4, 0.5, 0.9)
    opt_c = tf.keras.optimizers.Adam(1e-4, 0.5, 0.9)

    freq_w = tf.constant(freq_loss_weight, dtype=tf.float32)
    feat_w = tf.constant(feat_loss_weight, dtype=tf.float32)
    use_feat = templates is not None and window is not None
    tmpl_tf = tf.constant(np.asarray(templates, np.float32)) if use_feat else None
    ws, we = (int(window[0]), int(window[1])) if use_feat else (None, None)

    @tf.function
    def train_step(rx, ry):
        bs = tf.shape(rx)[0]
        for _ in range(n_critic_steps):
            nz = tf.random.normal([bs, latent_dim])
            with tf.GradientTape() as tape_c:
                fk = gen([nz, ry], training=True)
                d_loss = (tf.reduce_mean(crit([fk, ry], training=True)) -
                          tf.reduce_mean(crit([rx, ry], training=True)) +
                          gradient_penalty(crit, rx, fk, ry))
            opt_c.apply_gradients(zip(
                tape_c.gradient(d_loss, crit.trainable_variables),
                crit.trainable_variables))

        nz = tf.random.normal([bs, latent_dim])
        with tf.GradientTape() as tape_g:
            fk_g = gen([nz, ry], training=True)
            g_w = -tf.reduce_mean(crit([fk_g, ry], training=True))
            g_frq = frequency_domain_loss(rx, fk_g)
            g_feat = tf.constant(0.0)
            if use_feat:
                g_feat = component_feature_loss(rx, fk_g, ry, tmpl_tf, ws, we, num_classes)
            g_loss = g_w + freq_w * g_frq + feat_w * g_feat
        opt_g.apply_gradients(zip(
            tape_g.gradient(g_loss, gen.trainable_variables), gen.trainable_variables))
        return d_loss, g_w, g_frq, g_feat

    history = []
    best_g, best_weights, patience_cnt = float("inf"), None, 0
    rng = np.random.default_rng(0)

    for ep in range(epochs):
        perm = rng.permutation(X.shape[0])
        n_batches = max(1, X.shape[0] // batch_size)
        acc = np.zeros(4)
        for bi in range(n_batches):
            s, e = bi * batch_size, (bi + 1) * batch_size
            d, gw, gfrq, gft = train_step(
                tf.convert_to_tensor(X[perm[s:e]], tf.float32),
                tf.convert_to_tensor(labels_col[perm[s:e]], tf.int32))
            acc += [d.numpy(), gw.numpy(), gfrq.numpy(), gft.numpy()]
        acc /= n_batches
        g_total = acc[1] + freq_loss_weight * acc[2] + feat_loss_weight * acc[3]
        history.append(dict(epoch=ep + 1, critic=acc[0], gen_w=acc[1],
                            gen_freq=acc[2], gen_feat=acc[3], gen_total=g_total))
        assert np.isfinite(g_total), f"generator loss diverged at epoch {ep+1}"

        if verbose and ((ep + 1) % log_every == 0 or ep == 0):
            print(f"    ep {ep+1:4d}/{epochs} D={acc[0]:+.3f} Gw={acc[1]:+.3f} "
                  f"Gfrq={acc[2]:.4f} Gfeat={acc[3]:.4f}")

        if ep >= epochs // 4:
            if g_total < best_g - 0.1:
                best_g, best_weights, patience_cnt = g_total, gen.get_weights(), 0
            else:
                patience_cnt += 1
            if patience_cnt >= patience:
                if verbose:
                    print(f"    early stop at epoch {ep+1}")
                break

    if best_weights is not None:
        gen.set_weights(best_weights)
    return gen, history


def generate_synthetic(gen, num_samples, class_label, latent_dim=LATENT_DIM):
    """Synthesise `num_samples` exemplars of one class -> (n, channels, time).

    G0.5: any class, not targets only.
    """
    _require_tf()
    if gen is None or num_samples == 0:
        return np.empty((0,))
    return gen([tf.random.normal([num_samples, latent_dim]),
                tf.ones((num_samples, 1), tf.int32) * int(class_label)],
               training=False).numpy()


# =============================================================================
# Self-tests
# =============================================================================

def _selftest_data_units():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, NCHAN, TIME_POINTS)).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)

    for task, (lo, hi) in COMPONENT_WINDOWS_MS.items():
        s, e = component_window(task)
        assert e > s
        assert abs((s / FS * 1000 + TMIN_MS) - lo) < 1000 / FS
    # 130..200 ms with a -200 ms epoch start at 256 Hz -> samples 84..102 (18 = 70.3 ms).
    s170, e170 = component_window("n170")
    assert (s170, e170) == (84, 102), f"n170 window moved: {s170}:{e170}"
    assert abs((e170 - s170) / FS * 1000 - 70.0) < 1.0

    Xs, ys, prov = single_trials(X, y)
    assert Xs.shape == X.shape and prov["n_independent"] == 60

    # Averaging is correct, class-pure, and reports honest independence.
    Xa, ya, prov = subaverage(X, y, k=5, m=12, rng=np.random.default_rng(1))
    assert Xa.shape == (24, NCHAN, TIME_POINTS), Xa.shape
    assert (ya[:12] == 0).all() and (ya[12:] == 1).all()
    assert prov["n_independent"] == 6, prov
    # A k=1 subaverage of a single class must reproduce that trial exactly.
    one, oy, _ = subaverage(X[:2], np.array([0, 1]), k=1, m=1,
                            rng=np.random.default_rng(0))
    assert np.allclose(one[0], X[0]) and np.allclose(one[1], X[1])
    # Averaged exemplars must have lower variance than singles (SNR is the point).
    assert Xa.var() < X.var(), "sub-averaging did not reduce variance"

    tmpl = class_templates(X, y, s170, e170)
    assert tmpl.shape == (2, NCHAN, e170 - s170)
    assert np.allclose(tmpl[0], X[y == 0][:, :, s170:e170].mean(0), atol=1e-6)
    assert_template_provenance(tmpl, X, y, s170, e170)
    try:
        assert_template_provenance(tmpl, X[::-1] + 1.0, y[::-1], s170, e170)
    except AssertionError:
        pass
    else:
        raise AssertionError("provenance check failed to catch a mismatched template")

    print("data-unit tests OK")
    print(f"  n170 window {s170}:{e170} samples, p3b {component_window('p3b')}")
    print(f"  subavg(k=5,m=12) -> {prov['n_exemplars']} exemplars from "
          f"{prov['n_input_trials']} trials, {prov['n_independent']} independent per class")


def _selftest_model():
    if not HAVE_TF:
        print("model tests SKIPPED — no TensorFlow in this interpreter")
        return
    tf.random.set_seed(0)
    rng = np.random.default_rng(5)

    # G0.2 — shapes survive 295 samples without an architecture edit.
    gen = build_generator(NCHAN, TIME_POINTS, LATENT_DIM, 2, EMBEDDING_DIM)
    crit = build_critic(NCHAN, TIME_POINTS, 2, EMBEDDING_DIM)
    out = gen([tf.random.normal([4, LATENT_DIM]), tf.zeros((4, 1), tf.int32)])
    assert tuple(out.shape) == (4, NCHAN, TIME_POINTS), out.shape
    score = crit([out, tf.zeros((4, 1), tf.int32)])
    assert tuple(score.shape) == (4, 1), score.shape

    # G0.2 — the three stride-2 blocks must leave a sane width, not a 1-sample sliver.
    conv_widths = [l.output.shape[1] for l in crit.layers
                   if l.__class__.__name__ == "Conv1D"]
    head_width = max(l.output.shape[1] for l in crit.layers
                     if l.__class__.__name__ == "Concatenate")
    assert conv_widths == [148, 74, 37], f"critic downsampling changed: {conv_widths}"
    assert head_width == 37 * 256 + EMBEDDING_DIM, head_width

    # G0.5 — both classes are reachable and conditioning actually changes the output.
    a = generate_synthetic(gen, 6, 0)
    b = generate_synthetic(gen, 6, 1)
    assert a.shape == b.shape == (6, NCHAN, TIME_POINTS)
    assert not np.allclose(a.mean(0), b.mean(0)), "label conditioning has no effect"

    # G0.4 — the feature loss is zero exactly when it should be.
    s, e = component_window("n170")
    X = rng.normal(size=(16, NCHAN, TIME_POINTS)).astype(np.float32)
    y = np.array([0] * 8 + [1] * 8, dtype=np.int32)
    tmpl = class_templates(X, y, s, e)
    lab = tf.constant(y[:, None], tf.int32)

    # fake == real  ->  contrast term vanishes, template term does not (trials != mean).
    both = component_feature_loss(tf.constant(X), tf.constant(X), lab,
                                  tf.constant(tmpl), s, e).numpy()
    tmpl_only = np.mean([((X[y == c][:, :, s:e] - tmpl[c]) ** 2).mean() for c in (0, 1)])
    assert abs(both - tmpl_only) < 1e-4, f"contrast term not zero: {both} vs {tmpl_only}"

    # fake window == its own class template  ->  template term vanishes.
    Xf = X.copy()
    for c in (0, 1):
        Xf[y == c, :, s:e] = tmpl[c]
    only_contrast = component_feature_loss(tf.constant(X), tf.constant(Xf), lab,
                                           tf.constant(tmpl), s, e).numpy()
    assert only_contrast > 0, "expected a non-zero contrast penalty here"
    zero_both = component_feature_loss(tf.constant(Xf), tf.constant(Xf), lab,
                                       tf.constant(tmpl), s, e).numpy()
    assert zero_both < 1e-5, f"both terms should vanish, got {zero_both}"

    # Integration smoke test: a few real epochs, a handful of epochs, finite losses.
    g2, hist = train_cgan(X, y, TIME_POINTS, epochs=4, batch_size=8,
                          templates=tmpl, window=(s, e), verbose=False)
    assert len(hist) == 4
    assert all(np.isfinite([h["gen_total"] for h in hist]))
    assert generate_synthetic(g2, 3, 1).shape == (3, NCHAN, TIME_POINTS)

    print("model tests OK")
    print(f"  generator {TIME_POINTS} samples via crop from "
          f"{((TIME_POINTS + 7) // 8) * 8}")
    print(f"  critic {TIME_POINTS} -> {' -> '.join(map(str, conv_widths))} samples, "
          f"head {head_width} (= 37x256 + {EMBEDDING_DIM} label)")
    print(f"  feature loss: template-only {tmpl_only:.4f}, both-zero {zero_both:.2e}")
    print(f"  4-epoch smoke run, final gen_total {hist[-1]['gen_total']:+.3f}, "
          f"GPUs {len(tf.config.list_physical_devices('GPU'))}")


if __name__ == "__main__":
    _selftest_data_units()
    _selftest_model()
