"""P300 baseline v3: 16 channels, 0.1-15 Hz, GEDAI, linked mastoids, xDAWN + LDA.

Spec and freeze criteria: BASELINE_PLAN.md. Stages:

  python p300_baseline_v3.py preprocess [sub ses ...]   cleaned continuous -> _baseline_cache/
  python p300_baseline_v3.py checks                     filter, P300 vs GEDAI, mastoid ratio, yield
  python p300_baseline_v3.py classify                   within-session CV, both marker sets

The cache holds all 16 channels at 80 Hz after GEDAI (on its internal average reference) and
interpolation; the reference is applied at epoch time, so linked mastoids and the 14-channel
average (check 3 only) come from the same cleaned data.
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p300_multisite as pm                       # noqa: E402
from erp_common import _read_named_channels      # noqa: E402

CH16 = ["F7", "F3", "Fz", "F4", "F8", "FCz", "C3", "C4", "Pz", "POz",
        "PO7", "PO8", "P7", "P8", "M1", "M2"]
CH14 = CH16[:14]
MAST = ["M1", "M2"]
L_FREQ, H_FREQ = 0.1, 15.0
FS_CLEAN = 256
FS = 80
TS = np.arange(-round(0.1 * FS), round(0.8 * FS) + 1)          # -8..64 (73 samples)
BL = TS <= 0
FEAT = (TS >= round(0.15 * FS)) & (TS < round(0.5 * FS))         # 28 samples
FEAT_SENS = (TS >= round(0.2 * FS)) & (TS < round(0.6 * FS))     # sensitivity window
P3 = (TS >= round(0.3 * FS)) & (TS <= round(0.6 * FS))
PTP = 100.0
MIN_TARGETS = 20

# bad-channel thresholds (fixed in BASELINE_PLAN.md before any result)
FLAT_SD = 1.0
NOISY_Z = 5.0
NOISY_RATIO = 2.0
MAST_CORR = 0.2

CACHE = os.path.join(HERE, "_baseline_cache")
OUT = HERE
VARIANTS = ("gedai", "nogedai")


# ---------------------------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------------------------
def _rs(x, fs_in, fs_out):
    g = np.gcd(int(fs_in), int(fs_out))
    return x if int(fs_in) == int(fs_out) else resample_poly(x, int(fs_out) // g, int(fs_in) // g, axis=-1)


def _robust_sd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=-1, keepdims=True)), axis=-1)


def bad_channels(x_rec: np.ndarray) -> list[tuple[str, str, float]]:
    """x_rec: (16, n) filtered µV at 256 Hz on the recording reference.

    Revised 7 Oct 2026, before any classification (BASELINE_PLAN.md, threshold review):
    - noisy needs z > 5 AND SD > 2x the median channel. With 16 channels the MAD of channel
      SDs is ~10-15% of the median, so healthy mastoids (1.4-1.9x) reached z 4-7 on z alone.
    - mastoid-pair correlation is computed after average reference, as the median over 2 s
      windows. On the recording reference it is low at SHE by construction (online reference
      next to M2), and a whole-recording r is dominated by artefact bursts.
    """
    bads = []
    car = x_rec - x_rec.mean(axis=0, keepdims=True)
    sd = _robust_sd(car)
    med = np.median(sd)
    mad = 1.4826 * np.median(np.abs(sd - med))
    for i, c in enumerate(CH16):
        if sd[i] < FLAT_SD:
            bads.append((c, "flat", float(sd[i])))
        elif mad > 0 and (sd[i] - med) / mad > NOISY_Z and sd[i] > NOISY_RATIO * med:
            bads.append((c, "noisy", float(sd[i] / med)))
    m1, m2 = car[CH16.index("M1")], car[CH16.index("M2")]
    w = 2 * FS_CLEAN
    r = np.median([np.corrcoef(m1[k * w:(k + 1) * w], m2[k * w:(k + 1) * w])[0, 1]
                   for k in range(len(m1) // w)])
    if r < MAST_CORR:
        for c in MAST:
            if c not in [b[0] for b in bads]:
                bads.append((c, "mastoid_pair_corr", float(r)))
    return bads


def _raw(x_uv, fs, names):
    import mne
    raw = mne.io.RawArray(x_uv * 1e-6, mne.create_info(names, fs, "eeg"), verbose=False)
    raw.set_montage("standard_1005", verbose=False)
    return raw


def preprocess(sub: str, ses: str) -> dict:
    """Steps 1-9 of BASELINE_PLAN.md for both variants. Returns a session log row."""
    import mne
    os.makedirs(CACHE, exist_ok=True)
    row = dict(subject=sub, session=ses, site=pm.site_of(ses))
    p = pm.eeg_prefix(sub, ses)
    if p is None:
        return {**row, "status": "no_eeg"}
    ch, fs0 = _read_named_channels(p, CH16)
    miss = [c for c in CH16 if c not in ch]
    if miss:
        return {**row, "status": f"missing:{'/'.join(miss)}"}
    x = np.vstack([ch[c] for c in CH16])
    row["fs_orig"] = fs0

    # 2. filter at the original fs (MNE zero-phase FIR), 3. resample to 256 Hz
    raw = _raw(x, fs0, CH16)
    raw.filter(L_FREQ, H_FREQ, method="fir", phase="zero", fir_design="firwin", verbose=False)
    x = _rs(raw.get_data() * 1e6, fs0, FS_CLEAN)

    # 4-5. bad channels, session drop rules
    bads = bad_channels(x)
    row["bads"] = ";".join(f"{c}:{why}:{v:.2f}" for c, why, v in bads)
    bad_names = [b[0] for b in bads]
    if any(c in MAST for c in bad_names):
        return {**row, "status": "drop_bad_mastoid"}
    if len(bad_names) > 1:
        return {**row, "status": "drop_too_many_bad"}
    good = [c for c in CH16 if c not in bad_names]

    for variant in VARIANTS:
        raw = _raw(x, FS_CLEAN, CH16)
        with raw.info._unlock():
            raw.info["highpass"], raw.info["lowpass"] = L_FREQ, H_FREQ
        raw.set_eeg_reference("average", verbose=False)       # GEDAI's leadfield assumes it
        if variant == "gedai":
            from gedai import Gedai
            sub_raw = raw.copy().pick(good)
            g = Gedai()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g.fit_raw(sub_raw, verbose=False)
                clean = g.transform_raw(sub_raw, verbose=False)
            cd = clean.get_data()
            assert cd.shape == (len(good), raw.n_times), (cd.shape, raw.n_times)
            data = raw.get_data()
            data[[CH16.index(c) for c in good]] = cd
            raw = mne.io.RawArray(data, raw.info, verbose=False)
        # 7. interpolate the (at most one) bad non-mastoid channel
        if bad_names:
            raw.info["bads"] = bad_names
            raw.interpolate_bads(reset_bads=True, verbose=False)
        # 9. 80 Hz, cache all 16 channels (reference applied at epoch time)
        y = _rs(raw.get_data() * 1e6, FS_CLEAN, FS).astype(np.float32)
        np.savez(os.path.join(CACHE, f"{sub}_{ses}_{variant}.npz"), x=y, fs_orig=fs0)
    return {**row, "status": "ok"}


# ---------------------------------------------------------------------------------------------
# epochs
# ---------------------------------------------------------------------------------------------
def load_clean(sub, ses, variant):
    z = np.load(os.path.join(CACHE, f"{sub}_{ses}_{variant}.npz"))
    return z["x"].astype(float), float(z["fs_orig"])


def reref(x16, ref="lm"):
    if ref == "lm":
        r = x16[[CH16.index(c) for c in MAST]].mean(axis=0)
    elif ref == "avg14":
        r = x16[:14].mean(axis=0)
    else:
        raise ValueError(ref)
    return x16[:14] - r


def _cut(x, s80):
    """(n, ch, len(TS)) baselined; NaN where the window leaves the recording."""
    out = np.full((len(s80), x.shape[0], len(TS)), np.nan)
    for k, s in enumerate(s80):
        if s + TS[0] < 0 or s + TS[-1] >= x.shape[1]:
            continue
        ep = x[:, s + TS]
        out[k] = ep - ep[:, BL].mean(axis=1, keepdims=True)
    return out


def epochs(sub, ses, variant, ref="lm"):
    """Epochs for both marker sets over the common clean-trial mask (steps 10-11)."""
    x16, fs0 = load_clean(sub, ses, variant)
    x = reref(x16, ref)
    tt = pm.trial_table(sub, ses)
    to80 = lambda s: np.round(s * FS / fs0).astype(int)          # noqa: E731
    ep = {"corrected": _cut(x, to80(tt.cor_sample.values)),
          "uncorrected": _cut(x, to80(tt.raw_sample.values))}
    ok = np.ones(len(tt), bool)
    for e in ep.values():
        with np.errstate(invalid="ignore"):
            ok &= np.isfinite(e).all(axis=(1, 2)) & (np.ptp(e, axis=2).max(axis=1) < PTP)
    return {k: e[ok] for k, e in ep.items()}, tt.label.values[ok], len(tt)


# ---------------------------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------------------------
def make_clf(nfilter=4, feat=FEAT):
    from pyriemann.spatialfilters import Xdawn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    return make_pipeline(
        FunctionTransformer(lambda X: X[:, :, feat]),
        Xdawn(nfilter=nfilter),
        FunctionTransformer(lambda X: X.reshape(len(X), -1)),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )


def cv_scores(X, y, clf, n_repeats=10, seed=0):
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    r = cross_validate(clf, X, (y == 2).astype(int), cv=cv,
                       scoring=("balanced_accuracy", "roc_auc"), n_jobs=1)
    ba = r["test_balanced_accuracy"].reshape(n_repeats, 5).mean(axis=1)
    auc = r["test_roc_auc"].reshape(n_repeats, 5).mean(axis=1)
    return ba, auc


# ---------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------
def _sessions():
    inv = pm.inventory()
    return inv[inv.has_eeg][["subject", "session"]].itertuples(index=False, name=None)


def stage_preprocess(pairs=None):
    from joblib import Parallel, delayed
    pairs = list(pairs or _sessions())

    def one(s, e):
        try:
            return preprocess(s, e)
        except Exception as ex:                       # noqa: BLE001
            return dict(subject=s, session=e, site=pm.site_of(e), status=f"error:{ex}")
    rows = Parallel(n_jobs=8)(delayed(one)(s, e) for s, e in pairs)
    df = pd.DataFrame(rows)
    f = os.path.join(OUT, "baseline_sessions.csv")
    if os.path.exists(f):
        old = pd.read_csv(f)
        old = old[~old.set_index(["subject", "session"]).index.isin(df.set_index(["subject", "session"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    df = df.sort_values(["subject", "session"])
    df.to_csv(f, index=False)
    bad = [dict(subject=r.subject, session=r.session, site=r.site, channel=b.split(":")[0],
                criterion=b.split(":")[1], value=float(b.split(":")[2]))
           for r in df.itertuples() if isinstance(getattr(r, "bads", None), str) and r.bads
           for b in r.bads.split(";")]
    pd.DataFrame(bad, columns=["subject", "session", "site", "channel", "criterion", "value"]).to_csv(
        os.path.join(OUT, "baseline_bad_channels.csv"), index=False)
    print(df.to_string(index=False))


def _ok_sessions():
    df = pd.read_csv(os.path.join(OUT, "baseline_sessions.csv"))
    return df[df.status == "ok"]


def check_filter():
    import mne
    rows = []
    for fs in (256.0, 1000.0, 2048.0):
        h = mne.filter.create_filter(None, fs, L_FREQ, H_FREQ, method="fir", phase="zero",
                                     fir_design="firwin", verbose=False)
        n = len(h)
        sym = np.allclose(h, h[::-1])
        f = np.array([0.05, 0.1, 0.5, 1, 5, 10, 15, 20, 30])
        H = np.abs(np.exp(-2j * np.pi * np.outer(f, np.arange(n)) / fs) @ h)
        rows.append(dict(fs=fs, taps=n, length_s=n / fs, symmetric=sym,
                         **{f"gain_{v}Hz": round(g, 3) for v, g in zip(f, H)}))
    return pd.DataFrame(rows)


def stage_checks():
    ok = _ok_sessions()
    print("\n=== check 1: filter (symmetric taps = zero phase after centring) ===")
    print(check_filter().to_string(index=False))

    rows, ratio_rows = [], []
    for r in ok.itertuples():
        tt_n = None
        for variant in VARIANTS:
            try:
                ep, lab, n_all = epochs(r.subject, r.session, variant)
            except Exception as ex:                   # noqa: BLE001
                rows.append(dict(subject=r.subject, session=r.session, site=r.site, variant=variant,
                                 status=f"error:{ex}"))
                continue
            e = ep["corrected"]
            d = e[lab == 2].mean(0) - e[lab == 1].mean(0) if (lab == 2).any() and (lab == 1).any() else None
            pz = CH14.index("Pz")
            rows.append(dict(subject=r.subject, session=r.session, site=r.site, variant=variant, status="ok",
                             n_trials=n_all, n_kept=len(lab), n_targets=int((lab == 2).sum()),
                             yield_=len(lab) / n_all,
                             p3_pz=np.nan if d is None else d[pz, P3].mean(),
                             lat_pz=np.nan if d is None else TS[P3][np.argmax(d[pz, P3])] / FS * 1000))
            # check 3: same trials under both references, rejection on Pz (as mastoid_p300_check)
            x16, fs0 = load_clean(r.subject, r.session, variant)
            tt = pm.trial_table(r.subject, r.session)
            s80 = np.round(tt.cor_sample.values * FS / fs0).astype(int)
            eps = {ref: _cut(reref(x16, ref)[[pz]], s80)[:, 0] for ref in ("lm", "avg14")}
            good = np.ones(len(s80), bool)
            for e2 in eps.values():
                with np.errstate(invalid="ignore"):
                    good &= np.isfinite(e2).all(1) & (np.ptp(e2, axis=1) < PTP)
            lab2 = tt.label.values
            rr = dict(subject=r.subject, session=r.session, site=r.site, variant=variant,
                      n_t=int((good & (lab2 == 2)).sum()))
            for ref, e2 in eps.items():
                dd = e2[good & (lab2 == 2)].mean(0) - e2[good & (lab2 == 1)].mean(0)
                rr[f"p3_{ref}"] = dd[P3].mean()
            rr["ratio"] = rr["p3_lm"] / rr["p3_avg14"]
            ratio_rows.append(rr)
    ck = pd.DataFrame(rows)
    ck.to_csv(os.path.join(OUT, "baseline_checks.csv"), index=False)
    rt = pd.DataFrame(ratio_rows)
    rt["usable"] = (rt.n_t >= 15) & (rt.p3_avg14 > 1.0)
    rt.to_csv(os.path.join(OUT, "baseline_mastoid_ratio.csv"), index=False)

    okc = ck[ck.status == "ok"]
    print("\n=== check 2: P300 at Pz (target-nontarget, 0.3-0.6 s, linked mastoids) ===")
    w = okc.pivot_table(index=["subject", "session", "site"], columns="variant", values="p3_pz")
    print(okc.groupby(["site", "variant"])[["p3_pz", "lat_pz"]].median().round(2).unstack())
    print(f"all sessions: median P300 nogedai {w.nogedai.median():.2f} µV, gedai {w.gedai.median():.2f} µV; "
          f"median per-session gedai/nogedai = {(w.gedai / w.nogedai).median():.2f}; "
          f"grand-mean ratio = {w.gedai.mean() / w.nogedai.mean():.2f}")

    print("\n=== check 3: linked-mastoid stability (ratio lm/avg14 at Pz) ===")
    for v in VARIANTS:
        u = rt[(rt.variant == v) & rt.usable]
        print(f"{v:8s} usable {len(u)}/{(rt.variant == v).sum()}, collapsed (ratio<0.25) {int((u.ratio < 0.25).sum())}; "
              f"median ratio per site: " + ", ".join(f"{s} {g.ratio.median():.2f}" for s, g in u.groupby("site")))
    from scipy.stats import wilcoxon
    u = rt[(rt.variant == "gedai") & rt.usable]
    subj = u.groupby(["subject", "site"]).ratio.mean().unstack()
    for a, b in (("SHE", "LDS"), ("SHE", "MAN"), ("LDS", "MAN")):
        if a in subj and b in subj:
            d = (subj[a] - subj[b]).dropna()
            p = wilcoxon(d).pvalue if len(d) >= 5 else np.nan
            print(f"  gedai ratio {a}-{b}: median {d.median():+.2f}, p={p:.3f}, n={len(d)}")

    print("\n=== check 4: trial yield (100 µV p-p, 14 ch, both marker sets) ===")
    print(okc.groupby(["site", "variant"]).agg(yield_median=("yield_", "median"),
                                               targets_median=("n_targets", "median"),
                                               sessions_ge20=("n_targets", lambda s: int((s >= MIN_TARGETS).sum())),
                                               sessions=("n_targets", "size")).round(2))


def stage_classify():
    ok = _ok_sessions()
    rows = []
    configs = [("primary", "gedai", 4, FEAT), ("nogedai", "nogedai", 4, FEAT),
               ("window_0.2-0.6", "gedai", 4, FEAT_SENS), ("nfilter2", "gedai", 2, FEAT)]
    for r in ok.itertuples():
        for name, variant, nf, feat in configs:
            try:
                ep, lab, _ = epochs(r.subject, r.session, variant)
            except Exception as ex:                   # noqa: BLE001
                print(f"skip {r.subject} {r.session} {name}: {ex}")
                continue
            nt = int((lab == 2).sum())
            if nt < MIN_TARGETS:
                rows.append(dict(subject=r.subject, session=r.session, site=r.site, config=name,
                                 markers="-", n_targets=nt, status="excluded_min_targets"))
                continue
            for mk, X in ep.items():
                ba, auc = cv_scores(X, lab, make_clf(nf, feat))
                rows += [dict(subject=r.subject, session=r.session, site=r.site, config=name, markers=mk,
                              n_targets=nt, status="ok", repeat=i, bal_acc=ba[i], auc=auc[i])
                         for i in range(len(ba))]
                if name == "primary":                 # check 6: shuffled labels
                    rng = np.random.default_rng(0)
                    ba_s, auc_s = cv_scores(X, rng.permutation(lab), make_clf(nf, feat), n_repeats=2)
                    rows += [dict(subject=r.subject, session=r.session, site=r.site, config="shuffled",
                                  markers=mk, n_targets=nt, status="ok", repeat=i, bal_acc=ba_s[i], auc=auc_s[i])
                             for i in range(len(ba_s))]
        print(f"done {r.subject} {r.session}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "baseline_scores.csv"), index=False)
    s = df[df.status == "ok"].groupby(["config", "markers", "subject", "session", "site"])[["bal_acc", "auc"]].mean()
    print("\n=== session-mean scores: median over sessions (n sessions) ===")
    g = s.groupby(["config", "markers"])
    print(pd.concat([g.median().round(3), g.size().rename("n")], axis=1))
    print("\nprimary by site:")
    print(s.loc["primary"].groupby(["markers", "site"]).median().round(3))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "preprocess":
        a = sys.argv[2:]
        stage_preprocess(list(zip(a[::2], a[1::2])) if a else None)
    elif stage == "checks":
        stage_checks()
    elif stage == "classify":
        stage_classify()
    else:
        raise SystemExit(__doc__)
