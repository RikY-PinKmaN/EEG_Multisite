#!/usr/bin/env python3
"""
p300_baseline_report.py — summarise p300_baseline_v2.csv (variant x classifier x marker x regime).

Per configuration: mean balanced accuracy / AUC over training units, corrected vs uncorrected
(paired Wilcoxon over units, then over subjects), and the GEDAI-vs-mine contrast on corrected
within-session accuracy (the pre-registered preprocessing decision).
Usage: python p300_baseline_report.py [FILE]
"""
import os
import sys
import pandas as pd

from p300_report import HERE, table_markers, unit_means, paired

f = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "p300_baseline_v2.csv")
df = pd.read_csv(f)
tabs = []
for (v, c), g in df.groupby(["variant", "clf"]):
    t = table_markers(g, ["variant", "clf"]).drop(columns="arm")
    tabs.append(t.assign(variant=v, clf=c))
tab = pd.concat(tabs, ignore_index=True)
cols = ["variant", "clf", "regime", "units", "bacc_cor", "bacc_raw", "raw_minus_cor", "p_units",
        "p_subjects", "auc_cor", "auc_raw"]
tab = tab[cols]
tab.to_csv(f.replace(".csv", "_summary.csv"), index=False)
print(tab.round(3).to_string(index=False))

print("\nPreprocessing decision (corrected, within-session, GEDAI - mine, paired units):")
u = unit_means(df[(df.marker == "cor") & (df.regime == "within")], ["variant", "clf"])
for c in sorted(u.clf.unique()):
    for win in ("", "-w"):
        w = u[u.clf == c].pivot_table(index=["subject", "train_session"], columns="variant", values="bacc")
        a, b = "mine" + win, "gedai" + win
        if {a, b} <= set(w.columns):
            r = paired(w[a], w[b])
            n_a, n_b = w[a].notna().sum(), w[b].notna().sum()
            print(f"  {c:5s} {a:7s} {r['mean_a']:.3f}  {b:8s} {r['mean_b']:.3f}  diff {r['diff']:+.3f}"
                  f"  p = {r['p']:.3g}  (paired n = {r['n']}; units mine {n_a}, gedai {n_b})")
