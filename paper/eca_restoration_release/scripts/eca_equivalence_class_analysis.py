#!/usr/bin/env python3
"""
eca_equivalence_class_analysis.py -- sensitivity analysis for the Wolfram-class comparison.

The 256 elementary rules fall into 88 equivalence classes under left-right reflection and
0/1 conjugation. Equivalent rules have identical expected restoration, so the 256 rule-level
values are not independent. This script repeats the one-way ANOVA, Kruskal-Wallis test and
Holm-corrected pairwise Mann-Whitney comparisons with ONE value per equivalence class (n = 88).

Usage:
  python eca_equivalence_class_analysis.py rule_metrics_aggregated.csv \
         --rule-col rule --class-col wolfram_class --metric-col mean_relative_restoration
Requires numpy, pandas, scipy.
"""
import argparse, itertools, sys
import numpy as np, pandas as pd
from scipy import stats

def table(r): return [(r >> i) & 1 for i in range(8)]          # index = neighbourhood (l*4+c*2+r)
def reflect(r):
    b = table(r); return sum(b[((n & 1) << 2) | (n & 2) | (n >> 2)] << n for n in range(8))
def conjugate(r):
    b = table(r); return sum((1 - b[7 - n]) << n for n in range(8))
def eq_classes():
    seen, classes = {}, []
    for r in range(256):
        if r in seen: continue
        orbit = {r, reflect(r), conjugate(r), reflect(conjugate(r))}
        for m in orbit: seen[m] = len(classes)
        classes.append(sorted(orbit))
    return classes, seen

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); m = len(p); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--rule-col", default="rule")
    ap.add_argument("--class-col", default="wolfram_class"); ap.add_argument("--metric-col", default="mean_relative_restoration")
    ap.add_argument("--out", default="equivalence_class_sensitivity.csv"); a = ap.parse_args()
    classes, idx = eq_classes()
    assert len(classes) == 88, len(classes)
    d = pd.read_csv(a.csv)[[a.rule_col, a.class_col, a.metric_col]].rename(columns={a.rule_col: "rule", a.class_col: "cls", a.metric_col: "y"})
    assert d.rule.nunique() == 256, "expected 256 rules"
    d["eq"] = d.rule.map(idx)
    g = d.groupby("eq")
    bad = g.cls.nunique(); bad = bad[bad > 1]
    if len(bad): print("WARNING: class label differs within equivalence classes:", {int(k): classes[k] for k in bad.index})
    e = g.agg(cls=("cls", lambda s: s.mode().iat[0]), y=("y", "mean"), n_rules=("rule", "size"), spread=("y", lambda s: s.max() - s.min())).reset_index()
    print("equivalence classes: %d; mean within-class spread of restoration = %.4f (max %.4f)" % (len(e), e.spread.mean(), e.spread.max()))
    groups = [v.y.values for _, v in e.groupby("cls")]; names = [k for k, _ in e.groupby("cls")]
    F, pF = stats.f_oneway(*groups); H, pH = stats.kruskal(*groups)
    ss_b = sum(len(x) * (x.mean() - e.y.mean()) ** 2 for x in groups); ss_t = ((e.y - e.y.mean()) ** 2).sum()
    k, N = len(groups), len(e)
    print("n = %d, classes = %s, sizes = %s" % (N, names, [len(x) for x in groups]))
    print("ANOVA: F(%d, %d) = %.2f, p = %.3g; eta^2 = %.3f" % (k - 1, N - k, F, pF, ss_b / ss_t))
    ms_w = (ss_t - ss_b) / (N - k); omega2 = (ss_b - (k - 1) * ms_w) / (ss_t + ms_w)
    print("omega^2 = %.3f" % omega2)
    print("Kruskal-Wallis: H(%d) = %.2f, p = %.3g" % (k - 1, H, pH))
    rows = []
    for (i, j) in itertools.combinations(range(k), 2):
        U, p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
        rb = abs(2 * U / (len(groups[i]) * len(groups[j])) - 1)   # |rank-biserial correlation|
        rows.append((names[i], names[j], len(groups[i]), len(groups[j]), np.median(groups[i]), np.median(groups[j]), U, p, rb))
    out = pd.DataFrame(rows, columns=["class_a", "class_b", "n_a", "n_b", "median_a", "median_b", "U", "p", "rank_biserial_abs"])
    out["p_holm"] = holm(out.p.values); out.to_csv(a.out, index=False); print(out.round(4).to_string(index=False))
    print("pairwise comparisons significant after Holm (alpha 0.05): %d of %d" % ((out.p_holm < 0.05).sum(), len(out)))
if __name__ == "__main__": main()
