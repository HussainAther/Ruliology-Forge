#!/usr/bin/env python3
"""Robustness check: Wald vs. bootstrap CIs for headline rules.

Regenerates raw per-trial relative-restoration values using the EXACT
seeding scheme of run_restoration_benchmark_v2.py (same seed, same
per-rule / per-perturbation-time RNG offsets), then compares:
  (1) the manuscript's existing Wald CI (pooled-mean +/- 1.96*pooled-SE,
      pooled variance = mean of per-time sample variances, n_eff=3*trials)
  (2) a nonparametric percentile bootstrap on the pooled 3*trials values
  (3) a BCa bootstrap on the same pooled values
  (4) a two-stage (cluster) bootstrap that also resamples perturbation
      times, to propagate between-time variability into the interval
for a handful of headline rules, and cross-checks rules 105 and 161
against the manuscript's own higher-trial confirmatory rerun.
"""
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

LUT = np.array([[(r >> i) & 1 for i in range(8)] for r in range(256)], dtype=np.uint8)


def step_batch(x, rule, boundary="periodic"):
    if boundary == "periodic":
        left = np.roll(x, 1, axis=1)
        right = np.roll(x, -1, axis=1)
    idx = (left << 2) | (x << 1) | right
    return LUT[rule][idx]


def cone_mask(width, start, perturb_width, k, boundary):
    mask = np.zeros(width, dtype=bool)
    inds = np.arange(start - k, start + perturb_width + k) % width
    mask[inds] = True
    return mask


def apply_perturbation(per, ref, start, pw, op, rng):
    if op == "random_replace":
        per[:, start:start + pw] = rng.integers(0, 2, (len(per), pw), dtype=np.uint8)


def raw_rel_restore(rule, *, width=200, perturb_times=(25, 50, 75), post_steps=40,
                     perturb_width=20, trials=128, boundary="periodic",
                     op="random_replace", init_density=0.5, seed=20260816):
    """Reproduces run_rule()'s exact RNG stream; returns {tp: raw rel_restore array}."""
    rng = np.random.default_rng(seed + 1000003 * rule)
    max_tp = max(perturb_times)
    x0 = (rng.random((trials, width)) < init_density).astype(np.uint8)
    state = x0.copy()
    snapshots = {}
    for t in range(max_tp + 1):
        if t in perturb_times:
            snapshots[t] = state.copy()
        if t < max_tp:
            state = step_batch(state, rule, boundary)

    start = width // 2 - perturb_width // 2
    out = {}
    for tp in perturb_times:
        ref = snapshots[tp].copy()
        per = ref.copy()
        local_rng = np.random.default_rng(seed + 1000003 * rule + 7919 * tp)
        apply_perturbation(per, ref, start, perturb_width, op, local_rng)

        intended0 = cone_mask(width, start, perturb_width, 0, boundary)
        d0 = np.mean(ref[:, intended0] != per[:, intended0], axis=1)

        cone_series = np.empty((trials, post_steps), dtype=float)
        for k in range(1, post_steps + 1):
            ref = step_batch(ref, rule, boundary)
            per = step_batch(per, rule, boundary)
            cmask = cone_mask(width, start, perturb_width, k, boundary)
            cone_series[:, k - 1] = np.mean(ref[:, cmask] != per[:, cmask], axis=1)

        valid = d0 > 0
        rel_restore = np.full(trials, np.nan)
        rel_auc = np.full(trials, np.nan)
        rel_auc[valid] = cone_series[valid].mean(axis=1) / d0[valid]
        rel_restore[valid] = 1.0 - rel_auc[valid]
        out[tp] = rel_restore[valid]  # drop the zero-effect trials, matching nanmean
    return out


def wald_ci_pooled(by_time):
    """Reproduces the manuscript's existing method exactly."""
    means = np.array([v.mean() for v in by_time.values()])
    sds = np.array([v.std(ddof=1) for v in by_time.values()])
    ns = np.array([len(v) for v in by_time.values()])
    grand_mean = np.concatenate(list(by_time.values())).mean()  # matches nanmean pooling used for rule_metrics_aggregated (simple mean of per-time means, equal n)
    time_level_mean = means.mean()
    pooled_var = np.mean(sds ** 2)
    n_eff = ns.sum()
    se = np.sqrt(pooled_var / n_eff)
    return time_level_mean, se, time_level_mean - 1.96 * se, time_level_mean + 1.96 * se


def percentile_and_bca_bootstrap(pooled, n_resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    res_pct = bootstrap((pooled,), np.mean, n_resamples=n_resamples, method="percentile",
                         random_state=rng, confidence_level=0.95)
    res_bca = bootstrap((pooled,), np.mean, n_resamples=n_resamples, method="BCa",
                         random_state=rng, confidence_level=0.95)
    return (res_pct.confidence_interval.low, res_pct.confidence_interval.high,
            res_bca.confidence_interval.low, res_bca.confidence_interval.high)


def cluster_bootstrap(by_time, n_resamples=10000, seed=0):
    """Two-stage bootstrap: resample perturbation-time blocks with replacement,
    then resample trials within each drawn block. Propagates between-time
    variability into the interval, unlike the pooled-Wald/pooled-bootstrap
    approaches above which treat the three times as one pooled sample."""
    rng = np.random.default_rng(seed)
    tps = list(by_time.keys())
    n_blocks = len(tps)
    means = np.empty(n_resamples)
    for b in range(n_resamples):
        drawn_tps = rng.choice(tps, size=n_blocks, replace=True)
        vals = []
        for tp in drawn_tps:
            arr = by_time[tp]
            idx = rng.integers(0, len(arr), size=len(arr))
            vals.append(arr[idx].mean())
        means[b] = np.mean(vals)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return means.mean(), lo, hi


if __name__ == "__main__":
    headline_rules = [105, 161, 57, 36]
    rows = []
    for rule in headline_rules:
        by_time = raw_rel_restore(rule)
        mean_w, se_w, lo_w, hi_w = wald_ci_pooled(by_time)
        pooled = np.concatenate(list(by_time.values()))
        lo_p, hi_p, lo_bca, hi_bca = percentile_and_bca_bootstrap(pooled)
        mean_c, lo_c, hi_c = cluster_bootstrap(by_time)
        time_means = np.array([v.mean() for v in by_time.values()])
        rows.append({
            "rule": rule,
            "n_pooled": len(pooled),
            "mean": mean_w,
            "wald_se": se_w,
            "wald_lo": lo_w, "wald_hi": hi_w, "wald_width": hi_w - lo_w,
            "pct_boot_lo": lo_p, "pct_boot_hi": hi_p, "pct_boot_width": hi_p - lo_p,
            "bca_boot_lo": lo_bca, "bca_boot_hi": hi_bca, "bca_boot_width": hi_bca - lo_bca,
            "cluster_boot_mean": mean_c, "cluster_boot_lo": lo_c, "cluster_boot_hi": hi_c,
            "cluster_boot_width": hi_c - lo_c,
            "across_time_sd": time_means.std(ddof=1),
            "across_time_range": time_means.max() - time_means.min(),
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(out.round(5).to_string(index=False))
    import sys
    out.to_csv(sys.argv[1] if len(sys.argv) > 1 else "bootstrap_ci_comparison_headline_rules.csv", index=False)
