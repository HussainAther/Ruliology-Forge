#!/usr/bin/env python3
"""Computational Restoration Benchmark v2.

Adds causal-cone-normalized damage, signed relative restoration, exact
coalescence probability, recovery time, damage AUC, random initial-condition
ensembles, multiple perturbation times, and symmetry validation.

Portable default output: ./restoration_v2_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LUT = np.array([[(r >> i) & 1 for i in range(8)] for r in range(256)], dtype=np.uint8)


def step_batch(x: np.ndarray, rule: int, boundary: str = "periodic") -> np.ndarray:
    if boundary == "periodic":
        left = np.roll(x, 1, axis=1)
        right = np.roll(x, -1, axis=1)
    else:
        z = np.zeros((len(x), 1), dtype=np.uint8)
        left = np.concatenate([z, x[:, :-1]], axis=1)
        right = np.concatenate([x[:, 1:], z], axis=1)
    idx = (left << 2) | (x << 1) | right
    return LUT[rule][idx]


def cone_mask(width: int, start: int, perturb_width: int, k: int, boundary: str) -> np.ndarray:
    """Cells causally reachable k ECA steps after a width-p perturbation."""
    mask = np.zeros(width, dtype=bool)
    if boundary == "periodic":
        inds = np.arange(start - k, start + perturb_width + k) % width
        mask[inds] = True
    else:
        lo = max(0, start - k)
        hi = min(width, start + perturb_width + k)
        mask[lo:hi] = True
    return mask


def transform_rule(rule: int, kind: str) -> int:
    out = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        a, b, c = (i >> 2) & 1, (i >> 1) & 1, i & 1
        if kind == "reflect":
            j = (c << 2) | (b << 1) | a
            y = LUT[rule, j]
        elif kind == "conjugate":
            j = ((1 - a) << 2) | ((1 - b) << 1) | (1 - c)
            y = 1 - LUT[rule, j]
        else:
            raise ValueError(kind)
        out[i] = y
    return int(sum(int(out[i]) << i for i in range(8)))


def apply_perturbation(per: np.ndarray, ref: np.ndarray, start: int, pw: int,
                       op: str, rng: np.random.Generator) -> None:
    if op == "random_replace":
        per[:, start:start + pw] = rng.integers(0, 2, (len(per), pw), dtype=np.uint8)
    elif op == "flip_window":
        per[:, start:start + pw] ^= 1
    elif op == "single_flip":
        per[:, start + pw // 2] ^= 1
    else:
        raise ValueError(f"Unknown perturbation operator: {op}")


def run_rule(rule: int, *, width: int, perturb_times: list[int], post_steps: int,
             perturb_width: int, trials: int, boundary: str, op: str,
             init_density: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed + 1000003 * rule)
    max_tp = max(perturb_times)

    # One random initial-condition ensemble per rule; snapshots are taken at each tp.
    x0 = (rng.random((trials, width)) < init_density).astype(np.uint8)
    state = x0.copy()
    snapshots: dict[int, np.ndarray] = {}
    for t in range(max_tp + 1):
        if t in perturb_times:
            snapshots[t] = state.copy()
        if t < max_tp:
            state = step_batch(state, rule, boundary)

    rows = []
    start = width // 2 - perturb_width // 2

    for tp in perturb_times:
        ref = snapshots[tp].copy()
        per = ref.copy()
        local_rng = np.random.default_rng(seed + 1000003 * rule + 7919 * tp)
        apply_perturbation(per, ref, start, perturb_width, op, local_rng)

        intended0 = cone_mask(width, start, perturb_width, 0, boundary)
        d0 = np.mean(ref[:, intended0] != per[:, intended0], axis=1)

        cone_series = np.empty((trials, post_steps), dtype=float)
        full_series = np.empty((trials, post_steps), dtype=float)

        for k in range(1, post_steps + 1):
            ref = step_batch(ref, rule, boundary)
            per = step_batch(per, rule, boundary)
            cmask = cone_mask(width, start, perturb_width, k, boundary)
            cone_series[:, k - 1] = np.mean(ref[:, cmask] != per[:, cmask], axis=1)
            full_series[:, k - 1] = np.mean(ref != per, axis=1)

        # Signed relative restoration: 1 means complete average removal of initial
        # local damage; 0 means damage persists at its initial density; negative
        # means average amplification. Trials with d0=0 are excluded for this metric.
        valid = d0 > 0
        rel_auc = np.full(trials, np.nan)
        rel_restore = np.full(trials, np.nan)
        rel_auc[valid] = cone_series[valid].mean(axis=1) / d0[valid]
        rel_restore[valid] = 1.0 - rel_auc[valid]

        # Deterministic coalescence: once full states match, they remain matched.
        recovered = full_series == 0
        exact = np.any(recovered, axis=1)
        recovery_time = np.full(trials, np.nan)
        if np.any(exact):
            recovery_time[exact] = np.argmax(recovered[exact], axis=1) + 1

        # A bounded descriptive score inside the causal cone (not the main recovery metric).
        cone_similarity = 1.0 - cone_series.mean(axis=1)

        rows.append({
            "rule": rule,
            "perturb_time": tp,
            "trials": trials,
            "width": width,
            "post_steps": post_steps,
            "perturb_width": perturb_width,
            "boundary": boundary,
            "operator": op,
            "init_density": init_density,
            "initial_damage_mean": float(np.mean(d0)),
            "cone_damage_auc_mean": float(np.mean(cone_series)),
            "cone_damage_peak_mean": float(np.mean(np.max(cone_series, axis=1))),
            "cone_damage_final_mean": float(np.mean(cone_series[:, -1])),
            "cone_similarity_mean": float(np.mean(cone_similarity)),
            "relative_damage_auc_mean": float(np.nanmean(rel_auc)),
            "relative_restoration_mean": float(np.nanmean(rel_restore)),
            "relative_restoration_sd": float(np.nanstd(rel_restore, ddof=1)),
            "exact_coalescence_rate": float(np.mean(exact)),
            "recovery_time_mean_conditional": float(np.nanmean(recovery_time)) if np.any(exact) else np.nan,
            "recovery_time_median_conditional": float(np.nanmedian(recovery_time)) if np.any(exact) else np.nan,
            "full_damage_final_mean": float(np.mean(full_series[:, -1])),
            "zero_effect_perturbation_rate": float(np.mean(~valid)),
        })
    return rows


def aggregate_across_times(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "initial_damage_mean", "cone_damage_auc_mean", "cone_damage_peak_mean",
        "cone_damage_final_mean", "cone_similarity_mean", "relative_damage_auc_mean",
        "relative_restoration_mean", "exact_coalescence_rate",
        "recovery_time_mean_conditional", "full_damage_final_mean",
        "zero_effect_perturbation_rate",
    ]
    agg = df.groupby("rule", as_index=False)[metrics].mean(numeric_only=True)
    return agg


def symmetry_table(agg: pd.DataFrame) -> pd.DataFrame:
    by_rule = agg.set_index("rule")
    rows = []
    for r in range(256):
        for kind in ("reflect", "conjugate"):
            p = transform_rule(r, kind)
            rows.append({
                "rule": r,
                "transform": kind,
                "partner_rule": p,
                "delta_relative_restoration": abs(
                    by_rule.loc[r, "relative_restoration_mean"] - by_rule.loc[p, "relative_restoration_mean"]
                ),
                "delta_cone_auc": abs(
                    by_rule.loc[r, "cone_damage_auc_mean"] - by_rule.loc[p, "cone_damage_auc_mean"]
                ),
                "delta_coalescence": abs(
                    by_rule.loc[r, "exact_coalescence_rate"] - by_rule.loc[p, "exact_coalescence_rate"]
                ),
            })
    return pd.DataFrame(rows)


def timing_stability(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.pivot(index="rule", columns="perturb_time", values="relative_restoration_mean")
    times = list(piv.columns)
    rows = []
    for i, a in enumerate(times):
        for b in times[i + 1:]:
            rho, p = spearmanr(piv[a], piv[b], nan_policy="omit")
            rows.append({"time_a": a, "time_b": b, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="restoration_v2_outputs")
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--post-steps", type=int, default=40)
    ap.add_argument("--perturb-width", type=int, default=20)
    ap.add_argument("--perturb-times", default="25,50,75")
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    ap.add_argument("--operator", choices=["random_replace", "flip_window", "single_flip"], default="random_replace")
    ap.add_argument("--init-density", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pts = [int(x) for x in args.perturb_times.split(",")]

    rows = []
    for rule in range(256):
        rows.extend(run_rule(
            rule,
            width=args.width,
            perturb_times=pts,
            post_steps=args.post_steps,
            perturb_width=args.perturb_width,
            trials=args.trials,
            boundary=args.boundary,
            op=args.operator,
            init_density=args.init_density,
            seed=args.seed,
        ))
        if rule % 32 == 31:
            print(f"completed rules 0-{rule}")

    by_time = pd.DataFrame(rows)
    agg = aggregate_across_times(by_time)
    sym = symmetry_table(agg)
    stab = timing_stability(by_time)

    by_time.to_csv(out / "rule_metrics_by_perturbation_time.csv", index=False)
    agg.to_csv(out / "rule_metrics_aggregated.csv", index=False)
    sym.to_csv(out / "symmetry_validation.csv", index=False)
    stab.to_csv(out / "timing_rank_stability.csv", index=False)

    top = agg.sort_values("relative_restoration_mean", ascending=False).head(12)
    low = agg.sort_values("relative_restoration_mean", ascending=True).head(12)

    print("\nTOP RELATIVE RESTORATION")
    print(top[["rule", "relative_restoration_mean", "cone_damage_auc_mean", "exact_coalescence_rate"]].to_string(index=False))
    print("\nLOW RELATIVE RESTORATION")
    print(low[["rule", "relative_restoration_mean", "cone_damage_auc_mean", "exact_coalescence_rate"]].to_string(index=False))
    print("\nTIMING RANK STABILITY")
    print(stab.to_string(index=False))
    print("\nSYMMETRY VALIDATION (mean absolute deltas)")
    print(sym.groupby("transform")[["delta_relative_restoration", "delta_cone_auc", "delta_coalescence"]].mean().to_string())
    print(f"\nWrote outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
