#!/usr/bin/env python3
"""Generate a reproducible ECA restoration animation from the v2 benchmark.

This script imports the exact benchmark update, perturbation, and causal-cone
functions from run_restoration_benchmark_v2.py. It reconstructs the same seeded
trial ensemble used by the manuscript, selects one representative trial per
rule (closest to that rule's mean relative restoration at the chosen
perturbation time), and animates:

  1. reference spacetime trajectory,
  2. perturbed spacetime trajectory,
  3. reference-vs-perturbed disagreement with causal-cone boundary,
  4. live causal-cone damage D_C(tau).

Default rules:
  Rule 36  - strongly restorative Class II example
  Rule 161 - strongly non-restorative Class III example
  Rule 105 - confirmed amplifying Class III example

The defaults mirror the manuscript benchmark:
  width=200, perturb_width=20, perturb_time=50, post_steps=40,
  trials=128, periodic boundary, random-replacement perturbation,
  init_density=0.5, seed=20260816.

Usage:
    python generate_restoration_animation.py \
        --benchmark-script run_restoration_benchmark_v2.py \
        --out restoration_movie.gif

If ffmpeg is installed you can instead use --out restoration_movie.mp4.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle


def load_benchmark(path: Path):
    spec = importlib.util.spec_from_file_location("restoration_benchmark_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import benchmark script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct_rule_trials(
    bench,
    rule: int,
    *,
    width: int,
    perturb_time: int,
    post_steps: int,
    perturb_width: int,
    trials: int,
    boundary: str,
    operator: str,
    init_density: float,
    seed: int,
):
    """Reconstruct the exact seeded trial ensemble for one rule and one tp."""
    rng = np.random.default_rng(seed + 1000003 * rule)
    x0 = (rng.random((trials, width)) < init_density).astype(np.uint8)

    # Preserve the full pre-perturbation trajectory for visualization.
    states = [x0.copy()]
    state = x0.copy()
    for _ in range(perturb_time):
        state = bench.step_batch(state, rule, boundary)
        states.append(state.copy())

    ref = state.copy()
    per = ref.copy()
    start = width // 2 - perturb_width // 2

    local_rng = np.random.default_rng(seed + 1000003 * rule + 7919 * perturb_time)
    bench.apply_perturbation(per, ref, start, perturb_width, operator, local_rng)

    ref_post = [ref.copy()]
    per_post = [per.copy()]

    initial_mask = bench.cone_mask(width, start, perturb_width, 0, boundary)
    d0 = np.mean(ref[:, initial_mask] != per[:, initial_mask], axis=1)

    cone_series = np.empty((trials, post_steps), dtype=float)
    full_series = np.empty((trials, post_steps), dtype=float)

    ref_work = ref.copy()
    per_work = per.copy()
    for k in range(1, post_steps + 1):
        ref_work = bench.step_batch(ref_work, rule, boundary)
        per_work = bench.step_batch(per_work, rule, boundary)
        ref_post.append(ref_work.copy())
        per_post.append(per_work.copy())

        cmask = bench.cone_mask(width, start, perturb_width, k, boundary)
        cone_series[:, k - 1] = np.mean(ref_work[:, cmask] != per_work[:, cmask], axis=1)
        full_series[:, k - 1] = np.mean(ref_work != per_work, axis=1)

    valid = d0 > 0
    rel_restore = np.full(trials, np.nan)
    rel_restore[valid] = 1.0 - cone_series[valid].mean(axis=1) / d0[valid]

    mean_r = np.nanmean(rel_restore)
    candidates = np.where(np.isfinite(rel_restore))[0]
    representative = candidates[np.argmin(np.abs(rel_restore[candidates] - mean_r))]

    # Build one representative spacetime trajectory.
    pre = np.stack([s[representative] for s in states], axis=0)
    ref_after = np.stack([s[representative] for s in ref_post], axis=0)
    per_after = np.stack([s[representative] for s in per_post], axis=0)

    # Avoid duplicating t=tp in reference history.
    ref_space = np.concatenate([pre[:-1], ref_after], axis=0)
    per_space = np.concatenate([pre[:-1], per_after], axis=0)
    diff_space = ref_space != per_space

    rep_d0 = float(d0[representative])
    rep_cone = cone_series[representative]
    rep_r = float(rel_restore[representative])
    exact = bool(np.any(full_series[representative] == 0))
    recovery_time = None
    if exact:
        recovery_time = int(np.argmax(full_series[representative] == 0) + 1)

    return {
        "rule": rule,
        "representative_trial": int(representative),
        "rule_mean_restoration": float(mean_r),
        "trial_restoration": rep_r,
        "d0": rep_d0,
        "cone_damage": rep_cone,
        "exact": exact,
        "recovery_time": recovery_time,
        "reference": ref_space,
        "perturbed": per_space,
        "difference": diff_space,
        "start": start,
        "perturb_width": perturb_width,
        "perturb_time": perturb_time,
        "post_steps": post_steps,
        "width": width,
    }


def cone_edges(start: int, perturb_width: int, tau: int):
    """Unwrapped causal-cone boundaries; valid here because the main benchmark does not wrap."""
    return start - tau, start + perturb_width - 1 + tau


def make_animation(records, out: Path, fps: int, dpi: int):
    n_rules = len(records)
    width = records[0]["width"]
    tp = records[0]["perturb_time"]
    post_steps = records[0]["post_steps"]
    n_frames = tp + post_steps + 1

    fig = plt.figure(figsize=(15, 4.8 * n_rules), constrained_layout=True)
    gs = fig.add_gridspec(n_rules, 4, width_ratios=[1.0, 1.0, 1.0, 0.9])

    rows = []
    for row_idx, rec in enumerate(records):
        ax_ref = fig.add_subplot(gs[row_idx, 0])
        ax_per = fig.add_subplot(gs[row_idx, 1])
        ax_diff = fig.add_subplot(gs[row_idx, 2])
        ax_curve = fig.add_subplot(gs[row_idx, 3])

        blank = np.zeros((n_frames, width), dtype=float)
        im_ref = ax_ref.imshow(blank, aspect="auto", interpolation="nearest", origin="upper", vmin=0, vmax=1, cmap="gray_r")
        im_per = ax_per.imshow(blank, aspect="auto", interpolation="nearest", origin="upper", vmin=0, vmax=1, cmap="gray_r")
        im_diff = ax_diff.imshow(blank, aspect="auto", interpolation="nearest", origin="upper", vmin=0, vmax=1, cmap="binary")

        ax_ref.set_title(f"Rule {rec['rule']} — reference")
        ax_per.set_title("Perturbed")
        ax_diff.set_title("Disagreement + causal cone")

        for ax in (ax_ref, ax_per, ax_diff):
            ax.set_xlim(0, width - 1)
            ax.set_ylim(n_frames - 0.5, -0.5)
            ax.set_xlabel("Lattice position")
            ax.set_ylabel("Time")
            ax.axhline(tp - 0.5, linewidth=1.0, linestyle="--")

        # Highlight nominal perturbation window at t=tp.
        rect = Rectangle(
            (rec["start"] - 0.5, tp - 0.5),
            rec["perturb_width"],
            1.0,
            fill=False,
            linewidth=2.0,
        )
        ax_per.add_patch(rect)

        left_line, = ax_diff.plot([], [], linewidth=1.5)
        right_line, = ax_diff.plot([], [], linewidth=1.5)

        x_tau = np.arange(1, post_steps + 1)
        ax_curve.set_xlim(1, post_steps)
        ax_curve.set_ylim(0, 1)
        ax_curve.set_xlabel("Post-perturbation update τ")
        ax_curve.set_ylabel("Causal-cone damage D_C(τ)")
        curve, = ax_curve.plot([], [], linewidth=2.0)
        marker, = ax_curve.plot([], [], marker="o")
        ax_curve.axhline(rec["d0"], linestyle="--", linewidth=1.0)
        ax_curve.set_title(
            f"trial {rec['representative_trial']} | "
            f"R_trial={rec['trial_restoration']:.3f}\n"
            f"R_rule≈{rec['rule_mean_restoration']:.3f} | "
            + (f"coalesces at τ={rec['recovery_time']}" if rec["exact"] else f"no exact coalescence ≤{rec["post_steps"]}")
        )

        rows.append({
            "rec": rec,
            "im_ref": im_ref,
            "im_per": im_per,
            "im_diff": im_diff,
            "left_line": left_line,
            "right_line": right_line,
            "curve": curve,
            "marker": marker,
            "x_tau": x_tau,
        })

    fig.suptitle(
        "Computational restoration after a localized perturbation\n"
        "Exact v2 benchmark dynamics; dashed horizontal line marks perturbation time",
        fontsize=16,
    )

    def update(frame):
        artists = []
        for item in rows:
            rec = item["rec"]
            ref_show = np.zeros_like(rec["reference"], dtype=float)
            per_show = np.zeros_like(rec["perturbed"], dtype=float)
            diff_show = np.zeros_like(rec["difference"], dtype=float)

            ref_show[: frame + 1] = rec["reference"][: frame + 1]
            per_show[: frame + 1] = rec["perturbed"][: frame + 1]
            diff_show[: frame + 1] = rec["difference"][: frame + 1]

            item["im_ref"].set_data(ref_show)
            item["im_per"].set_data(per_show)
            item["im_diff"].set_data(diff_show)
            artists.extend([item["im_ref"], item["im_per"], item["im_diff"]])

            if frame >= tp:
                tau_max = frame - tp
                ys = np.arange(tp, frame + 1)
                lefts = []
                rights = []
                for y in ys:
                    tau = y - tp
                    l, r = cone_edges(rec["start"], rec["perturb_width"], tau)
                    lefts.append(l)
                    rights.append(r)
                item["left_line"].set_data(lefts, ys)
                item["right_line"].set_data(rights, ys)

                if tau_max >= 1:
                    n = min(tau_max, rec["post_steps"])
                    item["curve"].set_data(item["x_tau"][:n], rec["cone_damage"][:n])
                    item["marker"].set_data([n], [rec["cone_damage"][n - 1]])
                else:
                    item["curve"].set_data([], [])
                    item["marker"].set_data([], [])
            else:
                item["left_line"].set_data([], [])
                item["right_line"].set_data([], [])
                item["curve"].set_data([], [])
                item["marker"].set_data([], [])

            artists.extend([item["left_line"], item["right_line"], item["curve"], item["marker"]])

        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".gif":
        ani.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    elif suffix == ".mp4":
        ani.save(out, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
    else:
        raise ValueError("Output must end in .gif or .mp4")

    # Also save a static final frame for manuscript/supplementary use.
    update(n_frames - 1)
    static_path = out.with_name(out.stem + "_final_frame.png")
    fig.savefig(static_path, dpi=max(dpi, 180), bbox_inches="tight")
    plt.close(fig)
    return static_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-script", default="run_restoration_benchmark_v2.py")
    ap.add_argument("--out", default="Supplementary_Movie_S1.gif")
    ap.add_argument("--rules", default="36,161,105")
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--perturb-time", type=int, default=50)
    ap.add_argument("--post-steps", type=int, default=40)
    ap.add_argument("--perturb-width", type=int, default=20)
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    ap.add_argument("--operator", choices=["random_replace", "flip_window", "single_flip"], default="random_replace")
    ap.add_argument("--init-density", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    benchmark_path = Path(args.benchmark_script).resolve()
    bench = load_benchmark(benchmark_path)

    rules = [int(x.strip()) for x in args.rules.split(",") if x.strip()]
    records = []
    for rule in rules:
        rec = reconstruct_rule_trials(
            bench,
            rule,
            width=args.width,
            perturb_time=args.perturb_time,
            post_steps=args.post_steps,
            perturb_width=args.perturb_width,
            trials=args.trials,
            boundary=args.boundary,
            operator=args.operator,
            init_density=args.init_density,
            seed=args.seed,
        )
        records.append(rec)
        print(
            f"Rule {rule}: representative trial={rec['representative_trial']}, "
            f"trial R={rec['trial_restoration']:.4f}, "
            f"rule mean R={rec['rule_mean_restoration']:.4f}, "
            f"exact={rec['exact']}, recovery_time={rec['recovery_time']}"
        )

    out = Path(args.out).resolve()
    static_path = make_animation(records, out, fps=args.fps, dpi=args.dpi)
    print(f"Wrote animation: {out}")
    print(f"Wrote final-frame figure: {static_path}")


if __name__ == "__main__":
    main()
