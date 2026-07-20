"""Command-line interface for Ruliology Forge."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analysis import parameter_grid, summarize_scan
from .experiments import (
    ExperimentConfig,
    evolve_resilient_rules,
    run_parameter_sweep,
    run_perturbation_experiment,
    scan_rules,
)
from .plotting import plot_divergence, plot_trajectory


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=201)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--perturb-time", type=int, default=80)
    parser.add_argument("--perturb-radius", type=int, default=5)
    parser.add_argument("--perturbation", choices=["bit_flip", "void", "random_mix"], default="bit_flip")
    parser.add_argument("--perturbation-timing", choices=["before_update", "after_update"], default="after_update")
    parser.add_argument("--noise-probability", type=float, default=0.5)
    parser.add_argument("--initial-condition", choices=["single", "random"], default="single")
    parser.add_argument("--initial-density", type=float, default=0.5)
    parser.add_argument("--boundary", choices=["periodic", "fixed"], default="periodic")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--recovery-threshold", type=float, default=0.01)
    parser.add_argument("--recovery-persistence", type=int, default=5)
    parser.add_argument("--max-shift", type=int, default=10)


def _experiment_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "width": args.width,
        "steps": args.steps,
        "perturb_time": args.perturb_time,
        "perturb_radius": args.perturb_radius,
        "perturbation": args.perturbation,
        "perturbation_timing": args.perturbation_timing,
        "noise_probability": args.noise_probability,
        "initial_condition": args.initial_condition,
        "initial_density": args.initial_density,
        "boundary": args.boundary,
        "seed": args.seed,
        "recovery_threshold": args.recovery_threshold,
        "recovery_persistence": args.recovery_persistence,
        "max_shift": args.max_shift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="ruliology")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('ruliology-forge')}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment = subparsers.add_parser("experiment", help="Run one perturbation experiment.")
    experiment.add_argument("--rule", type=int, default=110)
    experiment.add_argument("--perturb-center", type=int, default=None)
    experiment.add_argument("--output-dir", type=Path, default=Path("results/experiment"))
    experiment.add_argument("--no-plots", action="store_true")
    experiment.add_argument("--save-arrays", action="store_true")
    _add_common_arguments(experiment)

    scan = subparsers.add_parser("scan", help="Scan ECA rules.")
    scan.add_argument("--start-rule", type=int, default=0)
    scan.add_argument("--end-rule", type=int, default=255)
    scan.add_argument("--repeats", type=int, default=1)
    scan.add_argument("--output", type=Path, default=Path("results/eca_scan.csv"))
    _add_common_arguments(scan)

    summarize = subparsers.add_parser("summarize", help="Aggregate a raw scan CSV.")
    summarize.add_argument("input", type=Path)
    summarize.add_argument("--output", type=Path, default=None)
    summarize.add_argument("--confidence", type=float, default=0.95)

    sweep = subparsers.add_parser("sweep", help="Run a parameter sweep.")
    sweep.add_argument("--rules", type=int, nargs="+", required=True)
    sweep.add_argument("--perturb-times", type=int, nargs="+", default=[80])
    sweep.add_argument("--perturb-radii", type=int, nargs="+", default=[5])
    sweep.add_argument("--initial-densities", type=float, nargs="+", default=[0.5])
    sweep.add_argument("--repeats", type=int, default=1)
    sweep.add_argument("--jobs", type=int, default=1)
    sweep.add_argument("--output", type=Path, default=Path("results/sweep_raw.csv"))
    sweep.add_argument("--summary-output", type=Path, default=Path("results/sweep_summary.csv"))
    _add_common_arguments(sweep)

    evolve = subparsers.add_parser("evolve", help="Search for resilient ECA rules.")
    evolve.add_argument("--population-size", type=int, default=32)
    evolve.add_argument("--generations", type=int, default=20)
    evolve.add_argument("--elite-fraction", type=float, default=0.25)
    evolve.add_argument("--mutation-rate", type=float, default=0.08)
    evolve.add_argument("--output", type=Path, default=Path("results/evolution.csv"))
    _add_common_arguments(evolve)

    args = parser.parse_args()

    if args.command == "experiment":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = run_perturbation_experiment(
            args.rule,
            perturb_center=args.perturb_center,
            **_experiment_kwargs(args),
        )
        if not args.no_plots:
            figures = [
                plot_trajectory(result.control, title=f"Rule {args.rule} control", perturb_time=result.perturb_time, save_path=args.output_dir / "control.png"),
                plot_trajectory(result.perturbed, title=f"Rule {args.rule} perturbed", perturb_time=result.perturb_time, save_path=args.output_dir / "perturbed.png"),
                plot_trajectory(result.difference, title=f"Rule {args.rule} XOR difference", perturb_time=result.perturb_time, save_path=args.output_dir / "difference.png"),
                plot_divergence(result.divergence, perturb_time=result.perturb_time, save_path=args.output_dir / "divergence.png"),
            ]
            for fig, _ in figures:
                plt.close(fig)
        if args.save_arrays:
            np.savez_compressed(
                args.output_dir / "trajectories.npz",
                control=result.control,
                perturbed=result.perturbed,
                difference=result.difference,
                divergence=result.divergence,
            )
        metadata = {
            **result.config.to_dict(),
            "initial_seed": result.initial_seed,
            "perturbation_seed": result.perturbation_seed,
            "software": {
                "ruliology_forge": version("ruliology-forge"),
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            **result.summary.to_dict(),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        print(f"Wrote experiment outputs to {args.output_dir}")

    elif args.command == "scan":
        if args.start_rule > args.end_rule:
            parser.error("--start-rule must be less than or equal to --end-rule")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        kwargs = _experiment_kwargs(args)
        seed = kwargs.pop("seed")
        rows = scan_rules(
            range(args.start_rule, args.end_rule + 1),
            repeats=args.repeats,
            seed=seed,
            **kwargs,
        )
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.output}")

    elif args.command == "summarize":
        with args.input.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["recovered"] = str(row.get("recovered", "")).lower() == "true"
        summary_rows = summarize_scan(rows, confidence=args.confidence)
        output = args.output or args.input.with_name(args.input.stem + "_summary.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(output, summary_rows)
        print(f"Wrote {len(summary_rows)} aggregate rows to {output}")

    elif args.command == "sweep":
        base = _experiment_kwargs(args)
        base.pop("seed", None)
        for key in ("perturb_time", "perturb_radius", "initial_density"):
            base.pop(key, None)
        configs = []
        for values in parameter_grid(
            rule=args.rules,
            perturb_time=args.perturb_times,
            perturb_radius=args.perturb_radii,
            initial_density=args.initial_densities,
        ):
            configs.append(ExperimentConfig(**base, seed=args.seed, **values))
        rows = run_parameter_sweep(configs, repeats=args.repeats, jobs=args.jobs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output, rows)
        summary_rows = summarize_scan(
            rows,
            group_by=("rule", "perturb_time", "perturb_radius", "initial_density"),
        )
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.summary_output, summary_rows)
        print(f"Wrote {len(rows)} raw rows to {args.output}")
        print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")

    elif args.command == "evolve":
        kwargs = _experiment_kwargs(args)
        seed = kwargs.pop("seed")
        history = evolve_resilient_rules(
            population_size=args.population_size,
            generations=args.generations,
            elite_fraction=args.elite_fraction,
            mutation_rate=args.mutation_rate,
            seed=seed,
            experiment_kwargs=kwargs,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output, history)
        print(f"Wrote {len(history)} generations to {args.output}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result set.")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
