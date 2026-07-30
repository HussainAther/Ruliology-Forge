"""Command-line interface for Ruliology Forge."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analysis import (
    AtlasAnalysisConfig,
    analyze_atlas_directory,
    parameter_grid,
    summarize_scan,
)
from .atlas import AtlasConfig, atlas_markdown, build_resilience_atlas
from .benchmarks import benchmark_markdown, benchmark_rules, standard_suite, timing_suite
from .experiments import (
    ExperimentConfig,
    evolve_resilient_rules,
    run_parameter_sweep,
    run_perturbation_experiment,
    scan_rules,
)
from .plotting import plot_divergence, plot_trajectory
from .validation import ValidationConfig, validate_resilience


def _package_version() -> str:
    try:
        return version("ruliology-forge")
    except PackageNotFoundError:
        return "0.6.0"


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
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
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


    analyze = subparsers.add_parser(
        "analyze", help="Analyze an existing resilience atlas and generate figures."
    )
    analyze.add_argument("--atlas-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, default=Path("results/resilience_analysis"))
    analyze.add_argument("--top", type=int, default=20)
    analyze.add_argument("--trivial-threshold", type=float, default=0.5)
    analyze.add_argument("--activity-floor", type=float, default=0.005)
    analyze.add_argument("--entropy-floor", type=float, default=0.05)
    analyze.add_argument("--anomaly-z", type=float, default=1.5)
    analyze.add_argument("--figure-dpi", type=int, default=220)

    validate = subparsers.add_parser(
        "validate-atlas", help="Validate resilience findings across experimental conditions."
    )
    validate.add_argument("--rules", type=int, nargs="+", default=list(range(256)))
    validate.add_argument("--widths", type=int, nargs="+", default=[101, 201, 401])
    validate.add_argument("--steps-values", type=int, nargs="+", default=[100, 200, 400])
    validate.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    validate.add_argument("--perturb-positions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    validate.add_argument("--boundary-modes", choices=["periodic", "fixed"], nargs="+", default=["periodic", "fixed"])
    validate.add_argument("--perturb-time-fraction", type=float, default=0.4)
    validate.add_argument("--perturb-radius", type=int, default=5)
    validate.add_argument("--perturbation", choices=["bit_flip", "void", "random_mix"], default="bit_flip")
    validate.add_argument("--initial-condition", choices=["single", "random"], default="single")
    validate.add_argument("--initial-density", type=float, default=0.5)
    validate.add_argument("--jobs", type=int, default=1)
    validate.add_argument("--recovery-threshold", type=float, default=0.01)
    validate.add_argument("--recovery-persistence", type=int, default=5)
    validate.add_argument("--max-shift", type=int, default=10)
    validate.add_argument("--classification-agreement", type=float, default=0.8)
    validate.add_argument("--rank-stability", type=float, default=0.75)
    validate.add_argument("--anomaly-reproduction", type=float, default=0.7)
    validate.add_argument("--figure-dpi", type=int, default=220)
    validate.add_argument("--output-dir", type=Path, default=Path("results/atlas_validation"))

    atlas = subparsers.add_parser(
        "atlas", help="Build a feature-enriched ECA resilience atlas."
    )
    atlas.add_argument("--rules", type=int, nargs="+", default=list(range(256)))
    atlas.add_argument("--suite", choices=["standard", "timing"], default="standard")
    atlas.add_argument("--timing-interval", type=int, default=10)
    atlas.add_argument("--repeats", type=int, default=20)
    atlas.add_argument("--jobs", type=int, default=1)
    atlas.add_argument("--output-dir", type=Path, default=Path("results/resilience_atlas"))
    atlas.add_argument("--top", type=int, default=20)
    _add_common_arguments(atlas)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark and rank ECA rules across a standard resilience suite.")
    benchmark.add_argument("--rules", type=int, nargs="+", default=list(range(256)))
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--jobs", type=int, default=1)
    benchmark.add_argument("--output-dir", type=Path, default=Path("results/benchmark"))
    benchmark.add_argument("--top", type=int, default=20)
    _add_common_arguments(benchmark)

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
                "ruliology_forge": _package_version(),
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


    elif args.command == "analyze":
        result = analyze_atlas_directory(
            args.atlas_dir,
            args.output_dir,
            config=AtlasAnalysisConfig(
                top_n=args.top,
                trivial_threshold=args.trivial_threshold,
                nontrivial_activity_floor=args.activity_floor,
                nontrivial_entropy_floor=args.entropy_floor,
                anomaly_z_threshold=args.anomaly_z,
                figure_dpi=args.figure_dpi,
            ),
        )
        print(
            f"Analyzed {result.manifest['source_rule_count']} rules and wrote outputs "
            f"to {args.output_dir}"
        )

    elif args.command == "validate-atlas":
        result = validate_resilience(
            ValidationConfig(
                rules=tuple(args.rules),
                widths=tuple(args.widths),
                steps_values=tuple(args.steps_values),
                seeds=tuple(args.seeds),
                perturb_positions=tuple(args.perturb_positions),
                boundaries=tuple(args.boundary_modes),
                perturb_time_fraction=args.perturb_time_fraction,
                perturb_radius=args.perturb_radius,
                perturbation=args.perturbation,
                initial_condition=args.initial_condition,
                initial_density=args.initial_density,
                jobs=args.jobs,
                recovery_threshold=args.recovery_threshold,
                recovery_persistence=args.recovery_persistence,
                max_shift=args.max_shift,
                classification_agreement_threshold=args.classification_agreement,
                rank_stability_threshold=args.rank_stability,
                anomaly_reproduction_threshold=args.anomaly_reproduction,
                figure_dpi=args.figure_dpi,
            ),
            args.output_dir,
        )
        print(
            f"Validated {result.manifest['rule_count']} rules across "
            f"{result.manifest['condition_count']} conditions; outputs: {args.output_dir}"
        )

    elif args.command == "atlas":
        if args.top <= 0:
            parser.error("--top must be positive")
        suite = (
            standard_suite(steps=args.steps)
            if args.suite == "standard"
            else timing_suite(
                steps=args.steps,
                interval=args.timing_interval,
                perturb_radius=args.perturb_radius,
            )
        )
        result = build_resilience_atlas(
            AtlasConfig(
                rules=tuple(args.rules),
                repeats=args.repeats,
                width=args.width,
                steps=args.steps,
                seed=args.seed,
                jobs=args.jobs,
                recovery_threshold=args.recovery_threshold,
                recovery_persistence=args.recovery_persistence,
                max_shift=args.max_shift,
            ),
            suite=suite,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output_dir / "atlas_observations.csv", result.observations)
        _write_csv(args.output_dir / "atlas_rule_profiles.csv", result.rule_profiles)
        _write_csv(args.output_dir / "atlas_correlations.csv", result.correlations)
        (args.output_dir / "atlas_manifest.json").write_text(
            json.dumps(result.manifest, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "atlas_report.md").write_text(
            atlas_markdown(result, top_n=args.top), encoding="utf-8"
        )
        print(f"Wrote resilience atlas outputs to {args.output_dir}")

    elif args.command == "benchmark":
        if args.top <= 0:
            parser.error("--top must be positive")
        suite = standard_suite(steps=args.steps)
        raw_rows, ranking_rows = benchmark_rules(
            args.rules,
            suite=suite,
            repeats=args.repeats,
            jobs=args.jobs,
            width=args.width,
            steps=args.steps,
            seed=args.seed,
            recovery_threshold=args.recovery_threshold,
            recovery_persistence=args.recovery_persistence,
            max_shift=args.max_shift,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output_dir / "benchmark_raw.csv", raw_rows)
        _write_csv(args.output_dir / "benchmark_ranking.csv", ranking_rows)
        (args.output_dir / "benchmark_report.md").write_text(
            benchmark_markdown(ranking_rows, suite=suite, top_n=args.top),
            encoding="utf-8",
        )
        manifest = {
            "suite": suite.name,
            "suite_fingerprint": suite.fingerprint(),
            "rules": args.rules,
            "repeats": args.repeats,
            "width": args.width,
            "steps": args.steps,
            "seed": args.seed,
            "scenario_count": len(suite.scenarios),
            "observation_count": len(raw_rows),
        }
        (args.output_dir / "benchmark_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote benchmark outputs to {args.output_dir}")

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
