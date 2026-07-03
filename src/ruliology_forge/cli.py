"""Command-line interface for Ruliology Forge."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .experiments import run_perturbation_experiment, scan_rules
from .plotting import plot_divergence, plot_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(prog="ruliology")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment = subparsers.add_parser("experiment", help="Run one perturbation experiment.")
    experiment.add_argument("--rule", type=int, default=110)
    experiment.add_argument("--width", type=int, default=201)
    experiment.add_argument("--steps", type=int, default=200)
    experiment.add_argument("--perturb-time", type=int, default=80)
    experiment.add_argument("--perturb-radius", type=int, default=5)
    experiment.add_argument("--perturbation", choices=["bit_flip", "void", "random_mix"], default="bit_flip")
    experiment.add_argument("--output-dir", type=Path, default=Path("results/experiment"))

    scan = subparsers.add_parser("scan", help="Scan ECA rules.")
    scan.add_argument("--start-rule", type=int, default=0)
    scan.add_argument("--end-rule", type=int, default=255)
    scan.add_argument("--width", type=int, default=201)
    scan.add_argument("--steps", type=int, default=200)
    scan.add_argument("--perturb-time", type=int, default=80)
    scan.add_argument("--perturb-radius", type=int, default=5)
    scan.add_argument("--output", type=Path, default=Path("results/eca_scan.csv"))

    args = parser.parse_args()

    if args.command == "experiment":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = run_perturbation_experiment(
            args.rule,
            width=args.width,
            steps=args.steps,
            perturb_time=args.perturb_time,
            perturb_radius=args.perturb_radius,
            perturbation=args.perturbation,
        )
        plot_trajectory(result.control, title=f"Rule {args.rule} control", save_path=args.output_dir / "control.png")
        plot_trajectory(result.perturbed, title=f"Rule {args.rule} perturbed", save_path=args.output_dir / "perturbed.png")
        plot_trajectory(result.difference, title=f"Rule {args.rule} XOR difference", save_path=args.output_dir / "difference.png")
        plot_divergence(result.divergence, save_path=args.output_dir / "divergence.png")
        print(f"R={result.restoration_coefficient:.4f}")
        print(f"Final restoration={result.final_restoration:.4f}")

    elif args.command == "scan":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = scan_rules(
            list(range(args.start_rule, args.end_rule + 1)),
            width=args.width,
            steps=args.steps,
            perturb_time=args.perturb_time,
            perturb_radius=args.perturb_radius,
        )
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.output}")
