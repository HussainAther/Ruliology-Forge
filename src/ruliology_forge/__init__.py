"""Ruliology Forge: computational resilience tools for rule-based systems."""

from .benchmarks import BenchmarkScenario, BenchmarkSuite, benchmark_rules, rank_benchmark_rows, standard_suite
from .eca import ECARule, initial_state, simulate_eca
from .experiments import ExperimentConfig, PerturbationResult, run_perturbation_experiment, scan_rules
from .metrics import (
    ResilienceSummary,
    divergence_auc,
    final_restoration,
    hamming_divergence,
    recovery_time,
    restoration_coefficient,
    scar_metrics,
    shift_tolerant_divergence,
    summarize_resilience,
    xor_difference,
)
from .perturb import perturb_state
from .plotting import plot_divergence, plot_trajectory

__all__ = [
    "BenchmarkScenario",
    "BenchmarkSuite",
    "ECARule",
    "ExperimentConfig",
    "PerturbationResult",
    "ResilienceSummary",
    "benchmark_rules",
    "divergence_auc",
    "final_restoration",
    "hamming_divergence",
    "initial_state",
    "perturb_state",
    "plot_divergence",
    "plot_trajectory",
    "rank_benchmark_rows",
    "recovery_time",
    "restoration_coefficient",
    "run_perturbation_experiment",
    "scan_rules",
    "scar_metrics",
    "shift_tolerant_divergence",
    "simulate_eca",
    "standard_suite",
    "summarize_resilience",
    "xor_difference",
]
