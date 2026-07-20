"""Ruliology Forge: computational resilience tools for rule-based systems."""

from .eca import ECARule, initial_state, simulate_eca
from .experiments import PerturbationResult, run_perturbation_experiment, scan_rules
from .metrics import (
    ResilienceSummary,
    divergence_auc,
    final_restoration,
    hamming_divergence,
    recovery_time,
    restoration_coefficient,
    summarize_resilience,
    xor_difference,
)
from .perturb import perturb_state
from .plotting import plot_divergence, plot_trajectory

__all__ = [
    "ECARule",
    "PerturbationResult",
    "ResilienceSummary",
    "divergence_auc",
    "final_restoration",
    "hamming_divergence",
    "initial_state",
    "perturb_state",
    "plot_divergence",
    "plot_trajectory",
    "recovery_time",
    "restoration_coefficient",
    "run_perturbation_experiment",
    "scan_rules",
    "simulate_eca",
    "summarize_resilience",
    "xor_difference",
]
