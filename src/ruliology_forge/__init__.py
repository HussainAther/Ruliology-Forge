"""Ruliology Forge: computational resilience tools for rule-based systems."""

from .eca import ECARule, initial_state, simulate_eca
from .experiments import PerturbationResult, run_perturbation_experiment, scan_rules
from .metrics import final_restoration, hamming_divergence, restoration_coefficient, xor_difference
from .perturb import perturb_state
from .plotting import plot_divergence, plot_trajectory

__all__ = [
    "ECARule",
    "PerturbationResult",
    "final_restoration",
    "hamming_divergence",
    "initial_state",
    "perturb_state",
    "plot_divergence",
    "plot_trajectory",
    "restoration_coefficient",
    "run_perturbation_experiment",
    "scan_rules",
    "simulate_eca",
    "xor_difference",
]
