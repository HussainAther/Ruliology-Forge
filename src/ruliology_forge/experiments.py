"""Experiment runners for Ruliology Forge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eca import BoundaryMode, InitialCondition, ECARule, initial_state, simulate_eca
from .metrics import final_restoration, hamming_divergence, restoration_coefficient, xor_difference
from .perturb import PerturbationKind, perturb_state


@dataclass(frozen=True)
class PerturbationResult:
    """Container for a control-vs-perturbed experiment."""

    rule: int
    control: np.ndarray
    perturbed: np.ndarray
    difference: np.ndarray
    divergence: np.ndarray
    restoration_coefficient: float
    final_restoration: float
    perturb_time: int
    perturb_radius: int
    perturbation: str


def run_perturbation_experiment(
    rule: int,
    *,
    width: int = 201,
    steps: int = 200,
    perturb_time: int = 80,
    perturb_center: int | None = None,
    perturb_radius: int = 5,
    perturbation: PerturbationKind = "bit_flip",
    initial_condition: InitialCondition = "single",
    boundary: BoundaryMode = "periodic",
    seed: int | None = None,
) -> PerturbationResult:
    """Run a control-vs-perturbed ECA experiment."""

    if not 0 <= perturb_time < steps:
        raise ValueError("perturb_time must be in the simulated time range.")

    init = initial_state(width, initial_condition, seed=seed)
    control = simulate_eca(rule, width, steps, init=init, boundary=boundary)

    perturbed = np.zeros_like(control)
    perturbed[0] = init
    eca_rule = ECARule(rule)

    for t in range(1, steps):
        previous = perturbed[t - 1]
        if t == perturb_time:
            previous = perturb_state(
                previous,
                center=perturb_center,
                radius=perturb_radius,
                kind=perturbation,
                seed=seed,
            )
        perturbed[t] = eca_rule.step(previous, boundary=boundary)

    difference = xor_difference(control, perturbed)
    divergence = hamming_divergence(control, perturbed)
    start = min(perturb_time + 1, steps - 1)

    return PerturbationResult(
        rule=rule,
        control=control,
        perturbed=perturbed,
        difference=difference,
        divergence=divergence,
        restoration_coefficient=restoration_coefficient(control, perturbed, start=start),
        final_restoration=final_restoration(control, perturbed),
        perturb_time=perturb_time,
        perturb_radius=perturb_radius,
        perturbation=perturbation,
    )


def scan_rules(
    rules: range | list[int] = range(256),
    *,
    width: int = 201,
    steps: int = 200,
    perturb_time: int = 80,
    perturb_radius: int = 5,
    perturbation: PerturbationKind = "bit_flip",
) -> list[dict[str, float | int | str]]:
    """Run perturbation experiments across many ECA rules."""

    rows: list[dict[str, float | int | str]] = []
    for rule in rules:
        result = run_perturbation_experiment(
            rule,
            width=width,
            steps=steps,
            perturb_time=perturb_time,
            perturb_radius=perturb_radius,
            perturbation=perturbation,
        )
        rows.append(
            {
                "rule": rule,
                "restoration_coefficient": result.restoration_coefficient,
                "final_restoration": result.final_restoration,
                "perturbation": perturbation,
                "perturb_time": perturb_time,
                "perturb_radius": perturb_radius,
            }
        )
    return rows
