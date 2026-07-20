"""Experiment runners for Ruliology Forge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .eca import BoundaryMode, InitialCondition, ECARule, initial_state, simulate_eca
from .metrics import ResilienceSummary, hamming_divergence, summarize_resilience, xor_difference
from .perturb import PerturbationKind, perturb_state


@dataclass(frozen=True)
class PerturbationResult:
    """Container for a control-vs-perturbed experiment."""

    rule: int
    control: np.ndarray
    perturbed: np.ndarray
    difference: np.ndarray
    divergence: np.ndarray
    summary: ResilienceSummary
    perturb_time: int
    perturb_center: int
    perturb_radius: int
    perturbation: str
    seed: int | None

    @property
    def restoration_coefficient(self) -> float:
        return self.summary.restoration_coefficient

    @property
    def final_restoration(self) -> float:
        return self.summary.final_restoration


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
    recovery_threshold: float = 0.01,
    recovery_persistence: int = 5,
) -> PerturbationResult:
    """Run a control-vs-perturbed ECA experiment."""

    if width <= 0:
        raise ValueError("width must be positive.")
    if not 0 <= perturb_time < steps:
        raise ValueError("perturb_time must be in the simulated time range.")

    center = width // 2 if perturb_center is None else perturb_center
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
                center=center,
                radius=perturb_radius,
                kind=perturbation,
                seed=seed,
            )
        perturbed[t] = eca_rule.step(previous, boundary=boundary)

    difference = xor_difference(control, perturbed)
    divergence = hamming_divergence(control, perturbed)
    start = min(perturb_time, steps - 1)
    summary = summarize_resilience(
        control,
        perturbed,
        start=start,
        recovery_threshold=recovery_threshold,
        recovery_persistence=recovery_persistence,
    )

    return PerturbationResult(
        rule=rule,
        control=control,
        perturbed=perturbed,
        difference=difference,
        divergence=divergence,
        summary=summary,
        perturb_time=perturb_time,
        perturb_center=center,
        perturb_radius=perturb_radius,
        perturbation=perturbation,
        seed=seed,
    )


def scan_rules(
    rules: Iterable[int] = range(256),
    *,
    width: int = 201,
    steps: int = 200,
    perturb_time: int = 80,
    perturb_radius: int = 5,
    perturbation: PerturbationKind = "bit_flip",
    initial_condition: InitialCondition = "single",
    boundary: BoundaryMode = "periodic",
    seed: int | None = None,
    repeats: int = 1,
    recovery_threshold: float = 0.01,
    recovery_persistence: int = 5,
) -> list[dict[str, float | int | str | bool | None]]:
    """Run perturbation experiments across ECA rules and optional repeats.

    Random experiments receive deterministic child seeds derived from ``seed``.
    Each output row represents one rule/repeat pair, preserving raw observations
    for later statistical analysis rather than silently averaging them.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    rows: list[dict[str, float | int | str | bool | None]] = []
    seed_sequence = np.random.SeedSequence(seed)
    rules_list = list(rules)
    child_seeds = seed_sequence.spawn(len(rules_list) * repeats)
    seed_index = 0

    for rule in rules_list:
        for repeat in range(repeats):
            child_seed = int(child_seeds[seed_index].generate_state(1)[0])
            seed_index += 1
            result = run_perturbation_experiment(
                rule,
                width=width,
                steps=steps,
                perturb_time=perturb_time,
                perturb_radius=perturb_radius,
                perturbation=perturbation,
                initial_condition=initial_condition,
                boundary=boundary,
                seed=child_seed,
                recovery_threshold=recovery_threshold,
                recovery_persistence=recovery_persistence,
            )
            rows.append(
                {
                    "rule": rule,
                    "repeat": repeat,
                    **result.summary.to_dict(),
                    "perturbation": perturbation,
                    "perturb_time": perturb_time,
                    "perturb_radius": perturb_radius,
                    "initial_condition": initial_condition,
                    "boundary": boundary,
                    "seed": child_seed,
                }
            )
    return rows
