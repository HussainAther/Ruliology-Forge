"""Experiment runners for Ruliology Forge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np

from .eca import BoundaryMode, InitialCondition, ECARule, initial_state, simulate_eca
from .metrics import ResilienceSummary, hamming_divergence, summarize_resilience, xor_difference
from .perturb import PerturbationKind, perturb_state

PerturbationTiming = Literal["before_update", "after_update"]


@dataclass(frozen=True)
class ExperimentConfig:
    rule: int
    width: int = 201
    steps: int = 200
    perturb_time: int = 80
    perturb_center: int | None = None
    perturb_radius: int = 5
    perturbation: PerturbationKind = "bit_flip"
    perturbation_timing: PerturbationTiming = "after_update"
    initial_condition: InitialCondition = "single"
    initial_density: float = 0.5
    boundary: BoundaryMode = "periodic"
    seed: int | None = None
    noise_probability: float = 0.5
    recovery_threshold: float = 0.01
    recovery_persistence: int = 5
    max_shift: int = 10

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationResult:
    rule: int
    control: np.ndarray
    perturbed: np.ndarray
    difference: np.ndarray
    divergence: np.ndarray
    summary: ResilienceSummary
    config: ExperimentConfig
    initial_seed: int
    perturbation_seed: int

    @property
    def restoration_coefficient(self) -> float:
        return self.summary.restoration_coefficient

    @property
    def final_restoration(self) -> float:
        return self.summary.final_restoration

    @property
    def perturb_time(self) -> int:
        return self.config.perturb_time

    @property
    def perturb_center(self) -> int:
        return self.config.width // 2 if self.config.perturb_center is None else self.config.perturb_center

    @property
    def perturb_radius(self) -> int:
        return self.config.perturb_radius

    @property
    def perturbation(self) -> str:
        return self.config.perturbation

    @property
    def seed(self) -> int | None:
        return self.config.seed


def run_perturbation_experiment(
    rule: int,
    *,
    width: int = 201,
    steps: int = 200,
    perturb_time: int = 80,
    perturb_center: int | None = None,
    perturb_radius: int = 5,
    perturbation: PerturbationKind = "bit_flip",
    perturbation_timing: PerturbationTiming = "after_update",
    initial_condition: InitialCondition = "single",
    initial_density: float = 0.5,
    boundary: BoundaryMode = "periodic",
    seed: int | None = None,
    noise_probability: float = 0.5,
    recovery_threshold: float = 0.01,
    recovery_persistence: int = 5,
    max_shift: int = 10,
) -> PerturbationResult:
    """Run a control-vs-perturbed ECA experiment."""

    config = ExperimentConfig(
        rule=rule,
        width=width,
        steps=steps,
        perturb_time=perturb_time,
        perturb_center=perturb_center,
        perturb_radius=perturb_radius,
        perturbation=perturbation,
        perturbation_timing=perturbation_timing,
        initial_condition=initial_condition,
        initial_density=initial_density,
        boundary=boundary,
        seed=seed,
        noise_probability=noise_probability,
        recovery_threshold=recovery_threshold,
        recovery_persistence=recovery_persistence,
        max_shift=max_shift,
    )
    _validate_config(config)
    center = width // 2 if perturb_center is None else perturb_center

    root = np.random.SeedSequence(seed)
    initial_child, perturbation_child = root.spawn(2)
    initial_seed = int(initial_child.generate_state(1)[0])
    perturbation_seed = int(perturbation_child.generate_state(1)[0])

    init = initial_state(width, initial_condition, density=initial_density, seed=initial_seed)
    control = simulate_eca(rule, width, steps, init=init, boundary=boundary)
    perturbed = np.zeros_like(control)
    perturbed[0] = init
    eca_rule = ECARule(rule)

    for t in range(1, steps):
        previous = perturbed[t - 1]
        if perturbation_timing == "before_update" and t == perturb_time:
            previous = perturb_state(
                previous,
                center=center,
                radius=perturb_radius,
                kind=perturbation,
                seed=perturbation_seed,
                noise_probability=noise_probability,
            )
        current = eca_rule.step(previous, boundary=boundary)
        if perturbation_timing == "after_update" and t == perturb_time:
            current = perturb_state(
                current,
                center=center,
                radius=perturb_radius,
                kind=perturbation,
                seed=perturbation_seed,
                noise_probability=noise_probability,
            )
        perturbed[t] = current

    difference = xor_difference(control, perturbed)
    divergence = hamming_divergence(control, perturbed)
    summary = summarize_resilience(
        control,
        perturbed,
        start=perturb_time,
        recovery_threshold=recovery_threshold,
        recovery_persistence=recovery_persistence,
        max_shift=max_shift,
        boundary=boundary,
    )
    return PerturbationResult(
        rule=rule,
        control=control,
        perturbed=perturbed,
        difference=difference,
        divergence=divergence,
        summary=summary,
        config=config,
        initial_seed=initial_seed,
        perturbation_seed=perturbation_seed,
    )


def scan_rules(
    rules: Iterable[int] = range(256),
    *,
    repeats: int = 1,
    **experiment_kwargs: object,
) -> list[dict[str, float | int | str | bool | None]]:
    """Run rule/repeat experiments while preserving raw observations."""

    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    base_seed = experiment_kwargs.pop("seed", None)
    rules_list = list(rules)
    root = np.random.SeedSequence(base_seed)
    children = root.spawn(len(rules_list) * repeats)
    rows: list[dict[str, float | int | str | bool | None]] = []
    index = 0
    for rule in rules_list:
        for repeat in range(repeats):
            child_seed = int(children[index].generate_state(1)[0])
            index += 1
            result = run_perturbation_experiment(
                rule,
                seed=child_seed,
                **experiment_kwargs,
            )
            rows.append(
                {
                    "rule": rule,
                    "repeat": repeat,
                    **result.summary.to_dict(),
                    **{k: v for k, v in result.config.to_dict().items() if k != "rule"},
                    "initial_seed": result.initial_seed,
                    "perturbation_seed": result.perturbation_seed,
                }
            )
    return rows


def _validate_config(config: ExperimentConfig) -> None:
    if config.width <= 0:
        raise ValueError("width must be positive.")
    if config.steps <= 0:
        raise ValueError("steps must be positive.")
    if not 1 <= config.perturb_time < config.steps:
        raise ValueError("perturb_time must be between 1 and steps - 1.")
    if config.perturb_radius < 0:
        raise ValueError("perturb_radius must be non-negative.")
    if config.perturb_center is not None and not 0 <= config.perturb_center < config.width:
        raise ValueError("perturb_center must be within the lattice.")
    if config.perturbation_timing not in {"before_update", "after_update"}:
        raise ValueError("invalid perturbation_timing.")
    if not 0 <= config.initial_density <= 1:
        raise ValueError("initial_density must be between 0 and 1.")
    if not 0 <= config.noise_probability <= 1:
        raise ValueError("noise_probability must be between 0 and 1.")
    if not 0 <= config.recovery_threshold <= 1:
        raise ValueError("recovery_threshold must be between 0 and 1.")
    if config.recovery_persistence <= 0:
        raise ValueError("recovery_persistence must be positive.")
    if config.max_shift < 0:
        raise ValueError("max_shift must be non-negative.")
