"""Benchmark suites and robust rule ranking for Ruliology Forge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable, Mapping, Sequence

import numpy as np

from .experiments import ExperimentConfig, run_parameter_sweep


@dataclass(frozen=True)
class BenchmarkScenario:
    """One named perturbation condition within a benchmark suite."""

    name: str
    perturbation: str = "bit_flip"
    perturb_time: int = 80
    perturb_radius: int = 5
    initial_condition: str = "single"
    initial_density: float = 0.5
    noise_probability: float = 0.5
    boundary: str = "periodic"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSuite:
    """A reproducible collection of perturbation scenarios."""

    name: str
    scenarios: tuple[BenchmarkScenario, ...]
    description: str = ""

    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "description": self.description,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()[:16]


def standard_suite(*, steps: int = 200) -> BenchmarkSuite:
    """Return a balanced starter suite for ECA resilience comparisons."""

    early = max(2, int(round(steps * 0.25)))
    middle = max(2, int(round(steps * 0.50)))
    late = min(steps - 1, max(2, int(round(steps * 0.75))))
    return BenchmarkSuite(
        name="eca-resilience-standard-v1",
        description=(
            "A compact suite spanning local flips, voids, stochastic damage, "
            "multiple injury times, random initial states, and fixed boundaries."
        ),
        scenarios=(
            BenchmarkScenario("local-flip-early", perturb_time=early, perturb_radius=1),
            BenchmarkScenario("local-flip-middle", perturb_time=middle, perturb_radius=5),
            BenchmarkScenario("wide-flip-late", perturb_time=late, perturb_radius=12),
            BenchmarkScenario(
                "local-void-middle",
                perturbation="void",
                perturb_time=middle,
                perturb_radius=5,
            ),
            BenchmarkScenario(
                "stochastic-noise-middle",
                perturbation="random_mix",
                perturb_time=middle,
                perturb_radius=12,
                noise_probability=0.35,
            ),
            BenchmarkScenario(
                "random-state-low-density",
                perturb_time=middle,
                perturb_radius=5,
                initial_condition="random",
                initial_density=0.25,
            ),
            BenchmarkScenario(
                "random-state-high-density",
                perturb_time=middle,
                perturb_radius=5,
                initial_condition="random",
                initial_density=0.75,
            ),
            BenchmarkScenario(
                "fixed-boundary-middle",
                perturb_time=middle,
                perturb_radius=5,
                boundary="fixed",
            ),
        ),
    )


def benchmark_rules(
    rules: Iterable[int],
    *,
    suite: BenchmarkSuite | None = None,
    repeats: int = 5,
    jobs: int = 1,
    width: int = 201,
    steps: int = 200,
    seed: int | None = None,
    recovery_threshold: float = 0.01,
    recovery_persistence: int = 5,
    max_shift: int = 10,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run a suite and return raw observations plus robust rule rankings."""

    suite = suite or standard_suite(steps=steps)
    rules_list = [int(rule) for rule in rules]
    if not rules_list:
        raise ValueError("at least one rule is required.")
    if any(rule < 0 or rule > 255 for rule in rules_list):
        raise ValueError("ECA rules must be between 0 and 255.")

    root = np.random.SeedSequence(seed)
    scenario_seeds = root.spawn(len(suite.scenarios))
    configs: list[ExperimentConfig] = []
    for scenario, scenario_seed in zip(suite.scenarios, scenario_seeds, strict=True):
        child_seed = int(scenario_seed.generate_state(1)[0])
        for rule in rules_list:
            configs.append(
                ExperimentConfig(
                    rule=rule,
                    width=width,
                    steps=steps,
                    perturb_time=scenario.perturb_time,
                    perturb_radius=scenario.perturb_radius,
                    perturbation=scenario.perturbation,  # type: ignore[arg-type]
                    initial_condition=scenario.initial_condition,  # type: ignore[arg-type]
                    initial_density=scenario.initial_density,
                    noise_probability=scenario.noise_probability,
                    boundary=scenario.boundary,  # type: ignore[arg-type]
                    seed=child_seed,
                    recovery_threshold=recovery_threshold,
                    recovery_persistence=recovery_persistence,
                    max_shift=max_shift,
                )
            )

    rows = run_parameter_sweep(configs, repeats=repeats, jobs=jobs)
    scenario_lookup = {
        (
            scenario.perturbation,
            scenario.perturb_time,
            scenario.perturb_radius,
            scenario.initial_condition,
            scenario.initial_density,
            scenario.noise_probability,
            scenario.boundary,
        ): scenario.name
        for scenario in suite.scenarios
    }
    for row in rows:
        key = (
            row["perturbation"],
            row["perturb_time"],
            row["perturb_radius"],
            row["initial_condition"],
            row["initial_density"],
            row["noise_probability"],
            row["boundary"],
        )
        row["scenario"] = scenario_lookup[key]
        row["suite"] = suite.name
        row["suite_fingerprint"] = suite.fingerprint()

    ranking = rank_benchmark_rows(rows)
    return rows, ranking


def rank_benchmark_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Rank rules using performance, consistency, recovery, and worst-case behavior.

    The score is intentionally transparent and bounded near [0, 1]:

    45% mean restoration, 20% worst-scenario restoration,
    15% recovery probability, 10% shift-tolerant restoration,
    10% consistency (one minus standard deviation).
    """

    if not rows:
        return []

    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (int(row["rule"]), str(row["scenario"]))
        grouped.setdefault(key, []).append(row)

    by_rule: dict[int, list[dict[str, float]]] = {}
    for (rule, _scenario), members in grouped.items():
        restoration = np.asarray(
            [float(member["restoration_coefficient"]) for member in members], dtype=float
        )
        recovery_probability = float(
            sum(bool(member.get("recovered")) for member in members) / len(members)
        )
        shift_restoration = float(
            np.mean([float(member["shift_tolerant_restoration"]) for member in members])
        )
        by_rule.setdefault(rule, []).append(
            {
                "mean_restoration": float(restoration.mean()),
                "recovery_probability": recovery_probability,
                "shift_restoration": shift_restoration,
            }
        )

    rankings: list[dict[str, object]] = []
    for rule, members in by_rule.items():
        mean_restorations = np.asarray(
            [member["mean_restoration"] for member in members], dtype=float
        )
        recovery_probabilities = np.asarray(
            [member["recovery_probability"] for member in members], dtype=float
        )
        shift_values = np.asarray(
            [member["shift_restoration"] for member in members], dtype=float
        )
        consistency = max(0.0, 1.0 - float(mean_restorations.std(ddof=0)))
        score = (
            0.45 * float(mean_restorations.mean())
            + 0.20 * float(mean_restorations.min())
            + 0.15 * float(recovery_probabilities.mean())
            + 0.10 * float(shift_values.mean())
            + 0.10 * consistency
        )
        raw_rule_rows = [row for row in rows if int(row["rule"]) == rule]
        rankings.append(
            {
                "rule": rule,
                "robustness_score": float(score),
                "mean_restoration": float(mean_restorations.mean()),
                "worst_scenario_restoration": float(mean_restorations.min()),
                "best_scenario_restoration": float(mean_restorations.max()),
                "recovery_probability": float(recovery_probabilities.mean()),
                "mean_shift_tolerant_restoration": float(shift_values.mean()),
                "scenario_consistency": consistency,
                "scenario_count": len(members),
                "observation_count": len(raw_rule_rows),
            }
        )
    rankings.sort(key=lambda row: (-float(row["robustness_score"]), int(row["rule"])))
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank
    return rankings


def benchmark_markdown(
    ranking: Sequence[Mapping[str, object]],
    *,
    suite: BenchmarkSuite,
    top_n: int = 20,
) -> str:
    """Render a compact, publication-friendly Markdown benchmark report."""

    top = list(ranking[:top_n])
    lines = [
        f"# Ruliology Forge Benchmark: {suite.name}",
        "",
        suite.description,
        "",
        f"Suite fingerprint: `{suite.fingerprint()}`",
        "",
        "## Ranking",
        "",
        "| Rank | Rule | Robustness | Mean restoration | Worst scenario | Recovery probability | Consistency |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            "| {rank} | {rule} | {robustness_score:.4f} | {mean_restoration:.4f} | "
            "{worst_scenario_restoration:.4f} | {recovery_probability:.4f} | "
            "{scenario_consistency:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Score definition",
            "",
            "The robustness score combines mean restoration (45%), worst-scenario "
            "restoration (20%), recovery probability (15%), shift-tolerant restoration "
            "(10%), and cross-scenario consistency (10%). Rankings are exploratory and "
            "should be validated with larger independent runs.",
            "",
        ]
    )
    return "\n".join(lines)
