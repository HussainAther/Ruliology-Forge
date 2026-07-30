"""Resilience-atlas construction and analysis.

The atlas joins descriptive features of an unperturbed control trajectory with
scenario-level perturbation outcomes.  It intentionally keeps raw observations
separate from per-rule profiles so later papers can revisit aggregation choices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np

from .benchmarks import BenchmarkScenario, BenchmarkSuite, standard_suite
from .experiments import ExperimentConfig, run_parameter_sweep
from .features import FEATURE_NAMES


@dataclass(frozen=True)
class AtlasConfig:
    """Configuration for a reproducible ECA resilience atlas."""

    rules: tuple[int, ...] = tuple(range(256))
    repeats: int = 20
    width: int = 201
    steps: int = 200
    seed: int | None = 42
    jobs: int = 1
    recovery_threshold: float = 0.01
    recovery_persistence: int = 5
    max_shift: int = 10

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AtlasResult:
    """Raw observations, rule profiles, correlations, and provenance."""

    observations: list[dict[str, object]]
    rule_profiles: list[dict[str, object]]
    correlations: list[dict[str, object]]
    manifest: dict[str, object]


def build_resilience_atlas(
    config: AtlasConfig,
    *,
    suite: BenchmarkSuite | None = None,
) -> AtlasResult:
    """Build a feature-enriched resilience atlas for the requested ECA rules."""

    _validate_atlas_config(config)
    suite = suite or standard_suite(steps=config.steps)
    _validate_suite(suite, steps=config.steps)

    root = np.random.SeedSequence(config.seed)
    scenario_seeds = root.spawn(len(suite.scenarios))
    configurations: list[ExperimentConfig] = []
    for scenario, scenario_seed in zip(suite.scenarios, scenario_seeds, strict=True):
        scenario_root_seed = int(scenario_seed.generate_state(1)[0])
        for rule in config.rules:
            configurations.append(
                ExperimentConfig(
                    rule=rule,
                    width=config.width,
                    steps=config.steps,
                    perturb_time=scenario.perturb_time,
                    perturb_radius=scenario.perturb_radius,
                    perturbation=scenario.perturbation,  # type: ignore[arg-type]
                    initial_condition=scenario.initial_condition,  # type: ignore[arg-type]
                    initial_density=scenario.initial_density,
                    noise_probability=scenario.noise_probability,
                    boundary=scenario.boundary,  # type: ignore[arg-type]
                    seed=scenario_root_seed,
                    recovery_threshold=config.recovery_threshold,
                    recovery_persistence=config.recovery_persistence,
                    max_shift=config.max_shift,
                )
            )

    observations = run_parameter_sweep(
        configurations,
        repeats=config.repeats,
        jobs=config.jobs,
        include_features=True,
    )
    scenario_lookup = {_scenario_key(scenario): scenario.name for scenario in suite.scenarios}
    fingerprint = suite.fingerprint()
    for row in observations:
        row["scenario"] = scenario_lookup[_row_scenario_key(row)]
        row["suite"] = suite.name
        row["suite_fingerprint"] = fingerprint

    observations.sort(
        key=lambda row: (int(row["rule"]), str(row["scenario"]), int(row["repeat"]))
    )
    profiles = aggregate_rule_profiles(observations)
    correlations = feature_outcome_correlations(observations)
    manifest = {
        "schema_version": "resilience-atlas-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite.name,
        "suite_fingerprint": fingerprint,
        "suite_description": suite.description,
        "scenarios": [scenario.to_dict() for scenario in suite.scenarios],
        "config": config.to_dict(),
        "observation_count": len(observations),
        "rule_profile_count": len(profiles),
        "feature_names": [f"control_{name}" for name in FEATURE_NAMES],
    }
    return AtlasResult(observations, profiles, correlations, manifest)


def aggregate_rule_profiles(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate scenario-level observations into one transparent profile per rule."""

    if not rows:
        return []
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["rule"]), []).append(row)

    profiles: list[dict[str, object]] = []
    for rule in sorted(grouped):
        members = grouped[rule]
        restoration = _numeric(members, "restoration_coefficient")
        shift_restoration = _numeric(members, "shift_tolerant_restoration")
        scar_duration = _numeric(members, "scar_duration")
        final_scar = _numeric(members, "final_scar_size")
        peak_divergence = _numeric(members, "peak_divergence")
        recovery_times = np.asarray(
            [float(row["recovery_time"]) for row in members if row.get("recovery_time") is not None],
            dtype=float,
        )
        scenarios: dict[str, list[float]] = {}
        for row in members:
            scenarios.setdefault(str(row["scenario"]), []).append(
                float(row["restoration_coefficient"])
            )
        scenario_means = np.asarray(
            [np.mean(values) for values in scenarios.values()], dtype=float
        )
        profile: dict[str, object] = {
            "rule": rule,
            "observation_count": len(members),
            "scenario_count": len(scenarios),
            "mean_restoration": float(restoration.mean()),
            "std_restoration": float(restoration.std(ddof=1)) if restoration.size > 1 else 0.0,
            "median_restoration": float(np.median(restoration)),
            "worst_scenario_restoration": float(scenario_means.min()),
            "best_scenario_restoration": float(scenario_means.max()),
            "recovery_probability": float(np.mean([bool(row.get("recovered")) for row in members])),
            "mean_recovery_time": float(recovery_times.mean()) if recovery_times.size else None,
            "mean_shift_tolerant_restoration": float(shift_restoration.mean()),
            "mean_peak_divergence": float(peak_divergence.mean()),
            "mean_scar_duration": float(scar_duration.mean()),
            "mean_final_scar_size": float(final_scar.mean()),
            "trivial_convergence_probability": float(
                np.mean([bool(row.get("control_trivial_convergence")) for row in members])
            ),
        }
        for feature in FEATURE_NAMES:
            values = _numeric(members, f"control_{feature}")
            profile[f"mean_control_{feature}"] = float(values.mean())
            profile[f"std_control_{feature}"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        profiles.append(profile)
    return profiles


def feature_outcome_correlations(
    rows: Sequence[Mapping[str, object]],
    *,
    outcomes: Sequence[str] = (
        "restoration_coefficient",
        "shift_tolerant_restoration",
        "peak_divergence",
        "scar_duration",
        "final_scar_size",
    ),
) -> list[dict[str, object]]:
    """Calculate Pearson and Spearman correlations for control features and outcomes."""

    if not rows:
        return []
    output: list[dict[str, object]] = []
    for feature in FEATURE_NAMES:
        feature_name = f"control_{feature}"
        for outcome in outcomes:
            pairs = [
                (float(row[feature_name]), float(row[outcome]))
                for row in rows
                if row.get(feature_name) is not None and row.get(outcome) is not None
            ]
            x = np.asarray([pair[0] for pair in pairs], dtype=float)
            y = np.asarray([pair[1] for pair in pairs], dtype=float)
            pearson = _safe_correlation(x, y)
            spearman = _safe_correlation(_average_ranks(x), _average_ranks(y))
            output.append(
                {
                    "feature": feature_name,
                    "outcome": outcome,
                    "n": int(x.size),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "absolute_spearman_rho": abs(spearman) if spearman is not None else None,
                }
            )
    output.sort(
        key=lambda row: (
            -float(row["absolute_spearman_rho"] or -1.0),
            str(row["outcome"]),
            str(row["feature"]),
        )
    )
    return output


def atlas_markdown(result: AtlasResult, *, top_n: int = 20) -> str:
    """Render a compact Markdown report from an atlas result."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    profiles = sorted(
        result.rule_profiles,
        key=lambda row: (-float(row["mean_restoration"]), int(row["rule"])),
    )[:top_n]
    correlations = result.correlations[:top_n]
    lines = [
        "# Ruliology Forge Resilience Atlas",
        "",
        f"Suite: `{result.manifest['suite']}`",
        f"Suite fingerprint: `{result.manifest['suite_fingerprint']}`",
        f"Observations: {result.manifest['observation_count']}",
        "",
        "## Highest mean restoration",
        "",
        "| Rule | Mean restoration | Worst scenario | Recovery probability | Trivial convergence |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in profiles:
        lines.append(
            "| {rule} | {mean_restoration:.4f} | {worst_scenario_restoration:.4f} | "
            "{recovery_probability:.4f} | {trivial_convergence_probability:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Strongest feature-outcome associations",
            "",
            "| Feature | Outcome | n | Pearson r | Spearman rho |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in correlations:
        pearson = "NA" if row["pearson_r"] is None else f"{float(row['pearson_r']):.4f}"
        spearman = (
            "NA" if row["spearman_rho"] is None else f"{float(row['spearman_rho']):.4f}"
        )
        lines.append(
            f"| {row['feature']} | {row['outcome']} | {row['n']} | {pearson} | {spearman} |"
        )
    lines.extend(
        [
            "",
            "Mean restoration is reported descriptively, not as a validated composite index. "
            "Rules with high trivial-convergence probability should be interpreted separately "
            "from rules that restore nontrivial structure.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_atlas_config(config: AtlasConfig) -> None:
    if not config.rules:
        raise ValueError("at least one rule is required.")
    if any(rule < 0 or rule > 255 for rule in config.rules):
        raise ValueError("ECA rules must be between 0 and 255.")
    if config.repeats <= 0 or config.jobs <= 0:
        raise ValueError("repeats and jobs must be positive.")
    if config.width <= 0 or config.steps <= 1:
        raise ValueError("width must be positive and steps must exceed one.")


def _validate_suite(suite: BenchmarkSuite, *, steps: int) -> None:
    if not suite.scenarios:
        raise ValueError("suite must contain at least one scenario.")
    if any(not 1 <= scenario.perturb_time < steps for scenario in suite.scenarios):
        raise ValueError("all scenario perturbation times must be between 1 and steps - 1.")


def _scenario_key(scenario: BenchmarkScenario) -> tuple[object, ...]:
    return (
        scenario.perturbation,
        scenario.perturb_time,
        scenario.perturb_radius,
        scenario.initial_condition,
        scenario.initial_density,
        scenario.noise_probability,
        scenario.boundary,
    )


def _row_scenario_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["perturbation"],
        row["perturb_time"],
        row["perturb_radius"],
        row["initial_condition"],
        row["initial_density"],
        row["noise_probability"],
        row["boundary"],
    )


def _numeric(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including tied values."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average
        start = stop
    return ranks
