"""Cross-condition validation for resilience-atlas findings."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analysis import AtlasAnalysisConfig, classify_rule_profiles, identify_statistical_anomalies
from .atlas import aggregate_rule_profiles
from .experiments import ExperimentConfig, run_parameter_sweep


@dataclass(frozen=True)
class ValidationConfig:
    rules: tuple[int, ...]
    widths: tuple[int, ...] = (101, 201, 401)
    steps_values: tuple[int, ...] = (100, 200, 400)
    seeds: tuple[int, ...] = (11, 23, 47)
    perturb_positions: tuple[float, ...] = (0.25, 0.5, 0.75)
    boundaries: tuple[str, ...] = ("periodic", "fixed")
    perturb_time_fraction: float = 0.4
    perturb_radius: int = 5
    perturbation: str = "bit_flip"
    initial_condition: str = "single"
    initial_density: float = 0.5
    jobs: int = 1
    recovery_threshold: float = 0.01
    recovery_persistence: int = 5
    max_shift: int = 10
    classification_agreement_threshold: float = 0.8
    rank_stability_threshold: float = 0.75
    anomaly_reproduction_threshold: float = 0.7
    figure_dpi: int = 220

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    validation_runs: list[dict[str, object]]
    rule_stability: list[dict[str, object]]
    rank_correlations: list[dict[str, object]]
    classification_stability: list[dict[str, object]]
    robust_findings: list[dict[str, object]]
    unstable_findings: list[dict[str, object]]
    report_markdown: str
    manifest: dict[str, object]


def validate_resilience(
    config: ValidationConfig,
    output_dir: str | Path | None = None,
) -> ValidationResult:
    """Rerun rules over a condition grid and quantify finding stability."""

    _validate_config(config)
    analysis_config = AtlasAnalysisConfig(figure_dpi=config.figure_dpi)
    conditions = _condition_grid(config)
    experiment_configs: list[ExperimentConfig] = []
    metadata: list[dict[str, object]] = []
    for condition in conditions:
        for rule in config.rules:
            experiment_configs.append(
                ExperimentConfig(
                    rule=rule,
                    width=int(condition["width"]),
                    steps=int(condition["steps"]),
                    perturb_time=int(condition["perturb_time"]),
                    perturb_center=int(condition["perturb_center"]),
                    perturb_radius=config.perturb_radius,
                    perturbation=config.perturbation,  # type: ignore[arg-type]
                    initial_condition=config.initial_condition,  # type: ignore[arg-type]
                    initial_density=config.initial_density,
                    boundary=str(condition["boundary"]),  # type: ignore[arg-type]
                    seed=int(condition["seed"]),
                    recovery_threshold=config.recovery_threshold,
                    recovery_persistence=config.recovery_persistence,
                    max_shift=config.max_shift,
                )
            )
            metadata.append(condition)

    rows = run_parameter_sweep(
        experiment_configs, repeats=1, jobs=config.jobs, include_features=True
    )
    for row, condition in zip(rows, metadata, strict=True):
        row.update(condition)
        row["scenario"] = condition["condition_id"]

    condition_profiles: dict[str, list[dict[str, object]]] = {}
    condition_anomalies: dict[str, set[int]] = {}
    condition_ranks: dict[str, dict[int, int]] = {}
    condition_classes: dict[str, dict[int, str]] = {}
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        members = [row for row in rows if row["condition_id"] == condition_id]
        enriched = classify_rule_profiles(
            aggregate_rule_profiles(members), config=analysis_config
        )
        ranked = sorted(
            enriched,
            key=lambda row: (-float(row["descriptive_resilience_score"]), int(row["rule"])),
        )
        condition_profiles[condition_id] = enriched
        condition_ranks[condition_id] = {
            int(row["rule"]): rank for rank, row in enumerate(ranked, start=1)
        }
        condition_classes[condition_id] = {
            int(row["rule"]): str(row["restoration_class"]) for row in enriched
        }
        condition_anomalies[condition_id] = {
            int(row["rule"])
            for row in identify_statistical_anomalies(enriched, config=analysis_config)
        }
        for row in members:
            rule = int(row["rule"])
            row["validation_rank"] = condition_ranks[condition_id][rule]
            row["validation_class"] = condition_classes[condition_id][rule]
            row["validation_anomaly"] = rule in condition_anomalies[condition_id]

    rank_correlations = _rank_correlations(condition_ranks)
    rule_stability, class_rows = _rule_stability(
        rows, condition_ranks, condition_classes, condition_anomalies, config
    )
    robust = [row for row in rule_stability if bool(row["robust_finding"])]
    unstable = [row for row in rule_stability if not bool(row["robust_finding"])]
    robust.sort(key=lambda row: (-float(row["stability_score"]), int(row["rule"])))
    unstable.sort(key=lambda row: (float(row["stability_score"]), int(row["rule"])))

    manifest = {
        "schema_version": "resilience-validation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "condition_count": len(conditions),
        "run_count": len(rows),
        "rule_count": len(config.rules),
        "robust_finding_count": len(robust),
        "unstable_finding_count": len(unstable),
    }
    report = validation_markdown(rule_stability, rank_correlations, robust, unstable, manifest)
    result = ValidationResult(
        rows, rule_stability, rank_correlations, class_rows, robust, unstable, report, manifest
    )
    if output_dir is not None:
        write_validation_outputs(result, output_dir, dpi=config.figure_dpi)
    return result


def write_validation_outputs(
    result: ValidationResult, output_dir: str | Path, *, dpi: int = 220
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_runs.csv": result.validation_runs,
        "rule_stability.csv": result.rule_stability,
        "rank_correlations.csv": result.rank_correlations,
        "classification_stability.csv": result.classification_stability,
        "robust_findings.csv": result.robust_findings,
        "unstable_findings.csv": result.unstable_findings,
    }
    for name, rows in tables.items():
        _write_csv(output / name, rows)
    (output / "validation_report.md").write_text(result.report_markdown, encoding="utf-8")
    (output / "validation_manifest.json").write_text(
        json.dumps(result.manifest, indent=2) + "\n", encoding="utf-8"
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    _plot_rank_heatmap(result.rank_correlations, figures / "rank_stability_heatmap", dpi)
    _plot_factor_sensitivity(result.validation_runs, "width", figures / "width_sensitivity", dpi)
    _plot_factor_sensitivity(result.validation_runs, "steps", figures / "duration_sensitivity", dpi)
    _plot_seed_variability(result.validation_runs, figures, dpi)
    _plot_classification_stability(result.rule_stability, figures, dpi)


def validation_markdown(
    stability: Sequence[Mapping[str, object]],
    correlations: Sequence[Mapping[str, object]],
    robust: Sequence[Mapping[str, object]],
    unstable: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> str:
    mean_corr = float(np.mean([float(row["spearman_rho"]) for row in correlations])) if correlations else float("nan")
    lines = [
        "# Resilience Validation Report",
        "",
        f"Validated **{manifest['rule_count']} rules** across **{manifest['condition_count']} conditions** "
        f"({manifest['run_count']} runs).",
        "",
        "## Summary",
        "",
        f"- Robust findings: **{len(robust)}**",
        f"- Unstable findings: **{len(unstable)}**",
        f"- Mean pairwise Spearman rank correlation: **{mean_corr:.3f}**",
        "",
        "A robust finding satisfies the configured classification-agreement, rank-stability, "
        "and anomaly-reproduction thresholds. These thresholds are recorded in the manifest.",
        "",
        "## Most stable rules",
        "",
        "| Rule | Dominant class | Agreement | Rank stability | Anomaly consistency |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in list(robust)[:15]:
        lines.append(
            f"| {int(row['rule'])} | {row['dominant_class']} | "
            f"{float(row['classification_agreement']):.3f} | "
            f"{float(row['rank_stability']):.3f} | "
            f"{float(row['anomaly_consistency']):.3f} |"
        )
    lines += ["", "## Least stable rules", "", "| Rule | Dominant class | Agreement | Rank range |", "|---:|---|---:|---:|"]
    for row in list(unstable)[:15]:
        lines.append(
            f"| {int(row['rule'])} | {row['dominant_class']} | "
            f"{float(row['classification_agreement']):.3f} | {int(row['rank_range'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Rules in `robust_findings.csv` are the safest candidates for scientific discussion. "
        "Rules in `unstable_findings.csv` are sensitivity results, not failures: their apparent "
        "resilience depends materially on lattice size, duration, seed, injury location, or boundary.",
        "",
    ]
    return "\n".join(lines)


def _condition_grid(config: ValidationConfig) -> list[dict[str, object]]:
    conditions = []
    index = 0
    for width, steps, seed, position, boundary in product(
        config.widths,
        config.steps_values,
        config.seeds,
        config.perturb_positions,
        config.boundaries,
    ):
        perturb_time = min(steps - 1, max(1, round(steps * config.perturb_time_fraction)))
        center = min(width - 1, max(0, round((width - 1) * position)))
        conditions.append(
            {
                "condition_id": f"c{index:05d}",
                "width": width,
                "steps": steps,
                "seed": seed,
                "position_fraction": position,
                "boundary": boundary,
                "perturb_time": perturb_time,
                "perturb_center": center,
            }
        )
        index += 1
    return conditions


def _rank_correlations(ranks: Mapping[str, Mapping[int, int]]) -> list[dict[str, object]]:
    ids = sorted(ranks)
    rows = []
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            common = sorted(set(ranks[left]) & set(ranks[right]))
            x = np.asarray([ranks[left][rule] for rule in common], dtype=float)
            y = np.asarray([ranks[right][rule] for rule in common], dtype=float)
            rho = _pearson(_rankdata(x), _rankdata(y))
            rows.append({"condition_a": left, "condition_b": right, "n_rules": len(common), "spearman_rho": rho})
    return rows


def _rule_stability(rows, ranks, classes, anomalies, config):
    output = []
    class_rows = []
    condition_count = len(ranks)
    for rule in config.rules:
        rule_rows = [row for row in rows if int(row["rule"]) == rule]
        restoration = np.asarray([float(row["restoration_coefficient"]) for row in rule_rows])
        rule_ranks = np.asarray([ranks[c][rule] for c in sorted(ranks)], dtype=float)
        class_values = [classes[c][rule] for c in sorted(classes)]
        counts = {name: class_values.count(name) for name in sorted(set(class_values))}
        dominant = max(counts, key=lambda name: (counts[name], name))
        agreement = counts[dominant] / condition_count
        rank_range = float(rule_ranks.max() - rule_ranks.min())
        normalized_range = rank_range / max(1, len(config.rules) - 1)
        rank_stability = max(0.0, 1.0 - normalized_range)
        anomaly_rate = sum(rule in anomalies[c] for c in anomalies) / condition_count
        anomaly_consistency = max(anomaly_rate, 1.0 - anomaly_rate)
        robust = (
            agreement >= config.classification_agreement_threshold
            and rank_stability >= config.rank_stability_threshold
            and anomaly_consistency >= config.anomaly_reproduction_threshold
        )
        stability_score = (agreement + rank_stability + anomaly_consistency) / 3.0
        output.append({
            "rule": rule,
            "condition_count": condition_count,
            "mean_restoration": float(restoration.mean()),
            "std_restoration": float(restoration.std(ddof=0)),
            "min_restoration": float(restoration.min()),
            "max_restoration": float(restoration.max()),
            "coefficient_of_variation": float(restoration.std(ddof=0) / restoration.mean()) if restoration.mean() else 0.0,
            "mean_rank": float(rule_ranks.mean()),
            "min_rank": int(rule_ranks.min()),
            "max_rank": int(rule_ranks.max()),
            "rank_range": int(rank_range),
            "rank_stability": rank_stability,
            "dominant_class": dominant,
            "classification_agreement": agreement,
            "anomaly_reproduction_rate": anomaly_rate,
            "anomaly_consistency": anomaly_consistency,
            "stability_score": stability_score,
            "robust_finding": robust,
        })
        for name, count in counts.items():
            class_rows.append({"rule": rule, "restoration_class": name, "count": count, "proportion": count / condition_count})
    output.sort(key=lambda row: int(row["rule"]))
    return output, class_rows


def _plot_rank_heatmap(rows, output, dpi):
    ids = sorted({str(r["condition_a"]) for r in rows} | {str(r["condition_b"]) for r in rows})
    matrix = np.eye(len(ids))
    lookup = {name: i for i, name in enumerate(ids)}
    for row in rows:
        i, j = lookup[str(row["condition_a"])], lookup[str(row["condition_b"])]
        matrix[i, j] = matrix[j, i] = float(row["spearman_rho"])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Pairwise ranking stability")
    ax.set_xlabel("Validation condition")
    ax.set_ylabel("Validation condition")
    if len(ids) <= 20:
        ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=6)
        ax.set_yticks(range(len(ids)), ids, fontsize=6)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=ax, label="Spearman rho")
    fig.tight_layout()
    _save_figure(fig, output, dpi)


def _plot_factor_sensitivity(rows, factor, output, dpi):
    values = sorted({row[factor] for row in rows})
    means = [np.mean([float(r["restoration_coefficient"]) for r in rows if r[factor] == value]) for value in values]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([str(v) for v in values], means, marker="o")
    ax.set_title(f"Restoration sensitivity to {factor}")
    ax.set_xlabel(factor.replace("_", " ").title())
    ax.set_ylabel("Mean restoration coefficient")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, output, dpi)


def _plot_seed_variability(rows, output, dpi):
    seeds = sorted({int(row["seed"]) for row in rows})
    data = [[float(r["restoration_coefficient"]) for r in rows if int(r["seed"]) == seed] for seed in seeds]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, tick_labels=[str(seed) for seed in seeds])
    ax.set_title("Seed variability")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Restoration coefficient")
    fig.tight_layout()
    _save_figure(fig, output / "seed_variability", dpi)


def _plot_classification_stability(rows, output, dpi):
    ordered = sorted(rows, key=lambda r: float(r["classification_agreement"]))
    fig, ax = plt.subplots(figsize=(8, max(4.5, len(ordered) * 0.12)))
    ax.barh([str(r["rule"]) for r in ordered], [float(r["classification_agreement"]) for r in ordered])
    ax.set_xlim(0, 1)
    ax.set_title("Classification stability by rule")
    ax.set_xlabel("Agreement proportion")
    ax.set_ylabel("Rule")
    fig.tight_layout()
    _save_figure(fig, output / "classification_stability", dpi)


def _save_figure(fig, base: Path, dpi: int):
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _validate_config(config: ValidationConfig):
    if not config.rules or any(rule < 0 or rule > 255 for rule in config.rules):
        raise ValueError("rules must contain ECA rule numbers from 0 to 255.")
    if not config.widths or any(width <= 2 for width in config.widths):
        raise ValueError("widths must exceed two.")
    if not config.steps_values or any(steps <= 2 for steps in config.steps_values):
        raise ValueError("steps_values must exceed two.")
    if not config.seeds:
        raise ValueError("at least one seed is required.")
    if not config.perturb_positions or any(not 0 <= position <= 1 for position in config.perturb_positions):
        raise ValueError("perturb_positions must lie between zero and one.")
    if not config.boundaries or any(boundary not in {"periodic", "fixed"} for boundary in config.boundaries):
        raise ValueError("boundaries must be periodic or fixed.")
    if not 0 < config.perturb_time_fraction < 1:
        raise ValueError("perturb_time_fraction must lie strictly between zero and one.")
    if config.jobs <= 0 or config.figure_dpi <= 0:
        raise ValueError("jobs and figure_dpi must be positive.")
    for value in (
        config.classification_agreement_threshold,
        config.rank_stability_threshold,
        config.anomaly_reproduction_threshold,
    ):
        if not 0 <= value <= 1:
            raise ValueError("stability thresholds must lie between zero and one.")
