"""Higher-level analysis utilities for Ruliology Forge.

This module contains two layers:

* lightweight aggregation helpers used by the scan/sweep CLI commands; and
* an Atlas Analysis pipeline that loads a completed resilience atlas, separates
  trivial from nontrivial restoration, ranks interpretable rule categories,
  identifies statistical anomalies, creates publication-ready figures, and
  writes a grounded Markdown findings report.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Existing scan-analysis helpers
# ---------------------------------------------------------------------------


def summarize_scan(
    rows: Iterable[Mapping[str, object]],
    *,
    group_by: Sequence[str] = ("rule",),
    confidence: float = 0.95,
) -> list[dict[str, object]]:
    """Aggregate raw scan rows with confidence intervals and recovery rates."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_by)].append(row)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    output: list[dict[str, object]] = []
    for key, members in sorted(groups.items(), key=lambda item: item[0]):
        vals = np.asarray([float(m["restoration_coefficient"]) for m in members], dtype=float)
        recovery_times = [
            float(m["recovery_time"])
            for m in members
            if m.get("recovery_time") not in (None, "")
        ]
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        margin = z * std / sqrt(vals.size) if vals.size > 1 else 0.0
        row: dict[str, object] = {name: value for name, value in zip(group_by, key)}
        row.update(
            {
                "n": int(vals.size),
                "mean_restoration": mean,
                "std_restoration": std,
                "median_restoration": float(np.median(vals)),
                "ci_lower": max(0.0, mean - margin),
                "ci_upper": min(1.0, mean + margin),
                "recovery_probability": float(
                    sum(bool(m.get("recovered")) for m in members) / len(members)
                ),
                "mean_recovery_time": float(np.mean(recovery_times)) if recovery_times else None,
                "mean_peak_divergence": float(
                    np.mean([float(m["peak_divergence"]) for m in members])
                ),
                "mean_final_scar_size": float(
                    np.mean([float(m["final_scar_size"]) for m in members])
                ),
            }
        )
        output.append(row)
    return output


def parameter_grid(**parameters: Sequence[object]) -> list[dict[str, object]]:
    """Create a deterministic Cartesian product of parameter values."""
    names = list(parameters)
    if any(len(parameters[name]) == 0 for name in names):
        raise ValueError("parameter values cannot be empty.")
    return [dict(zip(names, values)) for values in product(*(parameters[name] for name in names))]


# ---------------------------------------------------------------------------
# Atlas Analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasAnalysisConfig:
    """Configuration for an interpretable, reproducible atlas analysis."""

    top_n: int = 20
    trivial_threshold: float = 0.5
    nontrivial_activity_floor: float = 0.005
    nontrivial_entropy_floor: float = 0.05
    anomaly_z_threshold: float = 1.5
    figure_dpi: int = 220

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AtlasAnalysisResult:
    """All tabular outputs and report text produced by atlas analysis."""

    enriched_profiles: list[dict[str, object]]
    top_resilient: list[dict[str, object]]
    nontrivial_recoverers: list[dict[str, object]]
    trivial_recoverers: list[dict[str, object]]
    scar_formers: list[dict[str, object]]
    fragile_rules: list[dict[str, object]]
    anomalies: list[dict[str, object]]
    findings_markdown: str
    manifest: dict[str, object]


def analyze_atlas_directory(
    atlas_dir: str | Path,
    output_dir: str | Path,
    *,
    config: AtlasAnalysisConfig | None = None,
) -> AtlasAnalysisResult:
    """Load atlas CSV outputs, analyze them, and write all analysis artifacts.

    Parameters
    ----------
    atlas_dir:
        Directory containing ``atlas_rule_profiles.csv`` and, when available,
        ``atlas_observations.csv``, ``atlas_correlations.csv``, and
        ``atlas_manifest.json``.
    output_dir:
        Destination directory for figures, ranked CSVs, the findings report,
        and an analysis manifest.
    config:
        Thresholds and rendering settings. Defaults are deliberately explicit
        and recorded in the output manifest.
    """

    config = config or AtlasAnalysisConfig()
    _validate_analysis_config(config)
    atlas_path = Path(atlas_dir)
    output_path = Path(output_dir)
    profile_path = atlas_path / "atlas_rule_profiles.csv"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing required atlas file: {profile_path}")

    profiles = _read_typed_csv(profile_path)
    if not profiles:
        raise ValueError("atlas_rule_profiles.csv contains no data rows.")
    _require_columns(
        profiles,
        {
            "rule",
            "mean_restoration",
            "std_restoration",
            "worst_scenario_restoration",
            "recovery_probability",
            "mean_peak_divergence",
            "mean_scar_duration",
            "mean_final_scar_size",
            "trivial_convergence_probability",
            "mean_control_temporal_activity",
            "mean_control_row_entropy_mean",
            "mean_control_compression_ratio",
        },
        source=profile_path,
    )

    correlations = _read_optional_csv(atlas_path / "atlas_correlations.csv")
    observations = _read_optional_csv(atlas_path / "atlas_observations.csv")
    source_manifest = _read_optional_json(atlas_path / "atlas_manifest.json")

    enriched = classify_rule_profiles(profiles, config=config)
    categories = rank_rule_categories(enriched, config=config)
    anomalies = identify_statistical_anomalies(enriched, config=config)

    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_files = create_atlas_figures(
        enriched,
        correlations=correlations,
        observations=observations,
        output_dir=figures_dir,
        dpi=config.figure_dpi,
    )

    tables = {
        "enriched_rule_profiles.csv": enriched,
        "top_resilient_rules.csv": categories["top_resilient"],
        "nontrivial_recoverers.csv": categories["nontrivial_recoverers"],
        "trivial_recoverers.csv": categories["trivial_recoverers"],
        "scar_formers.csv": categories["scar_formers"],
        "fragile_rules.csv": categories["fragile_rules"],
        "anomalous_rules.csv": anomalies,
    }
    for filename, rows in tables.items():
        _write_csv_allow_empty(output_path / filename, rows)

    manifest = {
        "schema_version": "atlas-analysis-v1",
        "source_atlas_dir": str(atlas_path.resolve()),
        "source_suite": source_manifest.get("suite"),
        "source_suite_fingerprint": source_manifest.get("suite_fingerprint"),
        "source_rule_count": len(enriched),
        "source_observation_count": len(observations),
        "config": config.to_dict(),
        "category_counts": {
            "nontrivial_recoverers": len(categories["nontrivial_recoverers"]),
            "trivial_recoverers": len(categories["trivial_recoverers"]),
            "scar_formers": len(categories["scar_formers"]),
            "fragile_rules": len(categories["fragile_rules"]),
            "anomalies": len(anomalies),
        },
        "figures": [path.name for path in figure_files],
    }
    findings = findings_markdown(
        enriched,
        categories=categories,
        anomalies=anomalies,
        correlations=correlations,
        manifest=manifest,
        config=config,
    )
    (output_path / "findings.md").write_text(findings, encoding="utf-8")
    (output_path / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    return AtlasAnalysisResult(
        enriched_profiles=enriched,
        top_resilient=categories["top_resilient"],
        nontrivial_recoverers=categories["nontrivial_recoverers"],
        trivial_recoverers=categories["trivial_recoverers"],
        scar_formers=categories["scar_formers"],
        fragile_rules=categories["fragile_rules"],
        anomalies=anomalies,
        findings_markdown=findings,
        manifest=manifest,
    )


def classify_rule_profiles(
    profiles: Sequence[Mapping[str, object]],
    *,
    config: AtlasAnalysisConfig | None = None,
) -> list[dict[str, object]]:
    """Add transparent restoration classes and interpretable ranking scores."""

    config = config or AtlasAnalysisConfig()
    _validate_analysis_config(config)
    if not profiles:
        return []

    restoration = np.asarray([float(row["mean_restoration"]) for row in profiles])
    scar_duration = np.asarray([float(row["mean_scar_duration"]) for row in profiles])
    peak_divergence = np.asarray([float(row["mean_peak_divergence"]) for row in profiles])
    restoration_hi = float(np.quantile(restoration, 0.75))
    restoration_lo = float(np.quantile(restoration, 0.25))
    scar_hi = float(np.quantile(scar_duration, 0.75))
    divergence_hi = float(np.quantile(peak_divergence, 0.75))

    output: list[dict[str, object]] = []
    for source in profiles:
        row = dict(source)
        trivial_probability = float(row["trivial_convergence_probability"])
        activity = float(row["mean_control_temporal_activity"])
        entropy = float(row["mean_control_row_entropy_mean"])
        mean_restoration = float(row["mean_restoration"])
        worst_restoration = float(row["worst_scenario_restoration"])
        recovery_probability = float(row["recovery_probability"])
        duration = float(row["mean_scar_duration"])
        divergence = float(row["mean_peak_divergence"])
        final_scar = float(row["mean_final_scar_size"])

        is_trivial = trivial_probability >= config.trivial_threshold
        has_nontrivial_dynamics = (
            activity >= config.nontrivial_activity_floor
            and entropy >= config.nontrivial_entropy_floor
            and not is_trivial
        )
        robust = mean_restoration >= restoration_hi
        fragile = mean_restoration <= restoration_lo
        scar_former = (
            duration >= scar_hi
            and final_scar > 0
            and mean_restoration < restoration_hi
        )
        chaotic_amplifier = divergence >= divergence_hi and fragile

        if is_trivial:
            restoration_class = "trivial_recoverer"
        elif robust and has_nontrivial_dynamics:
            restoration_class = "nontrivial_recoverer"
        elif scar_former:
            restoration_class = "scar_former"
        elif chaotic_amplifier:
            restoration_class = "chaotic_amplifier"
        elif fragile:
            restoration_class = "fragile"
        else:
            restoration_class = "intermediate"

        # Transparent descriptive scores, not validated universal indices.
        resilience_score = (
            0.45 * mean_restoration
            + 0.25 * worst_restoration
            + 0.20 * recovery_probability
            + 0.10 * (1.0 - min(1.0, trivial_probability))
        )
        scar_score = (
            _safe_unit(duration, scar_duration)
            + _safe_unit(final_scar, np.asarray([float(p["mean_final_scar_size"]) for p in profiles]))
            + _safe_unit(divergence, peak_divergence)
        ) / 3.0

        row.update(
            {
                "is_trivial_recovery": is_trivial,
                "has_nontrivial_control_dynamics": has_nontrivial_dynamics,
                "restoration_class": restoration_class,
                "descriptive_resilience_score": float(resilience_score),
                "descriptive_scar_score": float(scar_score),
            }
        )
        output.append(row)

    output.sort(key=lambda row: int(row["rule"]))
    return output


def rank_rule_categories(
    profiles: Sequence[Mapping[str, object]],
    *,
    config: AtlasAnalysisConfig | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Return ranked, non-overwriting category tables for the findings report."""

    config = config or AtlasAnalysisConfig()
    top_n = config.top_n
    all_rows = [dict(row) for row in profiles]

    by_resilience = sorted(
        all_rows,
        key=lambda row: (
            -float(row["descriptive_resilience_score"]),
            -float(row["mean_restoration"]),
            int(row["rule"]),
        ),
    )
    nontrivial = [row for row in by_resilience if row["restoration_class"] == "nontrivial_recoverer"]
    trivial = [row for row in by_resilience if row["restoration_class"] == "trivial_recoverer"]
    scar = sorted(
        [row for row in all_rows if row["restoration_class"] == "scar_former"],
        key=lambda row: (-float(row["descriptive_scar_score"]), int(row["rule"])),
    )
    fragile = sorted(
        [
            row
            for row in all_rows
            if row["restoration_class"] in {"fragile", "chaotic_amplifier"}
        ],
        key=lambda row: (float(row["mean_restoration"]), -float(row["mean_peak_divergence"]), int(row["rule"])),
    )
    return {
        "top_resilient": by_resilience[:top_n],
        "nontrivial_recoverers": nontrivial[:top_n],
        "trivial_recoverers": trivial[:top_n],
        "scar_formers": scar[:top_n],
        "fragile_rules": fragile[:top_n],
    }


def identify_statistical_anomalies(
    profiles: Sequence[Mapping[str, object]],
    *,
    config: AtlasAnalysisConfig | None = None,
) -> list[dict[str, object]]:
    """Identify interpretable z-score and regression-residual anomalies.

    An anomaly is reported only when it meets an explicit rule. This avoids
    presenting every extreme value as a discovery.
    """

    config = config or AtlasAnalysisConfig()
    if len(profiles) < 3:
        return []

    restoration = np.asarray([float(row["mean_restoration"]) for row in profiles])
    entropy = np.asarray([float(row["mean_control_row_entropy_mean"]) for row in profiles])
    activity = np.asarray([float(row["mean_control_temporal_activity"]) for row in profiles])
    compression = np.asarray([float(row["mean_control_compression_ratio"]) for row in profiles])
    scar_duration = np.asarray([float(row["mean_scar_duration"]) for row in profiles])
    std_restoration = np.asarray([float(row["std_restoration"]) for row in profiles])

    complexity = np.mean(
        np.vstack([_zscore(entropy), _zscore(activity), _zscore(compression)]), axis=0
    )
    design = np.column_stack([np.ones(len(profiles)), complexity])
    coefficients, *_ = np.linalg.lstsq(design, restoration, rcond=None)
    residual = restoration - design @ coefficients

    z_restoration = _zscore(restoration)
    z_complexity = _zscore(complexity)
    z_residual = _zscore(residual)
    z_scar = _zscore(scar_duration)
    z_instability = _zscore(std_restoration)
    threshold = config.anomaly_z_threshold

    anomalies: list[dict[str, object]] = []
    for index, source in enumerate(profiles):
        labels: list[str] = []
        if z_restoration[index] >= threshold and z_complexity[index] >= threshold:
            labels.append("high_complexity_high_restoration")
        if z_residual[index] >= threshold:
            labels.append("restores_better_than_complexity_trend")
        if z_residual[index] <= -threshold:
            labels.append("restores_worse_than_complexity_trend")
        if z_scar[index] >= threshold:
            labels.append("exceptionally_persistent_scar")
        if z_instability[index] >= threshold:
            labels.append("scenario_sensitive_restoration")
        if not labels:
            continue
        row = dict(source)
        row.update(
            {
                "anomaly_labels": ";".join(labels),
                "complexity_z": float(z_complexity[index]),
                "restoration_z": float(z_restoration[index]),
                "restoration_residual_z": float(z_residual[index]),
                "scar_duration_z": float(z_scar[index]),
                "restoration_instability_z": float(z_instability[index]),
                "anomaly_strength": float(
                    max(
                        abs(z_complexity[index]),
                        abs(z_restoration[index]),
                        abs(z_residual[index]),
                        abs(z_scar[index]),
                        abs(z_instability[index]),
                    )
                ),
            }
        )
        anomalies.append(row)

    anomalies.sort(key=lambda row: (-float(row["anomaly_strength"]), int(row["rule"])))
    return anomalies[: config.top_n]


def create_atlas_figures(
    profiles: Sequence[Mapping[str, object]],
    *,
    correlations: Sequence[Mapping[str, object]],
    observations: Sequence[Mapping[str, object]],
    output_dir: Path,
    dpi: int = 220,
) -> list[Path]:
    """Create a compact set of publication-ready PNG and PDF figures."""

    if not profiles:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    ranked = sorted(profiles, key=lambda row: float(row["mean_restoration"]), reverse=True)
    top = ranked[: min(30, len(ranked))]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    labels = [str(int(row["rule"])) for row in reversed(top)]
    values = [float(row["mean_restoration"]) for row in reversed(top)]
    ax.barh(labels, values)
    ax.set_xlabel("Mean restoration coefficient")
    ax.set_ylabel("ECA rule")
    ax.set_xlim(0, 1.02)
    ax.set_title("Highest mean restoration across the atlas")
    ax.grid(axis="x", alpha=0.25)
    paths.extend(_save_figure(fig, output_dir / "restoration_by_rule", dpi))

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    x = [float(row["mean_control_row_entropy_mean"]) for row in profiles]
    y = [float(row["mean_restoration"]) for row in profiles]
    sizes = [35 + 70 * float(row["trivial_convergence_probability"]) for row in profiles]
    ax.scatter(x, y, s=sizes, alpha=0.72)
    for row, x_value, y_value in zip(profiles, x, y):
        if row.get("restoration_class") in {"nontrivial_recoverer", "trivial_recoverer"}:
            ax.annotate(str(int(row["rule"])), (x_value, y_value), fontsize=7, alpha=0.8)
    ax.set_xlabel("Mean control row entropy")
    ax.set_ylabel("Mean restoration coefficient")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Restoration versus control-trajectory entropy")
    ax.grid(alpha=0.25)
    ax.text(
        0.01,
        0.01,
        "Marker size increases with trivial-convergence probability",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    paths.extend(_save_figure(fig, output_dir / "restoration_vs_entropy", dpi))

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    x = [float(row["mean_scar_duration"]) for row in profiles]
    y = [float(row["mean_final_scar_size"]) for row in profiles]
    ax.scatter(x, y, alpha=0.72)
    for row, x_value, y_value in zip(profiles, x, y):
        if row.get("restoration_class") == "scar_former":
            ax.annotate(str(int(row["rule"])), (x_value, y_value), fontsize=7)
    ax.set_xlabel("Mean scar duration")
    ax.set_ylabel("Mean final scar size")
    ax.set_title("Persistence and extent of computational scars")
    ax.grid(alpha=0.25)
    paths.extend(_save_figure(fig, output_dir / "scar_persistence", dpi))

    if correlations:
        paths.extend(_correlation_heatmap(correlations, output_dir, dpi))

    timing_rows = [row for row in observations if row.get("perturb_time") is not None]
    unique_times = sorted({int(row["perturb_time"]) for row in timing_rows})
    unique_rules = sorted({int(row["rule"]) for row in timing_rows})
    if len(unique_times) >= 3 and unique_rules:
        paths.extend(_timing_heatmap(timing_rows, unique_rules, unique_times, output_dir, dpi))

    return paths


def findings_markdown(
    profiles: Sequence[Mapping[str, object]],
    *,
    categories: Mapping[str, Sequence[Mapping[str, object]]],
    anomalies: Sequence[Mapping[str, object]],
    correlations: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
    config: AtlasAnalysisConfig,
) -> str:
    """Generate a cautious, data-grounded Markdown findings report."""

    lines = [
        "# Ruliology Forge Atlas Analysis",
        "",
        "## Scope",
        "",
        f"This analysis covers **{len(profiles)} ECA rules** and "
        f"**{manifest.get('source_observation_count', 0)} scenario-level observations**.",
        "The categories below are descriptive screening tools, not validated universal classes.",
        "",
        "## Classification criteria",
        "",
        f"- A recovery is treated as trivial when its trivial-convergence probability is at least "
        f"{config.trivial_threshold:.2f}.",
        f"- Nontrivial control dynamics require temporal activity of at least "
        f"{config.nontrivial_activity_floor:.4f} and row entropy of at least "
        f"{config.nontrivial_entropy_floor:.4f}.",
        "- Robust, fragile, and scar-persistence cutoffs use within-atlas quartiles, so they are "
        "relative to the analyzed dataset.",
        "",
    ]

    _append_rule_table(lines, "Highest descriptive resilience", categories["top_resilient"])
    _append_rule_table(lines, "Nontrivial recoverers", categories["nontrivial_recoverers"])
    _append_rule_table(lines, "Trivial recoverers", categories["trivial_recoverers"])
    _append_rule_table(lines, "Persistent scar formers", categories["scar_formers"])
    _append_rule_table(lines, "Most fragile rules", categories["fragile_rules"])

    lines.extend(["## Statistical anomalies", ""])
    if anomalies:
        lines.extend(
            [
                "| Rule | Anomaly labels | Mean restoration | Complexity z | Residual z |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in anomalies:
            lines.append(
                f"| {int(row['rule'])} | {row['anomaly_labels']} | "
                f"{float(row['mean_restoration']):.4f} | "
                f"{float(row['complexity_z']):.2f} | "
                f"{float(row['restoration_residual_z']):.2f} |"
            )
    else:
        lines.append("No rules crossed the configured anomaly threshold.")
    lines.append("")

    valid_correlations = [
        row
        for row in correlations
        if row.get("absolute_spearman_rho") not in (None, "")
    ]
    valid_correlations.sort(
        key=lambda row: -float(row["absolute_spearman_rho"])
    )
    lines.extend(["## Strongest feature–outcome associations", ""])
    if valid_correlations:
        lines.extend(
            [
                "| Feature | Outcome | Spearman rho | n |",
                "|---|---|---:|---:|",
            ]
        )
        for row in valid_correlations[:10]:
            lines.append(
                f"| {row['feature']} | {row['outcome']} | "
                f"{float(row['spearman_rho']):.4f} | {int(row['n'])} |"
            )
    else:
        lines.append("No nonconstant feature–outcome correlations were available.")

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "1. High restoration is reported separately from trivial convergence.",
            "2. The descriptive resilience score is a ranking convenience, not a claimed law or "
            "validated composite index.",
            "3. Anomalies are hypothesis generators. They require replication across lattice "
            "widths, durations, seeds, boundary conditions, and perturbation regimes.",
            "4. Correlations are exploratory and do not establish causal mechanisms.",
            "",
            "## Generated figures",
            "",
        ]
    )
    for filename in manifest.get("figures", []):
        if str(filename).endswith(".png"):
            lines.append(f"- `figures/{filename}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_analysis_config(config: AtlasAnalysisConfig) -> None:
    if config.top_n <= 0:
        raise ValueError("top_n must be positive.")
    if not 0 <= config.trivial_threshold <= 1:
        raise ValueError("trivial_threshold must be between 0 and 1.")
    if config.nontrivial_activity_floor < 0 or config.nontrivial_entropy_floor < 0:
        raise ValueError("nontrivial dynamics floors cannot be negative.")
    if config.anomaly_z_threshold <= 0:
        raise ValueError("anomaly_z_threshold must be positive.")
    if config.figure_dpi < 72:
        raise ValueError("figure_dpi must be at least 72.")


def _safe_unit(value: float, population: np.ndarray) -> float:
    minimum = float(np.min(population))
    maximum = float(np.max(population))
    if maximum == minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def _zscore(values: np.ndarray) -> np.ndarray:
    standard_deviation = float(values.std(ddof=0))
    if standard_deviation == 0:
        return np.zeros_like(values, dtype=float)
    return (values - float(values.mean())) / standard_deviation


def _read_typed_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: _coerce_csv_value(value) for key, value in row.items()} for row in rows]


def _read_optional_csv(path: Path) -> list[dict[str, object]]:
    return _read_typed_csv(path) if path.exists() else []


def _read_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _coerce_csv_value(value: str | None) -> object:
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(character in value for character in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _require_columns(
    rows: Sequence[Mapping[str, object]],
    required: set[str],
    *,
    source: Path,
) -> None:
    available = set(rows[0])
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _write_csv_allow_empty(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    fig.tight_layout()
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _correlation_heatmap(
    correlations: Sequence[Mapping[str, object]],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    features = sorted({str(row["feature"]) for row in correlations})
    outcomes = sorted({str(row["outcome"]) for row in correlations})
    if not features or not outcomes:
        return []
    matrix = np.full((len(features), len(outcomes)), np.nan)
    feature_index = {name: index for index, name in enumerate(features)}
    outcome_index = {name: index for index, name in enumerate(outcomes)}
    for row in correlations:
        value = row.get("spearman_rho")
        if value in (None, ""):
            continue
        matrix[feature_index[str(row["feature"])], outcome_index[str(row["outcome"])]] = float(value)

    fig, ax = plt.subplots(figsize=(9.5, max(5.5, len(features) * 0.42)))
    image = ax.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(outcomes)), labels=[name.replace("_", " ") for name in outcomes], rotation=35, ha="right")
    ax.set_yticks(range(len(features)), labels=[name.replace("control_", "").replace("_", " ") for name in features])
    ax.set_title("Feature–outcome Spearman correlations")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Spearman rho")
    return _save_figure(fig, output_dir / "resilience_correlation_matrix", dpi)


def _timing_heatmap(
    observations: Sequence[Mapping[str, object]],
    rules: Sequence[int],
    times: Sequence[int],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in observations:
        grouped[(int(row["rule"]), int(row["perturb_time"]))].append(
            float(row["restoration_coefficient"])
        )
    matrix = np.full((len(rules), len(times)), np.nan)
    for row_index, rule in enumerate(rules):
        for column_index, time in enumerate(times):
            values = grouped.get((rule, time), [])
            if values:
                matrix[row_index, column_index] = float(np.mean(values))

    fig_height = min(18.0, max(6.0, len(rules) * 0.06))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(times)), labels=[str(time) for time in times], rotation=45, ha="right")
    if len(rules) <= 40:
        ax.set_yticks(range(len(rules)), labels=[str(rule) for rule in rules])
    else:
        tick_indices = np.linspace(0, len(rules) - 1, 12, dtype=int)
        ax.set_yticks(tick_indices, labels=[str(rules[index]) for index in tick_indices])
    ax.set_xlabel("Perturbation time")
    ax.set_ylabel("ECA rule")
    ax.set_title("Restoration across perturbation timing")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean restoration coefficient")
    return _save_figure(fig, output_dir / "timing_heatmap", dpi)


def _append_rule_table(
    lines: list[str],
    title: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    lines.extend([f"## {title}", ""])
    if not rows:
        lines.extend(["No rules met this category's configured criteria.", ""])
        return
    lines.extend(
        [
            "| Rule | Class | Mean restoration | Worst scenario | Recovery probability | Trivial convergence |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {int(row['rule'])} | {row['restoration_class']} | "
            f"{float(row['mean_restoration']):.4f} | "
            f"{float(row['worst_scenario_restoration']):.4f} | "
            f"{float(row['recovery_probability']):.4f} | "
            f"{float(row['trivial_convergence_probability']):.4f} |"
        )
    lines.append("")
