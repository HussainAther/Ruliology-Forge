from ruliology_forge.analysis import parameter_grid, summarize_scan
from ruliology_forge.experiments import ExperimentConfig, run_parameter_sweep


def test_parameter_grid_cartesian_product():
    grid = parameter_grid(rule=[30, 90], perturb_radius=[1, 2])
    assert len(grid) == 4
    assert {row["rule"] for row in grid} == {30, 90}


def test_summarize_scan():
    rows = [
        {"rule": 30, "restoration_coefficient": 0.8, "recovered": True, "recovery_time": 2,
         "peak_divergence": 0.2, "final_scar_size": 1},
        {"rule": 30, "restoration_coefficient": 0.6, "recovered": False, "recovery_time": None,
         "peak_divergence": 0.4, "final_scar_size": 3},
    ]
    summary = summarize_scan(rows)[0]
    assert summary["n"] == 2
    assert summary["mean_restoration"] == 0.7
    assert summary["recovery_probability"] == 0.5


def test_small_parameter_sweep():
    config = ExperimentConfig(rule=90, width=31, steps=20, perturb_time=5, perturb_radius=1)
    rows = run_parameter_sweep([config], repeats=2, jobs=1)
    assert len(rows) == 2
    assert all(row["rule"] == 90 for row in rows)

import csv
import json
from pathlib import Path

from ruliology_forge.analysis import (
    AtlasAnalysisConfig,
    analyze_atlas_directory,
    classify_rule_profiles,
    identify_statistical_anomalies,
)


def _profile(rule, restoration, trivial, entropy, activity, scar, peak, std=0.05):
    return {
        "rule": rule,
        "observation_count": 6,
        "scenario_count": 3,
        "mean_restoration": restoration,
        "std_restoration": std,
        "median_restoration": restoration,
        "worst_scenario_restoration": max(0.0, restoration - 0.1),
        "best_scenario_restoration": min(1.0, restoration + 0.05),
        "recovery_probability": restoration,
        "mean_recovery_time": 3.0,
        "mean_shift_tolerant_restoration": restoration,
        "mean_peak_divergence": peak,
        "mean_scar_duration": scar,
        "mean_final_scar_size": scar / 2,
        "trivial_convergence_probability": trivial,
        "mean_control_density": 0.4,
        "std_control_density": 0.0,
        "mean_control_row_entropy_mean": entropy,
        "std_control_row_entropy_mean": 0.0,
        "mean_control_row_entropy_std": 0.1,
        "std_control_row_entropy_std": 0.0,
        "mean_control_temporal_activity": activity,
        "std_control_temporal_activity": 0.0,
        "mean_control_horizontal_edge_density": activity,
        "std_control_horizontal_edge_density": 0.0,
        "mean_control_bilateral_symmetry": 0.5,
        "std_control_bilateral_symmetry": 0.0,
        "mean_control_mean_components": 2.0,
        "std_control_mean_components": 0.0,
        "mean_control_max_components": 4.0,
        "std_control_max_components": 0.0,
        "mean_control_occupied_fraction": 0.5,
        "std_control_occupied_fraction": 0.0,
        "mean_control_compression_ratio": entropy + 0.1,
        "std_control_compression_ratio": 0.0,
    }


def test_classification_separates_trivial_and_nontrivial_recovery():
    profiles = [
        _profile(1, 0.95, 0.0, 0.8, 0.4, 1.0, 0.1),
        _profile(2, 0.94, 1.0, 0.0, 0.0, 1.0, 0.1),
        _profile(3, 0.20, 0.0, 0.6, 0.3, 9.0, 0.8),
        _profile(4, 0.50, 0.0, 0.5, 0.2, 2.0, 0.4),
    ]
    rows = classify_rule_profiles(profiles)
    classes = {row["rule"]: row["restoration_class"] for row in rows}
    assert classes[1] == "nontrivial_recoverer"
    assert classes[2] == "trivial_recoverer"
    assert all("descriptive_resilience_score" in row for row in rows)


def test_anomaly_detection_returns_interpretable_labels():
    profiles = [
        _profile(i, 0.35 + i * 0.02, 0.0, 0.1 + i * 0.02, 0.05 + i * 0.01, 2.0, 0.4)
        for i in range(8)
    ]
    profiles.append(_profile(110, 0.99, 0.0, 0.95, 0.8, 20.0, 0.2, std=0.5))
    enriched = classify_rule_profiles(profiles)
    anomalies = identify_statistical_anomalies(
        enriched, config=AtlasAnalysisConfig(anomaly_z_threshold=1.2)
    )
    assert anomalies
    assert any(int(row["rule"]) == 110 for row in anomalies)
    assert all(row["anomaly_labels"] for row in anomalies)


def test_analyze_atlas_directory_writes_outputs(tmp_path: Path):
    atlas_dir = tmp_path / "atlas"
    output_dir = tmp_path / "analysis"
    atlas_dir.mkdir()
    profiles = [
        _profile(1, 0.95, 0.0, 0.8, 0.4, 1.0, 0.1),
        _profile(2, 0.94, 1.0, 0.0, 0.0, 1.0, 0.1),
        _profile(3, 0.20, 0.0, 0.6, 0.3, 9.0, 0.8),
        _profile(4, 0.50, 0.0, 0.5, 0.2, 2.0, 0.4),
    ]
    with (atlas_dir / "atlas_rule_profiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(profiles[0]))
        writer.writeheader()
        writer.writerows(profiles)
    correlations = [
        {
            "feature": "control_row_entropy_mean",
            "outcome": "restoration_coefficient",
            "n": 4,
            "pearson_r": 0.4,
            "spearman_rho": 0.5,
            "absolute_spearman_rho": 0.5,
        }
    ]
    with (atlas_dir / "atlas_correlations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(correlations[0]))
        writer.writeheader()
        writer.writerows(correlations)
    observations = []
    for rule in range(1, 5):
        for time in (10, 20, 30):
            observations.append(
                {"rule": rule, "perturb_time": time, "restoration_coefficient": rule / 5}
            )
    with (atlas_dir / "atlas_observations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    (atlas_dir / "atlas_manifest.json").write_text(
        json.dumps({"suite": "test", "suite_fingerprint": "abc"}), encoding="utf-8"
    )

    result = analyze_atlas_directory(
        atlas_dir,
        output_dir,
        config=AtlasAnalysisConfig(top_n=3, figure_dpi=80),
    )
    assert result.nontrivial_recoverers[0]["rule"] == 1
    assert (output_dir / "findings.md").exists()
    assert (output_dir / "analysis_manifest.json").exists()
    assert (output_dir / "nontrivial_recoverers.csv").exists()
    assert (output_dir / "figures" / "restoration_by_rule.png").exists()
    assert (output_dir / "figures" / "restoration_by_rule.pdf").exists()
    assert (output_dir / "figures" / "timing_heatmap.png").exists()
