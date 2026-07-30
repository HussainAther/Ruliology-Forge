import json
from pathlib import Path

from ruliology_forge.validation import ValidationConfig, validate_resilience


def test_validation_pipeline_writes_reproducible_outputs(tmp_path: Path):
    output = tmp_path / "validation"
    result = validate_resilience(
        ValidationConfig(
            rules=(0, 30, 90, 110),
            widths=(31, 41),
            steps_values=(20,),
            seeds=(3, 7),
            perturb_positions=(0.25, 0.5),
            boundaries=("periodic", "fixed"),
            perturb_time_fraction=0.3,
            perturb_radius=1,
            figure_dpi=80,
        ),
        output,
    )
    assert result.manifest["condition_count"] == 16
    assert result.manifest["run_count"] == 64
    assert len(result.rule_stability) == 4
    assert all(0 <= float(row["classification_agreement"]) <= 1 for row in result.rule_stability)
    expected = [
        "validation_runs.csv", "rule_stability.csv", "rank_correlations.csv",
        "classification_stability.csv", "robust_findings.csv", "unstable_findings.csv",
        "validation_report.md", "validation_manifest.json",
    ]
    assert all((output / name).exists() for name in expected)
    assert (output / "figures" / "rank_stability_heatmap.png").exists()
    assert (output / "figures" / "classification_stability.pdf").exists()
    manifest = json.loads((output / "validation_manifest.json").read_text())
    assert manifest["schema_version"] == "resilience-validation-v1"


def test_validation_rejects_invalid_position():
    try:
        validate_resilience(ValidationConfig(rules=(30,), perturb_positions=(1.2,)))
    except ValueError as exc:
        assert "perturb_positions" in str(exc)
    else:
        raise AssertionError("expected invalid position to fail")
