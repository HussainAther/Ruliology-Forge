"""Tests for the Version 1 Houdini export bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ruliology_forge import (
    HOUDINI_SCHEMA_VERSION,
    export_houdini_bundle,
    run_perturbation_experiment,
)


EXPECTED_FILES = {
    "manifest.json",
    "metrics.json",
    "trajectories.npz",
    "control_points.csv",
    "perturbed_points.csv",
    "difference_points.csv",
    "README.md",
    "schema_version.txt",
}


@pytest.fixture
def result():
    return run_perturbation_experiment(
        rule=110,
        width=41,
        steps=30,
        perturb_time=10,
        perturb_center=20,
        perturb_radius=1,
        perturbation="bit_flip",
        initial_condition="single",
        boundary="periodic",
        seed=42,
    )


def test_export_creates_expected_files(
    tmp_path: Path,
    result,
) -> None:
    bundle = export_houdini_bundle(
        result,
        tmp_path / "rule110_bundle",
    )

    names = {
        path.name
        for path in bundle.iterdir()
    }

    assert names == EXPECTED_FILES


def test_manifest_matches_experiment(
    tmp_path: Path,
    result,
) -> None:
    bundle = export_houdini_bundle(
        result,
        tmp_path / "rule110_bundle",
    )

    payload = json.loads(
        (bundle / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema"] == HOUDINI_SCHEMA_VERSION
    assert payload["rule"] == 110
    assert payload["width"] == 41
    assert payload["steps"] == 30
    assert payload["boundary"] == "periodic"
    assert payload["initial_condition"] == "single"
    assert payload["seed"] == 42
    assert payload["perturbation"] == "bit_flip"
    assert payload["perturbation_time"] if False else True
    assert payload["perturb_time"] == 10
    assert payload["perturb_center"] == 20
    assert payload["perturb_radius"] == 1

    assert payload["coordinate_system"] == {
        "x": "cell",
        "y": "layer",
        "z": "time",
    }

    assert payload["files"]["manifest"] == "manifest.json"
    assert payload["files"]["metrics"] == "metrics.json"
    assert payload["files"]["trajectories"] == "trajectories.npz"


def test_metrics_match_result(
    tmp_path: Path,
    result,
) -> None:
    bundle = export_houdini_bundle(
        result,
        tmp_path / "rule110_bundle",
    )

    metrics = json.loads(
        (bundle / "metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert metrics["restoration_coefficient"] == pytest.approx(
        result.summary.restoration_coefficient
    )

    assert metrics["final_restoration"] == pytest.approx(
        result.summary.final_restoration
    )

    assert metrics["peak_divergence"] == pytest.approx(
        result.summary.peak_divergence
    )

    assert metrics["mean_divergence"] == pytest.approx(
        result.summary.mean_divergence
    )

    assert metrics["divergence_auc"] == pytest.approx(
        result.summary.divergence_auc
    )

    assert metrics["recovered"] is result.summary.recovered
    assert metrics["recovery_time"] == result.summary.recovery_time

    assert metrics["scar"]["size"] == (
        result.summary.final_scar_size
    )

    assert metrics["scar"]["duration"] == (
        result.summary.scar_duration
    )

    assert metrics["scar"]["spread"] == (
        result.summary.scar_spread
    )

    assert metrics["scar"]["centroid_drift"] == pytest.approx(
        result.summary.scar_centroid_drift
    )


def test_npz_archive_is_lossless(
    tmp_path: Path,
    result,
) -> None:
    bundle = export_houdini_bundle(
        result,
        tmp_path / "rule110_bundle",
    )

    with np.load(
        bundle / "trajectories.npz"
    ) as archive:
        np.testing.assert_array_equal(
            archive["control"],
            result.control,
        )

        np.testing.assert_array_equal(
            archive["perturbed"],
            result.perturbed,
        )

        np.testing.assert_array_equal(
            archive["difference"],
            result.difference,
        )

        np.testing.assert_allclose(
            archive["divergence"],
            result.divergence,
        )


def test_control_csv_contains_only_active_cells(
    tmp_path: Path,
    result,
) -> None:
    bundle = export_houdini_bundle(
        result,
        tmp_path / "rule110_bundle",
    )

    with (
        bundle / "control_points.csv"
    ).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == int(result.control.sum())

    for row in rows:
        time = int(row["time"])
        cell = int(row["cell"])

        assert int(row["state"]) == 1
        assert result.control[time, cell] == 1


def test_perturbed_csv_contains_only_active_cells(
    tmp_path: Path,
    result,
) -> None:
    bundle = export_houdini_bundle(
        result,
        tmp_path / "rule110_bundle",
    )

    with (
        bundle / "perturbed_points.csv"
    ).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == int(result.perturbed.sum())

    for row in rows:
        time = int(row["time"])
        cell = int(row["cell"])

        assert int(row["state"]) == 1
        assert result.perturbed[time, cell] == 1


def test_difference_csv_matches_xor_cells(
    tmp_path: Path,
    result,
) -> None:
    bundle = export_houdini_bundle(
        result,
        tmp_path / "rule110_bundle",
    )

    with (
        bundle / "difference_points.csv"
    ).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == int(result.difference.sum())

    for row in rows:
        time = int(row["time"])
        cell = int(row["cell"])

        assert int(row["difference"]) == 1

        assert int(row["control"]) == int(
            result.control[time, cell]
        )

        assert int(row["perturbed"]) == int(
            result.perturbed[time, cell]
        )

        assert result.difference[time, cell] == 1

        assert float(row["divergence"]) == pytest.approx(
            result.divergence[time]
        )


def test_schema_version_file(
    tmp_path: Path,
    result,
) -> None:
    bundle = export_houdini_bundle(
        result,
        tmp_path / "rule110_bundle",
    )

    version = (
        bundle / "schema_version.txt"
    ).read_text(
        encoding="utf-8"
    )

    assert version == f"{HOUDINI_SCHEMA_VERSION}\n"


def test_readme_contains_bundle_summary(
    tmp_path: Path,
    result,
) -> None:
    bundle = export_houdini_bundle(
        result,
        tmp_path / "rule110_bundle",
    )

    text = (
        bundle / "README.md"
    ).read_text(
        encoding="utf-8"
    )

    assert "Rule: 110" in text
    assert "Width: 41" in text
    assert "Steps: 30" in text
    assert HOUDINI_SCHEMA_VERSION in text
    assert "X: lattice cell" in text
    assert "Z: simulation time" in text


def test_nonempty_directory_requires_overwrite(
    tmp_path: Path,
    result,
) -> None:
    output = tmp_path / "rule110_bundle"
    output.mkdir()
    (output / "existing.txt").write_text(
        "keep me",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        export_houdini_bundle(
            result,
            output,
        )


def test_overwrite_allows_existing_directory(
    tmp_path: Path,
    result,
) -> None:
    output = tmp_path / "rule110_bundle"
    output.mkdir()

    export_houdini_bundle(
        result,
        output,
        overwrite=True,
    )

    assert (
        output / "manifest.json"
    ).exists()


def test_export_is_deterministic(
    tmp_path: Path,
    result,
) -> None:
    first = export_houdini_bundle(
        result,
        tmp_path / "first",
    )

    second = export_houdini_bundle(
        result,
        tmp_path / "second",
    )

    text_files = [
        "manifest.json",
        "metrics.json",
        "control_points.csv",
        "perturbed_points.csv",
        "difference_points.csv",
        "README.md",
        "schema_version.txt",
    ]

    for name in text_files:
        assert (
            first / name
        ).read_bytes() == (
            second / name
        ).read_bytes()

    with np.load(
        first / "trajectories.npz"
    ) as archive_a:
        with np.load(
            second / "trajectories.npz"
        ) as archive_b:
            for key in (
                "control",
                "perturbed",
                "difference",
                "divergence",
            ):
                np.testing.assert_array_equal(
                    archive_a[key],
                    archive_b[key],
                )
