"""Export Ruliology Forge experiments as Houdini-ready data bundles."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .common import (
    validate_binary_trajectory,
    validate_trajectory_shapes,
    write_json,
)
from .schema import (
    BundleFiles,
    CoordinateSystem,
    HOUDINI_SCHEMA_VERSION,
    HoudiniBundleManifest,
)

if TYPE_CHECKING:
    from ruliology_forge.experiments import PerturbationResult


CONTROL_POINT_FIELDS = (
    "cell",
    "time",
    "state",
)

PERTURBED_POINT_FIELDS = (
    "cell",
    "time",
    "state",
)

DIFFERENCE_POINT_FIELDS = (
    "cell",
    "time",
    "difference",
    "control",
    "perturbed",
    "divergence",
)


def export_houdini_bundle(
    result: PerturbationResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export one perturbation experiment as a Version 1 Houdini bundle.

    Parameters
    ----------
    result:
        A completed Ruliology Forge perturbation experiment.
    output_dir:
        Destination directory for the bundle.
    overwrite:
        Permit replacement of files in a non-empty output directory.

    Returns
    -------
    pathlib.Path
        The resolved bundle directory.
    """

    bundle_dir = Path(output_dir).expanduser().resolve()

    _prepare_output_directory(
        bundle_dir,
        overwrite=overwrite,
    )

    control = validate_binary_trajectory(
        result.control,
        name="control",
    )

    perturbed = validate_binary_trajectory(
        result.perturbed,
        name="perturbed",
    )

    difference = validate_binary_trajectory(
        result.difference,
        name="difference",
    )

    validate_trajectory_shapes(
        control,
        perturbed,
        difference,
    )

    divergence = np.asarray(
        result.divergence,
        dtype=np.float64,
    )

    if divergence.ndim != 1:
        raise ValueError(
            "divergence must be a one-dimensional vector."
        )

    if divergence.shape[0] != control.shape[0]:
        raise ValueError(
            "divergence length must equal the trajectory step count."
        )

    expected_difference = np.bitwise_xor(
        control,
        perturbed,
    ).astype(np.uint8)

    if not np.array_equal(
        expected_difference,
        difference,
    ):
        raise ValueError(
            "difference does not equal control XOR perturbed."
        )

    expected_divergence = difference.mean(axis=1)

    if not np.allclose(
        expected_divergence,
        divergence,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "divergence does not equal mean difference per step."
        )

    files = BundleFiles()

    manifest = _build_manifest(
        result=result,
        control=control,
        files=files,
    )

    metrics = _build_metrics(result)

    write_json(
        bundle_dir / files.manifest,
        manifest.to_dict(),
    )

    write_json(
        bundle_dir / files.metrics,
        metrics,
    )

    _write_trajectory_archive(
        bundle_dir / files.trajectories,
        control=control,
        perturbed=perturbed,
        difference=difference,
        divergence=divergence,
    )

    _write_active_points(
        bundle_dir / files.control_points,
        trajectory=control,
        fields=CONTROL_POINT_FIELDS,
    )

    _write_active_points(
        bundle_dir / files.perturbed_points,
        trajectory=perturbed,
        fields=PERTURBED_POINT_FIELDS,
    )

    _write_difference_points(
        bundle_dir / files.difference_points,
        control=control,
        perturbed=perturbed,
        difference=difference,
        divergence=divergence,
    )

    _write_schema_version(
        bundle_dir / files.schema_version,
    )

    _write_bundle_readme(
        bundle_dir / files.readme,
        manifest=manifest,
        metrics=metrics,
    )

    return bundle_dir


def _prepare_output_directory(
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Create the output directory and protect existing data."""

    if output_dir.exists():
        existing = list(output_dir.iterdir())

        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Pass overwrite=True to replace bundle files."
            )
    else:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


def _build_manifest(
    *,
    result: PerturbationResult,
    control: np.ndarray,
    files: BundleFiles,
) -> HoudiniBundleManifest:
    """Create a manifest from an experiment result."""

    config = result.config
    steps, width = control.shape

    if config.width != width:
        raise ValueError(
            "Experiment config width does not match trajectory width."
        )

    if config.steps != steps:
        raise ValueError(
            "Experiment config steps do not match trajectory step count."
        )

    return HoudiniBundleManifest(
        schema=HOUDINI_SCHEMA_VERSION,
        rule=int(result.rule),
        width=int(width),
        steps=int(steps),
        boundary=str(config.boundary),
        initial_condition=str(config.initial_condition),
        initial_density=float(config.initial_density),
        seed=(
            None
            if config.seed is None
            else int(config.seed)
        ),
        initial_seed=int(result.initial_seed),
        perturbation_seed=int(result.perturbation_seed),
        perturbation=str(config.perturbation),
        perturbation_timing=str(
            config.perturbation_timing
        ),
        perturb_time=int(config.perturb_time),
        perturb_center=int(result.perturb_center),
        perturb_radius=int(config.perturb_radius),
        noise_probability=float(
            config.noise_probability
        ),
        recovery_threshold=float(
            config.recovery_threshold
        ),
        recovery_persistence=int(
            config.recovery_persistence
        ),
        max_shift=int(config.max_shift),
        coordinate_system=CoordinateSystem(),
        files=files,
    )


def _build_metrics(
    result: PerturbationResult,
) -> dict[str, object]:
    """Serialize all standard resilience metrics."""

    summary = result.summary

    return {
        "restoration_coefficient": float(
            summary.restoration_coefficient
        ),
        "final_restoration": float(
            summary.final_restoration
        ),
        "peak_divergence": float(
            summary.peak_divergence
        ),
        "mean_divergence": float(
            summary.mean_divergence
        ),
        "divergence_auc": float(
            summary.divergence_auc
        ),
        "recovery_time": (
            None
            if summary.recovery_time is None
            else int(summary.recovery_time)
        ),
        "recovered": bool(summary.recovered),
        "shift_tolerant_restoration": float(
            summary.shift_tolerant_restoration
        ),
        "best_final_shift": int(
            summary.best_final_shift
        ),
        "scar": {
            "size": int(
                summary.final_scar_size
            ),
            "duration": int(
                summary.scar_duration
            ),
            "spread": int(
                summary.scar_spread
            ),
            "centroid_drift": float(
                summary.scar_centroid_drift
            ),
        },
    }


def _write_trajectory_archive(
    path: Path,
    *,
    control: np.ndarray,
    perturbed: np.ndarray,
    difference: np.ndarray,
    divergence: np.ndarray,
) -> None:
    """Write lossless compressed NumPy trajectory data."""

    np.savez_compressed(
        path,
        control=control.astype(
            np.uint8,
            copy=False,
        ),
        perturbed=perturbed.astype(
            np.uint8,
            copy=False,
        ),
        difference=difference.astype(
            np.uint8,
            copy=False,
        ),
        divergence=divergence.astype(
            np.float64,
            copy=False,
        ),
    )


def _write_active_points(
    path: Path,
    *,
    trajectory: np.ndarray,
    fields: tuple[str, ...],
) -> None:
    """Write one CSV row for each active cell."""

    active_times, active_cells = np.nonzero(
        trajectory
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fields),
        )

        writer.writeheader()

        for time, cell in zip(
            active_times.tolist(),
            active_cells.tolist(),
            strict=True,
        ):
            writer.writerow(
                {
                    "cell": int(cell),
                    "time": int(time),
                    "state": 1,
                }
            )


def _write_difference_points(
    path: Path,
    *,
    control: np.ndarray,
    perturbed: np.ndarray,
    difference: np.ndarray,
    divergence: np.ndarray,
) -> None:
    """Write one CSV row for each control/perturbed disagreement."""

    changed_times, changed_cells = np.nonzero(
        difference
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                DIFFERENCE_POINT_FIELDS
            ),
        )

        writer.writeheader()

        for time, cell in zip(
            changed_times.tolist(),
            changed_cells.tolist(),
            strict=True,
        ):
            writer.writerow(
                {
                    "cell": int(cell),
                    "time": int(time),
                    "difference": 1,
                    "control": int(
                        control[time, cell]
                    ),
                    "perturbed": int(
                        perturbed[time, cell]
                    ),
                    "divergence": format(
                        float(divergence[time]),
                        ".17g",
                    ),
                }
            )


def _write_schema_version(
    path: Path,
) -> None:
    """Write a plain-text schema marker."""

    path.write_text(
        f"{HOUDINI_SCHEMA_VERSION}\n",
        encoding="utf-8",
    )


def _write_bundle_readme(
    path: Path,
    *,
    manifest: HoudiniBundleManifest,
    metrics: dict[str, object],
) -> None:
    """Write a human-readable summary of the bundle."""

    scar = metrics["scar"]

    if not isinstance(scar, dict):
        raise TypeError(
            "Expected the scar metrics payload to be a dictionary."
        )

    recovery_time = metrics["recovery_time"]

    recovery_text = (
        "not detected"
        if recovery_time is None
        else str(recovery_time)
    )

    text = f"""# Ruliology Forge Houdini Bundle

