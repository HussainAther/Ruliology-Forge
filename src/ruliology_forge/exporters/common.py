"""Shared serialization helpers for Ruliology Forge exporters."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert NumPy and dataclass values into JSON-compatible objects."""

    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value


def write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write an object as stable, human-readable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(payload),
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")


def validate_binary_trajectory(
    trajectory: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return a validated two-dimensional binary trajectory."""

    array = np.asarray(trajectory, dtype=np.uint8)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional: steps x width."
        )

    if not np.isin(array, [0, 1]).all():
        raise ValueError(
            f"{name} must contain only binary values 0 and 1."
        )

    return array


def validate_trajectory_shapes(
    control: np.ndarray,
    perturbed: np.ndarray,
    difference: np.ndarray,
) -> None:
    """Require all trajectory arrays to have identical dimensions."""

    if control.shape != perturbed.shape:
        raise ValueError(
            "control and perturbed trajectories must have identical shapes."
        )

    if control.shape != difference.shape:
        raise ValueError(
            "control and difference trajectories must have identical shapes."
        )
