"""Feature extraction for cellular-automata trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import zlib

import numpy as np


@dataclass(frozen=True)
class TrajectoryFeatures:
    """Compact morphological and information-theoretic trajectory profile."""

    density: float
    row_entropy_mean: float
    row_entropy_std: float
    temporal_activity: float
    horizontal_edge_density: float
    bilateral_symmetry: float
    mean_components: float
    max_components: int
    occupied_fraction: float
    compression_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


FEATURE_NAMES = tuple(TrajectoryFeatures.__dataclass_fields__)


def extract_features(trajectory: np.ndarray) -> TrajectoryFeatures:
    """Extract reusable descriptive features from a binary trajectory."""

    arr = _validate_trajectory(trajectory)
    row_density = arr.mean(axis=1)
    row_entropy = np.asarray([_binary_entropy(float(value)) for value in row_density])
    temporal_activity = (
        float(np.not_equal(arr[1:], arr[:-1]).mean()) if arr.shape[0] > 1 else 0.0
    )
    horizontal_edge_density = (
        float(np.not_equal(arr[:, 1:], arr[:, :-1]).mean()) if arr.shape[1] > 1 else 0.0
    )
    bilateral_symmetry = float(1.0 - np.not_equal(arr, arr[:, ::-1]).mean())
    components = np.asarray([_component_count(row) for row in arr], dtype=float)
    packed = np.packbits(arr, axis=None).tobytes()
    compressed = zlib.compress(packed, level=9)
    compression_ratio = float(len(compressed) / max(1, len(packed)))

    return TrajectoryFeatures(
        density=float(arr.mean()),
        row_entropy_mean=float(row_entropy.mean()),
        row_entropy_std=float(row_entropy.std()),
        temporal_activity=temporal_activity,
        horizontal_edge_density=horizontal_edge_density,
        bilateral_symmetry=bilateral_symmetry,
        mean_components=float(components.mean()),
        max_components=int(components.max(initial=0)),
        occupied_fraction=float(arr.any(axis=1).mean()),
        compression_ratio=compression_ratio,
    )


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(
        -probability * math.log2(probability)
        - (1.0 - probability) * math.log2(1.0 - probability)
    )


def _component_count(row: np.ndarray) -> int:
    if row.size == 0:
        return 0
    starts = row.astype(bool) & ~np.concatenate(([False], row[:-1].astype(bool)))
    return int(starts.sum())


def _validate_trajectory(trajectory: np.ndarray) -> np.ndarray:
    arr = np.asarray(trajectory, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("trajectory must be two-dimensional: steps x width.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("trajectory dimensions must be non-empty.")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("trajectory must contain only 0 and 1 values.")
    return arr
