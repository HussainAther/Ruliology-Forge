"""Schema definitions for portable Ruliology Forge export bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


HOUDINI_SCHEMA_VERSION = "ruliology-houdini-v1"


@dataclass(frozen=True)
class CoordinateSystem:
    """Logical mapping from simulation dimensions to 3D coordinates."""

    x: str = "cell"
    y: str = "layer"
    z: str = "time"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class BundleFiles:
    """Relative paths for files contained in a Houdini export bundle."""

    manifest: str = "manifest.json"
    metrics: str = "metrics.json"
    trajectories: str = "trajectories.npz"
    control_points: str = "control_points.csv"
    perturbed_points: str = "perturbed_points.csv"
    difference_points: str = "difference_points.csv"
    readme: str = "README.md"
    schema_version: str = "schema_version.txt"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class HoudiniBundleManifest:
    """Serializable description of one exported perturbation experiment."""

    schema: str
    rule: int
    width: int
    steps: int
    boundary: str
    initial_condition: str
    initial_density: float
    seed: int | None
    initial_seed: int
    perturbation_seed: int
    perturbation: str
    perturbation_timing: str
    perturb_time: int
    perturb_center: int
    perturb_radius: int
    noise_probability: float
    recovery_threshold: float
    recovery_persistence: int
    max_shift: int
    coordinate_system: CoordinateSystem
    files: BundleFiles

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["coordinate_system"] = self.coordinate_system.to_dict()
        payload["files"] = self.files.to_dict()
        return payload
