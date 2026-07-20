"""Perturbation operators for cellular automata states."""

from __future__ import annotations

from typing import Literal

import numpy as np

PerturbationKind = Literal["bit_flip", "void", "random_mix"]


def perturb_state(
    state: np.ndarray,
    *,
    center: int | None = None,
    radius: int = 3,
    kind: PerturbationKind = "bit_flip",
    seed: int | None = None,
    noise_probability: float = 0.5,
) -> np.ndarray:
    """Apply a localized perturbation to a binary state."""

    arr = np.asarray(state, dtype=np.uint8).copy()
    if arr.ndim != 1:
        raise ValueError("state must be one-dimensional.")
    if arr.size == 0:
        raise ValueError("state must not be empty.")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("state must contain only 0 and 1 values.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if not 0 <= noise_probability <= 1:
        raise ValueError("noise_probability must be between 0 and 1.")

    if center is None:
        center = arr.size // 2
    if not 0 <= center < arr.size:
        raise ValueError("center must be within the state.")

    start = max(0, center - radius)
    end = min(arr.size, center + radius + 1)

    if kind == "bit_flip":
        arr[start:end] = 1 - arr[start:end]
    elif kind == "void":
        arr[start:end] = 0
    elif kind == "random_mix":
        rng = np.random.default_rng(seed)
        mask = rng.random(end - start) < noise_probability
        region = arr[start:end]
        region[mask] = 1 - region[mask]
        arr[start:end] = region
    else:
        raise ValueError("kind must be 'bit_flip', 'void', or 'random_mix'.")

    return arr
