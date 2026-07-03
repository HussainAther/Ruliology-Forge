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
) -> np.ndarray:
    """Apply a localized perturbation to a binary state.

    Parameters
    ----------
    state:
        One-dimensional binary state.
    center:
        Center index of the perturbation. Defaults to the middle of the state.
    radius:
        Number of cells on each side of the center to perturb. The affected
        interval has width ``2 * radius + 1`` before clipping at boundaries.
    kind:
        ``bit_flip``, ``void``, or ``random_mix``.
    """

    arr = np.asarray(state, dtype=np.uint8).copy()
    if arr.ndim != 1:
        raise ValueError("state must be one-dimensional.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")

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
        arr[start:end] = rng.integers(0, 2, size=end - start, dtype=np.uint8)
    else:
        raise ValueError("kind must be 'bit_flip', 'void', or 'random_mix'.")

    return arr
