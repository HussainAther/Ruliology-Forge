"""Metrics for control-vs-perturbed cellular automata trajectories."""

from __future__ import annotations

import numpy as np


def xor_difference(control: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """Return binary XOR difference map between two trajectories."""

    control, perturbed = _validate_pair(control, perturbed)
    return np.bitwise_xor(control, perturbed).astype(np.uint8)


def hamming_divergence(control: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """Return normalized Hamming divergence for each time step."""

    diff = xor_difference(control, perturbed)
    return diff.mean(axis=1)


def restoration_coefficient(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
) -> float:
    """Compute ``R = 1 - mean(D(t))`` over a recovery window."""

    divergence = hamming_divergence(control, perturbed)
    if stop is None:
        stop = divergence.size
    if not 0 <= start < stop <= divergence.size:
        raise ValueError("invalid start/stop recovery window.")
    return float(1.0 - divergence[start:stop].mean())


def final_restoration(control: np.ndarray, perturbed: np.ndarray) -> float:
    """Return final-step restoration score: ``1 - D(t_final)``."""

    return float(1.0 - hamming_divergence(control, perturbed)[-1])


def _validate_pair(control: np.ndarray, perturbed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(control, dtype=np.uint8)
    p = np.asarray(perturbed, dtype=np.uint8)
    if c.shape != p.shape:
        raise ValueError("control and perturbed arrays must have the same shape.")
    if c.ndim != 2:
        raise ValueError("trajectories must be two-dimensional: steps x width.")
    return c, p
