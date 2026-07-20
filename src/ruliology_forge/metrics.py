"""Metrics for control-vs-perturbed cellular automata trajectories."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class ResilienceSummary:
    """Compact, serializable summary of a recovery experiment.

    ``recovery_time`` is measured from ``start`` and is ``None`` when the
    trajectory never remains below the requested divergence threshold.
    """

    restoration_coefficient: float
    final_restoration: float
    peak_divergence: float
    mean_divergence: float
    divergence_auc: float
    recovery_time: int | None
    recovered: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        """Return a JSON/CSV-friendly representation."""

        return asdict(self)


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
    start, stop = _validate_window(divergence, start, stop)
    return float(1.0 - divergence[start:stop].mean())


def final_restoration(control: np.ndarray, perturbed: np.ndarray) -> float:
    """Return final-step restoration score: ``1 - D(t_final)``."""

    return float(1.0 - hamming_divergence(control, perturbed)[-1])


def divergence_auc(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
    normalize: bool = True,
) -> float:
    """Return area under the divergence curve over a recovery window.

    By default the area is divided by the number of sampled time steps, making
    scores comparable across experiments of different duration. Lower is better.
    """

    divergence = hamming_divergence(control, perturbed)
    start, stop = _validate_window(divergence, start, stop)
    window = divergence[start:stop]
    area = (
        float((window[:-1] + window[1:]).sum() * 0.5)
        if window.size > 1
        else float(window[0])
    )
    if normalize and window.size > 1:
        area /= window.size - 1
    return area


def recovery_time(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    start: int = 0,
    threshold: float = 0.01,
    persistence: int = 5,
) -> int | None:
    """Return steps until divergence recovers and stays below ``threshold``.

    A recovery is accepted only when at least ``persistence`` consecutive samples
    are at or below the threshold. The returned value is relative to ``start``.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if persistence <= 0:
        raise ValueError("persistence must be positive.")

    divergence = hamming_divergence(control, perturbed)
    if not 0 <= start < divergence.size:
        raise ValueError("start must be within the trajectory.")

    below = divergence[start:] <= threshold
    if below.size < persistence:
        return None

    run = np.convolve(below.astype(np.int64), np.ones(persistence, dtype=np.int64), mode="valid")
    matches = np.flatnonzero(run == persistence)
    return int(matches[0]) if matches.size else None


def summarize_resilience(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
    recovery_threshold: float = 0.01,
    recovery_persistence: int = 5,
) -> ResilienceSummary:
    """Calculate a standard bundle of complementary resilience metrics."""

    divergence = hamming_divergence(control, perturbed)
    start, stop = _validate_window(divergence, start, stop)
    window = divergence[start:stop]
    rt = recovery_time(
        control,
        perturbed,
        start=start,
        threshold=recovery_threshold,
        persistence=recovery_persistence,
    )
    return ResilienceSummary(
        restoration_coefficient=float(1.0 - window.mean()),
        final_restoration=float(1.0 - divergence[-1]),
        peak_divergence=float(window.max()),
        mean_divergence=float(window.mean()),
        divergence_auc=divergence_auc(control, perturbed, start=start, stop=stop),
        recovery_time=rt,
        recovered=rt is not None,
    )


def _validate_pair(control: np.ndarray, perturbed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(control, dtype=np.uint8)
    p = np.asarray(perturbed, dtype=np.uint8)
    if c.shape != p.shape:
        raise ValueError("control and perturbed arrays must have the same shape.")
    if c.ndim != 2:
        raise ValueError("trajectories must be two-dimensional: steps x width.")
    if not np.isin(c, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("trajectories must contain only 0 and 1 values.")
    return c, p


def _validate_window(
    divergence: np.ndarray, start: int, stop: int | None
) -> tuple[int, int]:
    if stop is None:
        stop = divergence.size
    if not 0 <= start < stop <= divergence.size:
        raise ValueError("invalid start/stop recovery window.")
    return start, stop
