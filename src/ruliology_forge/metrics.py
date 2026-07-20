"""Metrics for control-vs-perturbed cellular automata trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ResilienceSummary:
    """Compact, serializable summary of a recovery experiment."""

    restoration_coefficient: float
    final_restoration: float
    peak_divergence: float
    mean_divergence: float
    divergence_auc: float
    recovery_time: int | None
    recovered: bool
    final_scar_size: int
    scar_duration: int
    scar_spread: int
    scar_centroid_drift: float
    shift_tolerant_restoration: float
    best_final_shift: int

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


def xor_difference(control: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    control, perturbed = _validate_pair(control, perturbed)
    return np.bitwise_xor(control, perturbed).astype(np.uint8)


def hamming_divergence(control: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    return xor_difference(control, perturbed).mean(axis=1)


def restoration_coefficient(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
) -> float:
    divergence = hamming_divergence(control, perturbed)
    start, stop = _validate_window(divergence, start, stop)
    return float(1.0 - divergence[start:stop].mean())


def final_restoration(control: np.ndarray, perturbed: np.ndarray) -> float:
    return float(1.0 - hamming_divergence(control, perturbed)[-1])


def divergence_auc(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
    normalize: bool = True,
) -> float:
    divergence = hamming_divergence(control, perturbed)
    return _divergence_auc_from_vector(divergence, start=start, stop=stop, normalize=normalize)


def recovery_time(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    start: int = 0,
    threshold: float = 0.01,
    persistence: int = 5,
) -> int | None:
    divergence = hamming_divergence(control, perturbed)
    return _recovery_time_from_vector(
        divergence,
        start=start,
        threshold=threshold,
        persistence=persistence,
    )


def scar_metrics(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    start: int = 0,
) -> dict[str, int | float]:
    """Measure the size, persistence, spread, and motion of the difference region."""

    diff = xor_difference(control, perturbed)
    if not 0 <= start < diff.shape[0]:
        raise ValueError("start must be within the trajectory.")
    post = diff[start:]
    occupied = post.any(axis=1)
    widths: list[int] = []
    centroids: list[float] = []
    for row in post:
        indices = np.flatnonzero(row)
        if indices.size:
            widths.append(int(indices[-1] - indices[0] + 1))
            centroids.append(float(indices.mean()))
    drift = 0.0 if len(centroids) < 2 else abs(centroids[-1] - centroids[0])
    return {
        "final_scar_size": int(post[-1].sum()),
        "scar_duration": int(occupied.sum()),
        "scar_spread": max(widths, default=0),
        "scar_centroid_drift": float(drift),
    }


def shift_tolerant_divergence(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    max_shift: int = 10,
    boundary: str = "periodic",
) -> tuple[np.ndarray, np.ndarray]:
    """Return minimum divergence and best shift for each row.

    Positive shifts roll the perturbed row to the right before comparison.
    For fixed boundaries, shifted-in cells are filled with zeros.
    """

    control, perturbed = _validate_pair(control, perturbed)
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative.")
    if boundary not in {"periodic", "fixed"}:
        raise ValueError("boundary must be 'periodic' or 'fixed'.")

    shifts = np.arange(-max_shift, max_shift + 1, dtype=int)
    scores = np.empty((shifts.size, control.shape[0]), dtype=float)
    for i, shift in enumerate(shifts):
        shifted = _shift_rows(perturbed, int(shift), boundary)
        scores[i] = np.not_equal(control, shifted).mean(axis=1)
    best_indices = scores.argmin(axis=0)
    times = np.arange(control.shape[0])
    return scores[best_indices, times], shifts[best_indices]


def summarize_resilience(
    control: np.ndarray,
    perturbed: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
    recovery_threshold: float = 0.01,
    recovery_persistence: int = 5,
    max_shift: int = 10,
    boundary: str = "periodic",
) -> ResilienceSummary:
    """Calculate a standard bundle of complementary resilience metrics."""

    divergence = hamming_divergence(control, perturbed)
    start, stop = _validate_window(divergence, start, stop)
    window = divergence[start:stop]
    rt = _recovery_time_from_vector(
        divergence,
        start=start,
        threshold=recovery_threshold,
        persistence=recovery_persistence,
    )
    scars = scar_metrics(control, perturbed, start=start)
    shifted_divergence, best_shifts = shift_tolerant_divergence(
        control,
        perturbed,
        max_shift=max_shift,
        boundary=boundary,
    )
    return ResilienceSummary(
        restoration_coefficient=float(1.0 - window.mean()),
        final_restoration=float(1.0 - divergence[-1]),
        peak_divergence=float(window.max()),
        mean_divergence=float(window.mean()),
        divergence_auc=_divergence_auc_from_vector(divergence, start=start, stop=stop),
        recovery_time=rt,
        recovered=rt is not None,
        shift_tolerant_restoration=float(1.0 - shifted_divergence[start:stop].mean()),
        best_final_shift=int(best_shifts[-1]),
        **scars,
    )


def _shift_rows(rows: np.ndarray, shift: int, boundary: str) -> np.ndarray:
    if shift == 0:
        return rows
    if boundary == "periodic":
        return np.roll(rows, shift, axis=1)
    shifted = np.zeros_like(rows)
    if shift > 0:
        shifted[:, shift:] = rows[:, :-shift]
    else:
        shifted[:, :shift] = rows[:, -shift:]
    return shifted


def _divergence_auc_from_vector(
    divergence: np.ndarray,
    *,
    start: int,
    stop: int | None,
    normalize: bool = True,
) -> float:
    start, stop = _validate_window(divergence, start, stop)
    window = divergence[start:stop]
    area = float(np.trapezoid(window)) if window.size > 1 else float(window[0])
    if normalize and window.size > 1:
        area /= window.size - 1
    return area


def _recovery_time_from_vector(
    divergence: np.ndarray,
    *,
    start: int,
    threshold: float,
    persistence: int,
) -> int | None:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if persistence <= 0:
        raise ValueError("persistence must be positive.")
    if not 0 <= start < divergence.size:
        raise ValueError("start must be within the trajectory.")
    below = divergence[start:] <= threshold
    if below.size < persistence:
        return None
    run = np.convolve(below.astype(np.int64), np.ones(persistence, dtype=np.int64), mode="valid")
    matches = np.flatnonzero(run == persistence)
    return int(matches[0]) if matches.size else None


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


def _validate_window(divergence: np.ndarray, start: int, stop: int | None) -> tuple[int, int]:
    if stop is None:
        stop = divergence.size
    if not 0 <= start < stop <= divergence.size:
        raise ValueError("invalid start/stop recovery window.")
    return start, stop
