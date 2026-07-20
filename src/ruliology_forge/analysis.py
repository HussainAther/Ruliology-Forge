"""Higher-level analysis utilities for Ruliology Forge."""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from math import sqrt
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import numpy as np


def summarize_scan(
    rows: Iterable[Mapping[str, object]],
    *,
    group_by: Sequence[str] = ("rule",),
    confidence: float = 0.95,
) -> list[dict[str, object]]:
    """Aggregate raw scan rows with confidence intervals and recovery rates."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_by)].append(row)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    output: list[dict[str, object]] = []
    for key, members in sorted(groups.items(), key=lambda item: item[0]):
        vals = np.asarray([float(m["restoration_coefficient"]) for m in members], dtype=float)
        recovery_times = [float(m["recovery_time"]) for m in members if m.get("recovery_time") not in (None, "")]
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        margin = z * std / sqrt(vals.size) if vals.size > 1 else 0.0
        row: dict[str, object] = {name: value for name, value in zip(group_by, key)}
        row.update({
            "n": int(vals.size),
            "mean_restoration": mean,
            "std_restoration": std,
            "median_restoration": float(np.median(vals)),
            "ci_lower": max(0.0, mean - margin),
            "ci_upper": min(1.0, mean + margin),
            "recovery_probability": float(sum(bool(m.get("recovered")) for m in members) / len(members)),
            "mean_recovery_time": float(np.mean(recovery_times)) if recovery_times else None,
            "mean_peak_divergence": float(np.mean([float(m["peak_divergence"]) for m in members])),
            "mean_final_scar_size": float(np.mean([float(m["final_scar_size"]) for m in members])),
        })
        output.append(row)
    return output


def parameter_grid(**parameters: Sequence[object]) -> list[dict[str, object]]:
    """Create a deterministic Cartesian product of parameter values."""
    names = list(parameters)
    if any(len(parameters[name]) == 0 for name in names):
        raise ValueError("parameter values cannot be empty.")
    return [dict(zip(names, values)) for values in product(*(parameters[name] for name in names))]
