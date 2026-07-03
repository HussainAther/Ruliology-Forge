"""Elementary cellular automata simulation tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BoundaryMode = Literal["periodic", "fixed"]
InitialCondition = Literal["single", "random"]


@dataclass(frozen=True)
class ECARule:
    """Elementary cellular automaton rule.

    Parameters
    ----------
    number:
        Wolfram ECA rule number, from 0 to 255.
    """

    number: int

    def __post_init__(self) -> None:
        if not 0 <= self.number <= 255:
            raise ValueError("ECA rule number must be between 0 and 255.")

    @property
    def table(self) -> np.ndarray:
        """Return lookup table indexed by neighborhood code 0..7.

        Neighborhood code is interpreted as binary ``111`` -> 7, ``110`` -> 6,
        ..., ``000`` -> 0, matching the usual Wolfram rule convention.
        """

        return np.array([(self.number >> i) & 1 for i in range(8)], dtype=np.uint8)

    def step(self, state: np.ndarray, boundary: BoundaryMode = "periodic") -> np.ndarray:
        """Advance a binary state by one generation."""

        state = _validate_state(state)
        if boundary == "periodic":
            left = np.roll(state, 1)
            center = state
            right = np.roll(state, -1)
        elif boundary == "fixed":
            padded = np.pad(state, 1, mode="constant", constant_values=0)
            left = padded[:-2]
            center = padded[1:-1]
            right = padded[2:]
        else:
            raise ValueError("boundary must be 'periodic' or 'fixed'.")

        codes = (left << 2) | (center << 1) | right
        return self.table[codes]


def initial_state(
    width: int,
    mode: InitialCondition = "single",
    *,
    density: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Create an initial binary state."""

    if width <= 0:
        raise ValueError("width must be positive.")

    if mode == "single":
        state = np.zeros(width, dtype=np.uint8)
        state[width // 2] = 1
        return state

    if mode == "random":
        if not 0 <= density <= 1:
            raise ValueError("density must be between 0 and 1.")
        rng = np.random.default_rng(seed)
        return (rng.random(width) < density).astype(np.uint8)

    raise ValueError("mode must be 'single' or 'random'.")


def simulate_eca(
    rule: int,
    width: int,
    steps: int,
    *,
    init: np.ndarray | None = None,
    initial_condition: InitialCondition = "single",
    boundary: BoundaryMode = "periodic",
    seed: int | None = None,
) -> np.ndarray:
    """Simulate an elementary cellular automaton.

    Returns an array with shape ``(steps, width)``. Row 0 is the initial state.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")

    if init is None:
        state = initial_state(width, initial_condition, seed=seed)
    else:
        state = _validate_state(init).copy()
        width = state.size

    trajectory = np.zeros((steps, width), dtype=np.uint8)
    trajectory[0] = state
    eca_rule = ECARule(rule)

    for t in range(1, steps):
        trajectory[t] = eca_rule.step(trajectory[t - 1], boundary=boundary)

    return trajectory


def _validate_state(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError("state must be one-dimensional.")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("state must contain only 0 and 1 values.")
    return arr
