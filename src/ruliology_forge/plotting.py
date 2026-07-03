"""Plotting helpers for cellular automata trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(
    trajectory: np.ndarray,
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """Plot a trajectory or difference map as a binary image."""

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(trajectory, interpolation="nearest", aspect="auto")
    ax.set_xlabel("Cell index")
    ax.set_ylabel("Time step")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig, ax


def plot_divergence(
    divergence: np.ndarray,
    *,
    title: str = "Divergence over time",
    save_path: str | Path | None = None,
):
    """Plot normalized Hamming divergence over time."""

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(divergence)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized Hamming divergence")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig, ax
