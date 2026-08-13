"""Plotting helpers for experiment runs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_xy(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series_name: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", label=series_name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if series_name:
        ax.legend()
    _save(fig, path)


def plot_multi(
    xs: Sequence[float],
    series: dict[str, Sequence[float]],
    *,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)
