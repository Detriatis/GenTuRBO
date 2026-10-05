"""Small shared primitives for protein publication tables and figures.

Budget selection and batch boundaries belong to the experiment's reader. These
helpers only extract recorded values, accumulate incumbents, and render them.
They deliberately do not import the model-dependent protein study package.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def row_score(row: Mapping[str, Any]) -> float | None:
    # Preserve privileged analysis-oracle precedence, including a nonfinite
    # first numeric value: it must not fall through to a different oracle.
    for key in ("score_for_analysis", "score", "normalized_fitness", "fitness", "reward"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def row_sequence(row: Mapping[str, Any]) -> str | list[Any] | None:
    for key in ("sequence_string", "decoded_sequence", "smiles", "sequence", "seq", "to_seq"):
        value = row.get(key)
        if isinstance(value, (str, list)) and value:
            return value
    return None


def incumbent_values(scores: Iterable[float | None], calls: Sequence[int]) -> np.ndarray:
    """Incumbents at one-based paid-call checkpoints; nonfinite scores do not win."""
    values = np.asarray(list(scores), dtype=float)
    running = np.maximum.accumulate(np.where(np.isfinite(values), values, -np.inf))
    selected: np.ndarray = running[np.asarray(calls, dtype=int) - 1]
    if not np.isfinite(selected).all():
        raise ValueError("Undefined incumbent at a requested paid-call checkpoint")
    return selected


def curve_statistics(runs: Mapping, seeds: Sequence[int], *, context: str) -> tuple[np.ndarray, ...]:
    calls = runs[seeds[0]][0]
    if any(not np.array_equal(runs[seed][0], calls) for seed in seeds):
        raise ValueError(f"Mismatched within-method call grids: {context}")
    matrix = np.stack([runs[seed][1] for seed in seeds])
    return calls, matrix.mean(axis=0), matrix.std(axis=0, ddof=1)


def write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_incumbent(
    axis: Any, calls: np.ndarray, mean: np.ndarray, sd: np.ndarray, *,
    colour: str, label: str, style: str, alpha: float, linewidth: float, zorder: int,
) -> None:
    """The same post-step mean and pointwise sample-SD band in both figures."""
    axis.fill_between(calls, mean - sd, mean + sd, step="post", color=colour, alpha=alpha, linewidth=0)
    axis.step(calls, mean, where="post", color=colour, linestyle=style,
              linewidth=linewidth, label=label, zorder=zorder)


def save_figure(figure: Any, output_dir: Path, stem: str, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight",
                   metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
