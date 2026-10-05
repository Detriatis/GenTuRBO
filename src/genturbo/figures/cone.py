#!/usr/bin/env python3
"""Regroup the existing Hamming-ball visitation figure into a feasibility cone.

This is a display-only transformation of the saved exhaustive support and
100,000 generator draws, not a new sampling run. Each sequence has the same
coordinate in both panels. The left distribution is uniform on the *whole*
radius-two ball, including its centre, not a two-edit-only mutation operator.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from genturbo.figures.visitation import (
    BallState,
    HammingBallVisitationError,
    load_data,
    radial_coordinates,
)

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Wedge


DEFAULT_INPUT = Path("runs/ehrlich_radial_search_v1/hamming_ball_visitation")
DEFAULT_OUTPUT = Path("output/pdf/ehrlich_infeasible_cone/ehrlich_infeasible_cone")


def cone_coordinates(
    records: Sequence[BallState],
    *,
    parent: Sequence[int],
    vocab_size: int,
    cone_angle: float = 110.0,
    cone_centre: float = 0.0,
) -> dict[tuple[int, ...], tuple[float, float]]:
    """Keep radial lanes, packing invalid states into one shared angular sector.

    Within each lane and feasibility group, preserve the old circular order.
    No visit counts enter this mapping, so density cannot influence geometry.
    Angles are schematic, while centre/one-edit/two-edit shells are unchanged.
    """
    if not 10.0 <= cone_angle <= 350.0 or not math.isfinite(cone_centre):
        raise HammingBallVisitationError("Cone angle must be in [10, 350] degrees")
    if len({record.sequence for record in records}) != len(records):
        raise HammingBallVisitationError("Support contains repeated sequences")
    roots = [record for record in records if record.distance == 0]
    if len(roots) != 1 or not roots[0].feasible or roots[0].sequence != tuple(parent):
        raise HammingBallVisitationError("The cone layout requires one feasible incumbent")

    original = radial_coordinates(records, parent=parent, vocab_size=vocab_size)
    groups = defaultdict(list)
    result = {tuple(parent): (0.0, 0.0)}
    start = math.radians(cone_centre - cone_angle / 2.0)
    width = math.radians(cone_angle)
    for record in records:
        if record.distance == 0:
            continue
        x, y = original[record.sequence]
        # Rounding joins values which differ only by trig round-off.
        radius = round(math.hypot(x, y), 12)
        phase = (math.atan2(y, x) - start) % (2.0 * math.pi)
        groups[(radius, record.feasible)].append((phase, record.sequence))

    padding = math.radians(1.5)
    for (radius, feasible), entries in groups.items():
        entries.sort()
        beginning = start + width if feasible else start
        span = 2.0 * math.pi - width if feasible else width
        for index, (_, sequence) in enumerate(entries):
            angle = beginning + padding + (index + 0.5) * (span - 2 * padding) / len(entries)
            result[sequence] = (radius * math.cos(angle), radius * math.sin(angle))
    if set(result) != {record.sequence for record in records}:
        raise HammingBallVisitationError("The cone layout lost a support state")
    return result


def proposal_probabilities(records: Sequence[BallState]) -> tuple[np.ndarray, np.ndarray]:
    """Normalise over all states, including invalid draws and returns to parent."""
    if not records:
        raise HammingBallVisitationError("The support is empty")
    totals = {record.num_samples for record in records}
    if len(totals) != 1 or next(iter(totals)) <= 0:
        raise HammingBallVisitationError("A shared positive sample total is required")
    if any(record.visit_count < 0 for record in records):
        raise HammingBallVisitationError("Visit counts cannot be negative")
    if sum(record.visit_count for record in records) != records[0].num_samples:
        raise HammingBallVisitationError("Visit counts do not sum to the sample total")
    uniform = np.full(len(records), 1.0 / len(records))
    generator = np.asarray([record.visit_probability for record in records])
    return uniform, generator


def comparison_summary(records: Sequence[BallState]) -> dict:
    uniform, generator = proposal_probabilities(records)
    invalid = np.asarray([not record.feasible for record in records])
    uniform_invalid = float(uniform[invalid].sum())
    generator_invalid = float(generator[invalid].sum())
    return {
        "support_size": len(records),
        "infeasible_support_count": int(invalid.sum()),
        "generator_sample_count": records[0].num_samples,
        "infeasible_generator_draws": sum(r.visit_count for r in records if not r.feasible),
        "infeasible_designs_visited": sum(r.visit_count > 0 and not r.feasible for r in records),
        "uniform_infeasible_probability": uniform_invalid,
        "generator_infeasible_probability": generator_invalid,
        "relative_reduction": 1.0 - generator_invalid / uniform_invalid if uniform_invalid else None,
        "uniform_to_generator_ratio": uniform_invalid / generator_invalid if generator_invalid else None,
        "generator_return_to_parent_probability": float(sum(r.visit_probability for r in records if r.distance == 0)),
        "comparison": "Exact uniform distribution on all radius-two support states vs empirical generator draws",
        "denominator": "All outcomes; invalid draws and returns to the incumbent are included",
        "scope": "One fixed incumbent and one saved generator sampling seed; not an optimisation benchmark",
    }


def generator_visibility(records: Sequence[BallState], cutoff_multiple: float = 0.0) -> np.ndarray:
    """Highlight visits with empirical probability >= cutoff_multiple / |support|.

    The cutoff affects only the right panel's foreground, never the full support,
    probability normalisation, colours, coordinates or infeasible-mass summaries.
    At zero, preserve the original behaviour of highlighting every visited state.
    """
    if not math.isfinite(cutoff_multiple) or cutoff_multiple < 0:
        raise HammingBallVisitationError("Cutoff multiple must be finite and non-negative")
    proposal_probabilities(records)  # Validate sample totals before applying a cutoff.
    counts = np.asarray([record.visit_count for record in records], dtype=np.int64)
    # Cross-multiplication includes exact equality at 1/N without division error.
    return (counts > 0) & (counts * len(records) >= cutoff_multiple * records[0].num_samples)


def cutoff_summary(records: Sequence[BallState], cutoff_multiple: float) -> dict:
    visible = generator_visibility(records, cutoff_multiple)
    _, probabilities = proposal_probabilities(records)
    invalid = np.asarray([not record.feasible for record in records])
    displayed_mass = float(probabilities[visible].sum())
    return {
        "cutoff_multiple_of_uniform": cutoff_multiple,
        "cutoff_probability": cutoff_multiple / len(records),
        "minimum_visits": max(1, math.ceil(cutoff_multiple * records[0].num_samples / len(records))),
        "highlighted_designs": int(visible.sum()),
        "highlighted_infeasible_designs": int((visible & invalid).sum()),
        "highlighted_probability_mass": displayed_mass,
        "hidden_probability_mass": float(probabilities[~visible].sum()),
        "highlighted_infeasible_probability_mass": float(probabilities[visible & invalid].sum()),
        "hidden_infeasible_probability_mass": float(probabilities[~visible & invalid].sum()),
        "rule": "Right-panel overlay only: empirical probability >= cutoff_multiple / full support size; no renormalisation",
    }


def plot_cone(
    records: Sequence[BallState],
    *,
    parent: Sequence[int],
    vocab_size: int,
    output_prefix: Path,
    cone_angle: float = 110.0,
    cone_centre: float = 0.0,
    dpi: int = 320,
    clean: bool = False,
    show_colourbars: bool = True,
    show_infeasible_bars: bool = True,
    cutoff_multiple: float = 0.0,
    paper_width: float | None = None,
) -> tuple[Path, Path]:
    expected = 1 + len(parent) * (vocab_size - 1) + math.comb(len(parent), 2) * (vocab_size - 1) ** 2
    if len(records) != expected:
        raise HammingBallVisitationError(f"Expected the complete {expected}-state Hamming ball")
    coordinates = cone_coordinates(records, parent=parent, vocab_size=vocab_size,
                                   cone_angle=cone_angle, cone_centre=cone_centre)
    distributions = proposal_probabilities(records)
    generator_foreground = generator_visibility(records, cutoff_multiple)
    summary = comparison_summary(records)
    xy = np.asarray([coordinates[record.sequence] for record in records])
    valid = np.asarray([record.feasible for record in records])
    distances = np.asarray([record.distance for record in records])
    positive = np.concatenate([p[p > 0] for p in distributions])
    norm = mcolors.LogNorm(vmin=min(1.0 / records[0].num_samples, float(positive.min())),
                          vmax=max(float(positive.max()), 2.0 / records[0].num_samples))
    feasible_cmap = matplotlib.colormaps["viridis"]
    invalid_cmap = mcolors.LinearSegmentedColormap.from_list(
        "infeasible_red", ["#e7a9ac", "#d96870", "#b3263a", "#63031b"]
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    if clean:
        # The wheels keep the same physical size in the two clean variants.
        # Remove the unused legend/footer space instead of leaving a blank band.
        height = 5.7 if show_colourbars else (5.15 if paper_width else 5.0)
        bottom_inches = 1.2 if show_colourbars else (0.65 if paper_width else 0.5)
        top, bottom = (bottom_inches + 4.4) / height, bottom_inches / height
        offsets = (0.17, 0.39, 0.58) if paper_width else (0.14, 0.31, 0.43)
        percent_y, counts_y, bar_y = [(bottom_inches - offset) / height for offset in offsets]
        colourbar_x, colourbar_width = 0.35, 0.30
        colourbar_ys = (0.45 / height, 0.25 / height)
        bar_height, colourbar_height = 0.0585 / height, 0.0975 / height
    else:
        height, top, bottom = 6.5, 0.93, 0.265
        percent_y, counts_y, bar_y = 0.237, 0.210, 0.185
        colourbar_x, colourbar_width = 0.58, 0.30
        colourbar_ys = (0.123, 0.091)
        bar_height, colourbar_height = 0.009, 0.015
    figure, axes = plt.subplots(1, 2, figsize=(11.0, height))
    figure.subplots_adjust(left=0.025, right=0.975, top=top, bottom=bottom, wspace=0.08)
    titles = ("(a) Uniform Hamming-ball sampling", "(b) Generator visitation")
    masses = (summary["uniform_infeasible_probability"], summary["generator_infeasible_probability"])
    start = cone_centre - cone_angle / 2.0
    for index, (axis, probabilities) in enumerate(zip(axes, distributions, strict=True)):
        axis.set(aspect="equal", xlim=(-1.03, 1.03), ylim=(-1.03, 1.03))
        axis.axis("off")
        if not clean:
            axis.set_title(titles[index], fontsize=11.5, fontweight="semibold", pad=10)
        axis.add_patch(Wedge((0, 0), 0.97, start, start + cone_angle,
                             facecolor="#fff9f9", edgecolor="none", zorder=0))
        for radius in (0.39, 0.97):
            axis.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor="#d2d8db",
                                     linewidth=0.55, zorder=0))
        for angle in (start, start + cone_angle):
            theta = math.radians(angle)
            axis.plot([0, 0.97 * math.cos(theta)], [0, 0.97 * math.sin(theta)],
                      color="#b94c59", linewidth=0.65, alpha=0.7, zorder=1)
        # Identical pale support remains visible, including unvisited invalid states.
        for mask, color in ((valid, "#d8dcde"), (~valid, "#f0dfe1")):
            marker_sizes = np.where(distances[mask] == 1, 3.4, 0.62)
            axis.scatter(xy[mask, 0], xy[mask, 1], s=marker_sizes, color=color,
                         linewidths=0, rasterized=True, zorder=2)
        foreground = probabilities > 0 if index == 0 else generator_foreground
        for mask, cmap in ((valid & foreground, feasible_cmap),
                           (~valid & foreground, invalid_cmap)):
            # Identical log colour/size transforms in both panels; no within-group normalisation.
            scaled = np.asarray(norm(probabilities[mask]))
            sizes = np.where(distances[mask] == 1, 3.0 + 13.0 * scaled,
                             0.50 + 3.2 * scaled)
            sizes = np.where(distances[mask] == 0, 24.0, sizes)
            axis.scatter(xy[mask, 0], xy[mask, 1], s=sizes, c=probabilities[mask],
                         cmap=cmap, norm=norm, linewidths=0, rasterized=True, zorder=3)
        position = axis.get_position()
        centre = position.x0 + position.width / 2
        figure.text(centre, percent_y, f"{100 * masses[index]:.2f}% infeasible",
                    ha="center", color="#a62136", fontsize=8 * 11 / paper_width if paper_width else 13,
                    fontweight="semibold")
        denominator = (f"{summary['infeasible_support_count']:,} / {len(records):,} designs"
                       if index == 0 else f"{summary['infeasible_generator_draws']:,} / {records[0].num_samples:,} proposals")
        figure.text(centre, counts_y, denominator, ha="center",
                    fontsize=7 * 11 / paper_width if paper_width else 8.3, color="#626b70")
        if show_infeasible_bars:
            bar = figure.add_axes((centre - 0.125, bar_y, 0.25, bar_height))
            bar.barh([0], [1], color="#edf0f1", height=1)
            bar.barh([0], [masses[index]], color="#bd3549", height=1)
            bar.set(xlim=(0, 1), ylim=(-0.5, 0.5))
            bar.axis("off")

    if not clean:
        figure.text(0.09, 0.125, "Faint dots retain every possible design.", fontsize=8.6, color="#596166")
        figure.text(0.09, 0.098, "Red sector groups all infeasible designs.", fontsize=8.6, color="#a62136")
        if show_infeasible_bars:
            figure.text(0.09, 0.071, "Bar length shows total infeasible probability.", fontsize=8.6, color="#596166")
        figure.text(0.5, 0.015, "Same coordinates in both panels; angular position is a schematic feasibility grouping.",
                    ha="center", fontsize=8.1, color="#596166")
    colourbar_specs = zip(colourbar_ys, (feasible_cmap, invalid_cmap), ("Feasible", "Infeasible"), strict=True)
    for y, cmap, label in colourbar_specs if show_colourbars else ():
        colorbar_axis = figure.add_axes((colourbar_x, y, colourbar_width, colourbar_height))
        cb = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis,
                            orientation="horizontal")
        cb.outline.set_linewidth(0.4)
        cb.ax.minorticks_off()
        if label == "Feasible":
            cb.set_ticks([])
        else:
            ticks = [p for p in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1) if norm.vmin <= p <= norm.vmax]
            cb.set_ticks(ticks, labels=[f"{100 * p:g}%" for p in ticks])
            cb.ax.tick_params(labelsize=7, length=2, width=0.45, pad=2)
            cb.set_label("Probability per design (shared log scale)", fontsize=8.1, labelpad=3)
        figure.text(colourbar_x + colourbar_width + 0.015, y + colourbar_height / 2,
                    label, va="center", fontsize=7.8, color="#596166")
    prefix = output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = prefix.with_suffix(".png"), prefix.with_suffix(".pdf")
    for path in paths:
        figure.savefig(path, dpi=dpi, bbox_inches=None if paper_width else "tight", facecolor="white")
    plt.close(figure)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cone-angle", type=float, default=110.0)
    parser.add_argument("--cone-centre", type=float, default=0.0)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--clean", action="store_true", help="Remove panel titles and explanatory notes")
    parser.add_argument("--no-colourbars", action="store_true", help="Omit probability colour scales")
    parser.add_argument("--no-infeasible-bars", action="store_true",
                        help="Omit infeasible-probability bars, retaining percentages and counts")
    parser.add_argument("--paper-width", type=float,
                        help="Placed width in inches: keep headline/count text at 8/7 pt after scaling")
    parser.add_argument("--cutoff-multiple", type=float, default=0.0,
                        help="Highlight generator p >= this multiple of 1/support_size; left reference stays unchanged")
    args = parser.parse_args(argv)
    if args.paper_width is not None and (not math.isfinite(args.paper_width) or args.paper_width <= 0
                                       or not args.clean or not args.no_colourbars):
        parser.error("--paper-width needs a positive width, --clean and --no-colourbars")
    source = args.input_dir / "hamming_ball_visitation.csv"
    metadata = json.loads((args.input_dir / "hamming_ball_visitation.json").read_text())
    records = load_data(source)
    png, pdf = plot_cone(records, parent=metadata["parent_sequence"], vocab_size=metadata["vocab_size"],
                        output_prefix=args.output_prefix, cone_angle=args.cone_angle,
                        cone_centre=args.cone_centre, dpi=args.dpi, clean=args.clean,
                        show_colourbars=not args.no_colourbars,
                        show_infeasible_bars=not args.no_infeasible_bars,
                        cutoff_multiple=args.cutoff_multiple,
                        paper_width=args.paper_width)
    summary = comparison_summary(records)
    summary.update(source_table=str(source.resolve()), source_metadata=metadata,
                   cone_angle_degrees=args.cone_angle, cone_centre_degrees=args.cone_centre,
                   layout="Shared radial lanes; angular order regrouped by feasibility independently of visit counts",
                   clean_layout=args.clean, show_colourbars=not args.no_colourbars,
                   show_infeasible_bars=not args.no_infeasible_bars,
                   display_cutoff=cutoff_summary(records, args.cutoff_multiple),
                   png=str(png), pdf=str(pdf))
    png.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("uniform_infeasible_probability", "generator_infeasible_probability",
                                                   "uniform_to_generator_ratio", "relative_reduction")}, indent=2))
    print(f"Wrote {png}\nWrote {pdf}")
    print(json.dumps(summary["display_cutoff"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
