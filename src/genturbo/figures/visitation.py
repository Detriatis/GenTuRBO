#!/usr/bin/env python3
"""Show a complete radius-two Hamming ball and generator visitation density.

The left panel contains every sequence within Hamming distance two of one
length-12 Ehrlich incumbent.  The right panel keeps the identical support and
coordinates, then overlays endpoints from the frozen generator after exactly
two positions are corrupted and only those positions are denoised.  This keeps
search-space size and proposal density as separate visual quantities.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/genturbo-cache"))
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/genturbo-mplconfig"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from genturbo.figures.one_edit import choose_incumbent, load_archive  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from core.utils import seed_everything  # noqa: E402
from ehrlich.ehrlich_reward import EhrlichReward  # noqa: E402
from ehrlich.ehrlich_score import is_feasible  # noqa: E402
from flows.ctmc_adapter import CTMCFlowAdapter  # noqa: E402
from processes.dfm_mask import _build_time_grid  # noqa: E402
from runners.common import restore_process_from_checkpoint  # noqa: E402

DEFAULT_SEARCH_CONFIG = Path(
    "configs/experiments/ehrlich_radial_search_v1/generator_one_edit.json"
)
DEFAULT_OUTPUT_DIR = Path("runs/ehrlich_radial_search_v1/hamming_ball_visitation")
DEFAULT_OUTPUT_PREFIX = "ehrlich_hamming_ball_visitation"
CSV_COLUMNS = (
    "sequence",
    "hamming_distance",
    "changed_positions_0based",
    "feasible",
    "visit_count",
    "visit_probability",
    "num_generator_samples",
)

SequenceTuple = tuple[int, ...]


class HammingBallVisitationError(ValueError):
    """Raised when the radius-two support or visitation data are inconsistent."""


@dataclass(frozen=True)
class BallState:
    sequence: SequenceTuple
    distance: int
    changed_positions: tuple[int, ...]
    feasible: bool
    visit_count: int
    num_samples: int

    @property
    def visit_probability(self) -> float:
        return self.visit_count / float(self.num_samples)


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != len(right):
        raise HammingBallVisitationError("Cannot compare sequences of different lengths")
    return sum(a != b for a, b in zip(left, right, strict=True))


def enumerate_radius_two_ball(
    parent: Sequence[int],
    *,
    vocab_size: int,
) -> tuple[SequenceTuple, ...]:
    """Enumerate every unique sequence at Hamming distance zero, one, or two."""

    root = tuple(int(token) for token in parent)
    if not root:
        raise HammingBallVisitationError("The parent sequence must be non-empty")
    if vocab_size <= 1:
        raise HammingBallVisitationError("vocab_size must be greater than one")
    if min(root) < 0 or max(root) >= vocab_size:
        raise HammingBallVisitationError("The parent contains a token outside the vocabulary")

    states: list[SequenceTuple] = [root]
    alternatives = [
        tuple(token for token in range(vocab_size) if token != root[position])
        for position in range(len(root))
    ]
    for position in range(len(root)):
        for token in alternatives[position]:
            child = list(root)
            child[position] = token
            states.append(tuple(child))
    for first, second in itertools.combinations(range(len(root)), 2):
        for first_token in alternatives[first]:
            for second_token in alternatives[second]:
                child = list(root)
                child[first] = first_token
                child[second] = second_token
                states.append(tuple(child))

    expected = (
        1
        + len(root) * (vocab_size - 1)
        + math.comb(len(root), 2) * (vocab_size - 1) ** 2
    )
    if len(states) != expected or len(set(states)) != expected:
        raise HammingBallVisitationError("Radius-two enumeration produced duplicate states")
    return tuple(states)


def draw_exact_corruptions(
    parent: torch.Tensor,
    *,
    batch_size: int,
    vocab_size: int,
    num_corruptions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a uniform interpolant conditioned on an exact corruption count."""

    root = parent.to(device=device, dtype=torch.long).view(1, -1)
    sequence_length = int(root.shape[1])
    if not 1 <= num_corruptions <= sequence_length:
        raise HammingBallVisitationError("num_corruptions must lie in [1, sequence_length]")
    if batch_size <= 0:
        raise HammingBallVisitationError("batch_size must be positive")

    # IID continuous priorities induce a uniform subset without replacement.
    positions = torch.rand((batch_size, sequence_length), device=device).topk(
        num_corruptions,
        dim=1,
        largest=False,
    ).indices
    row_indices = torch.arange(batch_size, device=device).view(-1, 1).expand_as(positions)
    originals = root.expand(batch_size, -1)[row_indices, positions]
    replacements = torch.randint(
        0,
        vocab_size - 1,
        (batch_size, num_corruptions),
        device=device,
    )
    replacements = replacements + (replacements >= originals).to(dtype=torch.long)
    corrupted = root.expand(batch_size, -1).clone()
    corrupted[row_indices, positions] = replacements
    update_mask = torch.zeros_like(corrupted, dtype=torch.bool)
    update_mask[row_indices, positions] = True
    return corrupted, update_mask


@torch.no_grad()
def sample_generator_visits(
    *,
    parent: Sequence[int],
    search_config: Mapping[str, Any],
    num_samples: int,
    batch_size: int,
    num_corruptions: int,
    seed: int,
    device_override: str | None,
) -> tuple[Counter[SequenceTuple], dict[str, Any]]:
    """Sample native masked generator endpoints at a fixed corruption count."""

    if num_samples <= 0:
        raise HammingBallVisitationError("num_samples must be positive")
    seed_everything(seed)
    sequence_length = len(parent)
    restored = restore_process_from_checkpoint(
        search_config["checkpoint"],
        config_path=search_config.get("config"),
        device_override=device_override or search_config.get("device"),
        sequence_length_override=sequence_length,
    )
    process = restored.process
    flow_steps = int(search_config.get("flow_steps", 20))
    max_time = float(search_config.get("max_time", 1.0))
    time_grid = _build_time_grid(
        flow_steps,
        max_time,
        schedule=str(getattr(process, "sample_time_schedule", "linear")),
        schedule_cfg=getattr(process, "sample_time_schedule_cfg", None) or {},
    )
    time_grid[0] = 0.0
    time_grid[-1] = max_time
    temperature_raw = search_config.get("temperature")
    temperature = (
        float(temperature_raw)
        if temperature_raw is not None
        else float(getattr(process, "temperature", 1.0))
    )
    adapter = CTMCFlowAdapter(
        process=process,
        steps=flow_steps,
        max_time=max_time,
        device=restored.device,
        stop_when_filled=bool(search_config.get("stop_when_filled", False)),
        temperature=temperature,
        time_grid=time_grid,
    )
    vocab_size = int(restored.process_cfg["vocab_size"])
    parent_tensor = torch.as_tensor(parent, dtype=torch.long)
    start_time = (1.0 - num_corruptions / float(sequence_length)) * max_time
    counts: Counter[SequenceTuple] = Counter()
    distance_counts: Counter[int] = Counter()
    for start in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - start)
        corrupted, update_mask = draw_exact_corruptions(
            parent_tensor,
            batch_size=current_batch_size,
            vocab_size=vocab_size,
            num_corruptions=num_corruptions,
            device=restored.device,
        )
        endpoints = adapter.rollout_from_batch(
            corrupted,
            start_time,
            update_mask=update_mask,
            t_is_time=True,
        )[-1].detach().cpu()
        for endpoint in endpoints.tolist():
            sequence = tuple(int(token) for token in endpoint)
            distance = hamming_distance(sequence, parent)
            if distance > num_corruptions:
                raise HammingBallVisitationError(
                    "A masked generator endpoint escaped the radius-two support"
                )
            counts[sequence] += 1
            distance_counts[distance] += 1
        print(f"[visitation] sampled {start + current_batch_size}/{num_samples}", flush=True)

    metadata = {
        "checkpoint": str(search_config["checkpoint"]),
        "training_config": str(search_config.get("config")),
        "device": str(restored.device),
        "sequence_length": sequence_length,
        "vocab_size": vocab_size,
        "flow_steps": flow_steps,
        "max_time": max_time,
        "start_time": start_time,
        "num_corruptions": num_corruptions,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "seed": seed,
        "unique_visited_states": len(counts),
        "distance_visit_counts": dict(sorted(distance_counts.items())),
        "proposal_semantics": (
            "two positions corrupted uniformly without replacement; different tokens "
            "drawn uniformly; frozen masked CTMC denoising on those positions only"
        ),
    }
    return counts, metadata


def score_feasibility(
    states: Sequence[SequenceTuple],
    *,
    reward_spec: Mapping[str, Any],
) -> list[bool]:
    if str(reward_spec.get("name", "")) != "ehrlich":
        raise HammingBallVisitationError("The visitation figure requires an Ehrlich reward")
    reward = EhrlichReward(
        instance_path=reward_spec["instance_path"],
        quantization_override=reward_spec.get("quantization_override"),
    )
    sequences = torch.as_tensor(states, dtype=torch.long)
    result = is_feasible(
        sequences,
        reward.instance.transition_mask,
        expected_seq_len=int(reward.instance.cfg.seq_len),
    )
    if not isinstance(result, torch.Tensor):
        raise HammingBallVisitationError("Batched feasibility did not return a tensor")
    return [bool(value) for value in result.detach().cpu().tolist()]


def build_records(
    *,
    parent: Sequence[int],
    states: Sequence[SequenceTuple],
    feasible: Sequence[bool],
    visit_counts: Mapping[SequenceTuple, int],
    num_samples: int,
) -> tuple[BallState, ...]:
    if len(states) != len(feasible):
        raise HammingBallVisitationError("State and feasibility counts do not match")
    root = tuple(int(token) for token in parent)
    records: list[BallState] = []
    for sequence, valid in zip(states, feasible, strict=True):
        changed = tuple(
            position
            for position, (token, origin) in enumerate(zip(sequence, root, strict=True))
            if token != origin
        )
        records.append(
            BallState(
                sequence=tuple(sequence),
                distance=len(changed),
                changed_positions=changed,
                feasible=bool(valid),
                visit_count=int(visit_counts.get(tuple(sequence), 0)),
                num_samples=num_samples,
            )
        )
    if sum(record.visit_count for record in records) != num_samples:
        raise HammingBallVisitationError("Visit counts do not sum to num_samples")
    return tuple(records)


def radial_coordinates(
    records: Sequence[BallState],
    *,
    parent: Sequence[int],
    vocab_size: int,
) -> dict[SequenceTuple, tuple[float, float]]:
    """Place radius-two states in position and position-pair sectors."""

    root = tuple(int(token) for token in parent)
    length = len(root)
    alternatives = {
        position: tuple(token for token in range(vocab_size) if token != root[position])
        for position in range(length)
    }
    alternative_rank = {
        position: {token: rank for rank, token in enumerate(tokens)}
        for position, tokens in alternatives.items()
    }
    coordinates: dict[SequenceTuple, tuple[float, float]] = {root: (0.0, 0.0)}

    position_gap = 0.055
    position_span = (2.0 * math.pi - length * position_gap) / float(length)
    for record in records:
        if record.distance != 1:
            continue
        position = record.changed_positions[0]
        token = record.sequence[position]
        rank = alternative_rank[position][token]
        phase = (
            0.5 * position_gap
            + position * (position_span + position_gap)
            + (rank + 0.5) * position_span / float(vocab_size - 1)
        )
        radius = 0.325
        angle = 0.5 * math.pi - phase
        coordinates[record.sequence] = (radius * math.cos(angle), radius * math.sin(angle))

    pairs = tuple(itertools.combinations(range(length), 2))
    pair_rank = {pair: rank for rank, pair in enumerate(pairs)}
    pair_gap = 0.006
    pair_span = (2.0 * math.pi - len(pairs) * pair_gap) / float(len(pairs))
    radial_min = 0.43
    radial_max = 0.95
    for record in records:
        if record.distance != 2:
            continue
        first, second = record.changed_positions
        first_rank = alternative_rank[first][record.sequence[first]]
        second_rank = alternative_rank[second][record.sequence[second]]
        sector = pair_rank[(first, second)]
        phase = (
            0.5 * pair_gap
            + sector * (pair_span + pair_gap)
            + (first_rank + 0.5) * pair_span / float(vocab_size - 1)
        )
        radius = radial_min + (second_rank + 0.5) * (
            radial_max - radial_min
        ) / float(vocab_size - 1)
        angle = 0.5 * math.pi - phase
        coordinates[record.sequence] = (radius * math.cos(angle), radius * math.sin(angle))

    if set(coordinates) != {record.sequence for record in records}:
        raise HammingBallVisitationError("Not every Hamming-ball state received a coordinate")
    return coordinates


def write_data(
    *,
    output_dir: Path,
    records: Sequence[BallState],
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hamming_ball_visitation.csv"
    json_path = output_dir / "hamming_ball_visitation.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sequence": json.dumps(record.sequence, separators=(",", ":")),
                    "hamming_distance": record.distance,
                    "changed_positions_0based": json.dumps(
                        record.changed_positions,
                        separators=(",", ":"),
                    ),
                    "feasible": str(record.feasible).lower(),
                    "visit_count": record.visit_count,
                    "visit_probability": record.visit_probability,
                    "num_generator_samples": record.num_samples,
                }
            )
    payload = dict(metadata)
    payload["schema_version"] = 1
    payload["support_table"] = str(csv_path)
    payload["support_size"] = len(records)
    payload["support_shell_counts"] = dict(Counter(record.distance for record in records))
    payload["feasible_support_count"] = sum(record.feasible for record in records)
    payload["infeasible_support_count"] = sum(not record.feasible for record in records)
    payload["infeasible_visit_count"] = sum(
        record.visit_count for record in records if not record.feasible
    )
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def load_data(csv_path: Path) -> tuple[BallState, ...]:
    records: list[BallState] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, raw in enumerate(csv.DictReader(handle), start=2):
            try:
                sequence = tuple(int(token) for token in json.loads(raw["sequence"]))
                changed = tuple(
                    int(position) for position in json.loads(raw["changed_positions_0based"])
                )
                num_samples = int(raw["num_generator_samples"])
                record = BallState(
                    sequence=sequence,
                    distance=int(raw["hamming_distance"]),
                    changed_positions=changed,
                    feasible=raw["feasible"].strip().lower() == "true",
                    visit_count=int(raw["visit_count"]),
                    num_samples=num_samples,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HammingBallVisitationError(
                    f"{csv_path}:{line_number} is not a valid visitation row"
                ) from exc
            records.append(record)
    if not records:
        raise HammingBallVisitationError(f"{csv_path} contains no records")
    sample_counts = {record.num_samples for record in records}
    if len(sample_counts) != 1 or sum(record.visit_count for record in records) != next(
        iter(sample_counts)
    ):
        raise HammingBallVisitationError("Loaded visit counts are not normalized to one sample total")
    return tuple(records)


def _scatter_support(
    axis: plt.Axes,
    records: Sequence[BallState],
    coordinates: Mapping[SequenceTuple, tuple[float, float]],
    *,
    muted: bool,
) -> None:
    for distance, size in ((1, 3.2), (2, 0.58)):
        for feasible, color in (
            (True, "#d2d9d8" if muted else "#344d51"),
            (False, "#edd6d9" if muted else "#c45262"),
        ):
            selected = [
                record
                for record in records
                if record.distance == distance and record.feasible is feasible
            ]
            axis.scatter(
                [coordinates[record.sequence][0] for record in selected],
                [coordinates[record.sequence][1] for record in selected],
                s=size,
                c=color,
                linewidths=0.0,
                alpha=0.88 if muted else 0.92,
                rasterized=distance == 2,
                zorder=1,
            )


def _visit_marker_sizes(records: Sequence[BallState], *, maximum: int) -> np.ndarray:
    values = np.asarray([record.visit_count for record in records], dtype=float)
    scaled = np.log1p(values) / math.log1p(maximum)
    base = np.asarray([5.0 if record.distance == 1 else 0.8 for record in records])
    amplitude = np.asarray([13.0 if record.distance == 1 else 3.2 for record in records])
    return base + amplitude * scaled


def plot_visitation(
    records: Sequence[BallState],
    *,
    parent: Sequence[int],
    vocab_size: int,
    output_prefix: Path,
    dpi: int,
) -> tuple[Path, Path]:
    support_size = len(records)
    expected_size = (
        1
        + len(parent) * (vocab_size - 1)
        + math.comb(len(parent), 2) * (vocab_size - 1) ** 2
    )
    if support_size != expected_size:
        raise HammingBallVisitationError(
            f"Observed {support_size} support states; expected {expected_size}"
        )
    coordinates = radial_coordinates(records, parent=parent, vocab_size=vocab_size)
    num_samples = records[0].num_samples
    maximum_visit_count = max(record.visit_count for record in records)
    visited = [record for record in records if record.visit_count > 0 and record.distance > 0]
    norm = mcolors.LogNorm(vmin=1, vmax=maximum_visit_count)
    cmap = matplotlib.colormaps["viridis"]

    prefix = output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.55))
    figure.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.17, wspace=0.035)
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.axis("off")
        axis.add_patch(
            plt.Circle((0.0, 0.0), 0.39, fill=False, color="#c7d0cf", linewidth=0.45, zorder=0)
        )
        axis.add_patch(
            plt.Circle((0.0, 0.0), 0.97, fill=False, color="#c7d0cf", linewidth=0.45, zorder=0)
        )

    axes[0].set_title(
        f"(a) Radius-two Hamming ball ({support_size:,} designs)",
        fontsize=10.8,
        fontweight="semibold",
        pad=7,
    )
    _scatter_support(axes[0], records, coordinates, muted=False)

    axes[1].set_title(
        f"(b) Generator visitation ({num_samples:,} proposals)",
        fontsize=10.8,
        fontweight="semibold",
        pad=7,
    )
    _scatter_support(axes[1], records, coordinates, muted=True)
    if visited:
        sizes = _visit_marker_sizes(visited, maximum=maximum_visit_count)
        axes[1].scatter(
            [coordinates[record.sequence][0] for record in visited],
            [coordinates[record.sequence][1] for record in visited],
            s=sizes,
            c=[record.visit_count for record in visited],
            cmap=cmap,
            norm=norm,
            linewidths=0.0,
            alpha=0.94,
            rasterized=True,
            zorder=3,
        )
        invalid_visited = [record for record in visited if not record.feasible]
        if invalid_visited:
            invalid_sizes = _visit_marker_sizes(
                invalid_visited,
                maximum=maximum_visit_count,
            )
            axes[1].scatter(
                [coordinates[record.sequence][0] for record in invalid_visited],
                [coordinates[record.sequence][1] for record in invalid_visited],
                s=invalid_sizes + 1.0,
                facecolors="none",
                edgecolors="#d23e52",
                linewidths=0.28,
                rasterized=True,
                zorder=4,
            )

    root = tuple(int(token) for token in parent)
    root_record = next(record for record in records if record.distance == 0)
    for axis in axes:
        root_color = "#344d51" if axis is axes[0] else cmap(norm(max(root_record.visit_count, 1)))
        axis.scatter(
            [coordinates[root][0]],
            [coordinates[root][1]],
            s=35.0,
            c=[root_color],
            edgecolors="#263537",
            linewidths=0.45,
            zorder=5,
        )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#344d51",
                markeredgecolor="none",
                markersize=4.0,
                label="feasible design",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#c45262",
                markeredgecolor="none",
                markersize=4.0,
                label="infeasible design",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.09, 0.064),
        ncol=2,
        columnspacing=1.2,
        frameon=False,
        fontsize=7.8,
    )
    colorbar_axis = figure.add_axes((0.57, 0.078, 0.32, 0.025))
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Generator visits per design (log scale)", fontsize=8.2, labelpad=3)
    ticks = [
        value
        for value in (1, 10, 100, 1000, 10000)
        if value <= maximum_visit_count
    ]
    colorbar.set_ticks(ticks)
    colorbar.ax.set_xticklabels([f"{value:,}" for value in ticks])
    colorbar.ax.tick_params(labelsize=6.8, length=2.2, width=0.5)
    colorbar.outline.set_linewidth(0.5)

    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH_CONFIG)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--archive-index", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=320)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    search_config = json.loads(args.search_config.read_text(encoding="utf-8"))
    archive_path = args.archive or Path(search_config["initial_batch_path"])
    archive = load_archive(archive_path)
    archive_index, parent_list, parent_score = choose_incumbent(
        archive,
        archive_index=args.archive_index,
    )
    parent = tuple(parent_list)
    training_config = json.loads(Path(search_config["config"]).read_text(encoding="utf-8"))
    process_config = training_config.get("process_cfg")
    if not isinstance(process_config, Mapping) or "vocab_size" not in process_config:
        raise HammingBallVisitationError("Training config is missing process_cfg.vocab_size")
    vocab_size = int(process_config["vocab_size"])
    csv_path = args.output_dir / "hamming_ball_visitation.csv"
    json_path = args.output_dir / "hamming_ball_visitation.json"
    if args.plot_only:
        records = load_data(csv_path)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        states = enumerate_radius_two_ball(parent, vocab_size=vocab_size)
        counts, sampling_metadata = sample_generator_visits(
            parent=parent,
            search_config=search_config,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_corruptions=2,
            seed=args.seed,
            device_override=args.device,
        )
        reward_spec = search_config.get("reward")
        if not isinstance(reward_spec, Mapping):
            raise HammingBallVisitationError("Search config is missing its reward mapping")
        feasible = score_feasibility(states, reward_spec=reward_spec)
        records = build_records(
            parent=parent,
            states=states,
            feasible=feasible,
            visit_counts=counts,
            num_samples=args.num_samples,
        )
        metadata = {
            **sampling_metadata,
            "archive": str(archive_path),
            "archive_index": archive_index,
            "parent_sequence": list(parent),
            "parent_score": parent_score,
        }
        csv_path, json_path = write_data(
            output_dir=args.output_dir,
            records=records,
            metadata=metadata,
        )

    output_prefix = args.output_prefix or args.output_dir / DEFAULT_OUTPUT_PREFIX
    png_path, pdf_path = plot_visitation(
        records,
        parent=parent,
        vocab_size=vocab_size,
        output_prefix=output_prefix,
        dpi=args.dpi,
    )
    shell_counts = Counter(record.distance for record in records)
    print(f"Support: {len(records):,} states ({dict(sorted(shell_counts.items()))})")
    print(f"Visited: {sum(record.visit_count > 0 for record in records):,} unique states")
    print(
        "Generator visits by shell: "
        + str(
            dict(
                sorted(
                    Counter(
                        record.distance
                        for record in records
                        for _ in range(record.visit_count)
                    ).items()
                )
            )
        )
    )
    print(f"Data: {csv_path.resolve()}")
    print(f"Metadata: {json_path.resolve()}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
