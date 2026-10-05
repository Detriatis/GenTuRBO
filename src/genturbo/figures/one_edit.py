#!/usr/bin/env python3
"""Compute a generator-guided one-edit kernel around one Ehrlich incumbent.

The random reference is the exact uniform distribution over the ``L * (V - 1)``
one-substitution endpoints.  By default, the learned distribution is propagated
exactly through the categorical transition matrix of every Euler update.  For
each position and possible current token, the model is evaluated in the fixed
parent context and the same transition row used by the search adapter is built.
Monte Carlo masked rollouts remain available as an optional validation mode.

The learned endpoint probabilities written to ``probability`` are unconditional
mass, so their sum plus generator restoration is one.  The directly comparable
distribution conditioned on realising one edit is retained in
``conditional_probability``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from core.utils import seed_everything
from flows.ctmc_adapter import CTMCFlowAdapter
from processes.dfm_mask import _build_time_grid, ctmc_rate_uniform
from runners.common import restore_process_from_checkpoint

DEFAULT_SEARCH_CONFIG = Path("configs/experiments/ehrlich_radial_search_v1/generator_one_edit.json")
DEFAULT_OUTPUT_DIR = Path("runs/ehrlich_radial_search_v1/kernel_comparison")
CSV_COLUMNS = (
    "method",
    "parent_sequence",
    "parent_score",
    "position",
    "position_1based",
    "original_token",
    "substituted_token",
    "probability",
    "conditional_probability",
    "unconditional_probability",
    "log_probability",
    "conditional_log_probability",
    "sample_count",
    "num_samples",
    "conditioning_sample_count",
    "retention_count",
    "retention_probability",
    "probability_estimator",
)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_archive(path: Path) -> list[dict[str, Any]]:
    """Load finite scored sequences from a plain or gzip JSONL archive."""

    rows: list[dict[str, Any]] = []
    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            sequence = raw.get("seq")
            score = raw.get("score")
            if not isinstance(sequence, list) or not sequence:
                raise ValueError(f"{path}:{line_number} is missing a non-empty seq list")
            if score is None or not math.isfinite(float(score)):
                continue
            rows.append({**raw, "seq": [int(token) for token in sequence], "score": float(score)})
    if not rows:
        raise ValueError(f"{path} contains no finite scored sequences")
    return rows


def choose_incumbent(
    archive: Sequence[Mapping[str, Any]],
    *,
    archive_index: int | None = None,
) -> tuple[int, list[int], float]:
    """Choose a fixed archive incumbent, preferring the most recent score tie."""

    if not archive:
        raise ValueError("archive must not be empty")
    if archive_index is None:
        best_score = max(float(row["score"]) for row in archive)
        # The survivor policy replaces the incumbent on score ties.  Mirroring
        # that rule here also recovers the parent used by the radial pilot.
        archive_index = max(idx for idx, row in enumerate(archive) if float(row["score"]) == best_score)
    if archive_index < 0 or archive_index >= len(archive):
        raise IndexError(f"archive_index={archive_index} is outside [0, {len(archive)})")
    selected = archive[archive_index]
    return archive_index, [int(token) for token in selected["seq"]], float(selected["score"])


def _parent_label(parent: Sequence[int]) -> str:
    return " ".join(str(int(token)) for token in parent)


def _outcome_rows(
    *,
    method: str,
    parent: Sequence[int],
    parent_score: float,
    vocab_size: int,
    counts: Mapping[tuple[int, int], int] | None,
    num_samples: int | None,
    conditioning_sample_count: int | None,
    retention_count: int,
    probability_estimator: str,
) -> list[dict[str, Any]]:
    length = len(parent)
    support_size = length * (vocab_size - 1)
    if length <= 0 or vocab_size <= 1:
        raise ValueError("one-edit support requires a non-empty parent and vocab_size > 1")
    if counts is not None and conditioning_sample_count is None:
        raise ValueError("sampled rows require conditioning_sample_count")
    if conditioning_sample_count is not None and conditioning_sample_count <= 0:
        raise ValueError("the generator produced no one-edit endpoints to condition on")

    retention_probability = (
        float(retention_count) / float(num_samples) if num_samples is not None and num_samples > 0 else 0.0
    )
    rows: list[dict[str, Any]] = []
    for position, original_token in enumerate(parent):
        for substituted_token in range(vocab_size):
            if substituted_token == int(original_token):
                continue
            if counts is None:
                sample_count: int | None = None
                unconditional_probability = 1.0 / float(support_size)
                conditional_probability = unconditional_probability
            else:
                sample_count = int(counts.get((position, substituted_token), 0))
                assert conditioning_sample_count is not None
                assert num_samples is not None
                unconditional_probability = float(sample_count) / float(num_samples)
                conditional_probability = float(sample_count) / float(conditioning_sample_count)
            probability = unconditional_probability
            rows.append(
                {
                    "method": method,
                    "parent_sequence": _parent_label(parent),
                    "parent_score": float(parent_score),
                    "position": int(position),
                    "position_1based": int(position + 1),
                    "original_token": int(original_token),
                    "substituted_token": int(substituted_token),
                    "probability": probability,
                    "conditional_probability": conditional_probability,
                    "unconditional_probability": unconditional_probability,
                    "log_probability": math.log(probability) if probability > 0.0 else None,
                    "conditional_log_probability": (
                        math.log(conditional_probability) if conditional_probability > 0.0 else None
                    ),
                    "sample_count": sample_count,
                    "num_samples": num_samples,
                    "conditioning_sample_count": conditioning_sample_count,
                    "retention_count": int(retention_count),
                    "retention_probability": retention_probability,
                    "probability_estimator": probability_estimator,
                }
            )
    return rows


def exact_random_one_edit_rows(
    parent: Sequence[int],
    *,
    parent_score: float,
    vocab_size: int,
) -> list[dict[str, Any]]:
    """Return exact uniform probabilities for every one-edit endpoint."""

    return _outcome_rows(
        method="random",
        parent=parent,
        parent_score=parent_score,
        vocab_size=vocab_size,
        counts=None,
        num_samples=None,
        conditioning_sample_count=None,
        retention_count=0,
        probability_estimator="exact_uniform_one_edit",
    )


def adapter_integration_intervals(time_grid: torch.Tensor, start_time: float) -> list[tuple[float, float]]:
    """Reproduce ``CTMCFlowAdapter``'s fractional-start Euler intervals."""

    grid = torch.as_tensor(time_grid, dtype=torch.float).cpu()
    if grid.ndim != 1 or grid.numel() < 2:
        raise ValueError("time_grid must be one-dimensional with at least two points")
    if not bool(torch.all(grid[1:] > grid[:-1]).item()):
        raise ValueError("time_grid must be strictly increasing")
    steps = int(grid.numel() - 1)
    minimum = float(grid[0].item())
    maximum = float(grid[-1].item())
    start = min(max(float(start_time), minimum), maximum)
    if start <= minimum:
        step_coordinate = 0.0
    elif start >= maximum:
        step_coordinate = float(steps)
    else:
        query = torch.tensor(start, dtype=grid.dtype)
        index_high = int(torch.searchsorted(grid, query, right=False).item())
        index_high = max(1, min(index_high, steps))
        index_low = index_high - 1
        time_low = float(grid[index_low].item())
        time_high = float(grid[index_high].item())
        alpha = (start - time_low) / max(time_high - time_low, 1e-12)
        step_coordinate = float(index_low) + min(max(alpha, 0.0), 1.0)

    def coordinate_to_time(coordinate: float) -> float:
        value = min(max(float(coordinate), 0.0), float(steps))
        low = int(math.floor(value))
        high = min(steps, low + 1)
        if low == high:
            return float(grid[low].item())
        alpha = value - float(low)
        return (1.0 - alpha) * float(grid[low].item()) + alpha * float(grid[high].item())

    intervals: list[tuple[float, float]] = []
    coordinate = step_coordinate
    while coordinate < float(steps):
        next_coordinate = min(float(steps), float(math.floor(coordinate) + 1.0))
        if next_coordinate <= coordinate:
            break
        time_start = coordinate_to_time(coordinate)
        time_end = coordinate_to_time(next_coordinate)
        intervals.append((time_start, time_end - time_start))
        coordinate = next_coordinate
    return intervals


def build_euler_transition_rows(rates: torch.Tensor, dt: float) -> torch.Tensor:
    """Build the normalized categorical Euler rows used by ``ctmc_euler_step``.

    ``rates`` has shape ``[L, V, V]``: position, current token, destination
    token.  Its diagonal is expected to be zero.  The result has the same shape
    and is row-stochastic over the destination-token dimension.
    """

    rate_tensor = torch.as_tensor(rates)
    if rate_tensor.ndim != 3 or rate_tensor.shape[1] != rate_tensor.shape[2]:
        raise ValueError("rates must have shape [positions, vocab_size, vocab_size]")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    vocab_size = int(rate_tensor.shape[-1])
    transition = (rate_tensor * float(dt)).clamp_min(0.0)
    stay_probability = (1.0 - transition.sum(dim=-1)).clamp_min(0.0)
    token_indices = torch.arange(vocab_size, device=transition.device)
    transition[:, token_indices, token_indices] += stay_probability
    return transition / transition.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def propagate_categorical_distribution(
    initial_distribution: torch.Tensor,
    transition_steps: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Apply position-specific categorical transition matrices in order."""

    distribution = torch.as_tensor(initial_distribution, dtype=torch.double).clone()
    if distribution.ndim != 2:
        raise ValueError("initial_distribution must have shape [positions, vocab_size]")
    if bool((distribution < 0.0).any().item()):
        raise ValueError("initial_distribution must be non-negative")
    if not torch.allclose(distribution.sum(dim=-1), torch.ones(distribution.shape[0], dtype=torch.double)):
        raise ValueError("initial_distribution rows must sum to one")
    for step_index, transition in enumerate(transition_steps):
        matrix = torch.as_tensor(transition, dtype=torch.double)
        expected_shape = (distribution.shape[0], distribution.shape[1], distribution.shape[1])
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"transition step {step_index} has shape {tuple(matrix.shape)}; expected {expected_shape}"
            )
        if bool((matrix < 0.0).any().item()):
            raise ValueError(f"transition step {step_index} contains negative probabilities")
        if not torch.allclose(
            matrix.sum(dim=-1),
            torch.ones(expected_shape[:2], dtype=torch.double),
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError(f"transition step {step_index} rows must sum to one")
        distribution = torch.einsum("lc,lcd->ld", distribution, matrix)
    return distribution


def exact_generator_one_edit_rows(
    parent: Sequence[int],
    site_endpoint_probabilities: torch.Tensor,
    *,
    parent_score: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert exact per-site endpoint distributions into the tidy kernel."""

    probabilities = torch.as_tensor(site_endpoint_probabilities, dtype=torch.double).cpu()
    length = len(parent)
    if probabilities.ndim != 2 or probabilities.shape[0] != length:
        raise ValueError(f"site_endpoint_probabilities must have shape [{length}, V]")
    vocab_size = int(probabilities.shape[1])
    if vocab_size <= 1 or any(token < 0 or token >= vocab_size for token in parent):
        raise ValueError("parent tokens must lie in a vocabulary with at least two tokens")
    if bool((probabilities < 0.0).any().item()):
        raise ValueError("site endpoint probabilities must be non-negative")
    if not torch.allclose(probabilities.sum(dim=-1), torch.ones(length, dtype=torch.double), atol=1e-8, rtol=1e-8):
        raise ValueError("site endpoint probability rows must sum to one")

    joint = probabilities / float(length)
    retention_probability = sum(float(joint[position, int(token)].item()) for position, token in enumerate(parent))
    one_edit_probability = 1.0 - retention_probability
    if one_edit_probability <= 0.0:
        raise ValueError("the exact generator kernel has no one-edit mass")

    rows: list[dict[str, Any]] = []
    conditional_values: list[float] = []
    for position, original_token in enumerate(parent):
        for substituted_token in range(vocab_size):
            if substituted_token == int(original_token):
                continue
            unconditional_probability = float(joint[position, substituted_token].item())
            conditional_probability = unconditional_probability / one_edit_probability
            conditional_values.append(conditional_probability)
            rows.append(
                {
                    "method": "generator",
                    "parent_sequence": _parent_label(parent),
                    "parent_score": float(parent_score),
                    "position": int(position),
                    "position_1based": int(position + 1),
                    "original_token": int(original_token),
                    "substituted_token": int(substituted_token),
                    "probability": unconditional_probability,
                    "conditional_probability": conditional_probability,
                    "unconditional_probability": unconditional_probability,
                    "log_probability": (
                        math.log(unconditional_probability) if unconditional_probability > 0.0 else None
                    ),
                    "conditional_log_probability": (
                        math.log(conditional_probability) if conditional_probability > 0.0 else None
                    ),
                    "sample_count": None,
                    "num_samples": None,
                    "conditioning_sample_count": None,
                    "retention_count": None,
                    "retention_probability": retention_probability,
                    "probability_estimator": "exact_masked_ctmc_categorical_propagation",
                }
            )

    nonzero_conditional = [value for value in conditional_values if value > 0.0]
    conditional_entropy = -sum(value * math.log(value) for value in nonzero_conditional)
    summary = {
        "num_rollouts": None,
        "one_edit_count": None,
        "one_edit_probability": one_edit_probability,
        "retention_count": None,
        "retention_probability": retention_probability,
        "observed_one_edit_outcomes": len(nonzero_conditional),
        "possible_one_edit_outcomes": length * (vocab_size - 1),
        "zero_count_one_edit_outcomes": len(conditional_values) - len(nonzero_conditional),
        "conditional_entropy_nats": conditional_entropy,
        "conditional_effective_support": math.exp(conditional_entropy),
        "minimum_conditional_probability": min(conditional_values),
        "maximum_conditional_probability": max(conditional_values),
    }
    return rows, summary


def generator_one_edit_rows(
    parent: Sequence[int],
    endpoints: torch.Tensor,
    *,
    parent_score: float,
    vocab_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate endpoint rollouts, conditioning probabilities on one edit."""

    parent_tensor = torch.as_tensor(parent, dtype=torch.long).view(1, -1)
    endpoint_tensor = torch.as_tensor(endpoints, dtype=torch.long).cpu()
    if endpoint_tensor.ndim != 2 or endpoint_tensor.shape[1] != parent_tensor.shape[1]:
        raise ValueError(
            f"endpoints must have shape [N, {parent_tensor.shape[1]}], got {tuple(endpoint_tensor.shape)}"
        )
    edits = endpoint_tensor.ne(parent_tensor).sum(dim=1)
    if bool((edits > 1).any().item()):
        raise ValueError("masked generator rollouts changed more than one position")
    retention_count = int((edits == 0).sum().item())
    changed = endpoint_tensor[edits == 1]
    counts: dict[tuple[int, int], int] = {}
    for endpoint in changed:
        changed_position = int(endpoint.ne(parent_tensor[0]).nonzero(as_tuple=False)[0, 0].item())
        substituted_token = int(endpoint[changed_position].item())
        key = (changed_position, substituted_token)
        counts[key] = counts.get(key, 0) + 1

    num_samples = int(endpoint_tensor.shape[0])
    one_edit_count = int(changed.shape[0])
    rows = _outcome_rows(
        method="generator",
        parent=parent,
        parent_score=parent_score,
        vocab_size=vocab_size,
        counts=counts,
        num_samples=num_samples,
        conditioning_sample_count=one_edit_count,
        retention_count=retention_count,
        probability_estimator="batched_masked_ctmc_monte_carlo",
    )
    summary = {
        "num_rollouts": num_samples,
        "one_edit_count": one_edit_count,
        "one_edit_probability": float(one_edit_count) / float(num_samples),
        "retention_count": retention_count,
        "retention_probability": float(retention_count) / float(num_samples),
        "observed_one_edit_outcomes": len(counts),
        "possible_one_edit_outcomes": len(parent) * (vocab_size - 1),
        "zero_count_one_edit_outcomes": len(parent) * (vocab_size - 1) - len(counts),
    }
    return rows, summary


def _draw_corrupted_batch(
    parent: torch.Tensor,
    *,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw the interpolant conditioned on exactly one actual corruption."""

    parent_device = parent.to(device=device, dtype=torch.long).view(1, -1)
    length = int(parent_device.shape[1])
    positions = torch.randint(0, length, (batch_size,), device=device)
    originals = parent_device[0, positions]
    # Draw uniformly from all tokens except the parent token without rejection.
    replacements = torch.randint(0, vocab_size - 1, (batch_size,), device=device)
    replacements = replacements + (replacements >= originals).to(dtype=torch.long)
    xt = parent_device.expand(batch_size, -1).clone()
    row_indices = torch.arange(batch_size, device=device)
    xt[row_indices, positions] = replacements
    update_mask = torch.zeros_like(xt, dtype=torch.bool)
    update_mask[row_indices, positions] = True
    return xt, update_mask


@torch.no_grad()
def exact_generator_endpoint_probabilities(
    *,
    parent: Sequence[int],
    search_config: Mapping[str, Any],
    device_override: str | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Propagate every one-site categorical state through the adapter grid."""

    sequence_length = len(parent)
    restored = restore_process_from_checkpoint(
        search_config["checkpoint"],
        config_path=search_config.get("config"),
        device_override=device_override or search_config.get("device"),
        sequence_length_override=sequence_length,
    )
    process = restored.process
    model = restored.model_for_state
    if model is None:
        raise RuntimeError("exact generator propagation requires a restored denoising model")
    model.eval()
    if str(getattr(process, "interpolant", "")) != "uniform":
        raise ValueError("exact one-edit propagation currently requires the uniform interpolant")

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
    start_time = (1.0 - 1.0 / float(sequence_length)) * max_time
    intervals = adapter_integration_intervals(time_grid, start_time)
    start_index_high = int(
        torch.searchsorted(time_grid.cpu(), torch.tensor(start_time, dtype=time_grid.dtype), right=False).item()
    )
    start_index_high = max(1, min(start_index_high, flow_steps))
    start_index_low = start_index_high - 1
    start_grid_low = float(time_grid[start_index_low].item())
    start_grid_high = float(time_grid[start_index_high].item())
    start_step_coordinate = float(start_index_low) + (start_time - start_grid_low) / max(
        start_grid_high - start_grid_low,
        1e-12,
    )
    temperature_raw = search_config.get("temperature")
    temperature = float(temperature_raw) if temperature_raw is not None else float(getattr(process, "temperature", 1.0))
    vocab_size = int(restored.process_cfg["vocab_size"])

    parent_tensor = torch.as_tensor(parent, dtype=torch.long, device=restored.device)
    positions = torch.arange(sequence_length, device=restored.device).repeat_interleave(vocab_size)
    current_tokens = torch.arange(vocab_size, device=restored.device).repeat(sequence_length)
    row_indices = torch.arange(sequence_length * vocab_size, device=restored.device)
    contexts = parent_tensor.view(1, -1).expand(sequence_length * vocab_size, -1).clone()
    contexts[row_indices, positions] = current_tokens

    initial_distribution = torch.full(
        (sequence_length, vocab_size),
        1.0 / float(vocab_size - 1),
        dtype=torch.double,
    )
    initial_distribution[
        torch.arange(sequence_length),
        torch.as_tensor(parent, dtype=torch.long),
    ] = 0.0
    distribution = initial_distribution
    row_indices_device = torch.arange(sequence_length * vocab_size, device=restored.device)
    transition_row_sums_max_error = 0.0
    propagated_row_sums_max_error = 0.0
    for step_index, (time_start, dt) in enumerate(intervals):
        time_batch = torch.full(
            (sequence_length * vocab_size,),
            time_start,
            device=restored.device,
        )
        logits = model(contexts, timesteps=time_batch)
        rates = ctmc_rate_uniform(
            contexts,
            t_scalar=time_start,
            logits=logits,
            temperature=temperature,
        )
        selected_rates = rates[row_indices_device, positions, :].reshape(
            sequence_length,
            vocab_size,
            vocab_size,
        )
        transition = build_euler_transition_rows(selected_rates, dt).detach().cpu().to(dtype=torch.double)
        transition_row_sums_max_error = max(
            transition_row_sums_max_error,
            float((transition.sum(dim=-1) - 1.0).abs().max().item()),
        )
        distribution = propagate_categorical_distribution(distribution, [transition])
        distribution_row_sums = distribution.sum(dim=-1, keepdim=True)
        propagated_row_sums_max_error = max(
            propagated_row_sums_max_error,
            float((distribution_row_sums - 1.0).abs().max().item()),
        )
        distribution = distribution / distribution_row_sums.clamp_min(1e-12)
        print(
            f"[kernel] propagated exact Euler update {step_index + 1}/{len(intervals)} "
            f"(t={time_start:.9f}, dt={dt:.9f})",
            flush=True,
        )

    metadata = {
        "checkpoint": str(search_config["checkpoint"]),
        "training_config": str(search_config.get("config")),
        "device": str(restored.device),
        "sequence_length": sequence_length,
        "vocab_size": vocab_size,
        "flow_steps": flow_steps,
        "max_time": max_time,
        "start_time": start_time,
        "start_step_coordinate": start_step_coordinate,
        "num_euler_updates": len(intervals),
        "integration_intervals": [
            {"time_start": float(time_start), "dt": float(dt)} for time_start, dt in intervals
        ],
        "corruption_distribution": "uniform position; uniform different token",
        "rollout_update_mask": "corrupted position only",
        "probability_estimator": "exact_masked_ctmc_categorical_propagation",
        "categorical_context_batch_size": sequence_length * vocab_size,
        "transition_row_sums_max_error": transition_row_sums_max_error,
        "propagated_row_sums_max_error_before_renormalization": propagated_row_sums_max_error,
    }
    return distribution, metadata


@torch.no_grad()
def sample_generator_endpoints(
    *,
    parent: Sequence[int],
    search_config: Mapping[str, Any],
    num_samples: int,
    batch_size: int,
    device_override: str | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Restore the pilot checkpoint and perform actual masked CTMC rollouts."""

    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")
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
    temperature = float(temperature_raw) if temperature_raw is not None else float(getattr(process, "temperature", 1.0))
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
    t_value = 1.0 - 1.0 / float(sequence_length)
    endpoint_chunks: list[torch.Tensor] = []
    for start in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - start)
        xt, update_mask = _draw_corrupted_batch(
            parent_tensor,
            batch_size=current_batch_size,
            vocab_size=vocab_size,
            device=restored.device,
        )
        endpoint = adapter.rollout_from_batch(
            xt,
            t_value * max_time,
            update_mask=update_mask,
            t_is_time=True,
        )[-1]
        endpoint_chunks.append(endpoint.detach().cpu())
        completed = start + current_batch_size
        print(f"[kernel] sampled {completed}/{num_samples} generator endpoints", flush=True)
    metadata = {
        "checkpoint": str(search_config["checkpoint"]),
        "training_config": str(search_config.get("config")),
        "device": str(restored.device),
        "sequence_length": sequence_length,
        "vocab_size": vocab_size,
        "flow_steps": flow_steps,
        "max_time": max_time,
        "start_time": t_value * max_time,
        "corruption_distribution": "uniform position; uniform different token",
        "rollout_update_mask": "corrupted position only",
        "probability_estimator": "batched_masked_ctmc_monte_carlo",
    }
    return torch.cat(endpoint_chunks, dim=0), metadata


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH_CONFIG)
    parser.add_argument("--archive", type=Path, default=None, help="Override the config's initial_batch_path.")
    parser.add_argument("--archive-index", type=int, default=None, help="Use this archive row instead of the latest maximum.")
    parser.add_argument(
        "--estimator",
        choices=("exact", "mc"),
        default="exact",
        help="Exact categorical propagation (default) or Monte Carlo rollout validation.",
    )
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main(argv=None) -> None:
    args = parse_args()
    seed_everything(args.seed)
    search_config = json.loads(args.search_config.read_text(encoding="utf-8"))
    archive_path = args.archive or Path(search_config["initial_batch_path"])
    archive = load_archive(archive_path)
    archive_index, parent, parent_score = choose_incumbent(archive, archive_index=args.archive_index)
    if args.estimator == "exact":
        site_endpoint_probabilities, rollout_metadata = exact_generator_endpoint_probabilities(
            parent=parent,
            search_config=search_config,
            device_override=args.device,
        )
        generator_rows, generator_summary = exact_generator_one_edit_rows(
            parent,
            site_endpoint_probabilities,
            parent_score=parent_score,
        )
    else:
        endpoints, rollout_metadata = sample_generator_endpoints(
            parent=parent,
            search_config=search_config,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device_override=args.device,
        )
        generator_rows, generator_summary = generator_one_edit_rows(
            parent,
            endpoints,
            parent_score=parent_score,
            vocab_size=int(rollout_metadata["vocab_size"]),
        )
    vocab_size = int(rollout_metadata["vocab_size"])
    random_rows = exact_random_one_edit_rows(parent, parent_score=parent_score, vocab_size=vocab_size)
    rows = random_rows + generator_rows
    output_dir = args.output_dir
    csv_path = output_dir / "one_edit_kernel.csv"
    json_path = output_dir / "one_edit_kernel.json"
    _write_csv(csv_path, rows)
    payload = {
        "schema_version": 2,
        "position_indexing": "position is zero-based; position_1based is one-based",
        "probability_definition": {
            "random": "exact mass over one-edit endpoints",
            "generator": "unconditional endpoint mass; sums to one minus retention",
            "conditional_probability": "one-edit endpoint mass conditioned on the generator changing the parent",
            "unconditional_probability": "alias of probability, retained to make the denominator explicit",
        },
        "seed": int(args.seed),
        "estimator": str(args.estimator),
        "search_config": str(args.search_config),
        "archive": str(archive_path),
        "archive_index": archive_index,
        "parent_sequence": parent,
        "parent_score": parent_score,
        "rollout": rollout_metadata,
        "generator_summary": generator_summary,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[kernel] parent archive row {archive_index}: {parent} (score={parent_score:g})")
    if generator_summary["num_rollouts"] is None:
        print(f"[kernel] exact generator retention {generator_summary['retention_probability']:.6%}")
        print(
            "[kernel] exact conditional effective support "
            f"{generator_summary['conditional_effective_support']:.6f}"
        )
    else:
        print(
            "[kernel] generator retention "
            f"{generator_summary['retention_count']}/{generator_summary['num_rollouts']} "
            f"({generator_summary['retention_probability']:.3%})"
        )
    print(f"[kernel] wrote {csv_path}")
    print(f"[kernel] wrote {json_path}")


if __name__ == "__main__":
    main()
