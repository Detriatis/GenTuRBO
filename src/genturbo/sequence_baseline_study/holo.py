"""Manifest-compatible execution of a generic prepaid-archive Holo GA arm.

Non-Ehrlich resolved cells provide top-level ``reward_adapter`` as
``{module, class, kwargs}``; the constructed class must expose ``score_batch``.
Sequence length, alphabet, prepaid archive size, and whole-generation budget
are resolved per cell. Existing Ehrlich cells retain their historical reward
mapping as a backward-compatible fallback.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines.holo_genetic_algorithm import HoloGeneticAlgorithmBaseline
from search.interpolant.manifest import RunManifestRecorder, build_run_manifest
from genturbo.io import iter_jsonl, open_text, write_jsonl

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class HoloProtocolError(RuntimeError):
    """Raised when a resolved Holo cell violates the locked study protocol."""


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoloProtocolError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HoloProtocolError(f"{path} must contain a JSON object")
    return value


def _initial_archive(
    path: Path,
    *,
    expected_count: int,
    sequence_length: int,
    vocab_size: int,
) -> tuple[list[dict[str, Any]], torch.Tensor, np.ndarray]:
    rows = list(iter_jsonl(path))
    if len(rows) != expected_count:
        raise HoloProtocolError(f"initial archive has {len(rows)} rows; expected {expected_count}")
    sequences: list[list[int]] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        sequence = row.get("seq")
        if (
            not isinstance(sequence, list)
            or len(sequence) != sequence_length
            or any(isinstance(token, bool) or not isinstance(token, int) for token in sequence)
            or any(int(token) < 0 or int(token) >= vocab_size for token in sequence)
        ):
            raise HoloProtocolError(f"initial archive row {index} has an invalid sequence")
        if row.get("oracle_evaluated", True) is not True:
            raise HoloProtocolError(f"initial archive row {index} is not a prepaid oracle label")
        raw_score = row.get("score")
        if raw_score is None:
            score = float("nan")
        elif isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise HoloProtocolError(f"initial archive row {index} has an invalid score")
        else:
            score = float(raw_score)
            if not math.isfinite(score):
                score = float("nan")
        sequences.append([int(token) for token in sequence])
        scores.append(score)
    return rows, torch.as_tensor(sequences, dtype=torch.long), np.asarray(scores, dtype=float)


def _json_line(stream: Any, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
    stream.write("\n")


def _sample_row(
    sequence: list[int],
    score: float,
    *,
    batch_id: int,
    seq_idx: int,
    alphabet: str = AMINO_ACIDS,
) -> dict[str, Any]:
    finite = math.isfinite(score)
    sequence_string = "".join(alphabet[token] for token in sequence)
    return {
        "batch_id": batch_id,
        "seq_idx": seq_idx,
        "seq": sequence,
        "seq_length": len(sequence),
        "score": score if finite else None,
        "score_is_finite": finite,
        "score_for_analysis": score if finite else None,
        "decoded_sequence": sequence_string,
        "sequence_string": sequence_string,
        "oracle_evaluated": True,
        "oracle_rejected": not finite,
        "sample_source": "holo_discrete_evolution",
        "smiles": sequence_string,
    }


def _nearest_initial(sequence: Sequence[int], initial: np.ndarray) -> tuple[list[int], int]:
    child = np.asarray(sequence, dtype=np.int64)
    distances = np.count_nonzero(initial != child[None, :], axis=1)
    index = int(np.argmin(distances))
    return initial[index].astype(int).tolist(), int(distances[index])


def _trace_row(
    sample: Mapping[str, Any],
    *,
    ancestor: list[int],
    edits: int,
    duplicate: bool,
) -> dict[str, Any]:
    return {
        **sample,
        "acquisition_candidate": True,
        "acquisition_selected": True,
        "duplicate_exact_proposal": duplicate,
        "duplicate_generated_leaf": False,
        "cached_generated_proposal": False,
        "invalid_generated_proposal": False,
        "expected_edits": float(edits),
        "proposal_logical_edits": edits,
        "ancestor_to_child_edits": edits,
        "ancestor_inference": "nearest_prepaid_archive_point",
        "ancestor_seq": ancestor,
        "from_seq": ancestor,
        "to_seq": sample["seq"],
    }


def _diagnostic_row(
    batch_id: int,
    samples: Sequence[Mapping[str, Any]],
    *,
    total: int,
    survivor_count: int,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    unique = {tuple(int(token) for token in row["seq"]) for row in samples}
    rejected = sum(row["oracle_rejected"] is True for row in samples)
    return {
        "batch_id": batch_id,
        "generated_proposals": len(samples),
        "deduped_proposals": len(unique),
        "duplicate_exact_proposals": len(samples) - len(unique),
        "duplicate_generated_leaves": 0,
        "cached_generated_proposals": 0,
        "invalid_generated_proposals": 0,
        "acquisition_batch_size": len(samples),
        "oracle_evaluated_count": len(samples),
        "oracle_rejected_count": rejected,
        "oracle_budget_debit_count": len(samples),
        "oracle_scored_count": len(samples),
        "oracle_privileged_scored_count": 0,
        "oracle_budget_total_after_iteration": total,
        "oracle_physical_total_after_iteration": total,
        "oracle_privileged_total_after_iteration": 0,
        "oracle_total_after_iteration": total,
        "proposal_batch_discarded": False,
        "proposal_operator": "holo_discrete_evolution",
        "holo_survivor_count": survivor_count,
        "holo_prepaid_initial_archive_count": int(run_metadata["initial_oracle_evaluations"]),
        "holo_prepaid_initial_scores": run_metadata["prepaid_initial_scores"],
        "holo_population_size": int(run_metadata["population_size"]),
        "holo_generation_count": int(run_metadata["generation_count"]),
        "holo_initial_survivor_count": int(run_metadata["initial_survivor_count"]),
        "holo_warm_start_policy": run_metadata["warm_start_policy"],
        "holo_operator_order": "survive_recombine_mutate_then_score",
        "holo_mutation_prob": float(run_metadata["mutation_prob"]),
        "holo_recombine_prob": float(run_metadata["recombine_prob"]),
        "holo_survival_quantile": float(run_metadata["survival_quantile"]),
    }


def _online_reward_adapter(
    resolved: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Any, dict[str, Path]]:
    """Instantiate ``reward_adapter`` while retaining the Ehrlich fallback.

    Generic cells declare a top-level ``reward_adapter`` mapping with non-empty
    ``module`` and ``class`` strings plus a ``kwargs`` mapping. The constructed
    object must expose ``score_batch``. Existing Ehrlich cells omit this mapping
    and continue to construct their historical reward from ``reward``.
    """

    adapter = resolved.get("reward_adapter")
    if adapter is None:
        from ehrlich.ehrlich_reward import EhrlichReward

        reward_spec = resolved["reward"]
        reward_path = _resolve(repo_root, reward_spec["instance_path"])
        return (
            EhrlichReward(
                instance_path=reward_path,
                quantization_override=int(reward_spec["quantization_override"]),
            ),
            {"reward_instance": reward_path},
        )
    if not isinstance(adapter, Mapping):
        raise HoloProtocolError("reward_adapter must be a mapping")
    module_name = adapter.get("module")
    class_name = adapter.get("class")
    kwargs = adapter.get("kwargs")
    if (
        not isinstance(module_name, str)
        or not module_name
        or not isinstance(class_name, str)
        or not class_name
        or not isinstance(kwargs, Mapping)
    ):
        raise HoloProtocolError(
            "reward_adapter requires non-empty module/class strings and a kwargs mapping"
        )
    try:
        adapter_class = getattr(importlib.import_module(module_name), class_name)
        reward = adapter_class(**dict(kwargs))
    except (AttributeError, ImportError, TypeError) as exc:
        raise HoloProtocolError(
            f"cannot construct Holo reward adapter {module_name}.{class_name}: {exc}"
        ) from exc
    if not callable(getattr(reward, "score_batch", None)):
        raise HoloProtocolError("Holo reward adapter must expose score_batch")
    input_paths: dict[str, Path] = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            candidate = _resolve(repo_root, value)
            if candidate.is_file():
                input_paths[f"reward_adapter_{key}"] = candidate
    return reward, input_paths


def run_online(search_config: Path | str, *, repo_root: Path | str) -> None:
    """Run one exact prepaid-archive Holo cell and emit the common study streams."""

    root = Path(repo_root).resolve()
    config_path = _resolve(root, search_config)
    resolved = _load_json(config_path)
    if resolved.get("runner") != "holo_ga" or resolved.get("metadata", {}).get("plot_label") != "holo_ga":
        raise HoloProtocolError("resolved config does not identify a Holo GA cell")
    protocol = resolved.get("holo")
    if not isinstance(protocol, dict):
        raise HoloProtocolError("resolved Holo protocol is missing")

    sequence_length = int(resolved["sequence_length"])
    population_size = int(protocol["population_size"])
    initial_count = int(resolved["initial_oracle_evaluations"])
    generation_count = int(resolved["num_iterations"])
    total_budget = int(resolved["max_oracle_evaluations"])
    alphabet = str(protocol.get("alphabet", AMINO_ACIDS))
    if (
        len(alphabet) != int(protocol["vocab_size"])
        or len(set(alphabet)) != len(alphabet)
        or int(resolved["batch_size"]) != population_size
        or int(resolved["al_oracle_batch_size"]) != population_size
        or initial_count + generation_count * population_size != total_budget
    ):
        raise HoloProtocolError("Holo cell is not an exact initial + whole-generation budget")

    initial_path = _resolve(root, resolved["initial_batch_path"])
    initial_rows, initial_sequences, initial_scores = _initial_archive(
        initial_path,
        expected_count=initial_count,
        sequence_length=sequence_length,
        vocab_size=int(protocol["vocab_size"]),
    )
    reward, reward_inputs = _online_reward_adapter(resolved, repo_root=root)
    baseline = HoloGeneticAlgorithmBaseline(
        population_size=population_size,
        mutation_prob=float(protocol["mutation_prob"]),
        recombine_prob=float(protocol["recombine_prob"]),
        survival_quantile=float(protocol["survival_quantile"]),
        vocab_size=int(protocol["vocab_size"]),
    )

    outputs = {
        key: _resolve(root, resolved[key])
        for key in ("initial_batch_out", "samples_out", "proposal_diagnostics_out", "proposal_traces_out")
    }
    manifest_path = _resolve(root, resolved["run_manifest_out"])
    command = [
        os.sys.executable,
        "-m",
        "genturbo.sequence_baseline_study.holo",
        "--search-config",
        str(config_path),
    ]
    manifest = build_run_manifest(
        resolved_config=resolved,
        command=command,
        seed=int(resolved["seed"]),
        input_paths={
            "runtime_config": _resolve(root, resolved["config"]),
            "initial_batch": initial_path,
            "search_config": config_path,
            **reward_inputs,
        },
        repo_root=root,
    )

    expected_new = generation_count * population_size
    with RunManifestRecorder(manifest, manifest_path, output_paths=outputs.values()) as recorder:
        write_jsonl(outputs["initial_batch_out"], iter(initial_rows))
        result = baseline.run(
            None,
            reward,
            budget=total_budget,
            batch_size=population_size,
            seed=int(resolved["seed"]),
            initial_population=initial_sequences,
            initial_scores=initial_scores,
            device=str(resolved["device"]),
        )
        if (
            result.oracle_calls != expected_new
            or len(result.samples) != expected_new
            or len(result.trace) != generation_count
            or int(result.metadata["total_charged_oracle_calls"]) != total_budget
        ):
            raise HoloProtocolError("Holo implementation did not consume the exact charged budget")

        initial_array = initial_sequences.detach().cpu().numpy()
        archive = {tuple(int(token) for token in row) for row in initial_array.tolist()}
        with (
            open_text(outputs["samples_out"], "wt") as samples_stream,
            open_text(outputs["proposal_diagnostics_out"], "wt") as diagnostics_stream,
            open_text(outputs["proposal_traces_out"], "wt") as traces_stream,
        ):
            for generation in range(generation_count):
                start = generation * population_size
                stop = start + population_size
                batch_rows: list[dict[str, Any]] = []
                for offset, (tensor, raw_score) in enumerate(
                    zip(result.samples[start:stop], result.scores[start:stop], strict=True)
                ):
                    sequence = [int(token) for token in tensor.detach().cpu().view(-1).tolist()]
                    sample = _sample_row(
                        sequence,
                        float(raw_score),
                        batch_id=generation,
                        seq_idx=initial_count + start + offset,
                        alphabet=alphabet,
                    )
                    ancestor, edits = _nearest_initial(sequence, initial_array)
                    key = tuple(sequence)
                    trace = _trace_row(sample, ancestor=ancestor, edits=edits, duplicate=key in archive)
                    archive.add(key)
                    batch_rows.append(sample)
                    _json_line(samples_stream, sample)
                    _json_line(traces_stream, trace)
                total = initial_count + stop
                _json_line(
                    diagnostics_stream,
                    _diagnostic_row(
                        generation,
                        batch_rows,
                        total=total,
                        survivor_count=int(result.trace[generation]["survivor_count"]),
                        run_metadata=result.metadata,
                    ),
                )
        recorder.manifest.trace_records_written = expected_new


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--search-config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_online(args.search_config, repo_root=args.repo_root)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())


__all__ = ["HoloProtocolError", "run_online"]
