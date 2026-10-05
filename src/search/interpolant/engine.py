"""Execution engine for interpolant CTMC search."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TextIO

import torch

from adapters.protein.validation import CANONICAL_AA, validate_sequence
from eval.seq_eval import score_batch
from flows.ctmc_adapter import CTMCFlowAdapter
from monitoring import TensorBoardMonitor
from processes.dfm_mask import _build_time_grid
from runners.common import (
    load_target_tensor,
    parse_reward_spec,
    restore_process_from_checkpoint,
    summarise_search_batch,
)
from search.interpolant.active_learning import (
    DEFAULT_PROTEIN_ONEHOT_ALPHABET,
    ActiveLearningConfig,
    EhrlichAdditiveSpacedNGramExpectedImprovementSelector,
    EhrlichDiffusionExpectedImprovementSelector,
    EhrlichOneHotExpectedImprovementSelector,
    EhrlichSpacedNGramExpectedImprovementSelector,
    OneHotTanimotoExpectedImprovementSelector,
)
from search.interpolant.config import OutputPolicy, SearchConfig, adapt_legacy_namespace_config
from search.interpolant.evaluator import (
    DiagnosticInjectedCandidate,
    ProposalChangeStats,
    ProposalEvaluation,
    ProposalTrace,
    evaluate_proposals,
)
from search.interpolant.manifest import RunManifestRecorder, build_run_manifest
from search.interpolant.output import LimitedJsonlWriter, compressed_jsonl_path, open_jsonl
from search.interpolant.policies import (
    BestSurvivorSelectionPolicy,
    InterpolantProposalPolicy,
    Proposal,
    ProposalBatch,
    resolve_candidate_config,
)
from search.interpolant.runtime import SearchRuntime
from search.interpolant.timing import TimingRecorder
from search.interpolant.utils import elite_position_probs, update_elite_pool
from surrogate.model import make_local_spaced_ngram_offsets
from toy_reward import ToyReward, resolve_reward
from genturbo.configuration import load_config


@dataclass(slots=True)
class MultiRegionState:
    region_id: int
    proposal_policy: InterpolantProposalPolicy
    selector: Any
    incumbent: torch.Tensor
    incumbent_score: float
    boundary_streak: int = 0
    inactive_streak: int = 0
    restart_count: int = 0


@dataclass(slots=True)
class MultiRegionCandidate:
    region_id: int
    proposal: Proposal
    trace: ProposalTrace


SelectorFactory = Callable[[ActiveLearningConfig], Any]


def _resolve_gfp_aav_protein(args: argparse.Namespace, runtime: SearchRuntime) -> Optional[str]:
    reward_spec = getattr(args, "reward", None)
    if isinstance(reward_spec, dict):
        protein = reward_spec.get("protein")
        if protein:
            return str(protein).strip().upper()
    reward_protein = getattr(runtime.reward_fn, "protein", None)
    if reward_protein:
        return str(reward_protein).strip().upper()
    task = str(getattr(args, "benchmark_task", "") or getattr(args, "protein_objective", "") or "").lower()
    if task.startswith("gfp"):
        return "GFP"
    if task.startswith("aav"):
        return "AAV"
    return None


def _resolve_benchmark_root(args: argparse.Namespace) -> Path:
    reward_spec = getattr(args, "reward", None)
    if isinstance(reward_spec, dict) and reward_spec.get("benchmark_root"):
        return Path(str(reward_spec["benchmark_root"])).expanduser()
    return Path("benchmarks/gfp_aav")


def _tokenize_protein_sequence_for_runtime(
    sequence: str,
    *,
    seq_len: int,
    alphabet: str,
    device: torch.device,
) -> torch.Tensor:
    clean = validate_sequence(
        sequence,
        min_len=int(seq_len),
        max_len=int(seq_len),
        alphabet=str(alphabet),
    )
    aa_to_idx = {aa: idx for idx, aa in enumerate(str(alphabet))}
    return torch.tensor(
        [[aa_to_idx[aa] for aa in clean]],
        dtype=torch.long,
        device=device,
    )


def _load_full_dataset_best_candidate(
    args: argparse.Namespace,
    runtime: SearchRuntime,
) -> Optional[DiagnosticInjectedCandidate]:
    if not bool(getattr(args, "diagnostic_inject_full_dataset_best", False)):
        return None

    source = str(getattr(args, "diagnostic_full_dataset_best_source", "dms") or "dms").strip().lower()
    if source not in {"dms", "full_dms", "ground_truth"}:
        raise ValueError("diagnostic_full_dataset_best_source currently supports only 'dms'")

    protein = _resolve_gfp_aav_protein(args, runtime)
    if protein not in {"GFP", "AAV"}:
        raise ValueError("diagnostic_inject_full_dataset_best requires a GFP/AAV benchmark task or reward protein.")

    explicit_csv = getattr(args, "diagnostic_full_dataset_best_csv", None)
    if explicit_csv:
        csv_path = Path(str(explicit_csv)).expanduser()
    else:
        csv_path = _resolve_benchmark_root(args) / "raw" / protein / "ground_truth.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Full-DMS ground-truth CSV not found: {csv_path}")

    best_sequence: Optional[str] = None
    best_score = float("-inf")
    best_row_idx: Optional[int] = None
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"Full-DMS CSV has no header: {csv_path}")
        missing = {"sequence", "score"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Full-DMS CSV {csv_path} missing required columns: {sorted(missing)}")
        for row_idx, row in enumerate(reader):
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                continue
            sequence = str(row["sequence"]).strip().upper()
            if not sequence:
                continue
            if score > best_score:
                best_score = score
                best_sequence = sequence
                best_row_idx = row_idx
    if best_sequence is None:
        raise ValueError(f"Full-DMS CSV did not contain any scored sequences: {csv_path}")

    alphabet = str(getattr(runtime.reward_fn, "decode_alphabet", CANONICAL_AA) or CANONICAL_AA)
    tokenized = _tokenize_protein_sequence_for_runtime(
        best_sequence,
        seq_len=int(runtime.seq_len),
        alphabet=alphabet,
        device=runtime.device,
    )
    print(
        "[diagnostic] injecting full-DMS best candidate into each acquisition pool: "
        f"protein={protein} row={best_row_idx} score={best_score:.6g} "
        f"sequence={best_sequence[:24]}{'...' if len(best_sequence) > 24 else ''}"
    )
    return DiagnosticInjectedCandidate(
        sequence=tokenized,
        sequence_string=best_sequence,
        source=f"{protein.lower()}_full_dms_best",
        full_dataset_score=float(best_score),
    )


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    raw_config: dict[str, Any] = {}
    if args.search_config:
        loaded = load_config(args.search_config)
        if not isinstance(loaded, dict):
            raise TypeError("search config must contain a JSON object")
        raw_config = loaded
    # Output-policy validation happens on the resolved CLI-over-config values.
    # This permits a CLI summary override of an archived full-trace config while
    # still rejecting an unbounded full trace before any output is created.
    resolved_for_validation = dict(raw_config)
    provided = set(getattr(args, "_provided", ()))
    for key in ("trace_level", "trace_record_limit", "compress_jsonl"):
        if key in provided:
            resolved_for_validation[key] = getattr(args, key)
    args, search_config = adapt_legacy_namespace_config(args, resolved_for_validation)
    output_policy = OutputPolicy(
        trace_level=str(getattr(args, "trace_level", None) or "summary").strip().lower(),  # type: ignore[arg-type]
        trace_record_limit=getattr(args, "trace_record_limit", None),
        compress_jsonl=(bool(args.compress_jsonl) if getattr(args, "compress_jsonl", None) is not None else True),
    )
    args._search_config = SearchConfig(
        values=search_config.values,
        benchmark_task=search_config.benchmark_task,
        metadata=search_config.metadata,
        output_policy=output_policy,
    )
    args._output_policy = output_policy
    args._trace_level_explicit = "trace_level" in raw_config or "trace_level" in provided
    objective_direction = getattr(args, "objective_direction", None)
    if objective_direction is not None and str(objective_direction).strip().lower() not in {"max", "maximize"}:
        raise ValueError("objective_direction currently must be 'maximize'")
    if args.seed is None:
        args.seed = 0
    torch.manual_seed(args.seed)
    return args


def _optional_int_attr(obj: Any, name: str) -> Optional[int]:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_protein_reward_spec(
    args: argparse.Namespace,
    *,
    process: Any,
    process_cfg: dict[str, Any],
    model_for_state: Optional[torch.nn.Module],
) -> dict[str, Any]:
    objective = str(getattr(args, "protein_objective", None) or "toy_amp").strip().lower()
    if objective not in {"toy_amp", "toy_amp_objective", "amp"}:
        raise ValueError("protein_objective must be one of: toy_amp")
    spec: dict[str, Any] = {"name": "protein_toy_amp"}
    cfg = getattr(process, "cfg", None)
    pad_token_id = _optional_int_attr(model_for_state, "pad_token_id") if model_for_state is not None else None
    if pad_token_id is None:
        raw_pad = process_cfg.get("pad_token_id")
        pad_token_id = int(raw_pad) if raw_pad is not None else None
    mask_token_id = _optional_int_attr(cfg, "mask_id")
    if mask_token_id is None:
        raw_mask = process_cfg.get("mask_id")
        mask_token_id = int(raw_mask) if raw_mask is not None else None
    if pad_token_id is not None:
        spec["pad_token_id"] = pad_token_id
    if mask_token_id is not None:
        spec["mask_token_id"] = mask_token_id
    if getattr(args, "protein_min_len", None) is not None:
        spec["min_len"] = int(args.protein_min_len)
    if getattr(args, "protein_max_len", None) is not None:
        spec["max_len"] = int(args.protein_max_len)
    return spec


def _build_runtime(args: argparse.Namespace) -> SearchRuntime:
    if args.checkpoint is None:
        raise ValueError("checkpoint must be provided (via CLI or search_config)")

    restored = restore_process_from_checkpoint(
        args.checkpoint,
        config_path=args.config,
        device_override=args.device,
        sequence_length_override=getattr(args, "sequence_length", None),
    )
    cfg = restored.config
    process_cfg = restored.process_cfg
    process = restored.process
    model_for_state = restored.model_for_state
    device = restored.device

    flow_steps = int(args.flow_steps or 200)
    max_time = float(args.max_time or 1.0)
    schedule = getattr(process, "sample_time_schedule", "linear")
    schedule_cfg = getattr(process, "sample_time_schedule_cfg", None) or {}
    time_grid_time = _build_time_grid(flow_steps, max_time, schedule=schedule, schedule_cfg=schedule_cfg)
    time_grid_time = time_grid_time.clone()
    time_grid_time[0] = 0.0
    time_grid_time[-1] = float(max_time)
    time_grid_norm = (time_grid_time / float(max_time)).to(device)

    temperature = float(args.temperature) if args.temperature is not None else getattr(process, "temperature", 1.0)
    adapter = CTMCFlowAdapter(
        process=process,
        steps=flow_steps,
        max_time=max_time,
        device=device,
        stop_when_filled=bool(args.stop_when_filled) if args.stop_when_filled is not None else False,
        temperature=temperature,
        time_grid=time_grid_time,
    )
    if cfg.get("process") == "sokoban_masked_diffusion":
        from sokoban.search import SokobanRolloutAdapter

        if flow_steps != 100 or max_time != 1.0:
            raise ValueError("Sokoban search requires flow_steps=100 and max_time=1")
        adapter = SokobanRolloutAdapter(process, temperature=temperature)

    seq_len = int(process_cfg.get("seq_len", getattr(process.cfg, "seq_len", 0)))
    vocab_size = int(process_cfg.get("vocab_size", getattr(process.cfg, "vocab_size", 0)))

    oracle_spec = getattr(args, "oracle", None)
    if oracle_spec is not None:
        reward_spec = parse_reward_spec(None, oracle_spec)
    else:
        reward_spec = parse_reward_spec(args.reward_config, args.reward)
    if reward_spec is None and (
        bool(getattr(args, "protein_mode", False)) or getattr(args, "protein_objective", None) is not None
    ):
        reward_spec = _default_protein_reward_spec(
            args,
            process=process,
            process_cfg=process_cfg,
            model_for_state=model_for_state,
        )
    reward_fn = resolve_reward(reward_spec) if reward_spec is not None else None
    if reward_fn is None:
        target = load_target_tensor(
            args.target_sequence,
            use_max=bool(args.target_max),
            seq_len=seq_len,
            vocab_size=vocab_size,
            prior_sampler=adapter.sample_prior,
        )
        reward_fn = ToyReward(target=target.to(device))

    transition_mask = getattr(process, "transition_mask", None)
    if transition_mask is None and args.transition_mask_path:
        loaded = torch.load(Path(args.transition_mask_path).expanduser(), map_location="cpu", weights_only=False)
        if isinstance(loaded, torch.Tensor):
            transition_mask = loaded
        elif isinstance(loaded, dict) and "transition_mask" in loaded:
            transition_mask = loaded["transition_mask"]
        else:
            transition_mask = getattr(loaded, "transition_mask", None)

    return SearchRuntime(
        cfg=cfg,
        process_cfg=process_cfg,
        process=process,
        model_for_state=model_for_state,
        device=device,
        adapter=adapter,
        flow_steps=flow_steps,
        max_time=max_time,
        time_grid_norm=time_grid_norm,
        seq_len=seq_len,
        vocab_size=vocab_size,
        reward_fn=reward_fn,
        transition_mask=transition_mask,
        temperature=temperature,
    )


def _ensure_samples_output(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    output_policy = getattr(args, "_output_policy", OutputPolicy())
    if args.samples_out is not None:
        if str(args.samples_out) != "/dev/null" and bool(output_policy.compress_jsonl):
            args.samples_out = str(compressed_jsonl_path(args.samples_out))
        return
    run_root = cfg.get("run_root")
    run_name = cfg.get("run_name")
    if not (run_root and run_name):
        return
    samples_dir = Path(run_root).expanduser() / str(run_name)
    extension = ".jsonl.gz" if bool(output_policy.compress_jsonl) else ".jsonl"
    samples_path = samples_dir / f"interpolant_samples{extension}"
    args.samples_out = str(samples_path)


def _open_outputs(
    args: argparse.Namespace, cfg: dict[str, Any]
) -> tuple[Optional[TensorBoardMonitor], Optional[TextIO]]:
    _ensure_samples_output(args, cfg)
    monitor = TensorBoardMonitor(args.log_dir) if args.log_dir else None
    samples_fp = None
    try:
        if args.samples_out:
            samples_path = Path(args.samples_out).expanduser()
            samples_fp = open_jsonl(samples_path, "wt")
    except BaseException:
        if monitor is not None:
            monitor.close()
        raise
    return monitor, samples_fp


def _derived_jsonl_path(samples_out: Optional[str], filename: str) -> Optional[Path]:
    if not samples_out:
        return None
    path = Path(samples_out).expanduser()
    if str(path) == "/dev/null":
        return None
    derived = path.parent / filename
    if path.suffix == ".gz" and derived.suffix != ".gz":
        derived = Path(f"{derived}.gz")
    return derived


def _open_jsonl_output(
    path: Optional[str | Path],
    *,
    record_limit: Optional[int] = None,
) -> Optional[TextIO | LimitedJsonlWriter]:
    if path is None:
        return None
    out_path = Path(path).expanduser()
    if str(out_path) == "/dev/null":
        return None
    return open_jsonl(out_path, "wt", record_limit=record_limit)


def _open_proposal_outputs(args: argparse.Namespace) -> tuple[Optional[TextIO], Optional[TextIO]]:
    diagnostics_path, traces_path, trace_limit = _proposal_output_paths(args)
    diagnostics_fp = _open_jsonl_output(diagnostics_path)
    try:
        traces_fp = _open_jsonl_output(traces_path, record_limit=trace_limit)
    except BaseException:
        if diagnostics_fp is not None:
            diagnostics_fp.close()
        raise
    return diagnostics_fp, traces_fp


def _compressed_policy_path(path: Optional[str | Path], output_policy: OutputPolicy) -> Optional[Path]:
    if path is None:
        return None
    value = Path(path).expanduser()
    if str(value) == "/dev/null":
        return None
    if output_policy.compress_jsonl and value.name.endswith(".jsonl"):
        return compressed_jsonl_path(value)
    return value


def _proposal_output_paths(
    args: argparse.Namespace,
) -> tuple[Optional[Path], Optional[Path], Optional[int]]:
    output_policy = getattr(args, "_output_policy", OutputPolicy())
    if output_policy.trace_level == "none":
        if getattr(args, "proposal_traces_out", None):
            raise ValueError("proposal_traces_out requires trace_level='full'")
        return None, None, None
    write_diagnostics = (
        bool(args.write_proposal_diagnostics)
        if getattr(args, "write_proposal_diagnostics", None) is not None
        else bool(_derived_jsonl_path(args.samples_out, "proposal_diagnostics.jsonl") is not None)
    )
    diagnostics_path = (
        Path(args.proposal_diagnostics_out).expanduser()
        if getattr(args, "proposal_diagnostics_out", None)
        else (_derived_jsonl_path(args.samples_out, "proposal_diagnostics.jsonl") if write_diagnostics else None)
    )
    diagnostics_path = _compressed_policy_path(diagnostics_path, output_policy)
    # Scoring a full proposal batch is an algorithm/diagnostic choice, not
    # permission to emit an unbounded trace.  Full trace output is always an
    # explicit, record-limited output-policy decision.
    write_full_traces = output_policy.trace_level == "full"
    if not write_full_traces and getattr(args, "proposal_traces_out", None):
        raise ValueError("proposal_traces_out requires trace_level='full'")
    traces_path = None
    if write_full_traces:
        traces_path = (
            Path(args.proposal_traces_out).expanduser()
            if getattr(args, "proposal_traces_out", None)
            else _derived_jsonl_path(args.samples_out, "proposal_traces.jsonl")
        )
        traces_path = _compressed_policy_path(traces_path, output_policy)
    trace_limit = output_policy.trace_record_limit if output_policy.trace_level == "full" else None
    return diagnostics_path, traces_path, trace_limit


def _open_timing_output(args: argparse.Namespace, *, profile_timings: bool) -> Optional[TextIO]:
    return _open_jsonl_output(_timing_output_path(args, profile_timings=profile_timings))


def _timing_output_path(args: argparse.Namespace, *, profile_timings: bool) -> Optional[Path]:
    if not bool(profile_timings):
        return None
    path = (
        Path(args.timings_out).expanduser()
        if getattr(args, "timings_out", None)
        else _derived_jsonl_path(args.samples_out, "timings.jsonl")
    )
    return _compressed_policy_path(path, getattr(args, "_output_policy", OutputPolicy()))


def _validate_output_paths(
    args: argparse.Namespace,
    *,
    manifest_path: Path,
    profile_timings: bool,
) -> None:
    """Reject colliding, non-regular, or occupied outputs before opening anything."""

    diagnostics_path, traces_path, _trace_limit = _proposal_output_paths(args)
    outputs: dict[str, Optional[Path]] = {
        "run_manifest_out": manifest_path,
        "samples_out": (
            Path(args.samples_out).expanduser()
            if getattr(args, "samples_out", None) and str(args.samples_out) != "/dev/null"
            else None
        ),
        "proposal_diagnostics_out": diagnostics_path,
        "proposal_traces_out": traces_path,
        "timings_out": _timing_output_path(args, profile_timings=profile_timings),
        "initial_batch_out": _resolve_initial_batch_output_path(args),
        "leaves_out": (Path(args.leaves_out).expanduser() if getattr(args, "leaves_out", None) else None),
    }
    seen: dict[Path, str] = {}
    occupied: list[tuple[str, Path]] = []
    for label, path in outputs.items():
        if path is None:
            continue
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"{label} must be a regular file target: {path}")
        canonical = path.resolve(strict=False)
        previous = seen.get(canonical)
        if previous is not None:
            raise ValueError(f"output path collision: {previous} and {label} both resolve to {path}")
        seen[canonical] = label
        if path.exists():
            occupied.append((label, path))
    raw_log_dir = getattr(args, "log_dir", None)
    if raw_log_dir:
        log_dir = Path(raw_log_dir).expanduser()
        if log_dir.is_symlink() or (log_dir.exists() and not log_dir.is_dir()):
            raise ValueError(f"log_dir must be a directory target: {log_dir}")
        canonical = log_dir.resolve(strict=False)
        previous = seen.get(canonical)
        if previous is not None:
            raise ValueError(f"output path collision: {previous} and log_dir both resolve to {log_dir}")
        seen[canonical] = "log_dir"
        if log_dir.exists():
            try:
                log_dir_occupied = next(log_dir.iterdir(), None) is not None
            except OSError as exc:
                raise ValueError(f"unable to inspect log_dir before opening it: {log_dir}") from exc
            if log_dir_occupied:
                occupied.append(("log_dir", log_dir))
    if occupied:
        details = ", ".join(f"{label}={path}" for label, path in occupied)
        raise FileExistsError(
            "refusing to overwrite an occupied interpolant search output set: "
            f"{details}. Choose a fresh samples_out path/output directory and an empty or new log_dir."
        )


@dataclass(slots=True)
class _SearchResources:
    stack: ExitStack
    monitor: Optional[TensorBoardMonitor]
    samples_fp: Optional[TextIO]
    proposal_diagnostics_fp: Optional[TextIO]
    proposal_traces_fp: Optional[TextIO]
    timing_fp: Optional[TextIO]

    def close(self) -> None:
        self.stack.__exit__(*sys.exc_info())


def _stream_path(stream: Any) -> Optional[Path]:
    if stream is None:
        return None
    name = getattr(stream, "name", None)
    if not name or str(name).startswith("<"):
        return None
    return Path(str(name)).expanduser()


def _reward_instance_input_path(args: argparse.Namespace) -> str | Path | None:
    """Return the file-backed reward instance used by the resolved run, if any."""

    oracle_spec = getattr(args, "oracle", None)
    reward_spec = (
        parse_reward_spec(None, oracle_spec)
        if oracle_spec is not None
        else parse_reward_spec(
            getattr(args, "reward_config", None),
            getattr(args, "reward", None),
        )
    )
    if isinstance(reward_spec, dict):
        instance_path = reward_spec.get("instance_path")
        if isinstance(instance_path, (str, Path)) and str(instance_path).strip():
            return instance_path
    return None


def _open_search_resources(
    args: argparse.Namespace,
    runtime: SearchRuntime,
    *,
    profile_timings: bool,
) -> _SearchResources:
    """Open all run resources atomically and register deterministic cleanup."""

    _ensure_samples_output(args, runtime.cfg)
    if getattr(args, "run_manifest_out", None):
        manifest_path: Optional[Path] = Path(args.run_manifest_out).expanduser()
    elif getattr(args, "samples_out", None) and str(args.samples_out) != "/dev/null":
        manifest_path = Path(str(args.samples_out)).expanduser().parent / "run_manifest.json"
    else:
        manifest_path = None
    if manifest_path is None:
        manifest_path = Path("runs/search/manifests") / (
            f"run_manifest_seed{int(getattr(args, 'seed', 0) or 0)}_pid{os.getpid()}_{uuid.uuid4().hex[:12]}.json"
        )

    _validate_output_paths(
        args,
        manifest_path=manifest_path,
        profile_timings=profile_timings,
    )

    search_config = getattr(args, "_search_config", None)
    if isinstance(search_config, SearchConfig):
        resolved_config = search_config.resolved_flat(args)
    else:
        resolved_config = {
            key: value for key, value in vars(args).items() if not key.startswith("_") and value is not None
        }
    manifest = build_run_manifest(
        resolved_config=resolved_config,
        command=[sys.executable, *sys.argv],
        seed=int(getattr(args, "seed", 0) or 0),
        input_paths={
            "checkpoint": getattr(args, "checkpoint", None),
            "search_config": getattr(args, "search_config", None),
            "runtime_config": getattr(args, "config", None),
            "reward_config": getattr(args, "reward_config", None),
            "reward_instance": _reward_instance_input_path(args),
            "transition_mask": getattr(args, "transition_mask_path", None),
            "initial_batch": getattr(args, "initial_batch_path", None),
            "target_sequence": getattr(args, "target_sequence", None),
            "diagnostic_full_dataset_best_csv": getattr(
                args,
                "diagnostic_full_dataset_best_csv",
                None,
            ),
        },
    )

    diagnostics_path, traces_path, _trace_limit = _proposal_output_paths(args)
    planned_outputs = [
        (
            Path(args.samples_out).expanduser()
            if getattr(args, "samples_out", None) and str(args.samples_out) != "/dev/null"
            else None
        ),
        diagnostics_path,
        traces_path,
        _timing_output_path(args, profile_timings=profile_timings),
        _resolve_initial_batch_output_path(args),
        Path(args.leaves_out).expanduser() if getattr(args, "leaves_out", None) else None,
    ]
    stack = ExitStack()
    recorder = RunManifestRecorder(
        manifest,
        manifest_path,
        output_paths=[path for path in planned_outputs if path is not None],
        only_changed_outputs=True,
    )
    stack.enter_context(recorder)
    monitor: Optional[TensorBoardMonitor] = None
    samples_fp: Optional[TextIO] = None
    proposal_diagnostics_fp: Optional[TextIO] = None
    proposal_traces_fp: Optional[TextIO] = None
    timing_fp: Optional[TextIO] = None
    try:
        monitor, samples_fp = _open_outputs(args, runtime.cfg)
        if monitor is not None:
            stack.callback(monitor.close)
        if samples_fp is not None:
            stack.enter_context(samples_fp)
        proposal_diagnostics_fp, proposal_traces_fp = _open_proposal_outputs(args)
        if proposal_diagnostics_fp is not None:
            stack.enter_context(proposal_diagnostics_fp)
        if proposal_traces_fp is not None:
            stack.enter_context(proposal_traces_fp)
        timing_fp = _open_timing_output(args, profile_timings=profile_timings)
        if timing_fp is not None:
            stack.enter_context(timing_fp)
    except BaseException:
        stack.__exit__(*sys.exc_info())
        raise

    recorder.trace_writer = proposal_traces_fp
    opened_output_paths = [
        _stream_path(samples_fp),
        _stream_path(proposal_diagnostics_fp),
        _stream_path(proposal_traces_fp),
        _stream_path(timing_fp),
    ]
    recorder.register_outputs(path for path in opened_output_paths if path is not None)
    return _SearchResources(
        stack=stack,
        monitor=monitor,
        samples_fp=samples_fp,
        proposal_diagnostics_fp=proposal_diagnostics_fp,
        proposal_traces_fp=proposal_traces_fp,
        timing_fp=timing_fp,
    )


def _write_timing_jsonl(
    fp: TextIO,
    *,
    event: str,
    iteration: Optional[int],
    seconds: dict[str, float],
    counts: Optional[dict[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    record: dict[str, Any] = {
        "event": str(event),
        "seconds": {str(key): float(value) for key, value in seconds.items()},
    }
    if iteration is not None:
        record["iteration"] = int(iteration)
    if counts:
        record["counts"] = {str(key): float(value) for key, value in counts.items()}
    if extra:
        record.update(extra)
    fp.write(json.dumps(record, sort_keys=True) + "\n")


def _resolve_initial_batch_output_path(args: argparse.Namespace) -> Optional[Path]:
    output_policy = getattr(args, "_output_policy", OutputPolicy())
    explicit = getattr(args, "initial_batch_out", None)
    if explicit:
        return _compressed_policy_path(explicit, output_policy)
    return _compressed_policy_path(
        _derived_jsonl_path(getattr(args, "samples_out", None), "initial_batch.jsonl"),
        output_policy,
    )


def _initial_num_samples_from_args(
    args: argparse.Namespace,
    *,
    sample_only: bool,
    sample_only_count: Optional[int],
    active_learning_cfg: Optional[ActiveLearningConfig],
) -> int:
    configured_initial = getattr(args, "initial_oracle_evaluations", None)
    if sample_only_count is not None:
        initial_num_samples = sample_only_count
    elif configured_initial is not None:
        initial_num_samples = int(configured_initial)
    else:
        initial_num_samples = int(args.batch_size or 256)
    if int(initial_num_samples) <= 0:
        raise ValueError("initial_oracle_evaluations must be > 0 when provided")
    if (
        not sample_only
        and active_learning_cfg is not None
        and bool(active_learning_cfg.initial_batch_matches_buffer_size)
    ):
        initial_num_samples = max(1, int(active_learning_cfg.buffer_size))
    return int(initial_num_samples)


def _load_initial_batch_jsonl(
    path: str | Path,
    *,
    device: torch.device,
    seq_len: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    rows: list[list[int]] = []
    scores: list[Optional[float]] = []
    input_path = Path(path).expanduser()
    with open_jsonl(input_path, "rt") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in initial_batch_path={input_path} line {line_no}") from exc
            seq = record.get("seq")
            if not isinstance(seq, list):
                raise ValueError(f"initial_batch_path={input_path} line {line_no} is missing list field 'seq'")
            values = [int(v) for v in seq]
            if len(values) != int(seq_len):
                raise ValueError(
                    f"initial_batch_path={input_path} line {line_no} has seq length {len(values)}, "
                    f"expected {int(seq_len)}"
                )
            rows.append(values)
            score_value = record.get("score")
            if score_value is None:
                # Initial archives deliberately serialize non-finite Ehrlich
                # rewards as JSON ``null``.  The explicit flag distinguishes
                # a known infeasible observation (-inf for this maximization
                # loop) from a historical row whose score is genuinely
                # absent and therefore still needs oracle evaluation.
                scores.append(float("-inf") if record.get("score_is_finite") is False else None)
                continue
            try:
                score_float = float(score_value)
            except (TypeError, ValueError):
                scores.append(None)
                continue
            scores.append(score_float if math.isfinite(score_float) else None)
    if not rows:
        raise ValueError(f"initial_batch_path={input_path} did not contain any sequences")
    batch = torch.as_tensor(rows, dtype=torch.long, device=device)
    if any(value is None for value in scores):
        return batch, None
    score_tensor = torch.as_tensor([float(value) for value in scores], dtype=torch.float, device=device)
    return batch, score_tensor


def _decode_sequences_for_output(reward_fn: Any, sequences: torch.Tensor) -> list[Optional[str]]:
    decode_batch = getattr(reward_fn, "decode_batch", None)
    batch_size = int(sequences.shape[0]) if sequences.ndim > 1 else 1
    if not callable(decode_batch):
        return [None] * batch_size
    try:
        decoded = list(decode_batch(sequences))
    except Exception as exc:
        raise RuntimeError(f"reward decoder failed while preparing {batch_size} output sequences") from exc
    if len(decoded) != batch_size:
        raise RuntimeError(f"reward decoder returned {len(decoded)} rows for {batch_size} output sequences")
    return [None if value is None else str(value) for value in decoded]


@dataclass(slots=True)
class _InitialSearchState:
    batch: torch.Tensor
    rewards: torch.Tensor
    oracle_evaluated: int
    sample_source: str
    loaded_rewards: Optional[torch.Tensor]
    timings: TimingRecorder


def _sample_initial_batch(
    args: argparse.Namespace,
    runtime: SearchRuntime,
    *,
    num_samples: int,
) -> torch.Tensor:
    """Sample an initial cohort, optionally chunking sample-only generation."""

    raw_chunk_size = getattr(args, "sample_only_batch_size", None)
    chunk_size = int(raw_chunk_size) if raw_chunk_size is not None else int(num_samples)
    chunks = [
        runtime.process.sample(
            num_samples=min(chunk_size, int(num_samples) - start),
            steps=runtime.adapter.T,
            device=runtime.device,
        )
        for start in range(0, int(num_samples), chunk_size)
    ]
    if not chunks:
        raise ValueError("initial sampling requires at least one sample")
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


def _initialize_search_state(
    *,
    args: argparse.Namespace,
    runtime: SearchRuntime,
    planned_num_samples: int,
    max_oracle_evaluations: Optional[int],
    pad_token_id: Optional[int],
    active_learning_selector: Optional[Any],
    selection_policy: BestSurvivorSelectionPolicy,
    profile_timings: bool,
    profile_sync_cuda: bool,
) -> _InitialSearchState:
    """Create and observe the initial archive while a run manifest is active."""

    initial_batch_path = getattr(args, "initial_batch_path", None)
    sample_source = "model_initial_batch"
    loaded_rewards: Optional[torch.Tensor] = None
    timer = TimingRecorder(
        enabled=profile_timings,
        sync_cuda=profile_sync_cuda,
        device=runtime.device,
    )
    total_start = timer.start()
    if initial_batch_path:
        with timer.measure("load_initial_batch_seconds"):
            batch, loaded_rewards = _load_initial_batch_jsonl(
                initial_batch_path,
                device=runtime.device,
                seq_len=runtime.seq_len,
            )
        if max_oracle_evaluations is not None and int(batch.shape[0]) > int(max_oracle_evaluations):
            retained = int(max_oracle_evaluations)
            print(
                "[search] truncating loaded initial archive to respect "
                f"max_oracle_evaluations={retained}: "
                f"loaded={int(batch.shape[0])} retained={retained}."
            )
            batch = batch[:retained]
            if loaded_rewards is not None:
                loaded_rewards = loaded_rewards[:retained]
        advance_rng = (
            bool(args.initial_batch_advance_rng)
            if getattr(args, "initial_batch_advance_rng", None) is not None
            else True
        )
        if advance_rng:
            with timer.measure("advance_rng_sample_seconds"):
                _ = _sample_initial_batch(
                    args,
                    runtime,
                    num_samples=planned_num_samples,
                )
        rescore = bool(getattr(args, "initial_batch_rescore_with_reward", False))
        if rescore:
            with timer.measure("initial_rescore_seconds"):
                rewards = _score_reward_batch_once(
                    batch,
                    runtime.reward_fn,
                    context="loaded initial-batch rescore",
                )
        elif loaded_rewards is None:
            with timer.measure("initial_score_seconds"):
                rewards = _score_reward_batch_once(
                    batch,
                    runtime.reward_fn,
                    context="loaded initial-batch scoring",
                )
        else:
            rewards = loaded_rewards
        sample_source = "loaded_initial_batch_rescored" if rescore else "loaded_initial_batch"
        print(
            "[search] loaded initial evaluated archive: "
            f"path={initial_batch_path} size={int(batch.shape[0])} "
            f"rescored={int(rescore)}"
        )
    else:
        with timer.measure("initial_sample_seconds"):
            batch = _sample_initial_batch(
                args,
                runtime,
                num_samples=planned_num_samples,
            )
        with timer.measure("initial_score_seconds"):
            rewards = _score_reward_batch_once(
                batch,
                runtime.reward_fn,
                context="initial model batch",
            )

    rewards = torch.as_tensor(rewards, dtype=torch.float, device=runtime.device).view(-1)
    initial_batch_out_path = _resolve_initial_batch_output_path(args)
    if initial_batch_out_path is not None:
        with timer.measure("write_initial_batch_seconds"):
            _write_initial_batch_jsonl(
                initial_batch_out_path,
                sequences=batch,
                scores=rewards,
                pad_token_id=pad_token_id,
                sample_source=sample_source,
                decoded_sequences=_decode_sequences_for_output(runtime.reward_fn, batch),
            )
    if active_learning_selector is not None:
        with timer.measure("initial_acquisition_observe_seconds"):
            _observe_selector_once(
                active_learning_selector,
                batch,
                rewards,
                context="initial active-learning batch",
            )
        if rewards.numel() > 0:
            best_idx = int(torch.argmax(rewards).item())
            selection_policy.best_overall = batch[best_idx : best_idx + 1].clone()
            selection_policy.best_reward = float(rewards[best_idx].item())

    timer.add_count("initial_batch_size", int(batch.shape[0]))
    timer.add_count("planned_initial_batch_size", int(planned_num_samples))
    timer.add_count("initial_rewards_loaded", 1 if initial_batch_path and loaded_rewards is not None else 0)
    timer.add_count(
        "initial_rewards_rescored",
        1 if initial_batch_path and bool(getattr(args, "initial_batch_rescore_with_reward", False)) else 0,
    )
    timer.stop("total_seconds", total_start)
    return _InitialSearchState(
        batch=batch,
        rewards=rewards,
        oracle_evaluated=int(batch.shape[0]),
        sample_source=sample_source,
        loaded_rewards=loaded_rewards,
        timings=timer,
    )


def _log_summary(
    monitor: Optional[TensorBoardMonitor],
    iteration: int,
    summary: dict[str, Optional[float]],
) -> None:
    if monitor is None:
        return
    monitor.log_iteration(
        iteration,
        {
            "search/unique_count": summary["unique_count"],
            "search/valid_fraction": summary["valid_fraction"],
            "search/top_score": summary["top_score"],
            "search/avg_score": summary["avg_score"],
            "search/unique_frac": summary["unique_frac"],
            "search/pairwise_hamming": summary["pairwise_hamming"],
            "search/unique_frac_valid": summary["unique_frac_valid"],
            "search/pairwise_hamming_valid": summary["pairwise_hamming_valid"],
        },
    )


def _mean_normalized_edit_error(
    traces: list[ProposalTrace],
    *,
    editable_seq_len_for_sample: Callable[[torch.Tensor], int],
) -> Optional[float]:
    if not traces:
        return None
    normalized_errors: list[float] = []
    for trace in traces:
        editable_seq_len = int(max(1, editable_seq_len_for_sample(trace.ancestor_seq)))
        error = float(trace.expected_edits) - float(trace.ancestor_to_child_edits)
        normalized_errors.append(error / float(editable_seq_len))
    if not normalized_errors:
        return None
    return float(sum(normalized_errors) / float(len(normalized_errors)))


def _log_change_stats(
    monitor: Optional[TensorBoardMonitor],
    iteration: int,
    stats: ProposalChangeStats,
) -> None:
    if monitor is None:
        return
    monitor.log_iteration(
        iteration,
        {
            "search/proposals_total": float(stats.generated_proposals),
            "search/proposals_deduped": float(stats.deduped_proposals),
            "search/proposals_duplicate_exact": float(stats.duplicate_exact_proposals),
            "search/proposals_duplicate_leaf": float(stats.duplicate_generated_leaves),
            "search/proposals_cached": float(stats.cached_generated_proposals),
            "search/proposals_invalid": float(stats.invalid_generated_proposals),
            "search/acquisition_batch_size": float(stats.acquisition_batch_size),
            "search/model_changes_total": float(stats.model_changes_total),
            "search/model_changes_corrupted": float(stats.model_changes_on_corrupted),
            "search/model_changes_uncorrupted": float(stats.model_changes_on_uncorrupted),
            "search/proposal_edits_logical_total": float(stats.proposal_logical_edits_total),
            "search/model_edits_logical_total": float(stats.model_logical_edits_total),
            "search/proposal_edits_dropped_logical": float(stats.dropped_proposal_logical_edits),
            "search/new_edits_logical": float(stats.introduced_logical_edits),
            "search/changed_props_only_corrupted": float(stats.changed_proposals_only_corrupted),
            "search/changed_props_with_uncorrupted": float(stats.changed_proposals_with_uncorrupted),
            "search/inferred_corrupt_masks": float(stats.inferred_corrupt_masks),
            "search/xt_retained_frac": stats.mean_retained_frac,
            "search/xt_retained_frac_corrupted": stats.mean_retained_frac_corrupted,
            "search/xt_retained_frac_uncorrupted": stats.mean_retained_frac_uncorrupted,
            "search/xt_retained_frac_logical": stats.mean_retained_frac_logical,
        },
    )
    for item in stats.retention_by_time:
        t_key = f"{int(round(float(item.t_value) * 1000.0)):04d}"
        monitor.log_iteration(
            iteration,
            {
                f"search/xt_retained_frac_by_t/{t_key}": item.mean_retained_frac,
                f"search/xt_retained_frac_corrupted_by_t/{t_key}": item.mean_retained_frac_corrupted,
                f"search/xt_retained_frac_uncorrupted_by_t/{t_key}": item.mean_retained_frac_uncorrupted,
                f"search/xt_retained_frac_logical_by_t/{t_key}": item.mean_retained_frac_logical,
            },
        )


def _fmt_opt(value: Optional[float]) -> str:
    if value is None:
        return "na"
    return f"{value:.3f}"


def _fmt_edit_count(value: Optional[float | int]) -> str:
    if value is None:
        return "na"
    numeric = float(value)
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def _fmt_debug_float(value: Optional[float]) -> str:
    if value is None:
        return "na"
    return f"{float(value):.3e}"


def _merge_proposal_batches(batches: list[ProposalBatch]) -> ProposalBatch:
    non_empty = [batch for batch in batches if batch is not None]
    if not non_empty:
        raise ValueError("cannot merge an empty proposal batch list")

    base_rows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    proposals: list[Proposal] = []
    generation_diagnostics: dict[str, Any] = {}
    top_reward = float("-inf")
    for batch in non_empty:
        top_reward = max(top_reward, float(batch.top_reward))
        for row_idx, base_idx in enumerate(batch.base_indices):
            idx = int(base_idx)
            if idx not in base_rows:
                base_rows[idx] = (
                    batch.base_samples[row_idx : row_idx + 1].detach().clone(),
                    batch.base_rewards[row_idx : row_idx + 1].detach().clone(),
                )
        proposals.extend(batch.proposals)
        for key, value in batch.generation_diagnostics.items():
            if key.endswith("_count") and isinstance(value, (int, float)):
                generation_diagnostics[key] = generation_diagnostics.get(key, 0) + value
            elif key not in generation_diagnostics:
                generation_diagnostics[key] = value
            elif generation_diagnostics[key] != value:
                raise ValueError(f"cannot merge inconsistent proposal diagnostic {key!r}")

    ordered_indices = sorted(base_rows.keys())
    if ordered_indices:
        base_samples = torch.cat([base_rows[idx][0] for idx in ordered_indices], dim=0)
        base_rewards = torch.cat([base_rows[idx][1].view(1) for idx in ordered_indices], dim=0)
    else:
        first = non_empty[0]
        base_samples = first.base_samples[:0].clone()
        base_rewards = first.base_rewards[:0].clone()
    return ProposalBatch(
        base_indices=ordered_indices,
        base_samples=base_samples,
        base_rewards=base_rewards,
        top_reward=top_reward if math.isfinite(top_reward) else float(non_empty[0].top_reward),
        proposals=proposals,
        generation_diagnostics=generation_diagnostics,
    )


def _short_smiles(value: Optional[str], *, limit: int = 48) -> str:
    if value is None:
        return "na"
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _iter_active_learning_debug_lines(
    traces: list[ProposalTrace],
    *,
    sample_count: int,
    seed: int,
    iteration: int,
    active_learning_selector: Optional[Any] = None,
) -> list[str]:
    if sample_count <= 0:
        return []
    eligible = [trace for trace in traces if trace.surrogate_mean is not None and trace.surrogate_std is not None]
    if not eligible:
        return ["[al-debug] no surrogate predictions available for current proposal batch"]
    rng = random.Random((int(seed) + 1) * 1_000_003 + int(iteration))
    chosen = rng.sample(eligible, k=min(int(sample_count), len(eligible)))
    similarity_diagnostics = (
        active_learning_selector.similarity_diagnostics([trace.smiles for trace in chosen], top_k=5)
        if active_learning_selector is not None
        else []
    )
    lines: list[str] = []
    for idx, trace in enumerate(chosen, start=1):
        sim_diag = similarity_diagnostics[idx - 1] if idx - 1 < len(similarity_diagnostics) else None
        variance = float(trace.surrogate_std) ** 2 if trace.surrogate_std is not None else None
        lines.append(
            "[al-debug] "
            f"sample={idx}/{len(chosen)} "
            f"mean={_fmt_debug_float(trace.surrogate_mean)} "
            f"var={_fmt_debug_float(variance)} "
            f"ei={_fmt_debug_float(trace.expected_improvement)} "
            f"acq={_fmt_debug_float(trace.acquisition_value)} "
            f"logk={_fmt_debug_float(trace.local_kernel_log_prob)} "
            f"oracle={int(bool(trace.oracle_evaluated))} "
            f"inbuf="
            f"{'na' if sim_diag is None or sim_diag.in_buffer is None else int(bool(sim_diag.in_buffer))} "
            f"sim_max={_fmt_debug_float(None if sim_diag is None else sim_diag.max_similarity)} "
            f"sim_top5={_fmt_debug_float(None if sim_diag is None else sim_diag.mean_topk_similarity)} "
            f"smiles={_short_smiles(trace.smiles)}"
        )
    return lines


def _leaf_uniqueness_counts(traces: list[ProposalTrace]) -> tuple[int, int]:
    if not traces:
        return 0, 0
    unique_vs_parent: set[tuple[bytes, bytes]] = set()
    unique_vs_each_other: set[bytes] = set()
    for trace in traces:
        parent_key = trace.ancestor_seq.contiguous().numpy().tobytes()
        leaf_key = trace.to_seq.contiguous().numpy().tobytes()
        unique_vs_parent.add((parent_key, leaf_key))
        unique_vs_each_other.add(leaf_key)
    return len(unique_vs_parent), len(unique_vs_each_other)


def _aggregate_edit_counts(traces: list[ProposalTrace]) -> tuple[Optional[float], Optional[float]]:
    if not traces:
        return None, None
    denom = float(len(traces))
    expected_mean = sum(float(trace.expected_edits) for trace in traces) / denom
    actual_mean = sum(float(trace.ancestor_to_child_edits) for trace in traces) / denom
    return expected_mean, actual_mean


def _sequence_length(seq: torch.Tensor, pad_token_id: Optional[int]) -> int:
    seq_cpu = seq.detach().cpu().view(-1)
    if pad_token_id is None:
        return int(seq_cpu.numel())
    return int((seq_cpu != int(pad_token_id)).sum().item())


def _fmt_length_summary(traces: list[ProposalTrace], pad_token_id: Optional[int]) -> str:
    if not traces:
        return "na"
    deltas: list[int] = []
    best_trace = max(traces, key=lambda trace: float(trace.score))
    for trace in traces:
        parent_len = _sequence_length(trace.ancestor_seq, pad_token_id)
        child_len = _sequence_length(trace.to_seq, pad_token_id)
        deltas.append(child_len - parent_len)
    best_parent_len = _sequence_length(best_trace.ancestor_seq, pad_token_id)
    best_child_len = _sequence_length(best_trace.to_seq, pad_token_id)
    mean_delta = sum(float(delta) for delta in deltas) / float(len(deltas))
    min_delta = min(deltas)
    max_delta = max(deltas)
    return f"{best_parent_len}->{best_child_len} avg={mean_delta:+.2f} range=[{min_delta:+d},{max_delta:+d}]"


def _fmt_time_summary(traces: list[ProposalTrace]) -> str:
    if not traces:
        return "na"
    t_values = [float(trace.t_value) for trace in traces]
    t_min = min(t_values)
    t_max = max(t_values)
    if abs(t_max - t_min) <= 1e-9:
        return f"{t_min:.3f}"
    return f"{t_min:.3f}-{t_max:.3f}"


def _retention_preview(runtime: SearchRuntime, stats: ProposalChangeStats, max_items: int = 3) -> str:
    if not stats.retention_by_time:
        return ""
    parts: list[str] = []
    for item in stats.retention_by_time[:max_items]:
        t_val = float(item.t_value)
        parts.append(
            f"{t_val:.3f}:all={item.mean_retained_frac:.3f},"
            f"corr={_fmt_opt(item.mean_retained_frac_corrupted)},"
            f"uncorr={_fmt_opt(item.mean_retained_frac_uncorrupted)},n={item.count}"
        )
    suffix = " ..." if len(stats.retention_by_time) > max_items else ""
    return f" by_t[{'; '.join(parts)}{suffix}]"


def _write_transition_jsonl(
    fp: TextIO,
    *,
    iteration: int,
    traces: list[ProposalTrace],
    start_idx: int = 0,
    pad_token_id: Optional[int] = None,
) -> None:
    for idx, trace in enumerate(traces):
        to_seq_tensor = trace.to_seq.detach().cpu()
        to_seq = to_seq_tensor.tolist()
        score = float(trace.score)
        is_finite = math.isfinite(score)
        diagnostic_score = trace.diagnostic_score
        diagnostic_score_finite = diagnostic_score is not None and math.isfinite(float(diagnostic_score))
        score_for_analysis = float(diagnostic_score) if diagnostic_score_finite else (score if is_finite else None)
        if pad_token_id is None:
            seq_length = int(to_seq_tensor.numel())
        else:
            seq_length = int((to_seq_tensor != int(pad_token_id)).sum().item())
        record = {
            "batch_id": int(iteration),
            "seq_idx": int(start_idx + idx),
            "seq": to_seq,
            "seq_length": seq_length,
            "score": score if is_finite else None,
            "score_is_finite": is_finite,
            "score_for_analysis": score_for_analysis,
            "decoded_sequence": trace.smiles,
            "sequence_string": trace.smiles,
            "oracle_evaluated": bool(trace.oracle_evaluated),
            "acquisition_candidate": bool(trace.acquisition_candidate),
            "acquisition_selected": bool(trace.acquisition_selected),
            "duplicate_exact_proposal": bool(trace.duplicate_exact_proposal),
            "duplicate_generated_leaf": bool(trace.duplicate_generated_leaf),
            "cached_generated_proposal": bool(trace.cached_generated_proposal),
            "invalid_generated_proposal": bool(trace.invalid_generated_proposal),
            "oracle_rejected": bool(trace.oracle_rejected),
            "t_value": float(trace.t_value),
            "expected_edits": float(trace.expected_edits),
            "smiles": trace.smiles,
            "surrogate_mean": trace.surrogate_mean,
            "surrogate_std": trace.surrogate_std,
            "expected_improvement": trace.expected_improvement,
            "acquisition_value": trace.acquisition_value,
            "local_kernel_log_prob": trace.local_kernel_log_prob,
            "local_kernel_log_prob_sequence": trace.local_kernel_log_prob_sequence,
            "diagnostic_score": float(diagnostic_score) if diagnostic_score_finite else None,
            "diagnostic_score_evaluated": bool(trace.diagnostic_score_evaluated),
            "full_batch_rank_by_score": trace.full_batch_rank_by_score,
            "diagnostic_injected_global_best": bool(trace.diagnostic_injected_global_best),
            "diagnostic_injected_source": trace.diagnostic_injected_source,
            "diagnostic_injected_full_dataset_score": trace.diagnostic_injected_full_dataset_score,
            "region_id": trace.region_id,
            "proposal_corrupted_tokens": int(trace.proposal_corrupted_tokens),
            "model_changes_total": int(trace.model_changes_total),
            "model_changes_on_corrupted": int(trace.model_changes_on_corrupted),
            "model_changes_on_uncorrupted": int(trace.model_changes_on_uncorrupted),
            "proposal_logical_edits": int(trace.proposal_logical_edits),
            "model_logical_edits": int(trace.model_logical_edits),
            "retained_proposal_logical_edits": int(trace.retained_proposal_logical_edits),
            "dropped_proposal_logical_edits": int(trace.dropped_proposal_logical_edits),
            "introduced_logical_edits": int(trace.introduced_logical_edits),
            "retained_proposal_logical_frac": trace.retained_proposal_logical_frac,
            "ancestor_to_child_edits": int(trace.ancestor_to_child_edits),
            "ancestor_seq": trace.ancestor_seq.tolist(),
            "from_seq": trace.from_seq.tolist(),
            "to_seq": to_seq,
            "parent_seq_hash": trace.parent_seq_hash,
            "parent_score": trace.parent_score,
            "parent_archive_index": trace.parent_archive_index,
            "parent_archive_probability": trace.parent_archive_probability,
            "parent_selection_log_prob": trace.parent_selection_log_prob,
            "archive_size": trace.archive_size,
            "archive_snapshot_id": trace.archive_snapshot_id,
            "hamming_conditional_log_prob": trace.hamming_conditional_log_prob,
            "proposal_log_prob": trace.proposal_log_prob,
            "proposal_probability_method": trace.proposal_probability_method,
        }
        fp.write(json.dumps(record) + "\n")


def _write_sample_only_jsonl(
    fp: TextIO,
    *,
    sequences: torch.Tensor,
    scores: torch.Tensor,
    batch_id: int = 0,
    start_idx: int = 0,
    pad_token_id: Optional[int] = None,
    decoded_sequences: Optional[list[Optional[str]]] = None,
) -> None:
    seq_cpu = sequences.detach().cpu()
    score_cpu = scores.detach().cpu()
    batch_size = int(seq_cpu.shape[0]) if seq_cpu.ndim > 1 else 1
    for idx in range(batch_size):
        seq_tensor = seq_cpu[idx] if seq_cpu.ndim > 1 else seq_cpu
        seq = seq_tensor.tolist()
        score_value = float(score_cpu[idx].item()) if score_cpu.ndim > 0 else float(score_cpu.item())
        is_finite = math.isfinite(score_value)
        decoded_sequence = (
            decoded_sequences[idx] if decoded_sequences is not None and idx < len(decoded_sequences) else None
        )
        if pad_token_id is None:
            seq_length = int(seq_tensor.numel())
        else:
            seq_length = int((seq_tensor != int(pad_token_id)).sum().item())
        record = {
            "batch_id": int(batch_id),
            "seq_idx": int(start_idx + idx),
            "seq": seq,
            "seq_length": seq_length,
            "score": score_value if is_finite else None,
            "score_is_finite": is_finite,
            "decoded_sequence": decoded_sequence,
            "sequence_string": decoded_sequence,
            "oracle_evaluated": True,
            "sample_only": True,
            "sample_source": "model",
            "t_value": None,
            "expected_edits": None,
            "smiles": decoded_sequence,
            "surrogate_mean": None,
            "surrogate_std": None,
            "expected_improvement": None,
            "acquisition_value": None,
            "region_id": None,
            "proposal_corrupted_tokens": 0,
            "model_changes_total": 0,
            "model_changes_on_corrupted": 0,
            "model_changes_on_uncorrupted": 0,
            "proposal_logical_edits": 0,
            "model_logical_edits": 0,
            "retained_proposal_logical_edits": 0,
            "dropped_proposal_logical_edits": 0,
            "introduced_logical_edits": 0,
            "retained_proposal_logical_frac": None,
            "ancestor_to_child_edits": 0,
            "ancestor_seq": seq,
            "from_seq": seq,
            "to_seq": seq,
        }
        fp.write(json.dumps(record) + "\n")


def _write_initial_batch_jsonl(
    path: str | Path,
    *,
    sequences: torch.Tensor,
    scores: torch.Tensor,
    pad_token_id: Optional[int] = None,
    sample_source: str = "model_initial_batch",
    decoded_sequences: Optional[list[Optional[str]]] = None,
) -> None:
    out_path = Path(path).expanduser()
    if str(out_path) == "/dev/null":
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_jsonl(out_path, "wt") as fp:
        seq_cpu = sequences.detach().cpu()
        score_cpu = scores.detach().cpu().view(-1)
        for idx in range(int(seq_cpu.shape[0])):
            seq_tensor = seq_cpu[idx]
            seq = seq_tensor.tolist()
            score_value = float(score_cpu[idx].item())
            is_finite = math.isfinite(score_value)
            decoded_sequence = (
                decoded_sequences[idx] if decoded_sequences is not None and idx < len(decoded_sequences) else None
            )
            if pad_token_id is None:
                seq_length = int(seq_tensor.numel())
            else:
                seq_length = int((seq_tensor != int(pad_token_id)).sum().item())
            record = {
                "initial_batch": True,
                "batch_id": -1,
                "seq_idx": int(idx),
                "seq": seq,
                "seq_length": seq_length,
                "score": score_value if is_finite else None,
                "score_is_finite": is_finite,
                "decoded_sequence": decoded_sequence,
                "sequence_string": decoded_sequence,
                "oracle_evaluated": True,
                "smiles": decoded_sequence,
                "sample_source": str(sample_source),
            }
            fp.write(json.dumps(record) + "\n")


def _write_proposal_diagnostics_jsonl(
    fp: TextIO,
    *,
    iteration: int,
    eval_result: ProposalEvaluation,
    oracle_budget_total: int,
    oracle_physical_total: int,
    oracle_privileged_total: int,
    resample_count: int = 0,
    collapse_resample_count: int = 0,
    discarded: bool = False,
    collapse_topup_requested: int = 0,
    collapse_topup_generated: int = 0,
) -> None:
    stats = eval_result.change_stats
    generation = eval_result.proposal_generation_diagnostics
    record = {
        "batch_id": int(iteration),
        "generated_proposals": int(stats.generated_proposals),
        "deduped_proposals": int(stats.deduped_proposals),
        "duplicate_exact_proposals": int(stats.duplicate_exact_proposals),
        "duplicate_generated_leaves": int(stats.duplicate_generated_leaves),
        "cached_generated_proposals": int(stats.cached_generated_proposals),
        "invalid_generated_proposals": int(stats.invalid_generated_proposals),
        "acquisition_batch_size": int(stats.acquisition_batch_size),
        "oracle_evaluated_count": int(eval_result.oracle_evaluated_count),
        "oracle_rejected_count": int(eval_result.oracle_rejected_count),
        "oracle_budget_debit_count": int(eval_result.oracle_budget_debit_count),
        "oracle_scored_count": int(eval_result.oracle_scored_count),
        "oracle_privileged_scored_count": int(eval_result.oracle_privileged_scored_count),
        "oracle_budget_total_after_iteration": int(oracle_budget_total),
        "oracle_physical_total_after_iteration": int(oracle_physical_total),
        "oracle_privileged_total_after_iteration": int(oracle_privileged_total),
        # Historical consumers use this as the run's budget counter. Under the
        # default physical mode the value is unchanged; selected_archive mode
        # intentionally aliases the selected/archive budget instead.
        "oracle_total_after_iteration": int(oracle_budget_total),
        "active_learning_threshold_exhausted": bool(eval_result.active_learning_threshold_exhausted),
        "active_learning_max_expected_improvement": eval_result.active_learning_max_expected_improvement,
        "active_learning_num_above_threshold": int(eval_result.active_learning_num_above_threshold),
        "resample_count": int(resample_count),
        "collapse_resample_count": int(collapse_resample_count),
        "proposal_batch_discarded": bool(discarded),
        "proposal_collapse_detected": bool(eval_result.proposal_collapse_detected),
        "proposal_collapse_acquisition_fraction": eval_result.proposal_collapse_acquisition_fraction,
        "proposal_collapse_min_acquisition_fraction": eval_result.proposal_collapse_min_acquisition_fraction,
        "proposal_collapse_min_acquisition_count": eval_result.proposal_collapse_min_acquisition_count,
        "proposal_collapse_target_count": eval_result.proposal_collapse_target_count,
        "proposal_collapse_topup_requested": int(collapse_topup_requested),
        "proposal_collapse_topup_generated": int(collapse_topup_generated),
        "full_batch_scored_count": int(eval_result.full_batch_scored_count),
        "full_batch_top_score": eval_result.full_batch_top_score,
        "full_batch_top_smiles": eval_result.full_batch_top_smiles,
        "full_batch_top_t_value": eval_result.full_batch_top_t_value,
        "full_batch_top_selected": eval_result.full_batch_top_selected,
        "full_batch_best_selected_score": eval_result.full_batch_best_selected_score,
        "diagnostic_injected_global_best_present": bool(eval_result.diagnostic_injected_global_best_present),
        "diagnostic_injected_global_best_acquisition_candidate": bool(
            eval_result.diagnostic_injected_global_best_acquisition_candidate
        ),
        "diagnostic_injected_global_best_selected": bool(eval_result.diagnostic_injected_global_best_selected),
        "diagnostic_injected_global_best_rank_by_acquisition": (
            eval_result.diagnostic_injected_global_best_rank_by_acquisition
        ),
        "diagnostic_injected_global_best_surrogate_mean": (eval_result.diagnostic_injected_global_best_surrogate_mean),
        "diagnostic_injected_global_best_surrogate_std": (eval_result.diagnostic_injected_global_best_surrogate_std),
        "diagnostic_injected_global_best_expected_improvement": (
            eval_result.diagnostic_injected_global_best_expected_improvement
        ),
        "diagnostic_injected_global_best_acquisition_value": (
            eval_result.diagnostic_injected_global_best_acquisition_value
        ),
        "diagnostic_injected_global_best_sequence": (eval_result.diagnostic_injected_global_best_sequence),
        "diagnostic_injected_global_best_full_dataset_score": (
            eval_result.diagnostic_injected_global_best_full_dataset_score
        ),
        "filter_by_time": [
            {
                "t_value": float(item.t_value),
                "generated_proposals": int(item.generated_proposals),
                "deduped_proposals": int(item.deduped_proposals),
                "duplicate_exact_proposals": int(item.duplicate_exact_proposals),
                "duplicate_generated_leaves": int(item.duplicate_generated_leaves),
                "cached_generated_proposals": int(item.cached_generated_proposals),
                "invalid_generated_proposals": int(item.invalid_generated_proposals),
                "acquisition_batch_size": int(item.acquisition_batch_size),
            }
            for item in stats.filter_by_time
        ],
        "retention_by_time": [
            {
                "t_value": float(item.t_value),
                "count": int(item.count),
                "mean_retained_frac": float(item.mean_retained_frac),
                "mean_retained_frac_corrupted": item.mean_retained_frac_corrupted,
                "mean_retained_frac_uncorrupted": item.mean_retained_frac_uncorrupted,
                "mean_retained_frac_logical": item.mean_retained_frac_logical,
            }
            for item in stats.retention_by_time
        ],
    }
    if generation:
        if generation.get("selection") == "top_surrogate_mean":
            record["proposal_generation"] = dict(generation)
        record.update(
            {
                "proposal_operator": generation.get("operator"),
                "proposal_surrogate_query_count": generation.get("surrogate_query_count"),
                "proposal_surrogate_root_query_count": generation.get("surrogate_root_query_count"),
                "proposal_surrogate_mutant_query_count": generation.get("surrogate_mutant_query_count"),
                "proposal_paid_target_count": generation.get("paid_target_count"),
                "proposal_unique_generated_count": generation.get("unique_generated_count"),
                "proposal_children_per_selection": generation.get("children_per_selection"),
                "proposal_ucb_beta": generation.get("ucb_beta"),
                "proposal_parent_temperature": generation.get("parent_temperature"),
                "proposal_crossover_rate": generation.get("crossover_rate"),
                "proposal_mutation_operator": generation.get("mutation_operator"),
                "proposal_mutation_mask_probability": generation.get("mutation_mask_probability"),
                "proposal_generator_noise_time": generation.get("generator_noise_time"),
                "proposal_generator_mask_enforced": generation.get("generator_mask_enforced"),
                "proposal_generator_mutation_draw_count": generation.get(
                    "generator_mutation_draw_count"
                ),
                "proposal_generator_rollout_batch_count": generation.get(
                    "generator_rollout_batch_count"
                ),
                "proposal_generator_rollout_row_count": generation.get(
                    "generator_rollout_row_count"
                ),
                "proposal_generator_novelty_rejection_count": generation.get(
                    "generator_novelty_rejection_count"
                ),
                "proposal_generator_zero_mask_count": generation.get(
                    "generator_zero_mask_count"
                ),
            }
        )
    fp.write(json.dumps(record) + "\n")


def _score_batch_with_global_cache(
    batch: torch.Tensor,
    *,
    reward_fn: Any,
    device: torch.device,
    score_cache: dict[bytes, float],
) -> tuple[torch.Tensor, list[bool]]:
    if batch.ndim != 2:
        raise ValueError(f"_score_batch_with_global_cache expects [B, L], got {tuple(batch.shape)}")
    batch_cpu = batch.detach().to("cpu").contiguous()
    cached_scores: list[Optional[float]] = []
    uncached_rows: list[int] = []
    for row_idx, row in enumerate(batch_cpu):
        key = row.numpy().tobytes()
        cached_score = score_cache.get(key)
        cached_scores.append(cached_score)
        if cached_score is None:
            uncached_rows.append(row_idx)

    scores = torch.empty((batch.shape[0],), dtype=torch.float, device=device)
    oracle_mask = [False for _ in range(int(batch.shape[0]))]
    if uncached_rows:
        oracle_batch = batch[uncached_rows]
        oracle_scores = _score_reward_batch_once(
            oracle_batch,
            reward_fn,
            context="global score-cache population",
        )
        for local_idx, row_idx in enumerate(uncached_rows):
            score_value = float(oracle_scores[local_idx].item())
            scores[row_idx] = score_value
            oracle_mask[row_idx] = True
            key = batch_cpu[row_idx].numpy().tobytes()
            score_cache[key] = score_value
    for row_idx, cached_score in enumerate(cached_scores):
        if cached_score is None:
            continue
        scores[row_idx] = float(cached_score)
    return scores, oracle_mask


def _score_reward_batch_once(
    batch: torch.Tensor,
    reward_fn: Any,
    *,
    context: str,
) -> torch.Tensor:
    """Score a batch once; never retry a stateful oracle after failure."""

    expected = int(batch.shape[0]) if batch.ndim > 1 else 1
    try:
        scores = score_batch(batch, reward_fn)
    except Exception as exc:
        raise RuntimeError(f"oracle scoring failed during {context} for {expected} candidate(s)") from exc
    scores = torch.as_tensor(scores, dtype=torch.float, device=batch.device).view(-1)
    if int(scores.numel()) != expected:
        raise RuntimeError(
            f"oracle scoring during {context} returned {int(scores.numel())} score(s) for {expected} candidate(s)"
        )
    return scores


def _observe_selector_once(
    selector: Any,
    batch: torch.Tensor,
    scores: torch.Tensor,
    *,
    context: str,
) -> None:
    try:
        selector.observe_oracle_batch(batch, scores)
    except Exception as exc:
        raise RuntimeError(
            f"oracle observation failed during {context} for {int(batch.shape[0])} candidate(s)"
        ) from exc


def _make_active_learning_selector_factory(
    *,
    args: argparse.Namespace,
    reward_fn: Any,
    al_feature_space: str,
    seq_len: Optional[int] = None,
) -> tuple[SelectorFactory, str]:
    feature_space = str(al_feature_space or "onehot_tanimoto").strip().lower()
    if feature_space == "ehrlich_onehot":

        def _factory(config: ActiveLearningConfig) -> Any:
            return EhrlichOneHotExpectedImprovementSelector(
                reward_fn=reward_fn,
                config=config,
            )

        instance = getattr(reward_fn, "instance", None)
        instance_cfg = getattr(instance, "cfg", None)
        display_len = getattr(instance_cfg, "seq_len", seq_len or "auto")
        display_vocab = getattr(instance_cfg, "vocab_size", "auto")
        return _factory, f"ehrlich_onehot=len{display_len}/vocab{display_vocab}"

    if feature_space == "ehrlich_diffusion":
        diffusion_tau = float(
            getattr(args, "al_ehrlich_diffusion_tau", None)
            if getattr(args, "al_ehrlich_diffusion_tau", None) is not None
            else 1.0
        )
        if not math.isfinite(diffusion_tau) or diffusion_tau <= 0.0:
            raise ValueError("al_ehrlich_diffusion_tau must be finite and positive")
        diffusion_combine = str(getattr(args, "al_ehrlich_diffusion_combine", None) or "mean").strip().lower()
        if diffusion_combine not in {"mean", "sum", "product"}:
            raise ValueError("al_ehrlich_diffusion_combine must be one of: mean, sum, product")
        raw_transition_weights = getattr(
            args,
            "al_ehrlich_diffusion_use_transition_weights",
            None,
        )
        use_transition_weights = True if raw_transition_weights is None else bool(raw_transition_weights)

        def _factory(config: ActiveLearningConfig) -> Any:
            return EhrlichDiffusionExpectedImprovementSelector(
                reward_fn=reward_fn,
                config=config,
                tau=diffusion_tau,
                combine=diffusion_combine,
                use_transition_weights=use_transition_weights,
            )

        return (
            _factory,
            "ehrlich_diffusion="
            f"tau{diffusion_tau}/combine{diffusion_combine}/"
            f"transition_weights{int(use_transition_weights)}",
        )

    if feature_space == "ehrlich_local_spaced_ngram_additive":
        instance = getattr(reward_fn, "instance", None)
        instance_cfg = getattr(instance, "cfg", None)
        if instance is None or instance_cfg is None:
            raise TypeError("ehrlich_local_spaced_ngram_additive active learning requires reward_fn.instance")
        raw_orders = getattr(args, "al_ehrlich_spaced_ngram_orders", None)
        if raw_orders is None:
            raise ValueError("al_ehrlich_spaced_ngram_orders is required for ehrlich_local_spaced_ngram_additive")
        orders = [int(value) for value in raw_orders]
        if len(orders) < 2 or len(orders) != len(set(orders)):
            raise ValueError("al_ehrlich_spaced_ngram_orders must contain at least two unique orders")
        raw_windows = getattr(args, "al_ehrlich_spaced_ngram_windows", None)
        if raw_windows is None:
            shared_window = int(getattr(args, "al_ehrlich_spaced_ngram_window", None) or 8)
            windows = [shared_window] * len(orders)
        else:
            windows = [int(value) for value in raw_windows]
        if len(windows) != len(orders):
            raise ValueError("al_ehrlich_spaced_ngram_windows must contain one window per order")
        if any(order < 2 or order > window for order, window in zip(orders, windows, strict=True)):
            raise ValueError("each al_ehrlich_spaced_ngram_order must be between 2 and its local window")
        offset_groups = [
            make_local_spaced_ngram_offsets(order, window) for order, window in zip(orders, windows, strict=True)
        ]
        if any(int(offsets.max().item()) >= int(instance_cfg.seq_len) for offsets in offset_groups):
            raise ValueError("additive spaced n-gram offsets do not fit the Ehrlich sequence length")
        raw_weights = getattr(args, "al_ehrlich_spaced_ngram_weights", None)
        weights = [1.0 / len(orders)] * len(orders) if raw_weights is None else [float(value) for value in raw_weights]
        if len(weights) != len(orders) or any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
            raise ValueError("al_ehrlich_spaced_ngram_weights must contain one finite positive value per order")
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]

        def _factory(config: ActiveLearningConfig) -> Any:
            return EhrlichAdditiveSpacedNGramExpectedImprovementSelector(
                reward_fn=reward_fn,
                config=config,
                offset_groups=offset_groups,
                weights=weights,
            )

        order_display = ",".join(str(order) for order in orders)
        window_display = ",".join(str(window) for window in windows)
        weight_display = ",".join(f"{weight:.12g}" for weight in weights)
        return (
            _factory,
            f"ehrlich_additive_spaced_ngram=orders{order_display}/windows{window_display}/weights{weight_display}",
        )

    if feature_space in {
        "ehrlich_spaced_ngram",
        "ehrlich_spaced_pair",
        "ehrlich_spaced_trigram",
        "ehrlich_local_spaced_ngram",
        "ehrlich_trigram",
    }:
        instance = getattr(reward_fn, "instance", None)
        instance_cfg = getattr(instance, "cfg", None)
        if instance is None or instance_cfg is None:
            raise TypeError("ehrlich_spaced_ngram active learning requires reward_fn.instance")
        if feature_space == "ehrlich_spaced_pair":
            default_pattern_source = "local_pair"
        elif feature_space == "ehrlich_spaced_trigram":
            default_pattern_source = "local_trigram"
        elif feature_space == "ehrlich_local_spaced_ngram":
            default_pattern_source = "local_all"
        elif feature_space == "ehrlich_trigram":
            default_pattern_source = "regular"
        else:
            default_pattern_source = "instance"
        pattern_source = (
            str(getattr(args, "al_ehrlich_spaced_ngram_patterns", None) or default_pattern_source).strip().lower()
        )
        requested_n = getattr(args, "al_ehrlich_spaced_ngram_n", None)
        if feature_space == "ehrlich_local_spaced_ngram" and requested_n is None:
            raise ValueError("al_ehrlich_spaced_ngram_n is required for ehrlich_local_spaced_ngram")
        default_n = (
            3
            if feature_space in {"ehrlich_trigram", "ehrlich_spaced_trigram"}
            else getattr(instance_cfg, "motif_len", 0)
        )
        n = int(default_n if requested_n is None else requested_n)
        gap = int(getattr(args, "al_ehrlich_spaced_ngram_gap", None) or 1)
        window = int(getattr(args, "al_ehrlich_spaced_ngram_window", None) or 8)
        if n < 1:
            raise ValueError("al_ehrlich_spaced_ngram_n must be positive")
        if gap < 1:
            raise ValueError("al_ehrlich_spaced_ngram_gap must be positive")
        if pattern_source == "instance":
            offsets = torch.as_tensor(instance.spacings, dtype=torch.long)
            if int(offsets.shape[1]) != n:
                raise ValueError(
                    "al_ehrlich_spaced_ngram_n must equal the instance motif length when patterns=instance"
                )
        elif pattern_source == "regular":
            offsets = torch.arange(n, dtype=torch.long).mul(gap).view(1, -1)
        elif pattern_source == "local_pair":
            if requested_n not in {None, 2}:
                raise ValueError("al_ehrlich_spaced_ngram_n must be 2 when patterns=local_pair")
            offsets = make_local_spaced_ngram_offsets(2, window)
        elif pattern_source == "local_trigram":
            if requested_n not in {None, 3}:
                raise ValueError("al_ehrlich_spaced_ngram_n must be 3 when patterns=local_trigram")
            offsets = make_local_spaced_ngram_offsets(3, window)
        elif pattern_source == "local_all":
            offsets = make_local_spaced_ngram_offsets(n, window)
        else:
            raise ValueError(
                "al_ehrlich_spaced_ngram_patterns must be one of: "
                "instance, regular, local_pair, local_trigram, local_all"
            )
        if int(offsets.max().item()) >= int(instance_cfg.seq_len):
            raise ValueError("spaced n-gram offsets do not fit the Ehrlich sequence length")

        def _factory(config: ActiveLearningConfig) -> Any:
            return EhrlichSpacedNGramExpectedImprovementSelector(
                reward_fn=reward_fn,
                config=config,
                offsets=offsets,
            )

        offset_display = ",".join("-".join(str(int(v)) for v in row) for row in offsets.tolist())
        return _factory, f"ehrlich_spaced_ngram=patterns{pattern_source}/offsets{offset_display}"

    if feature_space in {"onehot_tanimoto", "protein_onehot_tanimoto", "protein_onehot", "onehot"}:

        def _resolve_onehot_max_len(config: ActiveLearningConfig) -> int:
            candidates = [
                config.onehot_max_len,
                getattr(args, "al_onehot_max_len", None),
                seq_len,
                getattr(args, "protein_max_len", None),
                getattr(reward_fn, "max_len", None),
            ]
            for value in candidates:
                if value is None:
                    continue
                max_len = int(value)
                if max_len > 0:
                    return max_len
            raise ValueError(
                "onehot_tanimoto active learning requires a positive one-hot length; "
                "set al_onehot_max_len or protein_max_len"
            )

        def _factory(config: ActiveLearningConfig) -> Any:
            config.onehot_max_len = _resolve_onehot_max_len(config)
            config.onehot_alphabet = str(
                config.onehot_alphabet or getattr(reward_fn, "alphabet", None) or DEFAULT_PROTEIN_ONEHOT_ALPHABET
            )
            if config.gp_outputscale is None:
                config.gp_outputscale = 1.0
            config.gp_learn_outputscale = False
            return OneHotTanimotoExpectedImprovementSelector(
                reward_fn=reward_fn,
                config=config,
            )

        display_len = (
            getattr(args, "al_onehot_max_len", None)
            or seq_len
            or getattr(args, "protein_max_len", None)
            or getattr(reward_fn, "max_len", None)
            or "auto"
        )
        display_scale = getattr(args, "al_gp_outputscale", None)
        if display_scale is None:
            display_scale = 1.0
        return _factory, f"onehot_tanimoto=max_len{display_len}/fixed_scale{display_scale}"

    if feature_space in {"kermut", "protein_kermut"}:
        from adapters.protein.esm import ESM2Featurizer
        from adapters.protein.kermut import KermutExpectedImprovementSelector, KermutResources

        resource_path = getattr(args, "al_kermut_resources", None)
        if not resource_path:
            raise ValueError("Kermut requires --al-kermut-resources with aligned ProteinMPNN probabilities and coordinates")
        resources = KermutResources.load(resource_path, expected_length=seq_len)
        model_name = str(getattr(args, "al_esm_model", None) or "esm2_t33_650M_UR50D")
        shared_kermut: dict[str, ESM2Featurizer] = {}

        def _factory(config: ActiveLearningConfig) -> Any:
            if "featurizer" not in shared_kermut:
                shared_kermut["featurizer"] = ESM2Featurizer(
                    model_name=model_name,
                    device=str(getattr(args, "device", None) or config.gp_device),
                    batch_size=int(getattr(args, "al_esm_batch_size", None) or 16),
                )
            return KermutExpectedImprovementSelector(
                reward_fn=reward_fn,
                config=config,
                resources=resources,
                featurizer=shared_kermut["featurizer"],
            )

        return _factory, f"kermut=resources{resource_path}/esm_model{model_name}"

    if feature_space in {"esm", "protein_esm", "esm2"}:
        from adapters.protein.esm import ESM2Featurizer
        from adapters.protein.selector import ProteinEmbeddingExpectedImprovementSelector

        model_name = str(getattr(args, "al_esm_model", None) or "esm2_t30_150M_UR50D")
        batch_size = int(getattr(args, "al_esm_batch_size", None) or 16)
        shared: dict[str, ESM2Featurizer] = {}

        def _factory(config: ActiveLearningConfig) -> Any:
            featurizer = shared.get("featurizer")
            if featurizer is None:
                featurizer = ESM2Featurizer(
                    model_name=model_name,
                    device=str(getattr(args, "device", None) or config.gp_device),
                    batch_size=batch_size,
                )
                shared["featurizer"] = featurizer
            return ProteinEmbeddingExpectedImprovementSelector(
                reward_fn=reward_fn,
                config=config,
                featurizer=featurizer,
            )

        return _factory, f"esm_model={model_name}"




    raise ValueError(
        "al_feature_space must be one of: onehot_tanimoto, "
        "ehrlich_onehot, ehrlich_diffusion, ehrlich_spaced_pair, esm, kermut"
    )


def _make_multi_region_selector_config(base_cfg: ActiveLearningConfig) -> ActiveLearningConfig:
    cfg = copy.deepcopy(base_cfg)
    cfg.enabled = True
    cfg.initial_batch_matches_buffer_size = False
    cfg.sequential_selection = False
    cfg.local_penalization = False
    cfg.acquisition = "thompson"
    cfg.max_resample_batches = 0
    cfg.ei_threshold = None
    cfg.min_train_size = 1
    return cfg


def _normalize_multi_region_score_mode(name: Any) -> str:
    value = str(name or "absolute").strip().lower()
    if value in {"absolute", "raw", "thompson"}:
        return "absolute"
    if value in {"improvement", "relative", "local_improvement", "thompson_minus_incumbent"}:
        return "improvement"
    raise ValueError("multi_region_score_mode must be one of: absolute, improvement")


def _make_multi_region_state(
    *,
    region_id: int,
    args: argparse.Namespace,
    runtime: SearchRuntime,
    candidate_cfg: Any,
    trust_region: Any,
    selector_cfg: ActiveLearningConfig,
    selector_factory: SelectorFactory,
    global_score_cache: dict[bytes, float],
) -> tuple[MultiRegionState, int]:
    proposal_policy = InterpolantProposalPolicy(
        args=args,
        runtime=runtime,
        candidate=copy.deepcopy(candidate_cfg),
        trust_region=copy.deepcopy(trust_region),
    )
    selector = selector_factory(copy.deepcopy(selector_cfg))
    incumbent = runtime.process.sample(
        num_samples=1,
        steps=runtime.adapter.T,
        device=runtime.device,
    )
    incumbent_scores, oracle_mask = _score_batch_with_global_cache(
        incumbent,
        reward_fn=runtime.reward_fn,
        device=runtime.device,
        score_cache=global_score_cache,
    )
    _observe_selector_once(
        selector,
        incumbent,
        incumbent_scores,
        context=f"multi-region {region_id} initialization",
    )
    return (
        MultiRegionState(
            region_id=int(region_id),
            proposal_policy=proposal_policy,
            selector=selector,
            incumbent=incumbent.clone(),
            incumbent_score=float(incumbent_scores[0].item()),
        ),
        int(sum(1 for item in oracle_mask if item)),
    )


def _region_is_collapsed(region: MultiRegionState, *, tol: float = 1e-12) -> bool:
    tr_strategy = region.proposal_policy.time_policy.tr_strategy
    if tr_strategy is None:
        return False
    config = tr_strategy.config
    domain = str(getattr(config, "domain", "edits") or "edits").lower()
    if domain == "time":
        mean_time = getattr(config, "mean_time", None)
        time_max = getattr(config, "time_max", None)
        # Time-domain scalar trust regions shrink in noise-radius space, so the
        # data-side collapse still appears as mean_time reaching time_max.
        return mean_time is not None and time_max is not None and float(mean_time) >= float(time_max) - tol
    mean_edits = getattr(config, "mean_edits", None)
    edits_min = getattr(config, "edits_min", None)
    return mean_edits is not None and edits_min is not None and float(mean_edits) <= float(edits_min) + tol


def _region_state_value(region: MultiRegionState) -> Optional[float]:
    tr_strategy = region.proposal_policy.time_policy.tr_strategy
    if tr_strategy is None:
        return None
    config = tr_strategy.config
    domain = str(getattr(config, "domain", "edits") or "edits").lower()
    if domain == "time":
        mean_time = getattr(config, "mean_time", None)
        return None if mean_time is None else float(mean_time)
    mean_edits = getattr(config, "mean_edits", None)
    return None if mean_edits is None else float(mean_edits)


def _run_multi_region_thompson_union_search(
    *,
    args: argparse.Namespace,
    runtime: SearchRuntime,
    candidate_cfg: Any,
    trust_region: Any,
    active_learning_cfg: ActiveLearningConfig,
    selector_factory: SelectorFactory,
    monitor: Optional[TensorBoardMonitor],
    samples_fp: Optional[TextIO],
    rollout_use_corrupt_mask: bool,
    rollout_denoise_steps: Optional[int],
    legacy_hamming_tr_error: bool,
    early_stop_patience: Optional[int],
    max_oracle_evaluations: Optional[int],
    pad_token_id: Optional[int],
) -> tuple[Optional[torch.Tensor], float, list[torch.Tensor]]:
    num_regions = int(getattr(args, "num_trust_regions", None) or 1)
    if num_regions <= 1:
        raise RuntimeError("_run_multi_region_thompson_union_search requires num_trust_regions > 1")
    if max_oracle_evaluations is not None and max_oracle_evaluations < num_regions:
        raise ValueError(
            "max_oracle_evaluations must be at least num_trust_regions because "
            "each region requires one initial oracle evaluation"
        )
    restart_patience = int(args.tr_restart_patience) if args.tr_restart_patience is not None else 10
    if restart_patience <= 0:
        raise ValueError("tr_restart_patience must be > 0 when provided")
    multi_region_score_mode = _normalize_multi_region_score_mode(getattr(args, "multi_region_score_mode", None))

    selector_cfg = _make_multi_region_selector_config(active_learning_cfg)
    if active_learning_cfg.sequential_selection:
        print("[search] multi-region Thompson-over-union overrides al_sequential_selection=False.")
    if active_learning_cfg.local_penalization:
        print("[search] multi-region Thompson-over-union ignores al_local_penalization.")
    if str(active_learning_cfg.acquisition or "").strip().lower() != "thompson":
        print("[search] multi-region Thompson-over-union overrides al_acquisition=thompson.")
    if active_learning_cfg.ei_threshold is not None:
        print("[search] multi-region Thompson-over-union ignores al_ei_threshold.")

    global_score_cache: dict[bytes, float] = {}
    regions: list[MultiRegionState] = []
    total_oracle_evaluated = 0
    for region_id in range(num_regions):
        region, added_oracle = _make_multi_region_state(
            region_id=region_id,
            args=args,
            runtime=runtime,
            candidate_cfg=candidate_cfg,
            trust_region=trust_region,
            selector_cfg=selector_cfg,
            selector_factory=selector_factory,
            global_score_cache=global_score_cache,
        )
        regions.append(region)
        total_oracle_evaluated += int(added_oracle)
    init_parts = [
        (
            f"r{region.region_id}:"
            f"score={float(region.incumbent_score):.3f},"
            f"len={_sequence_length(region.incumbent[0], pad_token_id)}"
        )
        for region in regions
    ]
    print("[search] multi-region init: " + " ".join(init_parts))

    best_region = max(regions, key=lambda item: float(item.incumbent_score))
    best_overall = best_region.incumbent.clone()
    best_reward = float(best_region.incumbent_score)
    leaves_log: list[torch.Tensor] = []

    num_iterations = args.num_iterations or 1
    no_improve_iters = 0
    if max_oracle_evaluations is not None and total_oracle_evaluated >= max_oracle_evaluations:
        print(
            "[search] stopping before search iterations: "
            f"max_oracle_evaluations={max_oracle_evaluations} reached "
            f"(oracle_total={total_oracle_evaluated})."
        )
        num_iterations = 0

    for it in range(num_iterations):
        prev_best = float(best_reward)
        incumbent_batch = torch.cat([region.incumbent for region in regions], dim=0)
        incumbent_rewards = torch.as_tensor(
            [float(region.incumbent_score) for region in regions],
            dtype=torch.float,
            device=runtime.device,
        )
        summary = summarise_search_batch(incumbent_batch, incumbent_rewards)
        _log_summary(monitor, it, summary)

        candidates_by_region: dict[int, list[MultiRegionCandidate]] = {}
        per_region_candidate_count = {region.region_id: 0 for region in regions}
        for region in regions:
            base_rewards = torch.as_tensor([float(region.incumbent_score)], dtype=torch.float, device=runtime.device)
            proposal_batch = region.proposal_policy.propose(
                iteration=it,
                batch=region.incumbent,
                rewards=base_rewards,
                elite_probs=None,
            )
            eval_result = evaluate_proposals(
                proposal_batch,
                adapter=runtime.adapter,
                reward_fn=runtime.reward_fn,
                track_trust_region=False,
                rollout_use_corrupt_mask=rollout_use_corrupt_mask,
                rollout_denoise_steps=rollout_denoise_steps,
                legacy_hamming_tr_error=legacy_hamming_tr_error,
                active_learning_selector=region.selector,
                active_learning_budget_override=0,
            )
            per_region_candidate_count[region.region_id] = len(eval_result.proposal_traces_all)
            region_candidates_by_sequence: dict[bytes, MultiRegionCandidate] = {}
            for proposal_idx, trace in enumerate(eval_result.proposal_traces_all):
                trace.region_id = int(region.region_id)
                if trace.acquisition_value is None:
                    continue
                seq_key = trace.to_seq.contiguous().numpy().tobytes()
                candidate = MultiRegionCandidate(
                    region_id=int(region.region_id),
                    proposal=proposal_batch.proposals[proposal_idx],
                    trace=trace,
                )
                current = region_candidates_by_sequence.get(seq_key)
                if current is None or float(candidate.trace.acquisition_value) > float(current.trace.acquisition_value):
                    region_candidates_by_sequence[seq_key] = candidate
            candidates_by_region[int(region.region_id)] = list(region_candidates_by_sequence.values())

        union_candidate_count = len(
            {
                candidate.trace.to_seq.contiguous().numpy().tobytes()
                for region_candidates in candidates_by_region.values()
                for candidate in region_candidates
            }
        )
        global_budget = max(1, int(active_learning_cfg.oracle_batch_size))
        if max_oracle_evaluations is not None:
            global_budget = min(
                global_budget,
                max(0, int(max_oracle_evaluations) - int(total_oracle_evaluated)),
            )
        remaining_by_region = {
            int(region_id): list(region_candidates) for region_id, region_candidates in candidates_by_region.items()
        }
        selected_candidates: list[MultiRegionCandidate] = []
        for _ in range(global_budget):
            best_choice: tuple[float, float, int, int, MultiRegionCandidate] | None = None
            for region in regions:
                remaining = remaining_by_region.get(int(region.region_id), [])
                if not remaining:
                    continue
                candidate_batch = torch.cat(
                    [item.trace.to_seq.view(1, -1).to(device=runtime.device, dtype=torch.long) for item in remaining],
                    dim=0,
                )
                selection = region.selector.select(candidate_batch, budget=1)
                if not selection.selected_indices:
                    continue
                local_idx = int(selection.selected_indices[0])
                if local_idx < 0 or local_idx >= len(remaining):
                    continue
                candidate = remaining[local_idx]
                candidate.trace.smiles = selection.smiles[local_idx]
                candidate.trace.surrogate_mean = selection.surrogate_mean[local_idx]
                candidate.trace.surrogate_std = selection.surrogate_std[local_idx]
                candidate.trace.expected_improvement = selection.expected_improvement[local_idx]
                candidate.trace.acquisition_value = selection.acquisition_value[local_idx]
                acquisition_value = (
                    float(candidate.trace.acquisition_value)
                    if candidate.trace.acquisition_value is not None
                    else float("-inf")
                )
                regional_priority = acquisition_value
                if multi_region_score_mode == "improvement" and math.isfinite(acquisition_value):
                    regional_priority = acquisition_value - float(region.incumbent_score)
                expected_improvement = (
                    float(candidate.trace.expected_improvement)
                    if candidate.trace.expected_improvement is not None
                    else float("-inf")
                )
                choice = (
                    regional_priority,
                    acquisition_value,
                    expected_improvement,
                    -int(region.region_id),
                    -int(local_idx),
                    candidate,
                )
                if best_choice is None or choice > best_choice:
                    best_choice = choice
            if best_choice is None:
                break
            chosen = best_choice[-1]
            selected_candidates.append(chosen)
            chosen_key = chosen.trace.to_seq.contiguous().numpy().tobytes()
            for region_id, remaining in remaining_by_region.items():
                remaining_by_region[region_id] = [
                    item for item in remaining if item.trace.to_seq.contiguous().numpy().tobytes() != chosen_key
                ]
        selected_by_region: dict[int, list[MultiRegionCandidate]] = {}
        for item in selected_candidates:
            selected_by_region.setdefault(int(item.region_id), []).append(item)

        selected_traces_all: list[ProposalTrace] = []
        oracle_evaluated_count = 0
        region_selected_counts = {region.region_id: 0 for region in regions}
        region_fresh_counts = {region.region_id: 0 for region in regions}
        region_cached_counts = {region.region_id: 0 for region in regions}
        region_restart_count = 0
        for region in regions:
            region_items = selected_by_region.get(int(region.region_id), [])
            region_selected_counts[region.region_id] = len(region_items)
            if not region_items:
                region.inactive_streak += 1
                region.boundary_streak = region.boundary_streak + 1 if _region_is_collapsed(region) else 0
                continue

            selected_batch = torch.cat(
                [item.trace.to_seq.view(1, -1).to(device=runtime.device, dtype=torch.long) for item in region_items],
                dim=0,
            )
            selected_scores, oracle_mask = _score_batch_with_global_cache(
                selected_batch,
                reward_fn=runtime.reward_fn,
                device=runtime.device,
                score_cache=global_score_cache,
            )
            _observe_selector_once(
                region.selector,
                selected_batch,
                selected_scores,
                context=f"multi-region {region.region_id} acquisition",
            )
            region.proposal_policy.update_from_evaluations(
                proposals=[item.proposal for item in region_items],
                proposal_rewards=selected_scores,
                base_rewards=torch.as_tensor([float(region.incumbent_score)], dtype=torch.float, device=runtime.device),
            )
            fresh_count = int(sum(1 for item in oracle_mask if item))
            region_fresh_counts[region.region_id] = fresh_count
            region_cached_counts[region.region_id] = len(region_items) - fresh_count

            for row_idx, item in enumerate(region_items):
                item.trace.score = float(selected_scores[row_idx].item())
                item.trace.oracle_evaluated = bool(oracle_mask[row_idx])
                item.trace.region_id = int(region.region_id)
                selected_traces_all.append(item.trace)
                oracle_evaluated_count += int(bool(oracle_mask[row_idx]))

            region_best_idx = int(torch.argmax(selected_scores).item())
            region_best_score = float(selected_scores[region_best_idx].item())
            region_best_item = region_items[region_best_idx]
            improved = region_best_score > float(region.incumbent_score) + float(trust_region.eps)
            mean_edit_error = _mean_normalized_edit_error(
                [item.trace for item in region_items],
                editable_seq_len_for_sample=region.proposal_policy.time_policy.editable_seq_len_for_sample,
            )
            region.proposal_policy.update_trust_region(
                best_time=float(region_best_item.trace.t_value),
                best_edits=float(region_best_item.trace.expected_edits),
                improved=improved,
                mean_edit_error=mean_edit_error,
            )
            region.inactive_streak = 0
            region.boundary_streak = _region_is_collapsed(region) and (region.boundary_streak + 1) or 0
            if improved:
                region.incumbent = selected_batch[region_best_idx : region_best_idx + 1].clone()
                region.incumbent_score = region_best_score
                if region_best_score > best_reward:
                    best_reward = region_best_score
                    best_overall = region.incumbent.clone()

        if (
            max_oracle_evaluations is not None
            and total_oracle_evaluated + int(oracle_evaluated_count) > max_oracle_evaluations
        ):
            raise RuntimeError(
                "multi-region evaluation exceeded the remaining oracle budget: "
                f"oracle_total={total_oracle_evaluated} "
                f"evaluated={oracle_evaluated_count} "
                f"max_oracle_evaluations={max_oracle_evaluations}"
            )
        total_oracle_evaluated += int(oracle_evaluated_count)

        for idx, region in enumerate(regions):
            restart_due_to_boundary = region.boundary_streak >= restart_patience
            if not restart_due_to_boundary:
                continue
            if max_oracle_evaluations is not None and total_oracle_evaluated >= max_oracle_evaluations:
                break
            print(
                "[search] multi-region restart: "
                f"region={region.region_id} reason=collapsed "
                f"boundary_streak={region.boundary_streak} inactive_streak={region.inactive_streak} "
                f"old_len={_sequence_length(region.incumbent[0], pad_token_id)} "
                f"old_score={float(region.incumbent_score):.3f}"
            )
            restarted, added_oracle = _make_multi_region_state(
                region_id=int(region.region_id),
                args=args,
                runtime=runtime,
                candidate_cfg=candidate_cfg,
                trust_region=trust_region,
                selector_cfg=selector_cfg,
                global_score_cache=global_score_cache,
            )
            restarted.restart_count = int(region.restart_count) + 1
            regions[idx] = restarted
            total_oracle_evaluated += int(added_oracle)
            region_restart_count += 1
            print(
                "[search] multi-region restart-seed: "
                f"region={restarted.region_id} "
                f"new_len={_sequence_length(restarted.incumbent[0], pad_token_id)} "
                f"new_score={float(restarted.incumbent_score):.3f} "
                f"oracle_seed={int(added_oracle)} "
                f"restart_count={restarted.restart_count}"
            )
            if restarted.incumbent_score > best_reward:
                best_reward = float(restarted.incumbent_score)
                best_overall = restarted.incumbent.clone()

        if float(best_reward) > prev_best + float(trust_region.eps):
            no_improve_iters = 0
        else:
            no_improve_iters += 1
        leaves_log.append(best_overall.clone())

        if samples_fp is not None and selected_traces_all:
            _write_transition_jsonl(
                samples_fp,
                iteration=it,
                traces=[trace for trace in selected_traces_all if trace.oracle_evaluated],
                start_idx=0,
                pad_token_id=pad_token_id,
            )
            samples_fp.flush()

        unique_vs_parent, unique_vs_each_other = _leaf_uniqueness_counts(selected_traces_all)
        expected_total, actual_total = _aggregate_edit_counts(selected_traces_all)
        best_len = _sequence_length(best_overall[0], pad_token_id) if best_overall is not None else None
        state_parts = [f"r{region.region_id}:{_fmt_opt(_region_state_value(region))}" for region in regions]
        selected_parts = [
            f"r{region.region_id}:{region_selected_counts.get(region.region_id, 0)}" for region in regions
        ]
        fresh_parts = [f"r{region.region_id}:{region_fresh_counts.get(region.region_id, 0)}" for region in regions]
        cached_parts = [f"r{region.region_id}:{region_cached_counts.get(region.region_id, 0)}" for region in regions]
        candidate_parts = [
            f"r{region.region_id}:{per_region_candidate_count.get(region.region_id, 0)}" for region in regions
        ]
        length_parts = [
            f"r{region.region_id}:{_sequence_length(region.incumbent[0], pad_token_id)}" for region in regions
        ]
        summary_parts = [
            f"[iter {it + 1:04d}]",
            "mode=multi_region_thompson",
            f"best={best_reward:.3f}",
            f"leaves={len(selected_traces_all) if selected_traces_all else 1}",
            f"uniq_p={unique_vs_parent}",
            f"uniq_l={unique_vs_each_other}",
            f"exp_ed={_fmt_edit_count(expected_total)}",
            f"act_ed={_fmt_edit_count(actual_total)}",
            f"t={_fmt_time_summary(selected_traces_all)}",
            f"len={best_len if best_len is not None else 'na'}",
            f"len_prop={_fmt_length_summary(selected_traces_all, pad_token_id)}",
            f"oracle={oracle_evaluated_count}/{max(1, union_candidate_count)}",
            f"oracle_total={total_oracle_evaluated}",
            f"regions={num_regions}",
            f"cand={','.join(candidate_parts)}",
            f"sel={','.join(selected_parts)}",
            f"fresh={','.join(fresh_parts)}",
            f"cached={','.join(cached_parts)}",
            f"len_r={','.join(length_parts)}",
            f"state={','.join(state_parts)}",
            f"restarts={region_restart_count}",
        ]
        print(" | ".join(summary_parts))

        if args.print_leaves and best_overall is not None:
            tokens = " ".join(str(int(v)) for v in best_overall[0].tolist())
            leaf_len = _sequence_length(best_overall[0], pad_token_id)
            print(f"leaf_len={leaf_len} leaf=[{tokens}]")
        if early_stop_patience is not None and no_improve_iters >= early_stop_patience:
            print(
                "[search] early stopping: "
                f"best score unchanged for {no_improve_iters} iterations "
                f"(patience={early_stop_patience})."
            )
            break
        if max_oracle_evaluations is not None and total_oracle_evaluated >= max_oracle_evaluations:
            print(
                "[search] stopping: "
                f"max_oracle_evaluations={max_oracle_evaluations} reached "
                f"(oracle_total={total_oracle_evaluated})."
            )
            break

    if monitor is not None and regions:
        incumbent_batch = torch.cat([region.incumbent for region in regions], dim=0)
        incumbent_rewards = torch.as_tensor(
            [float(region.incumbent_score) for region in regions],
            dtype=torch.float,
            device=runtime.device,
        )
        summary = summarise_search_batch(incumbent_batch, incumbent_rewards)
        _log_summary(monitor, num_iterations, summary)
    return best_overall, best_reward, leaves_log


def run_interpolant_search(args: argparse.Namespace) -> None:
    args = _prepare_args(args)
    if getattr(args, "feature_space", None) is not None and getattr(args, "al_feature_space", None) is None:
        args.al_feature_space = str(args.feature_space)
    if bool(getattr(args, "protein_mode", False)):
        if getattr(args, "active_learning", None) is None:
            args.active_learning = True
        if getattr(args, "al_feature_space", None) is None:
            args.al_feature_space = "esm"
    runtime = _build_runtime(args)
    expected_sequence_length = getattr(args, "sequence_length", None)
    if expected_sequence_length is not None and int(expected_sequence_length) != int(runtime.seq_len):
        raise ValueError(
            f"sequence_length={int(expected_sequence_length)} does not match restored process seq_len={runtime.seq_len}"
        )
    pad_token_id = getattr(runtime.model_for_state, "pad_token_id", None)
    rollout_denoise_steps = None
    if args.rollout_denoise_steps is not None:
        rollout_denoise_steps = int(args.rollout_denoise_steps)
        if rollout_denoise_steps <= 0:
            raise ValueError("rollout-denoise-steps must be > 0 when provided")
    candidate_cfg, trust_region = resolve_candidate_config(args, runtime)
    profile_timings = bool(args.profile_timings) if getattr(args, "profile_timings", None) is not None else False
    profile_sync_cuda = bool(args.profile_sync_cuda) if getattr(args, "profile_sync_cuda", None) is not None else False
    score_full_proposal_batch = (
        bool(args.al_score_full_proposal_batch)
        if getattr(args, "al_score_full_proposal_batch", None) is not None
        else False
    )
    if score_full_proposal_batch:
        print(
            "[search] al_score_full_proposal_batch enabled: "
            "all acquisition-eligible proposals are scored for diagnostics, "
            "but only selected proposals update the GP/search state."
        )
    active_learning_selector: Optional[Any] = None
    active_learning_cfg: Optional[ActiveLearningConfig] = None
    pad_edit_enabled = bool(getattr(runtime.model_for_state, "pad_edit_enabled", False))
    if args.rollout_use_corrupt_mask is None:
        rollout_use_corrupt_mask = pad_edit_enabled
    else:
        rollout_use_corrupt_mask = bool(args.rollout_use_corrupt_mask)
    if rollout_use_corrupt_mask:
        print("[search] rollout_use_corrupt_mask enabled: only proposal-corrupted positions may update during rollout.")
    elif pad_edit_enabled:
        print(
            "[search] warning: pad_edit_mode is enabled but rollout_use_corrupt_mask is disabled; "
            "denoising rollout may expand many pad positions beyond the proposal edits."
        )
    legacy_hamming_tr_error = bool(args.legacy_hamming_tr_error) if args.legacy_hamming_tr_error is not None else False
    if legacy_hamming_tr_error:
        print("[search] legacy_hamming_tr_error enabled: trust-region edit residuals use plain Hamming distance.")
    if rollout_denoise_steps is not None:
        print(
            f"[search] rollout_denoise_steps={rollout_denoise_steps}: "
            "exact integration substeps over the remaining interval from sampled t."
        )
    early_stop_patience = int(args.early_stop_patience) if args.early_stop_patience is not None else None
    if early_stop_patience is not None and early_stop_patience <= 0:
        raise ValueError("early_stop_patience must be > 0 when provided")
    if early_stop_patience is not None:
        print(f"[search] early_stop_patience={early_stop_patience}")
    max_oracle_evaluations = int(args.max_oracle_evaluations) if args.max_oracle_evaluations is not None else None
    if max_oracle_evaluations is not None and max_oracle_evaluations <= 0:
        raise ValueError("max_oracle_evaluations must be > 0 when provided")
    if max_oracle_evaluations is not None:
        print(f"[search] max_oracle_evaluations={max_oracle_evaluations}")
    use_active_learning = bool(args.active_learning) if args.active_learning is not None else False
    ehrlich_black_box_feasibility = bool(getattr(args, "al_ehrlich_black_box_feasibility", False))
    oracle_best_budget_mode = str(getattr(args, "al_oracle_best_budget_mode", None) or "physical").strip().lower()
    if oracle_best_budget_mode not in {"physical", "selected_archive"}:
        raise ValueError("al_oracle_best_budget_mode must be one of: physical, selected_archive")
    args.al_oracle_best_budget_mode = oracle_best_budget_mode
    if oracle_best_budget_mode == "selected_archive" and not use_active_learning:
        raise ValueError("al_oracle_best_budget_mode='selected_archive' requires active_learning")
    al_feature_space = str(getattr(args, "al_feature_space", None) or "onehot_tanimoto").strip().lower()
    if ehrlich_black_box_feasibility and not use_active_learning:
        raise ValueError("al_ehrlich_black_box_feasibility requires active_learning")
    if ehrlich_black_box_feasibility and not al_feature_space.startswith("ehrlich_"):
        raise ValueError("al_ehrlich_black_box_feasibility requires an Ehrlich feature space")
    active_learning_selector_factory: Optional[SelectorFactory] = None
    if use_active_learning:
        active_learning_cfg = ActiveLearningConfig(
            enabled=True,
            initial_batch_matches_buffer_size=(
                bool(args.al_initial_batch_matches_buffer_size)
                if args.al_initial_batch_matches_buffer_size is not None
                else False
            ),
            oracle_batch_size=int(args.al_oracle_batch_size)
            if args.al_oracle_batch_size is not None
            else int(args.batch_size or 256),
            oracle_best_budget_mode=oracle_best_budget_mode,
            sequential_selection=bool(args.al_sequential_selection)
            if args.al_sequential_selection is not None
            else False,
            local_penalization=bool(args.al_local_penalization) if args.al_local_penalization is not None else False,
            local_penalization_strength=(
                float(args.al_local_penalization_strength) if args.al_local_penalization_strength is not None else 1.0
            ),
            local_penalization_power=(
                float(args.al_local_penalization_power) if args.al_local_penalization_power is not None else 2.0
            ),
            acquisition=str(args.al_acquisition or "expected_improvement"),
            qei_mc_samples=int(args.al_qei_mc_samples) if args.al_qei_mc_samples is not None else 128,
            explore_batch_size=int(args.al_explore_batch_size) if args.al_explore_batch_size is not None else 0,
            exploit_batch_size=(int(args.al_exploit_batch_size) if args.al_exploit_batch_size is not None else None),
            wei_lambda=float(args.al_wei_lambda) if args.al_wei_lambda is not None else 0.0,
            wei_standardize=(bool(args.al_wei_standardize) if args.al_wei_standardize is not None else False),
            wei_eps=float(args.al_wei_eps) if args.al_wei_eps is not None else 1e-12,
            thompson_above_incumbent_only=(
                bool(args.al_thompson_above_incumbent_only)
                if args.al_thompson_above_incumbent_only is not None
                else False
            ),
            ei_threshold=float(args.al_ei_threshold) if args.al_ei_threshold is not None else None,
            debug_random_sample_count=(
                int(args.al_debug_random_sample_count) if args.al_debug_random_sample_count is not None else 0
            ),
            max_resample_batches=int(args.al_max_resample_batches) if args.al_max_resample_batches is not None else 0,
            buffer_size=int(args.al_buffer_size) if args.al_buffer_size is not None else 256,
            buffer_mode=str(args.al_buffer_mode or "recent"),
            buffer_diversity_max_similarity=(
                float(args.al_buffer_diversity_max_similarity)
                if args.al_buffer_diversity_max_similarity is not None
                else None
            ),
            onehot_max_len=(
                int(args.al_onehot_max_len) if getattr(args, "al_onehot_max_len", None) is not None else None
            ),
            onehot_alphabet=str(getattr(args, "al_onehot_alphabet", None) or DEFAULT_PROTEIN_ONEHOT_ALPHABET),
            gp_device=str(args.al_gp_device or "cpu"),
            gp_dtype=str(args.al_gp_dtype or "double"),
            gp_jitter=float(args.al_gp_jitter) if args.al_gp_jitter is not None else None,
            gp_noise_variance=(float(args.al_gp_noise_variance) if args.al_gp_noise_variance is not None else None),
            gp_outputscale=float(args.al_gp_outputscale) if args.al_gp_outputscale is not None else None,
            gp_learn_outputscale=(
                bool(args.al_gp_learn_outputscale) if args.al_gp_learn_outputscale is not None else True
            ),
            gp_fit_mean_and_noise_only=bool(getattr(args, "al_gp_fit_mean_and_noise_only", False)),
            gp_hyperparameter_refit_interval=int(getattr(args, "al_gp_hyperparameter_refit_interval", 1)),
            gp_lengthscale=(
                float(args.al_gp_lengthscale) if getattr(args, "al_gp_lengthscale", None) is not None else None
            ),
            gp_learn_lengthscale=(
                bool(args.al_gp_learn_lengthscale)
                if getattr(args, "al_gp_learn_lengthscale", None) is not None
                else False
            ),
            ehrlich_black_box_feasibility=ehrlich_black_box_feasibility,
        )
        active_learning_selector_factory, feature_display = _make_active_learning_selector_factory(
            args=args,
            reward_fn=runtime.reward_fn,
            al_feature_space=al_feature_space,
            seq_len=runtime.seq_len,
        )
        active_learning_selector = active_learning_selector_factory(active_learning_cfg)
        if (
            oracle_best_budget_mode == "selected_archive"
            and str(getattr(active_learning_selector, "_acquisition", "") or "") != "oracle_best"
        ):
            raise ValueError("al_oracle_best_budget_mode='selected_archive' requires al_acquisition='oracle_best'")
        if al_feature_space in {"onehot_tanimoto", "protein_onehot_tanimoto", "protein_onehot", "onehot"}:
            feature_display = (
                f"onehot_tanimoto=max_len{active_learning_cfg.onehot_max_len}/"
                f"alphabet{len(active_learning_cfg.onehot_alphabet)}/"
                f"fixed_scale{active_learning_cfg.gp_outputscale if active_learning_cfg.gp_outputscale is not None else 1.0}"
            )
        elif al_feature_space in {"esm", "protein_esm", "esm2"}:
            feature_display = f"{feature_display}/batch{int(getattr(args, 'al_esm_batch_size', None) or 16)}"
        ei_threshold_display = (
            "off"
            if active_learning_cfg.ei_threshold is None or float(active_learning_cfg.ei_threshold) <= 0.0
            else active_learning_cfg.ei_threshold
        )
        print(
            "[search] active_learning enabled: "
            f"initial_batch_matches_buffer_size={active_learning_cfg.initial_batch_matches_buffer_size} "
            f"oracle_batch_size={active_learning_cfg.oracle_batch_size} "
            f"oracle_best_budget_mode={active_learning_cfg.oracle_best_budget_mode} "
            f"sequential_selection={active_learning_cfg.sequential_selection} "
            f"local_penalization={active_learning_cfg.local_penalization} "
            f"local_penalization_strength={active_learning_cfg.local_penalization_strength} "
            f"local_penalization_power={active_learning_cfg.local_penalization_power} "
            f"acquisition={active_learning_cfg.acquisition} "
            f"qei_mc_samples={active_learning_cfg.qei_mc_samples} "
            f"explore_batch_size={active_learning_cfg.explore_batch_size} "
            f"exploit_batch_size={active_learning_cfg.exploit_batch_size} "
            f"wei_lambda={active_learning_cfg.wei_lambda} "
            f"wei_standardize={active_learning_cfg.wei_standardize} "
            f"thompson_above_incumbent_only={active_learning_cfg.thompson_above_incumbent_only} "
            f"ei_threshold={ei_threshold_display} "
            f"debug_random_sample_count={active_learning_cfg.debug_random_sample_count} "
            f"max_resample_batches={active_learning_cfg.max_resample_batches} "
            f"buffer_size={active_learning_cfg.buffer_size} "
            f"buffer_mode={active_learning_cfg.buffer_mode} "
            f"buffer_diversity_max_similarity={active_learning_cfg.buffer_diversity_max_similarity} "
            f"ehrlich_black_box_feasibility={active_learning_cfg.ehrlich_black_box_feasibility} "
            f"gp_jitter={active_learning_cfg.gp_jitter} "
            f"gp_noise_variance={active_learning_cfg.gp_noise_variance if active_learning_cfg.gp_noise_variance is not None else 'learn'} "
            f"gp_outputscale={('learn' if active_learning_cfg.gp_learn_outputscale else (active_learning_cfg.gp_outputscale if active_learning_cfg.gp_outputscale is not None else 1.0))} "
            f"gp_hyperparameter_refit_interval={active_learning_cfg.gp_hyperparameter_refit_interval} "
            f"gp_lengthscale={('learn' if active_learning_cfg.gp_learn_lengthscale else (active_learning_cfg.gp_lengthscale if active_learning_cfg.gp_lengthscale is not None else 'sqrt_dim'))} "
            f"feature_space={al_feature_space} "
            f"{feature_display}"
        )
    diagnostic_injected_candidate = _load_full_dataset_best_candidate(args, runtime)
    diagnostic_stop_on_injected_selection = (
        bool(args.diagnostic_stop_on_injected_selection)
        if getattr(args, "diagnostic_stop_on_injected_selection", None) is not None
        else False
    )
    diagnostic_injection_affects_collapse = (
        bool(args.diagnostic_injection_affects_collapse)
        if getattr(args, "diagnostic_injection_affects_collapse", None) is not None
        else False
    )
    if diagnostic_injected_candidate is not None:
        if not use_active_learning:
            raise ValueError("diagnostic_inject_full_dataset_best requires active_learning=true")
        print(
            "[diagnostic] full-DMS best injection enabled: "
            f"stop_on_selection={int(diagnostic_stop_on_injected_selection)} "
            f"affects_collapse={int(diagnostic_injection_affects_collapse)}"
        )
    proposal_collapse_min_acq_fraction = (
        float(args.proposal_collapse_min_acq_frac)
        if getattr(args, "proposal_collapse_min_acq_frac", None) is not None
        else None
    )
    proposal_collapse_min_acq_count = (
        int(args.proposal_collapse_min_acq_count)
        if getattr(args, "proposal_collapse_min_acq_count", None) is not None
        else None
    )
    proposal_collapse_threshold_configured = (
        proposal_collapse_min_acq_fraction is not None or proposal_collapse_min_acq_count is not None
    )
    proposal_collapse_max_resample_batches = (
        int(args.proposal_collapse_max_resample_batches)
        if getattr(args, "proposal_collapse_max_resample_batches", None) is not None
        else (1 if proposal_collapse_threshold_configured else 0)
    )
    proposal_collapse_top_up = (
        bool(args.proposal_collapse_top_up) if getattr(args, "proposal_collapse_top_up", None) is not None else False
    )
    if proposal_collapse_min_acq_fraction is not None and not (0.0 < proposal_collapse_min_acq_fraction <= 1.0):
        raise ValueError("proposal_collapse_min_acq_frac must be in (0, 1]")
    if proposal_collapse_min_acq_count is not None and proposal_collapse_min_acq_count < 1:
        raise ValueError("proposal_collapse_min_acq_count must be >= 1")
    if proposal_collapse_max_resample_batches < 0:
        raise ValueError("proposal_collapse_max_resample_batches must be >= 0")
    if proposal_collapse_threshold_configured and not use_active_learning:
        raise ValueError("proposal collapse detection requires active_learning=true")
    proposal_collapse_strategy = str(candidate_cfg.strategy).strip().lower()
    proposal_collapse_supported_strategies = {"random", "trust_region"}
    if (
        proposal_collapse_threshold_configured
        and proposal_collapse_strategy not in proposal_collapse_supported_strategies
    ):
        raise ValueError("proposal collapse detection requires candidate_strategy=random or trust_region")
    proposal_collapse_enabled = (
        proposal_collapse_threshold_configured
        and proposal_collapse_max_resample_batches > 0
        and use_active_learning
        and proposal_collapse_strategy in proposal_collapse_supported_strategies
    )
    if proposal_collapse_threshold_configured:
        if proposal_collapse_top_up:
            collapse_action = (
                "keep_candidates_expand_trust_region_once_and_top_up"
                if proposal_collapse_strategy == "trust_region"
                else "keep_candidates_redraw_random_times_and_top_up"
            )
        else:
            collapse_action = (
                "discard_batch_expand_trust_region_once_and_resample"
                if proposal_collapse_strategy == "trust_region"
                else "discard_batch_redraw_random_times_and_resample"
            )
        print(
            "[search] proposal collapse detection enabled: "
            f"min_acq_frac={proposal_collapse_min_acq_fraction} "
            f"min_acq_count={proposal_collapse_min_acq_count} "
            f"max_resample_batches={proposal_collapse_max_resample_batches} "
            f"top_up={proposal_collapse_top_up} "
            f"action={collapse_action}"
        )
    num_trust_regions = int(getattr(args, "num_trust_regions", None) or 1)
    if proposal_collapse_threshold_configured and num_trust_regions > 1:
        raise ValueError("proposal collapse detection is only implemented for single-region runs")
    if num_trust_regions > 1:
        if not use_active_learning or active_learning_cfg is None or active_learning_selector_factory is None:
            raise ValueError("num_trust_regions > 1 requires active_learning=true")
        if str(candidate_cfg.strategy).strip().lower() != "trust_region":
            raise ValueError("num_trust_regions > 1 currently requires candidate_strategy=trust_region")
        restart_patience_display = int(args.tr_restart_patience) if args.tr_restart_patience is not None else 10
        multi_region_score_mode = _normalize_multi_region_score_mode(getattr(args, "multi_region_score_mode", None))
        print(
            "[search] multi-region Thompson-over-union enabled: "
            f"num_trust_regions={num_trust_regions} "
            f"tr_restart_patience={restart_patience_display} "
            "allocation=slotwise_thompson "
            f"score_mode={multi_region_score_mode} "
            "restart_mode=boundary_only "
            "restart_seed=pure_noise"
        )
        resources = _open_search_resources(args, runtime, profile_timings=profile_timings)
        monitor = resources.monitor
        samples_fp = resources.samples_fp
        proposal_diagnostics_fp = resources.proposal_diagnostics_fp
        proposal_traces_fp = resources.proposal_traces_fp
        timing_fp = resources.timing_fp
        if profile_timings:
            timing_path = getattr(timing_fp, "name", None) if timing_fp is not None else "<disabled>"
            print(f"[search] profile_timings enabled: sync_cuda={profile_sync_cuda} timings_out={timing_path}")
        try:
            best_overall, best_reward, leaves_log = _run_multi_region_thompson_union_search(
                args=args,
                runtime=runtime,
                candidate_cfg=candidate_cfg,
                trust_region=trust_region,
                active_learning_cfg=active_learning_cfg,
                selector_factory=active_learning_selector_factory,
                monitor=monitor,
                samples_fp=samples_fp,
                rollout_use_corrupt_mask=rollout_use_corrupt_mask,
                rollout_denoise_steps=rollout_denoise_steps,
                legacy_hamming_tr_error=legacy_hamming_tr_error,
                early_stop_patience=early_stop_patience,
                max_oracle_evaluations=max_oracle_evaluations,
                pad_token_id=pad_token_id,
            )
            if best_overall is not None:
                print("Best sequence found:", " ".join(str(int(v)) for v in best_overall[0].tolist()))
                print(f"Best reward: {best_reward:.3f}")
            else:
                print("No sequences were found (unexpected).")
            if args.leaves_out and leaves_log:
                out_path = Path(args.leaves_out).expanduser()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"leaves": torch.cat(leaves_log, dim=0)}, out_path)
                print(f"Saved leaves to {out_path}")
        finally:
            resources.close()
        return
    proposal_policy = InterpolantProposalPolicy(
        args=args,
        runtime=runtime,
        candidate=candidate_cfg,
        trust_region=trust_region,
        active_learning_selector=active_learning_selector,
    )
    selection_policy = BestSurvivorSelectionPolicy()

    elite_prior_k = int(args.elite_prior_k) if args.elite_prior_k is not None else None
    elite_prior_alpha = float(args.elite_prior_alpha)
    elite_prior_min = int(args.elite_prior_min)
    elite_prior_weighted = bool(args.elite_prior_weighted) if args.elite_prior_weighted is not None else False
    elite_prior_mix = float(args.elite_prior_mix)

    sample_only_arg = getattr(args, "sample_only", None)
    sample_only_count_arg = getattr(args, "sample_only_count", None)
    sample_only = bool(sample_only_arg) if sample_only_arg is not None else False
    sample_only_count = int(sample_only_count_arg) if sample_only_count_arg is not None else None
    if sample_only_count is not None and sample_only_count <= 0:
        raise ValueError("sample_only_count must be > 0 when provided")
    sample_only_batch_size_arg = getattr(args, "sample_only_batch_size", None)
    if sample_only_batch_size_arg is not None:
        if not sample_only:
            raise ValueError("sample_only_batch_size requires sample_only=true")
        if int(sample_only_batch_size_arg) <= 0:
            raise ValueError("sample_only_batch_size must be > 0 when provided")
    if sample_only:
        print(
            "[search] sample_only enabled: "
            f"sampling and scoring {sample_only_count or int(args.batch_size or 256)} model samples "
            "with no optimisation iterations."
        )

    planned_initial_num_samples = _initial_num_samples_from_args(
        args,
        sample_only=sample_only,
        sample_only_count=sample_only_count,
        active_learning_cfg=active_learning_cfg,
    )
    if max_oracle_evaluations is not None and planned_initial_num_samples > max_oracle_evaluations:
        print(
            "[search] capping initial oracle evaluations: "
            f"requested={planned_initial_num_samples} "
            f"max_oracle_evaluations={max_oracle_evaluations}."
        )
        planned_initial_num_samples = int(max_oracle_evaluations)
    resources = _open_search_resources(args, runtime, profile_timings=profile_timings)
    monitor = resources.monitor
    samples_fp = resources.samples_fp
    proposal_diagnostics_fp = resources.proposal_diagnostics_fp
    proposal_traces_fp = resources.proposal_traces_fp
    timing_fp = resources.timing_fp
    try:
        initial_state = _initialize_search_state(
            args=args,
            runtime=runtime,
            planned_num_samples=planned_initial_num_samples,
            max_oracle_evaluations=max_oracle_evaluations,
            pad_token_id=pad_token_id,
            active_learning_selector=active_learning_selector,
            selection_policy=selection_policy,
            profile_timings=profile_timings,
            profile_sync_cuda=profile_sync_cuda,
        )
    except BaseException:
        resources.close()
        raise
    current_batch = initial_state.batch
    current_rewards = initial_state.rewards
    # The initial archive is both selected and physically scored, so all budget
    # modes begin with identical budget and physical totals and no privileged
    # oracle calls.
    total_oracle_evaluated = initial_state.oracle_evaluated
    total_oracle_physical = initial_state.oracle_evaluated
    total_oracle_privileged = 0
    initial_sample_source = initial_state.sample_source
    initial_timer = initial_state.timings
    if profile_timings:
        timing_path = getattr(timing_fp, "name", None) if timing_fp is not None else "<disabled>"
        print(f"[search] profile_timings enabled: sync_cuda={profile_sync_cuda} timings_out={timing_path}")
    if timing_fp is not None:
        _write_timing_jsonl(
            timing_fp,
            event="initial_batch",
            iteration=None,
            seconds=initial_timer.seconds,
            counts=initial_timer.counts,
            extra={"sample_source": initial_sample_source},
        )
        timing_fp.flush()

    if sample_only:
        try:
            if samples_fp is not None:
                _write_sample_only_jsonl(
                    samples_fp,
                    sequences=current_batch,
                    scores=current_rewards,
                    batch_id=0,
                    pad_token_id=pad_token_id,
                    decoded_sequences=_decode_sequences_for_output(runtime.reward_fn, current_batch),
                )
                samples_fp.flush()
            if monitor is not None:
                summary = summarise_search_batch(current_batch, current_rewards)
                _log_summary(monitor, 0, summary)
            if current_rewards.numel() > 0:
                best_idx = int(torch.argmax(current_rewards).item())
                best_seq = current_batch[best_idx : best_idx + 1]
                best_reward = float(current_rewards[best_idx].item())
                print("Best sequence found:", " ".join(str(int(v)) for v in best_seq[0].tolist()))
                print(f"Best reward: {best_reward:.3f}")
            else:
                print("No sequences were sampled.")
        finally:
            resources.close()
        return

    elite_pool: list[torch.Tensor] = []
    elite_scores: list[float] = []
    elite_probs: Optional[torch.Tensor] = None

    num_iterations = args.num_iterations or 1
    fill_exact_oracle_budget = (
        active_learning_cfg is not None
        and max_oracle_evaluations is not None
        and total_oracle_evaluated
        + num_iterations * int(active_learning_cfg.oracle_batch_size)
        == max_oracle_evaluations
    )
    iteration_limit = num_iterations * 2 if fill_exact_oracle_budget else num_iterations
    no_improve_iters = 0
    if max_oracle_evaluations is not None and total_oracle_evaluated >= max_oracle_evaluations:
        print(
            "[search] stopping before search iterations: "
            f"max_oracle_evaluations={max_oracle_evaluations} reached "
            f"(oracle_total={total_oracle_evaluated})."
        )
        num_iterations = 0
        iteration_limit = 0
    try:
        for it in range(iteration_limit):
            if it == num_iterations:
                print(
                    "[search] extending exact-budget schedule after underfilled batches: "
                    f"oracle_total={total_oracle_evaluated} "
                    f"max_oracle_evaluations={max_oracle_evaluations}."
                )
            iter_timer = TimingRecorder(
                enabled=profile_timings,
                sync_cuda=profile_sync_cuda,
                device=runtime.device,
            )
            iter_total_start = iter_timer.start()
            if current_rewards is not None:
                rewards = current_rewards
            else:
                with iter_timer.measure("reward_refresh_seconds"):
                    rewards = _score_reward_batch_once(
                        current_batch,
                        runtime.reward_fn,
                        context="iteration reward refresh",
                    )

            with iter_timer.measure("summary_log_seconds"):
                summary = summarise_search_batch(current_batch, rewards)
                _log_summary(monitor, it, summary)

            if elite_prior_k is not None and rewards.numel() > 0:
                elite_pool, elite_scores = update_elite_pool(
                    elite_pool,
                    elite_scores,
                    current_batch,
                    rewards,
                    k=elite_prior_k,
                )
                if len(elite_pool) >= elite_prior_min:
                    elite_probs = elite_position_probs(
                        elite_pool,
                        vocab_size=runtime.vocab_size,
                        alpha=elite_prior_alpha,
                        device=runtime.device,
                        weights=elite_scores if elite_prior_weighted else None,
                        mix=elite_prior_mix,
                    )
                else:
                    elite_probs = None

            proposal_batch = None
            eval_result = None
            resample_count = 0
            collapse_resample_count = 0
            proposal_collapse_target_count: Optional[int] = None
            max_resample_batches = (
                int(active_learning_cfg.max_resample_batches) if active_learning_cfg is not None else 0
            )
            while True:
                if proposal_batch is None:
                    with iter_timer.measure("proposal_generate_seconds"):
                        proposal_batch = proposal_policy.propose(
                            iteration=it,
                            batch=current_batch,
                            rewards=rewards,
                            elite_probs=elite_probs,
                        )
                    iter_timer.add_count("proposal_batches_generated", 1)
                    proposal_collapse_target_count = int(len(proposal_batch.proposals))
                collapse_block_selection = proposal_collapse_enabled and (
                    not proposal_collapse_top_up or collapse_resample_count < proposal_collapse_max_resample_batches
                )
                eval_result = evaluate_proposals(
                    proposal_batch,
                    adapter=runtime.adapter,
                    reward_fn=runtime.reward_fn,
                    track_trust_region=(candidate_cfg.strategy == "trust_region"),
                    rollout_use_corrupt_mask=rollout_use_corrupt_mask,
                    rollout_denoise_steps=rollout_denoise_steps,
                    legacy_hamming_tr_error=legacy_hamming_tr_error,
                    active_learning_selector=active_learning_selector,
                    active_learning_budget_override=(
                        None
                        if max_oracle_evaluations is None
                        else max(0, int(max_oracle_evaluations) - int(total_oracle_evaluated))
                    ),
                    score_full_proposal_batch=score_full_proposal_batch,
                    diagnostic_injected_candidate=diagnostic_injected_candidate,
                    diagnostic_injection_affects_collapse=diagnostic_injection_affects_collapse,
                    proposal_collapse_min_acq_fraction=(
                        proposal_collapse_min_acq_fraction if proposal_collapse_enabled else None
                    ),
                    proposal_collapse_min_acq_count=(
                        proposal_collapse_min_acq_count if proposal_collapse_enabled else None
                    ),
                    proposal_collapse_target_count=(
                        proposal_collapse_target_count
                        if proposal_collapse_enabled and proposal_collapse_top_up
                        else None
                    ),
                    proposal_collapse_block_selection=collapse_block_selection,
                    profile_timings=profile_timings,
                    profile_sync_cuda=profile_sync_cuda,
                )
                attempt_oracle_budget_debit = int(eval_result.oracle_budget_debit_count)
                attempt_oracle_scored_count = int(eval_result.oracle_scored_count)
                attempt_oracle_privileged_count = int(eval_result.oracle_privileged_scored_count)
                if (
                    attempt_oracle_budget_debit < 0
                    or attempt_oracle_scored_count < 0
                    or attempt_oracle_privileged_count < 0
                ):
                    raise RuntimeError(
                        "proposal evaluation returned a negative oracle counter: "
                        f"budget_debit={attempt_oracle_budget_debit} "
                        f"scored={attempt_oracle_scored_count} "
                        f"privileged={attempt_oracle_privileged_count}"
                    )
                if attempt_oracle_budget_debit + attempt_oracle_privileged_count != attempt_oracle_scored_count:
                    raise RuntimeError(
                        "proposal evaluation returned inconsistent oracle counters: "
                        f"budget_debit={attempt_oracle_budget_debit} "
                        f"privileged={attempt_oracle_privileged_count} "
                        f"physical_scored={attempt_oracle_scored_count}"
                    )
                if (
                    max_oracle_evaluations is not None
                    and total_oracle_evaluated + attempt_oracle_budget_debit > max_oracle_evaluations
                ):
                    raise RuntimeError(
                        "proposal evaluation exceeded the remaining oracle budget: "
                        f"oracle_total={total_oracle_evaluated} "
                        f"budget_debit={attempt_oracle_budget_debit} "
                        f"physical_scored={attempt_oracle_scored_count} "
                        f"max_oracle_evaluations={max_oracle_evaluations}"
                    )
                # Debit every attempt immediately. In the default physical mode
                # this is every scored row. The explicit selected_archive mode
                # debits only rows admitted to the archive while retaining
                # separate physical and privileged totals.
                total_oracle_evaluated += attempt_oracle_budget_debit
                total_oracle_physical += attempt_oracle_scored_count
                total_oracle_privileged += attempt_oracle_privileged_count
                iter_timer.add_count("proposal_evaluations", 1)
                iter_timer.add_seconds_mapping(eval_result.timing_seconds, prefix="evaluate.")
                iter_timer.add_counts_mapping(eval_result.timing_counts, prefix="evaluate.")
                should_collapse_resample = (
                    proposal_collapse_enabled
                    and bool(eval_result.proposal_collapse_detected)
                    and collapse_resample_count < proposal_collapse_max_resample_batches
                )
                if should_collapse_resample:
                    collapse_resample_count += 1
                    if not proposal_collapse_top_up:
                        if proposal_diagnostics_fp is not None:
                            _write_proposal_diagnostics_jsonl(
                                proposal_diagnostics_fp,
                                iteration=it,
                                eval_result=eval_result,
                                oracle_budget_total=total_oracle_evaluated,
                                oracle_physical_total=total_oracle_physical,
                                oracle_privileged_total=total_oracle_privileged,
                                resample_count=resample_count,
                                collapse_resample_count=collapse_resample_count,
                                discarded=True,
                            )
                            proposal_diagnostics_fp.flush()
                        collapse_log_parts = [
                            "[search] proposal collapse resample "
                            f"{collapse_resample_count}/{proposal_collapse_max_resample_batches}:",
                            (
                                f"acq={eval_result.change_stats.acquisition_batch_size}/"
                                f"{max(1, eval_result.change_stats.generated_proposals)}"
                            ),
                            f"min_acq={eval_result.proposal_collapse_min_acquisition_count}",
                        ]
                        if proposal_collapse_strategy == "trust_region":
                            old_state, new_state = proposal_policy.expand_trust_region_once()
                            collapse_log_parts.append(f"tr_state={_fmt_opt(old_state)}->{_fmt_opt(new_state)}")
                        else:
                            collapse_log_parts.append("action=redraw_random_times")
                        print(" ".join(collapse_log_parts))
                        proposal_batch = None
                        proposal_collapse_target_count = None
                        continue

                    target_count = int(
                        proposal_collapse_target_count
                        if proposal_collapse_target_count is not None
                        else max(1, eval_result.change_stats.generated_proposals)
                    )
                    if eval_result.proposal_collapse_min_acquisition_count is not None:
                        target_count = max(target_count, int(eval_result.proposal_collapse_min_acquisition_count))
                    topup_requested = max(0, target_count - int(eval_result.change_stats.acquisition_batch_size))
                    collapse_log_parts = [
                        "[search] proposal collapse top-up "
                        f"{collapse_resample_count}/{proposal_collapse_max_resample_batches}:",
                        (f"acq={eval_result.change_stats.acquisition_batch_size}/{target_count}"),
                        f"min_acq={eval_result.proposal_collapse_min_acquisition_count}",
                        f"request={topup_requested}",
                    ]
                    if proposal_collapse_strategy == "trust_region":
                        old_state, new_state = proposal_policy.expand_trust_region_once()
                        collapse_log_parts.append(f"tr_state={_fmt_opt(old_state)}->{_fmt_opt(new_state)}")
                    else:
                        collapse_log_parts.append("action=redraw_random_times")
                    topup_generated = 0
                    if topup_requested <= 0:
                        if proposal_diagnostics_fp is not None:
                            _write_proposal_diagnostics_jsonl(
                                proposal_diagnostics_fp,
                                iteration=it,
                                eval_result=eval_result,
                                oracle_budget_total=total_oracle_evaluated,
                                oracle_physical_total=total_oracle_physical,
                                oracle_privileged_total=total_oracle_privileged,
                                resample_count=resample_count,
                                collapse_resample_count=collapse_resample_count,
                                discarded=False,
                                collapse_topup_requested=topup_requested,
                                collapse_topup_generated=topup_generated,
                            )
                            proposal_diagnostics_fp.flush()
                        print(" ".join(collapse_log_parts))
                        continue
                    with iter_timer.measure("proposal_generate_seconds"):
                        topup_batch = proposal_policy.propose(
                            iteration=it,
                            batch=current_batch,
                            rewards=rewards,
                            elite_probs=elite_probs,
                            max_proposals=topup_requested,
                        )
                    iter_timer.add_count("proposal_batches_generated", 1)
                    topup_generated = int(len(topup_batch.proposals))
                    proposal_batch = _merge_proposal_batches([proposal_batch, topup_batch])
                    if proposal_diagnostics_fp is not None:
                        _write_proposal_diagnostics_jsonl(
                            proposal_diagnostics_fp,
                            iteration=it,
                            eval_result=eval_result,
                            oracle_budget_total=total_oracle_evaluated,
                            oracle_physical_total=total_oracle_physical,
                            oracle_privileged_total=total_oracle_privileged,
                            resample_count=resample_count,
                            collapse_resample_count=collapse_resample_count,
                            discarded=False,
                            collapse_topup_requested=topup_requested,
                            collapse_topup_generated=topup_generated,
                        )
                        proposal_diagnostics_fp.flush()
                    collapse_log_parts.append(f"generated={len(topup_batch.proposals)}")
                    print(" ".join(collapse_log_parts))
                    continue
                should_resample = (
                    active_learning_selector is not None
                    and not bool(eval_result.proposal_collapse_detected)
                    and bool(eval_result.active_learning_threshold_exhausted)
                    and int(eval_result.oracle_evaluated_count) == 0
                    and resample_count < max_resample_batches
                )
                if not should_resample:
                    break
                resample_count += 1
                if proposal_diagnostics_fp is not None:
                    _write_proposal_diagnostics_jsonl(
                        proposal_diagnostics_fp,
                        iteration=it,
                        eval_result=eval_result,
                        oracle_budget_total=total_oracle_evaluated,
                        oracle_physical_total=total_oracle_physical,
                        oracle_privileged_total=total_oracle_privileged,
                        resample_count=resample_count,
                        collapse_resample_count=collapse_resample_count,
                        discarded=True,
                    )
                    proposal_diagnostics_fp.flush()
                print(
                    "[search] active_learning resample "
                    f"{resample_count}/{max_resample_batches}: "
                    f"no candidate passed active-learning gate "
                    f"(max_ei={_fmt_debug_float(eval_result.active_learning_max_expected_improvement)}, "
                    f"above={eval_result.active_learning_num_above_threshold})."
                )
                proposal_batch = None
                proposal_collapse_target_count = None
            assert proposal_batch is not None
            assert eval_result is not None
            with iter_timer.measure("write_outputs_seconds"):
                if proposal_diagnostics_fp is not None:
                    _write_proposal_diagnostics_jsonl(
                        proposal_diagnostics_fp,
                        iteration=it,
                        eval_result=eval_result,
                        oracle_budget_total=total_oracle_evaluated,
                        oracle_physical_total=total_oracle_physical,
                        oracle_privileged_total=total_oracle_privileged,
                        resample_count=resample_count,
                        collapse_resample_count=collapse_resample_count,
                    )
                    proposal_diagnostics_fp.flush()
                if proposal_traces_fp is not None:
                    _write_transition_jsonl(
                        proposal_traces_fp,
                        iteration=it,
                        traces=eval_result.proposal_traces_all,
                        start_idx=0,
                        pad_token_id=pad_token_id,
                    )
                    proposal_traces_fp.flush()
                if samples_fp is not None:
                    traces_to_write = (
                        [trace for trace in eval_result.proposal_traces if trace.oracle_evaluated]
                        if active_learning_selector is not None
                        else eval_result.proposal_traces_all
                    )
                    _write_transition_jsonl(
                        samples_fp,
                        iteration=it,
                        traces=traces_to_write,
                        start_idx=int(current_batch.shape[0]),
                        pad_token_id=pad_token_id,
                    )
                    samples_fp.flush()
            with iter_timer.measure("policy_update_seconds"):
                _log_change_stats(monitor, it, eval_result.change_stats)
                proposal_policy.update_from_evaluations(
                    proposals=eval_result.evaluated_proposals,
                    proposal_rewards=eval_result.evaluated_rewards,
                    base_rewards=rewards,
                )

            with iter_timer.measure("selection_update_seconds"):
                if active_learning_selector is not None and not eval_result.candidate_leaves:
                    prev_best = float(selection_policy.best_reward)
                    current_batch = current_batch
                    current_rewards = current_rewards
                else:
                    current_batch, prev_best = selection_policy.update(
                        candidate_leaves=eval_result.candidate_leaves,
                        best_candidate_state=eval_result.best_candidate_state,
                        best_candidate_reward=eval_result.best_candidate_reward,
                    )
                    next_rewards: list[float] = []
                    if selection_policy.best_overall is not None:
                        next_rewards.append(float(selection_policy.best_reward))
                    next_rewards.extend(float(v) for v in eval_result.candidate_rewards.detach().cpu().tolist())
                    current_rewards = torch.as_tensor(next_rewards, dtype=torch.float, device=runtime.device)

            chosen_time = (
                eval_result.tr_best_t if candidate_cfg.strategy == "trust_region" else eval_result.best_candidate_t
            )
            improved = selection_policy.best_reward > prev_best + trust_region.eps
            if improved:
                no_improve_iters = 0
            else:
                no_improve_iters += 1
            best_len = (
                _sequence_length(selection_policy.best_overall[0], pad_token_id)
                if selection_policy.best_overall is not None
                else None
            )
            mean_edit_error = _mean_normalized_edit_error(
                eval_result.proposal_traces,
                editable_seq_len_for_sample=proposal_policy.time_policy.editable_seq_len_for_sample,
            )
            if (not bool(eval_result.proposal_collapse_detected)) or int(eval_result.oracle_evaluated_count) > 0:
                with iter_timer.measure("trust_region_update_seconds"):
                    proposal_policy.update_trust_region(
                        best_time=chosen_time,
                        best_edits=(
                            eval_result.tr_best_edits
                            if candidate_cfg.strategy == "trust_region"
                            else eval_result.best_candidate_edits
                        ),
                        improved=improved,
                        mean_edit_error=mean_edit_error,
                    )

            num_leaves = len(eval_result.candidate_leaves) if eval_result.candidate_leaves else 1
            unique_vs_parent, unique_vs_each_other = _leaf_uniqueness_counts(eval_result.proposal_traces)
            expected_total, actual_total = _aggregate_edit_counts(eval_result.proposal_traces)
            summary_parts = [
                f"[iter {it + 1:04d}]",
                f"best={selection_policy.best_reward:.3f}",
                f"leaves={num_leaves}",
                f"uniq_p={unique_vs_parent}",
                f"uniq_l={unique_vs_each_other}",
                f"exp_ed={_fmt_edit_count(expected_total)}",
                f"act_ed={_fmt_edit_count(actual_total)}",
                f"t={_fmt_time_summary(eval_result.proposal_traces)}",
                f"len={best_len if best_len is not None else 'na'}",
                f"len_prop={_fmt_length_summary(eval_result.proposal_traces, pad_token_id)}",
                f"invalid={eval_result.change_stats.invalid_generated_proposals}/{max(1, eval_result.change_stats.generated_proposals)}",
                f"dup={eval_result.change_stats.duplicate_generated_leaves}/{max(1, eval_result.change_stats.generated_proposals)}",
            ]
            if active_learning_selector is not None:
                buffer_stats = active_learning_selector.last_buffer_stats
                summary_parts.append(
                    f"oracle={eval_result.oracle_evaluated_count}/{max(1, len(eval_result.proposal_traces_all))}"
                )
                summary_parts.append(
                    f"cached={eval_result.change_stats.cached_generated_proposals}/{max(1, eval_result.change_stats.generated_proposals)}"
                )
                summary_parts.append(
                    f"acq={eval_result.change_stats.acquisition_batch_size}/{max(1, eval_result.change_stats.generated_proposals)}"
                )
                if eval_result.proposal_collapse_target_count is not None:
                    summary_parts.append(f"collapse_target={int(eval_result.proposal_collapse_target_count)}")
                if eval_result.proposal_collapse_detected:
                    summary_parts.append("collapse=1")
                    summary_parts.append(
                        f"collapse_acq_frac={_fmt_debug_float(eval_result.proposal_collapse_acquisition_fraction)}"
                    )
                if collapse_resample_count > 0:
                    summary_parts.append(f"collapse_resamples={collapse_resample_count}")
                summary_parts.append(f"oracle_total={total_oracle_evaluated}")
                if oracle_best_budget_mode == "selected_archive":
                    summary_parts.append(f"oracle_physical_total={total_oracle_physical}")
                    summary_parts.append(f"oracle_privileged_total={total_oracle_privileged}")
                summary_parts.append(f"ei_max={_fmt_debug_float(eval_result.active_learning_max_expected_improvement)}")
                summary_parts.append(f"gate+={int(eval_result.active_learning_num_above_threshold)}")
                summary_parts.append(
                    "gpbuf="
                    f"{active_learning_selector.buffer_size}"
                    f"(pre={buffer_stats.pre_size},valid={buffer_stats.valid_rows},"
                    f"add={buffer_stats.admitted},swap={buffer_stats.replaced},rej={buffer_stats.rejected_similar})"
                )
                if resample_count > 0:
                    summary_parts.append(f"resamples={resample_count}")
                if eval_result.full_batch_top_selected is not None:
                    summary_parts.append(f"full_top_selected={int(bool(eval_result.full_batch_top_selected))}")
                    summary_parts.append(f"full_top={_fmt_debug_float(eval_result.full_batch_top_score)}")
                if eval_result.diagnostic_injected_global_best_present:
                    summary_parts.append(
                        f"inj_best_sel={int(bool(eval_result.diagnostic_injected_global_best_selected))}"
                    )
                    summary_parts.append(
                        "inj_rank="
                        f"{eval_result.diagnostic_injected_global_best_rank_by_acquisition if eval_result.diagnostic_injected_global_best_rank_by_acquisition is not None else 'na'}"
                    )
                    summary_parts.append(
                        f"inj_ei={_fmt_debug_float(eval_result.diagnostic_injected_global_best_expected_improvement)}"
                    )
            summary_line = " | ".join(summary_parts)
            print(summary_line)
            if (
                active_learning_selector is not None
                and eval_result.full_batch_top_selected is False
                and eval_result.full_batch_top_score is not None
            ):
                print(
                    "[al-diagnostic] GP missed full-batch top scorer: "
                    f"score={_fmt_debug_float(eval_result.full_batch_top_score)} "
                    f"best_selected={_fmt_debug_float(eval_result.full_batch_best_selected_score)} "
                    f"t={_fmt_opt(eval_result.full_batch_top_t_value)} "
                    f"smiles={_short_smiles(eval_result.full_batch_top_smiles)}"
                )
            if active_learning_cfg is not None and int(active_learning_cfg.debug_random_sample_count) > 0:
                for debug_line in _iter_active_learning_debug_lines(
                    eval_result.proposal_traces_all,
                    sample_count=int(active_learning_cfg.debug_random_sample_count),
                    seed=int(args.seed or 0),
                    iteration=int(it),
                    active_learning_selector=active_learning_selector,
                ):
                    print(debug_line)
            if args.print_leaves:
                tokens = " ".join(str(int(v)) for v in eval_result.best_candidate_state[0].tolist())
                leaf_len = _sequence_length(eval_result.best_candidate_state[0], pad_token_id)
                print(f"leaf_len={leaf_len} leaf=[{tokens}]")
            iter_timer.add_count("oracle_total", total_oracle_evaluated)
            iter_timer.add_count("oracle_budget_total", total_oracle_evaluated)
            iter_timer.add_count("oracle_physical_total", total_oracle_physical)
            iter_timer.add_count("oracle_privileged_total", total_oracle_privileged)
            iter_timer.add_count("resample_count", resample_count)
            iter_timer.add_count("collapse_resample_count", collapse_resample_count)
            iter_timer.add_count("best_reward", selection_policy.best_reward)
            iter_timer.add_count("current_batch_size", int(current_batch.shape[0]))
            iter_timer.stop("iteration_total_seconds", iter_total_start)
            if timing_fp is not None:
                _write_timing_jsonl(
                    timing_fp,
                    event="iteration",
                    iteration=it,
                    seconds=iter_timer.seconds,
                    counts=iter_timer.counts,
                    extra={
                        "best_reward": float(selection_policy.best_reward),
                        "oracle_total": int(total_oracle_evaluated),
                        "oracle_budget_total": int(total_oracle_evaluated),
                        "oracle_physical_total": int(total_oracle_physical),
                        "oracle_privileged_total": int(total_oracle_privileged),
                        "proposal_collapse_detected": bool(eval_result.proposal_collapse_detected),
                    },
                )
                timing_fp.flush()
            if diagnostic_stop_on_injected_selection and bool(eval_result.diagnostic_injected_global_best_selected):
                print(
                    "[diagnostic] stopping: injected full-DMS best candidate was selected "
                    f"(oracle_total={total_oracle_evaluated}, "
                    f"rank={eval_result.diagnostic_injected_global_best_rank_by_acquisition})."
                )
                break
            if early_stop_patience is not None and no_improve_iters >= early_stop_patience:
                print(
                    "[search] early stopping: "
                    f"best score unchanged for {no_improve_iters} iterations "
                    f"(patience={early_stop_patience})."
                )
                break
            if max_oracle_evaluations is not None and total_oracle_evaluated >= max_oracle_evaluations:
                print(
                    "[search] stopping: "
                    f"max_oracle_evaluations={max_oracle_evaluations} reached "
                    f"(oracle_total={total_oracle_evaluated})."
                )
                break

        if samples_fp is not None or monitor is not None:
            rewards = (
                current_rewards
                if current_rewards is not None
                else _score_reward_batch_once(
                    current_batch,
                    runtime.reward_fn,
                    context="final summary",
                )
            )
            if monitor is not None:
                summary = summarise_search_batch(current_batch, rewards)
                _log_summary(monitor, num_iterations, summary)
        if selection_policy.best_overall is not None:
            print(
                "Best sequence found:",
                " ".join(str(int(v)) for v in selection_policy.best_overall[0].tolist()),
            )
            print(f"Best reward: {selection_policy.best_reward:.3f}")
        else:
            print("No sequences were found (unexpected).")

        if args.leaves_out and selection_policy.leaves_log:
            out_path = Path(args.leaves_out).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"leaves": torch.cat(selection_policy.leaves_log, dim=0)}, out_path)
            print(f"Saved leaves to {out_path}")
    finally:
        resources.close()


__all__ = ["run_interpolant_search"]
