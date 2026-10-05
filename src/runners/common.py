"""Shared helpers for runner entrypoints.

This module centralizes common runner concerns:
- loading JSON and merging search config overrides
- restoring process/model state from a checkpoint
- reward/target spec parsing
- JSONL record writing for scored batches
- lightweight search-batch summary metrics
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TextIO

import torch
from torch import Tensor, nn

from core.registry import build as build_from_registry
from core.utils import load_checkpoint_metadata, maybe_load_checkpoint, resolve_device
from eval.seq_eval import mean_pairwise_hamming, unique_fraction


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    json_path = Path(path).expanduser()
    return json.loads(json_path.read_text())


def merge_namespace_config(args: argparse.Namespace, cfg: Mapping[str, Any]) -> argparse.Namespace:
    """Apply config values onto argparse namespace unless explicitly overridden."""
    defaults = getattr(args, "_defaults", {})
    provided = set(getattr(args, "_provided", ()))
    for key, val in cfg.items():
        if not hasattr(args, key):
            continue
        if key in provided:
            continue
        cur = getattr(args, key)
        if cur is None or (key in defaults and cur == defaults[key]):
            setattr(args, key, val)
    return args


def load_run_config(path: Optional[str], metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Load a training/runtime config from an explicit path or checkpoint metadata."""
    if path is not None:
        return load_json(path)
    if metadata is not None and isinstance(metadata.get("config"), dict):
        return dict(metadata["config"])
    raise ValueError("No config provided and checkpoint metadata did not contain a config.")


def build_model(name: Optional[str], cfg: Mapping[str, Any]) -> Optional[nn.Module]:
    """Build model from registry."""
    if name is None:
        return None
    return build_from_registry("model", name, **dict(cfg))


def build_process(name: str, model: Optional[nn.Module], cfg: Mapping[str, Any]) -> Any:
    """Build process from registry."""
    process_cfg = dict(cfg)
    if model is None:
        return build_from_registry("process", name, model=None, **process_cfg)
    return build_from_registry("process", name, model=model, **process_cfg)


def set_process_data_shape(process: Any, data_shape: Optional[Any]) -> None:
    """Apply recorded data shape to a restored process when available."""
    if data_shape is None or not isinstance(data_shape, (list, tuple)):
        return
    shape = tuple(int(dim) for dim in data_shape)
    setter = getattr(process, "set_data_shape", None)
    if callable(setter):
        setter(shape)
        return
    if hasattr(process, "_data_shape"):
        process._data_shape = shape


def _apply_sequence_length_override(
    *,
    sequence_length: Optional[int],
    process_name: str,
    process_cfg: dict[str, Any],
    model_name: Optional[str],
    model_cfg: dict[str, Any],
) -> bool:
    """Apply a safe length override for a position-free convolutional CTMC."""

    if sequence_length is None:
        return False
    target = int(sequence_length)
    if target < 1:
        raise ValueError("sequence_length_override must be positive")
    current_process = int(process_cfg.get("seq_len", target))
    current_model = int(model_cfg.get("seq_len", target))
    if target == current_process == current_model:
        return False
    if process_name != "flow_mask_ctmc" or model_name != "conv1d_denoiser":
        raise ValueError("sequence length can be overridden only for a flow_mask_ctmc position-free conv1d_denoiser")
    position_embedding = str(model_cfg.get("pos_emb", "none") or "none").strip().lower()
    if position_embedding not in {"none", "off", "false"}:
        raise ValueError("sequence length cannot be overridden for a model with positional embeddings")
    if current_process != current_model:
        raise ValueError(f"runtime process/model sequence lengths disagree: {current_process} versus {current_model}")
    process_cfg["seq_len"] = target
    model_cfg["seq_len"] = target
    return True


@dataclass
class RestoredProcess:
    checkpoint_path: Path
    metadata: Optional[dict[str, Any]]
    config: dict[str, Any]
    process_cfg: dict[str, Any]
    model_cfg: dict[str, Any]
    model_name: Optional[str]
    process_name: str
    device: torch.device
    model: Optional[nn.Module]
    process: Any
    model_for_state: Optional[nn.Module]


def restore_process_from_checkpoint(
    checkpoint: str | Path,
    *,
    config_path: Optional[str] = None,
    device_override: Optional[str | torch.device] = None,
    sequence_length_override: Optional[int] = None,
) -> RestoredProcess:
    """Rebuild process/model from checkpoint metadata and restore model weights."""
    ckpt_path = Path(checkpoint).expanduser()
    metadata = load_checkpoint_metadata(ckpt_path)
    cfg = dict(load_run_config(config_path, metadata))

    process_name = str(cfg["process"])
    process_cfg = dict(cfg.get("process_cfg", {}))
    model_name = cfg.get("model")
    model_cfg = dict(cfg.get("model_cfg", {}))
    length_was_overridden = _apply_sequence_length_override(
        sequence_length=sequence_length_override,
        process_name=process_name,
        process_cfg=process_cfg,
        model_name=model_name,
        model_cfg=model_cfg,
    )
    cfg["process_cfg"] = process_cfg
    cfg["model_cfg"] = model_cfg

    device = resolve_device(device_override or cfg.get("device"))
    process_cfg["device"] = device

    model = build_model(model_name, model_cfg)
    process = build_process(process_name, model, process_cfg)
    if metadata is not None and not length_was_overridden:
        set_process_data_shape(process, metadata.get("data_shape"))

    model_for_state = process.model if hasattr(process, "model") else model
    skip_checkpoint_load = bool(getattr(model_for_state, "_skip_genturbo_checkpoint_load", False))
    if model_for_state is not None and not skip_checkpoint_load:
        maybe_load_checkpoint(str(ckpt_path), model_for_state)
        model_for_state.eval()
    elif model_for_state is not None:
        model_for_state.eval()

    return RestoredProcess(
        checkpoint_path=ckpt_path,
        metadata=metadata,
        config=cfg,
        process_cfg=process_cfg,
        model_cfg=model_cfg,
        model_name=model_name,
        process_name=process_name,
        device=device,
        model=model,
        process=process,
        model_for_state=model_for_state,
    )


def parse_reward_spec(
    reward_config_path: Optional[str],
    reward_spec: Optional[Any],
) -> Optional[dict[str, Any]]:
    """Parse reward config from a file path or JSON/dict CLI input."""
    if reward_config_path:
        loaded = load_json(reward_config_path)
        if not isinstance(loaded, dict):
            raise TypeError("reward config file must contain a JSON object")
        return loaded
    if reward_spec is None:
        return None
    if isinstance(reward_spec, str):
        parsed = json.loads(reward_spec)
    elif isinstance(reward_spec, Mapping):
        parsed = dict(reward_spec)
    else:
        raise TypeError("reward must be a JSON string or object")
    if not isinstance(parsed, dict):
        raise TypeError("reward must decode to a JSON object")
    return parsed


def load_target_tensor(
    path: Optional[str],
    *,
    use_max: bool,
    seq_len: int,
    vocab_size: int,
    prior_sampler: Optional[Callable[[], Tensor]] = None,
) -> Tensor:
    """Load target tensor from disk or derive fallback target."""
    if path:
        target = torch.load(Path(path).expanduser(), map_location="cpu")
        if isinstance(target, dict) and "target" in target:
            target = target["target"]
        if not isinstance(target, Tensor):
            raise ValueError("Target file must contain a tensor or dict with key 'target'")
        return target.long()
    if use_max:
        return torch.full((seq_len,), vocab_size - 1, dtype=torch.long)
    if prior_sampler is not None:
        return prior_sampler().long().cpu()
    raise ValueError("No reward provided and no target sequence specified.")


def write_scored_jsonl(
    fp: TextIO,
    sequences: Tensor,
    scores: Tensor,
    *,
    batch_id: int,
    start_idx: int = 0,
    pad_token_id: Optional[int] = None,
) -> None:
    """Write sequence/score records to JSONL."""
    seq_cpu = sequences.detach().cpu()
    score_cpu = scores.detach().cpu()
    batch_size = seq_cpu.shape[0] if seq_cpu.ndim > 1 else 1
    for idx in range(batch_size):
        seq_tensor = seq_cpu[idx] if seq_cpu.ndim > 1 else seq_cpu
        seq = seq_tensor.tolist()
        score_val = float(score_cpu[idx].item()) if score_cpu.ndim > 0 else float(score_cpu.item())
        is_finite = math.isfinite(score_val)
        if pad_token_id is None:
            seq_length = int(seq_tensor.numel())
        else:
            seq_length = int((seq_tensor != int(pad_token_id)).sum().item())
        record = {
            "batch_id": int(batch_id),
            "seq_idx": int(start_idx + idx),
            "seq": seq,
            "seq_length": seq_length,
            "score": score_val if is_finite else None,
            "score_is_finite": is_finite,
        }
        fp.write(json.dumps(record) + "\n")


def summarise_search_batch(sequences: Tensor, scores: Tensor) -> dict[str, Optional[float]]:
    """Compute common diversity/validity stats for search batches."""
    batch_cpu = sequences.detach().cpu()
    scores_cpu = scores.detach().cpu()
    batch_size = batch_cpu.shape[0] if batch_cpu.ndim > 1 else (1 if batch_cpu.numel() > 0 else 0)

    if batch_cpu.ndim > 1 and batch_size > 0:
        unique_count = float(torch.unique(batch_cpu, dim=0).shape[0])
    else:
        unique_count = float(1 if batch_size > 0 else 0)

    summary: dict[str, Optional[float]] = {
        "unique_count": unique_count,
        "unique_frac": unique_fraction(batch_cpu),
        "pairwise_hamming": mean_pairwise_hamming(batch_cpu),
        "valid_fraction": 0.0,
        "top_score": None,
        "avg_score": None,
        "unique_frac_valid": None,
        "pairwise_hamming_valid": None,
    }

    if scores_cpu.numel() == 0:
        return summary

    finite_mask = torch.isfinite(scores_cpu)
    summary["valid_fraction"] = float(finite_mask.float().mean().item())
    if not finite_mask.any():
        return summary

    valid_scores = scores_cpu[finite_mask]
    valid_batch = batch_cpu[finite_mask] if batch_cpu.ndim > 1 else batch_cpu.view(1, -1)
    summary["top_score"] = float(valid_scores.max().item())
    summary["avg_score"] = float(valid_scores.mean().item())
    summary["unique_frac_valid"] = unique_fraction(valid_batch)
    summary["pairwise_hamming_valid"] = mean_pairwise_hamming(valid_batch)
    return summary


__all__ = [
    "RestoredProcess",
    "build_model",
    "build_process",
    "load_json",
    "load_run_config",
    "load_target_tensor",
    "merge_namespace_config",
    "parse_reward_spec",
    "restore_process_from_checkpoint",
    "set_process_data_shape",
    "summarise_search_batch",
    "write_scored_jsonl",
]
