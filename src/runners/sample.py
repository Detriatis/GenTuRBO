"""Sampling script for trained GenTuRBO processes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, cast

import torch
from core.utils import (
    load_checkpoint_metadata,
    maybe_load_checkpoint,
    resolve_device,
)
from eval.seq_eval import score_batch, summarise_rewards
from toy_reward import resolve_reward
from runners.common import (
    build_model,
    build_process,
    parse_reward_spec,
    set_process_data_shape,
    write_scored_jsonl,
)
from search.interpolant.output import compressed_jsonl_path, open_jsonl


@dataclass
class SampleConfig:
    process: str = "flow_matching"
    process_cfg: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = "mlp"
    model_cfg: Dict[str, Any] = field(default_factory=lambda: {"in_dim": 2, "out_dim": 2})
    checkpoint_path: Optional[str] = None
    num_samples: int = 1024
    steps: int = 100
    batch_size: Optional[int] = None
    seed: Optional[int] = None
    device: Optional[str] = None
    output_path: Optional[str] = None
    jsonl_path: Optional[str] = None
    reward: Optional[Dict[str, Any]] = None
    prior_invalid: Optional[bool] = None
    prior_invalid_mode: Optional[str] = None


def _maybe_apply_checkpoint_config(
    cfg: SampleConfig,
    metadata: dict[str, Any],
    *,
    override_flags: dict[str, bool],
) -> None:
    saved_cfg = metadata.get("config")
    if not isinstance(saved_cfg, dict):
        return
    if not override_flags["process"]:
        cfg.process = saved_cfg.get("process", cfg.process)
    if not override_flags["process_cfg"]:
        cfg.process_cfg = saved_cfg.get("process_cfg", cfg.process_cfg)
    if not override_flags["model"]:
        cfg.model = saved_cfg.get("model", cfg.model)
    if not override_flags["model_cfg"]:
        cfg.model_cfg = saved_cfg.get("model_cfg", cfg.model_cfg)
    run_dir = saved_cfg.get("run_dir")
    if cfg.output_path is None and isinstance(run_dir, str):
        cfg.output_path = str(Path(run_dir).expanduser() / "samples.pt")


@torch.no_grad()
def sample(config: SampleConfig, metadata: Optional[dict[str, Any]] = None) -> torch.Tensor:
    device = resolve_device(config.device)
    model = build_model(config.model, config.model_cfg)
    process = build_process(config.process, model, {**config.process_cfg, "device": device})
    if metadata:
        set_process_data_shape(process, metadata.get("data_shape"))
    model_for_state = process.model if hasattr(process, "model") else model
    pad_token_id = getattr(model_for_state, "pad_token_id", None)
    if config.checkpoint_path and model_for_state is not None:
        maybe_load_checkpoint(config.checkpoint_path, model_for_state)
    reward_fn = resolve_reward(config.reward) if config.reward is not None else None
    samples_chunks: list[torch.Tensor] = []
    rewards_chunks: list[torch.Tensor] = []
    batch_size = int(config.batch_size) if config.batch_size is not None else config.num_samples
    if batch_size <= 0:
        raise ValueError("batch_size must be positive when provided")
    remaining = config.num_samples
    jsonl_fp: Optional[TextIO] = None
    if config.jsonl_path:
        jsonl_path = Path(config.jsonl_path).expanduser()
        if str(jsonl_path) != "/dev/null":
            jsonl_path = compressed_jsonl_path(jsonl_path)
            config.jsonl_path = str(jsonl_path)
            jsonl_fp = cast(TextIO, open_jsonl(jsonl_path, "wt"))

    try:
        batch_id = 0
        seq_idx = 0
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            generated = process.sample(
                current_batch,
                steps=config.steps,
                device=device,
                prior_invalid=config.prior_invalid,
                prior_invalid_mode=config.prior_invalid_mode,
            )
            samples_chunks.append(generated.detach().cpu())
            rewards_tensor: Optional[torch.Tensor] = None
            if reward_fn is not None:
                if generated.ndim < 2:
                    raise ValueError(
                        f"Reward evaluation expects generated shape [B, L], got {tuple(generated.shape)}"
                    )
                rewards_tensor = score_batch(generated, reward_fn)
                rewards_chunks.append(rewards_tensor.detach().cpu())
            if jsonl_fp is not None:
                if rewards_tensor is None:
                    raise ValueError("jsonl_path requires reward scoring; provide a reward spec")
                write_scored_jsonl(
                    jsonl_fp,
                    generated,
                    rewards_tensor,
                    batch_id=batch_id,
                    start_idx=seq_idx,
                    pad_token_id=pad_token_id,
                )
                jsonl_fp.flush()
            remaining -= current_batch
            seq_idx += current_batch
            batch_id += 1
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    generated = torch.cat(samples_chunks, dim=0) if samples_chunks else torch.empty((0,))
    rewards_tensor = torch.cat(rewards_chunks, dim=0) if rewards_chunks else None
    reward_metrics: Optional[Dict[str, float]] = None
    if reward_fn is not None and rewards_tensor is not None:
        reward_metrics = summarise_rewards(
            generated,
            rewards_tensor if rewards_tensor.ndim > 0 else rewards_tensor.view(1),
        )
        print(f"Reward metrics: {reward_metrics}")
    if config.jsonl_path:
        print(f"Wrote JSONL samples to {config.jsonl_path}")
    print(generated[0:10])
    if config.output_path:
        out_path = Path(config.output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {"samples": generated.cpu(), "metadata": metadata or {}}
        if rewards_tensor is not None:
            payload["rewards"] = rewards_tensor.cpu()
            if reward_metrics is not None:
                payload["reward_metrics"] = reward_metrics
        torch.save(payload, out_path)
        print(f"Saved samples to {out_path}")
    return generated


def _load_config(path: Optional[str]) -> SampleConfig:
    if path is None:
        return SampleConfig()
    data = json.loads(Path(path).read_text())
    valid_keys = set(SampleConfig.__dataclass_fields__)
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    unknown = sorted(set(data) - valid_keys)
    if unknown:
        print(f"Ignoring unsupported sampling config keys: {unknown}")
    return SampleConfig(**filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained GenTuRBO process")
    parser.add_argument("--config", type=str, default=None, help="JSON config path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--num-samples", type=int, default=None, help="Override number of samples")
    parser.add_argument("--process", type=str, default=None, help="Override process name")
    parser.add_argument("--process-cfg", type=str, default=None, help="JSON string for process cfg overrides")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--model-cfg", type=str, default=None, help="JSON string for model cfg overrides")
    parser.add_argument("--steps", type=int, default=None, help="Override sampling steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Sample in chunks of this size")
    parser.add_argument("--device", type=str, default=None, help="Device for sampling (e.g., cuda:0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default=None, help="Path to save generated samples")
    parser.add_argument("--jsonl-output", type=str, default=None, help="Optional JSONL output path")
    parser.add_argument("--reward", type=str, default=None, help="JSON string for reward config")
    parser.add_argument("--reward-config", type=str, default=None, help="Path to JSON reward config")
    parser.add_argument(
        "--prior-invalid",
        action="store_true",
        help="Use invalid-transition prior when supported by the process.",
    )
    parser.add_argument(
        "--prior-invalid-mode",
        type=str,
        default=None,
        help="Invalid prior mode (e.g., sequential_both).",
    )
    args = parser.parse_args()
    cfg = _load_config(args.config)
    override_flags = {
        "process": args.process is not None,
        "process_cfg": args.process_cfg is not None,
        "model": args.model is not None,
        "model_cfg": args.model_cfg is not None,
    }
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.process:
        cfg.process = args.process
    if args.process_cfg:
        cfg.process_cfg = json.loads(args.process_cfg)
    if args.model:
        cfg.model = args.model
    if args.model_cfg:
        cfg.model_cfg = json.loads(args.model_cfg)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.reward = parse_reward_spec(args.reward_config, args.reward)
    if args.prior_invalid:
        cfg.prior_invalid = True
    if args.prior_invalid_mode is not None:
        cfg.prior_invalid_mode = args.prior_invalid_mode
    if args.output:
        cfg.output_path = args.output
    if args.jsonl_output:
        cfg.jsonl_path = args.jsonl_output
    metadata: Optional[dict[str, Any]] = None
    if cfg.checkpoint_path:
        metadata = load_checkpoint_metadata(Path(cfg.checkpoint_path).expanduser())
        if metadata:
            _maybe_apply_checkpoint_config(cfg, metadata, override_flags=override_flags)
    if cfg.output_path is None and cfg.checkpoint_path is not None:
        ckpt_path = Path(cfg.checkpoint_path).expanduser()
        cfg.output_path = str(ckpt_path.with_name("samples.pt"))
    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))
    sample(cfg, metadata=metadata)


if __name__ == "__main__":
    main()
