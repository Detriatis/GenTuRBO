"""Offline evaluation for generated sequences."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from eval.seq_eval import evaluate_sequences
from toy_reward import resolve_reward


@dataclass
class EvaluateConfig:
    input_path: str
    reward: Dict[str, Any]
    output_path: Optional[str] = None
    save_rewards: bool = True


def _load_sequences(path: Path) -> tuple[Tensor, Dict[str, Any]]:
    if path.suffix in {".npz", ".npy"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "samples" in data:
                arr = data["samples"]
            else:
                first_key = next(iter(data.files))
                arr = data[first_key]
        else:
            arr = data
        seqs = torch.from_numpy(arr)
        payload: Dict[str, Any] = {"samples": seqs}
        return seqs, payload

    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "samples" in blob:
        seqs = blob["samples"]
        payload = dict(blob)
        return seqs, payload
    if torch.is_tensor(blob):
        seqs = blob
        payload = {"samples": seqs}
        return seqs, payload
    raise ValueError(f"Unsupported samples format at {path}")


def evaluate(config: EvaluateConfig) -> dict[str, Any]:
    path = Path(config.input_path).expanduser()
    seqs, payload = _load_sequences(path)
    reward_fn = resolve_reward(config.reward)
    rewards, metrics = evaluate_sequences(seqs, reward_fn)

    out_path = Path(config.output_path).expanduser() if config.output_path else path
    if config.output_path is None and path.suffix in {".npz", ".npy"}:
        out_path = path.with_suffix(path.suffix + ".pt")
    metrics_path = out_path.with_suffix(out_path.suffix + ".metrics.json")

    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote reward metrics to {metrics_path}")

    if config.save_rewards:
        payload.update({"rewards": rewards, "reward_metrics": metrics})
        torch.save(payload, out_path)
        print(f"Saved rewards alongside samples at {out_path}")
    return {"metrics": metrics, "rewards_path": str(out_path), "metrics_path": str(metrics_path)}


def _parse_args() -> EvaluateConfig:
    parser = argparse.ArgumentParser(description="Score generated sequences with a reward function.")
    parser.add_argument("--input", type=str, required=True, help="Path to tensor/npz containing samples.")
    parser.add_argument("--reward", type=str, required=True, help='JSON string for reward spec {"name": ...}.')
    parser.add_argument("--reward-config", type=str, default=None, help="Path to JSON reward spec.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for saved rewards.")
    parser.add_argument("--no-save-rewards", action="store_true", help="Do not write rewards back to disk.")
    args = parser.parse_args()

    if args.reward_config:
        reward_spec = json.loads(Path(args.reward_config).read_text())
    else:
        reward_spec = json.loads(args.reward)

    return EvaluateConfig(
        input_path=args.input,
        reward=reward_spec,
        output_path=args.output,
        save_rewards=not args.no_save_rewards,
    )


def main() -> None:
    cfg = _parse_args()
    evaluate(cfg)


if __name__ == "__main__":
    main()
