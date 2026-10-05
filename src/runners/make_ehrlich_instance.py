"""Utility to build and save Ehrlich instances from a training config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from ehrlich.ehrlich_instance import (
    EhrlichConfig,
    construct_optimal_sequence,
    make_ehrlich_instance,
    make_ehrlich_instance_from_transition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Ehrlich instance + target from a train config.")
    parser.add_argument("--config", type=str, required=True, help="Training config JSON to read.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (defaults to run_root/run_name).")
    parser.add_argument(
        "--base-instance",
        type=str,
        default=None,
        help="Path to an existing EhrlichInstance to reuse transition probabilities/mask.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override instance seed (motif/spacing sampling).")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length.")
    parser.add_argument("--motif-count", type=int, default=None, help="Override motif count.")
    parser.add_argument("--motif-len", type=int, default=None, help="Override motif length.")
    parser.add_argument("--quantization", type=int, default=None, help="Override quantization.")
    parser.add_argument("--motif-from-sequence", action="store_true", help="Override to sample motifs from a base sequence.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def _resolve_out_dir(cfg: dict, override: Optional[str]) -> Path:
    if override is not None:
        return Path(override).expanduser()
    run_root = cfg.get("run_root", "runs")
    run_name = cfg.get("run_name", "ehrlich_instance")
    return Path(run_root).expanduser() / str(run_name)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser()
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("dataset") != "ehrlich_markov":
        raise ValueError("Config does not use dataset='ehrlich_markov'.")
    dataset_cfg = cfg.get("dataset_cfg") or {}
    inst_cfg_raw = dataset_cfg.get("instance_cfg")
    if not isinstance(inst_cfg_raw, dict):
        raise ValueError("dataset_cfg.instance_cfg must be a dict.")

    inst_cfg = dict(inst_cfg_raw)
    if args.motif_from_sequence:
        inst_cfg["motif_from_sequence"] = True
    if args.seq_len is not None:
        inst_cfg["seq_len"] = int(args.seq_len)
    if args.motif_count is not None:
        inst_cfg["motif_count"] = int(args.motif_count)
    if args.motif_len is not None:
        inst_cfg["motif_len"] = int(args.motif_len)
    if args.quantization is not None:
        inst_cfg["quantization"] = int(args.quantization)

    seed = int(args.seed) if args.seed is not None else int(dataset_cfg.get("instance_seed", 0))
    if args.base_instance:
        base = torch.load(Path(args.base_instance).expanduser(), map_location="cpu", weights_only=False)
        from ehrlich.ehrlich_instance import EhrlichInstance

        if not isinstance(base, EhrlichInstance):
            raise TypeError("base-instance did not contain an EhrlichInstance")
        base_v = int(base.transition_probs.shape[0])
        if "vocab_size" in inst_cfg:
            if int(inst_cfg["vocab_size"]) != base_v:
                raise ValueError("vocab_size must match base instance transition_probs size")
        else:
            inst_cfg["vocab_size"] = base_v
        inst = make_ehrlich_instance_from_transition(
            EhrlichConfig(**inst_cfg),
            seed=seed,
            transition_probs=base.transition_probs,
            transition_mask=base.transition_mask,
        )
    else:
        inst = make_ehrlich_instance(EhrlichConfig(**inst_cfg), seed=seed)
    out_dir = _resolve_out_dir(cfg, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst_path = out_dir / "dataset_instance.pt"
    target_path = out_dir / "target_seq.pt"
    if not args.force and (inst_path.exists() or target_path.exists()):
        raise FileExistsError("Output files already exist; use --force to overwrite.")

    torch.save(inst, inst_path)
    target = inst.base_sequence if inst.base_sequence is not None else construct_optimal_sequence(inst)
    torch.save(target, target_path)
    print(f"Wrote {inst_path} and {target_path}")


if __name__ == "__main__":
    main()
