"""Held-out reconstruction and solver diagnostics of unfiltered samples."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.utils import seed_everything
from data.boxoban import BoxobanConfig, BoxobanIterable
from runners.common import restore_process_from_checkpoint
from sokoban.assess import add_solver_arguments, solver_config
from sokoban.boards import MASK_ID, SEQ_LEN, decode_board, structural_errors
from sokoban.metrics import assess_boards


@torch.no_grad()
def reconstruction_report(model, batches, *, mask_counts=(1, 4, 16, 32, 64, 100), seed: int = 0) -> dict:
    """Exact-k masking at t=1-k/100 (GenTuRBO time runs noise -> clean).

    Report non-wall accuracy separately so predicting the majority wall tile
    cannot masquerade as learning entities. The model's train/eval mode is
    restored and mask draws use a private CPU generator.
    """
    if not mask_counts or any(not 1 <= k <= SEQ_LEN for k in mask_counts):
        raise ValueError("mask counts must be in [1, 100]")
    device = next(model.parameters()).device
    rng = torch.Generator().manual_seed(seed)
    totals = {k: {"loss": 0.0, "correct": 0, "tokens": 0, "nonwall_correct": 0, "nonwall_tokens": 0}
              for k in mask_counts}
    was_training = model.training
    model.eval()
    levels = 0
    try:
        for batch in batches:
            clean = batch["x"].to(device)
            levels += len(clean)
            for k in mask_counts:
                positions = torch.rand(len(clean), SEQ_LEN, generator=rng).argsort(dim=1)[:, :k]
                mask = torch.zeros(len(clean), SEQ_LEN, dtype=torch.bool).scatter_(1, positions, True).to(device)
                noisy = clean.masked_fill(mask, MASK_ID)
                time = torch.full((len(clean),), 1.0 - k / SEQ_LEN, device=device)
                logits = model(noisy, timesteps=time)
                predictions = logits.argmax(-1)
                nonwall = mask & (clean != 0)
                values = totals[k]
                values["loss"] += float(F.cross_entropy(logits[mask], clean[mask], reduction="sum"))
                values["correct"] += int(((predictions == clean) & mask).sum())
                values["tokens"] += int(mask.sum())
                values["nonwall_correct"] += int(((predictions == clean) & nonwall).sum())
                values["nonwall_tokens"] += int(nonwall.sum())
    finally:
        model.train(was_training)
    if not levels:
        raise ValueError("no validation boards supplied")
    return {
        "levels": levels,
        "seed": seed,
        "by_mask_count": {
            str(k): {
                "masked_cross_entropy": v["loss"] / v["tokens"],
                "masked_accuracy": v["correct"] / v["tokens"],
                "masked_tokens": v["tokens"],
                "nonwall_accuracy": v["nonwall_correct"] / v["nonwall_tokens"] if v["nonwall_tokens"] else None,
                "nonwall_tokens": v["nonwall_tokens"],
            }
            for k, v in totals.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", default="runs/data/boxoban")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-solver", action="store_true")
    add_solver_arguments(parser)
    args = parser.parse_args()
    if min(args.batch_size, args.validation_batches, args.steps) <= 0 or args.num_samples < 0:
        parser.error("batch sizes and steps must be positive; num-samples must be nonnegative")
    config = solver_config(args)
    if not np.isfinite(args.repair_max_seconds) or args.repair_max_seconds <= 0:
        parser.error("repair-max-seconds must be positive and finite")
    seed_everything(args.seed)
    restored = restore_process_from_checkpoint(args.checkpoint, device_override=args.device)
    if restored.config.get("dataset") != "boxoban" or restored.process.interpolant != "mask":
        parser.error("expected a masked Boxoban checkpoint")
    model = restored.model_for_state
    model.eval().requires_grad_(False)
    validation = BoxobanIterable(BoxobanConfig(
        root=args.data_root, split="valid", batch_size=args.batch_size,
        seed=args.seed, shuffle=True, augment=False,
    ))
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "reconstruction": reconstruction_report(
            model, islice(validation, args.validation_batches), seed=args.seed,
        ),
    }
    boards = []
    for start in range(0, args.num_samples, args.batch_size):
        samples = restored.process.sample(
            min(args.batch_size, args.num_samples - start), steps=args.steps, show_progress=False,
        ).cpu().numpy()
        boards.extend(samples)
    errors = [structural_errors(board) for board in boards]
    report["generation"] = {
        "num_samples": len(boards),
        "steps": args.steps,
        "seed": args.seed,
        "structurally_valid": sum(not error for error in errors),
        "unique_raw_boards": len({board.tobytes() for board in boards}),
        "solvability": "not evaluated",
        "reward": "not evaluated",
        "filtering": "none; raw samples retained including structural failures",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "boards.npy", np.asarray(boards, dtype=np.uint8).reshape(-1, SEQ_LEN))
    (args.output / "boards.txt").write_text("\n\n".join(
        f"; {i} structural={'ok' if not error else '; '.join(error)}\n{decode_board(board)}"
        for i, (board, error) in enumerate(zip(boards, errors, strict=True))
    ) + "\n")
    if not args.skip_solver:
        metrics, records = assess_boards(boards, config, repair=args.one_wall_repair,
                                         repair_max_seconds=args.repair_max_seconds)
        report["generation"]["solvability"] = metrics
        (args.output / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
