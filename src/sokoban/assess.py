"""Evaluate saved raw Sokoban boards without loading a model or using a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sokoban.boards import MASK_ID, TILE_TO_ID
from sokoban.metrics import assess_boards, compare_difficulty
from sokoban.prepare import sha256
from sokoban.solver import SolverConfig


def load_boards(path: Path) -> tuple[list[str], np.ndarray]:
    """Preserve wrong entity counts and masks; reject broken file framing."""
    if path.suffix == ".npy":
        boards = np.load(path, mmap_mode="r", allow_pickle=False)
        if boards.ndim != 2 or boards.shape[1] != 100 or not np.issubdtype(boards.dtype, np.integer):
            raise ValueError("expected an integer [N,100] array")
        return [str(i) for i in range(len(boards))], boards
    alphabet = {**TILE_TO_ID, "?": MASK_ID}
    records, ids, rows = [], [], []
    board_id = None

    def finish():
        if len(rows) != 10 or any(len(row) != 10 for row in rows):
            raise ValueError(f"board {board_id}: expected ten rows of ten characters")
        try:
            records.append([alphabet[tile] for row in rows for tile in row])
        except KeyError as exc:
            raise ValueError(f"board {board_id}: unknown tile {exc.args[0]!r}") from exc
        ids.append(board_id)

    for line in path.read_text().splitlines():
        if line.startswith(";"):
            if board_id is not None:
                finish()
            header = line[1:].strip().split()
            if not header:
                raise ValueError("missing board id")
            board_id, rows = header[0], []
        elif line != "":
            if board_id is None:
                raise ValueError("board before first '; id' header")
            rows.append(line)
    if board_id is not None:
        finish()
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate board ids")
    return ids, np.asarray(records, dtype=np.uint8).reshape(-1, 100)


def add_solver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-states", type=int, default=200_000)
    parser.add_argument("--max-seconds", type=float, default=30.0,
                        help="per-board time cap; 0 disables it for state-budget reproducibility")
    parser.add_argument("--max-discovered-states", type=int, default=1_000_000)
    parser.add_argument("--one-wall-repair", action="store_true", help="separate optional repair diagnostic")
    parser.add_argument("--repair-max-seconds", type=float, default=60.0)


def solver_config(args) -> SolverConfig:
    return SolverConfig(args.max_states, args.max_seconds if args.max_seconds != 0 else None,
                        args.max_discovered_states)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=Path, required=True, help="raw boards.txt or integer [N,100] .npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="reference .npy or boards.txt for difficulty calibration")
    parser.add_argument("--reference-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    add_solver_arguments(parser)
    args = parser.parse_args()
    if args.reference_size < 1:
        parser.error("reference-size must be positive")
    if not np.isfinite(args.repair_max_seconds) or args.repair_max_seconds <= 0:
        parser.error("repair-max-seconds must be positive and finite")
    config = solver_config(args)
    ids, boards = load_boards(args.boards)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "records.jsonl").open("w") as output:
        count = 0

        def progress(record):
            nonlocal count
            count += 1
            output.write(json.dumps(record) + "\n")
            output.flush()
            if count % 32 == 0 or count == len(boards):
                print(f"Assessed {count}/{len(boards)} boards", flush=True)

        summary, records = assess_boards(boards, config, ids=ids, repair=args.one_wall_repair,
                                         repair_max_seconds=args.repair_max_seconds, on_record=progress)
    summary["source"] = {"path": str(args.boards.resolve()), "sha256": sha256(args.boards)}
    if args.reference:
        ref_ids, reference = load_boards(args.reference)
        size = min(args.reference_size, len(reference))
        indices = np.random.default_rng(args.seed).choice(len(reference), size=size, replace=False)
        reference_summary, reference_records = assess_boards(reference[indices], config,
                                                              ids=[ref_ids[i] for i in indices])
        reference_summary["source"] = {"path": str(args.reference.resolve()), "sha256": sha256(args.reference),
                                       "indices": indices.tolist(), "seed": args.seed}
        summary["reference"] = reference_summary
        summary["difficulty_comparison"] = compare_difficulty(records, reference_records)
        (args.output / "reference_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reference_records))
        (args.output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
