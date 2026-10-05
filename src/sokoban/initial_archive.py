"""Validate and reuse shared Sokoban initial puzzles across execution devices."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from sokoban.planning import solution_switches
from sokoban.prepare import sha256
from sokoban.search import CANONICAL_VERSION, board_key, canonical_board


def load_initial_archives(cfg, source, checkpoint_sha256):
    """Validate every requested seed before creating any run output."""
    if cfg.objective != "box_switches":
        raise ValueError("saved initial archives require objective=box_switches")
    source = Path(source)
    original = json.loads((source / "config.json").read_text())
    provenance = json.loads((source / "provenance.json").read_text())
    if len(checkpoint_sha256) != 64 or provenance["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("initial archive checkpoint does not match the expected checkpoint")
    for field in ("initial_size", "objective", "solver_max_states", "solver_max_discovered_states", "solver_max_seconds"):
        if original[field] != getattr(cfg, field):
            raise ValueError(f"initial archive protocol mismatch: {field}")
    archives = {}
    for seed in cfg.seeds:
        if seed not in original["seeds"]:
            raise ValueError(f"seed {seed} is not recorded in the initial source configuration")
        folder = source / f"seed{seed}"
        paths = [folder / name for name in ("initial.npy", "initial_records.json", "initial_report.json")]
        if not all(path.is_file() for path in paths):
            raise ValueError(f"missing saved initial archive for seed {seed}; generate shared starts first")
        initial = np.load(paths[0], allow_pickle=False)
        records = json.loads(paths[1].read_text())
        if initial.shape != (cfg.initial_size, 100) or not np.issubdtype(initial.dtype, np.integer):
            raise ValueError(f"invalid initial board array for seed {seed}")
        if len(records) != cfg.initial_size:
            raise ValueError(f"invalid initial label count for seed {seed}")
        for board, record in zip(initial, records, strict=True):
            if (record["board"] != board.tolist() or record["key"] != board_key(board)
                    or record["canonical_board"] != canonical_board(board).tolist()
                    or record.get("canonical_version") != CANONICAL_VERSION
                    or record.get("objective") != cfg.objective):
                raise ValueError(f"initial board/label identity mismatch for seed {seed}")
            if record["status"] == "solved":
                replay = solution_switches(record["canonical_board"], record["solution"])
                if record["objective_status"] == "exact":
                    planning = record["planning"]
                    if not (record["score"] == planning["lower_bound"] == planning["upper_bound"] == replay["switches"]):
                        raise ValueError(f"inconsistent exact initial score for seed {seed}")
        archives[seed] = (torch.as_tensor(initial, dtype=torch.long), records,
                          {name.name: sha256(name) for name in paths})
    return archives



def load_initial_sources(cfg, sources, checkpoint_sha256):
    """Resolve each requested seed exactly once; validate all before a run starts."""
    sources = [Path(source).resolve() for source in sources]
    owners = {}
    for seed in cfg.seeds:
        matches = [source for source in sources if (source / f"seed{seed}").is_dir()]
        if len(matches) != 1:
            raise ValueError(f"seed {seed} must have exactly one initial source; found {len(matches)}")
        owners[seed] = matches[0]
    archives = {}
    for source in sources:
        seeds = [seed for seed in cfg.seeds if owners[seed] == source]
        if seeds:
            loaded = load_initial_archives(replace(cfg, seeds=seeds), source, checkpoint_sha256)
            archives.update({seed: (*values, source) for seed, values in loaded.items()})
    return archives
