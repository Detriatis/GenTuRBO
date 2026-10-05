"""Sokoban oracle and conditional rollout interfaces for frozen-model search."""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

from processes.sokoban_masked import sample_one_cell
from sokoban.boards import structural_errors
from sokoban.planning import PLANNING_VERSION, PlanningConfig, minimum_switches
from sokoban.solver import SolverConfig, _cells, _geometry, _reachable, solve

CANONICAL_VERSION = "d4-worker-region-v1"
DISPLAY_ALPHABET = "ABCDEFGH"  # Seven tiles plus mask, without whitespace.


def canonical_board(tokens) -> np.ndarray:
    """Collapse square symmetries and free worker movement, without moving boxes.

    Invalid boards retain their literal representation. Valid boards are solved
    in this fixed orientation so rotations cannot change the objective merely
    through deterministic BFS tie order. Certificates refer to this board.
    """
    board = np.asarray(tokens)
    if board.size != 100 or not np.issubdtype(board.dtype, np.integer):
        raise ValueError("expected 100 integer tokens")
    board = board.reshape(100).astype(np.int64)
    if structural_errors(board):
        return board.copy()
    forms = []
    for reflected in (board.reshape(10, 10), board.reshape(10, 10)[:, ::-1]):
        for k in range(4):
            candidate = np.rot90(reflected, k).flatten().copy()
            floor, boxes, _, player = _geometry(candidate)
            canonical_player = next(_cells(_reachable(player, floor & ~boxes)))
            candidate[player] = 4 if candidate[player] == 6 else 1
            candidate[canonical_player] = 6 if candidate[canonical_player] == 4 else 2
            forms.append(candidate)
    return min(forms, key=lambda x: tuple(x.tolist()))


def board_key(tokens) -> str:
    return hashlib.sha256(canonical_board(tokens).astype("<i8").tobytes()).hexdigest()


def _solve_task(task):
    board, config = task
    result = solve(board, config).to_dict()
    result["score"] = math.log1p(result["expanded_states"]) if result["status"] == "solved" else -1.0
    result["canonical_board"] = board.tolist()
    return result


def _switch_task(task):
    board, config = task
    planning = minimum_switches(board, config=config)
    known_solvable = planning.solution is not None
    failure = planning.status in {"structurally_invalid", "proven_unsolvable"}
    return {
        "status": "solved" if known_solvable else planning.status,
        "objective_status": planning.status,
        "objective": "box_switches",
        "reason": planning.reason,
        "score": planning.value if planning.status == "exact" else (-1.0 if failure else None),
        "box_switches": planning.value,
        "planning": planning.to_dict(),
        "solution": planning.solution,
        "solution_verified": known_solvable,
        "solution_pushes": planning.solution_pushes,
        # The switch-optimal witness need not minimise pushes. Its search work
        # is not the previous BFS-effort metric either.
        "minimum_pushes": None,
        "expanded_states": None,
        "difficulty_expanded_states": None,
        "canonical_board": board.tolist(),
    }


class SokobanReward:
    """Finite objective with separate feasibility/censoring and paid-query logs.

    Cached and invalid queries still count against the experiment's evaluation
    budget. Only confirmed solutions earn log(1+E); unknown means censored,
    not proven infeasible. Decode/features never call the solver.
    """

    alphabet = decode_alphabet = DISPLAY_ALPHABET
    max_len = sequence_length = 100
    task_function = staticmethod(_solve_task)

    def __init__(self, *, max_states=200_000, max_seconds=30.0,
                 max_discovered_states=1_000_000, workers=1):
        self.config = SolverConfig(max_states, max_seconds, max_discovered_states)
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.pool = None
        self.cache = {}
        self.records = []
        self.physical_solves = 0

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def decode_batch(self, batch):
        rows = batch.detach().cpu().numpy().reshape(-1, 100)
        return ["".join(DISPLAY_ALPHABET[int(v)] for v in canonical_board(row))
                if np.all((row >= 0) & (row < 8)) else None for row in rows]

    def score_batch(self, batch):
        if batch.ndim not in (1, 2) or batch.shape[-1] != 100 or batch.dtype != torch.long:
            raise ValueError("expected long [100] or [B,100]")
        rows = batch.detach().cpu().numpy().reshape(-1, 100)
        canonical = [canonical_board(row) for row in rows]
        keys = [hashlib.sha256(row.astype("<i8").tobytes()).hexdigest() for row in canonical]
        missing = {key: row for key, row in zip(keys, canonical, strict=False) if key not in self.cache}
        if self.workers > 1 and self.pool is None:
            self.pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"))
        tasks = [(board, self.config) for board in missing.values()]
        results = map(self.task_function, tasks) if self.pool is None else self.pool.map(self.task_function, tasks)
        previously_seen = set(self.cache)
        self.cache.update(zip(missing, results, strict=False))
        self.physical_solves += len(missing)
        values = []
        for row, key in zip(rows, keys, strict=False):
            result = dict(self.cache[key])
            result.update(query=len(self.records) + 1, key=key, cache_hit=key in previously_seen,
                          board=row.tolist(), canonical_version=CANONICAL_VERSION)
            self.records.append(result)
            values.append(float("nan") if result["score"] is None else result["score"])
            previously_seen.add(key)
        scores = torch.tensor(values, dtype=torch.float, device=batch.device)
        return scores[0] if batch.ndim == 1 else scores

    def __call__(self, board):
        return float(self.score_batch(board))

    def metadata(self):
        return {"objective": "log1p(expanded_states) for solved; -1 otherwise",
                "canonical_version": CANONICAL_VERSION, "solver": asdict(self.config),
                "query_count": len(self.records), "physical_solves": self.physical_solves}


class SokobanSwitchReward(SokobanReward):
    """Maximise proven minimum box switches; omit censored values from GP data.

    JSON records use null for an unresolved score. The tensor API uses NaN,
    which the shared active-learning selector explicitly skips. Proven failures
    receive -1. Solvability and objective exactness are distinct properties.
    """

    task_function = staticmethod(_switch_task)

    def __init__(self, *, max_states=200_000, max_seconds=10.0,
                 max_discovered_states=400_000, workers=1):
        super().__init__(max_states=max_states, max_seconds=max_seconds,
                         max_discovered_states=max_discovered_states, workers=workers)
        self.config = PlanningConfig(max_states, max_discovered_states, max_seconds)

    def metadata(self):
        return {"objective": "box_switches", "reward": "exact minimum box switches; proven failures -1",
                "unresolved": "charged, cached, excluded from GP labels and exact incumbent archive",
                "canonical_version": CANONICAL_VERSION, "planning_version": PLANNING_VERSION,
                "solver": asdict(self.config), "query_count": len(self.records),
                "physical_solves": self.physical_solves}


class SokobanRolloutAdapter:
    """The shared RolloutAdapter protocol using exactly one reveal per call."""

    T = 100

    def __init__(self, process, *, temperature=1.0, precision="float32", seed=None):
        self.process = process
        self.device = torch.device(process.device)
        self.temperature = temperature
        if precision not in {"float32", "float16"}:
            raise ValueError("precision must be float32 or float16")
        if precision == "float16" and self.device.type != "cuda":
            raise ValueError("float16 sampling requires CUDA")
        self.precision = precision
        process.model.eval().requires_grad_(False)
        self.generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)

    def sample_prior(self):
        return torch.full((100,), 7, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def complete(self, masked, *, on_step=None):
        context = torch.autocast("cuda", dtype=torch.float16) if self.precision == "float16" else nullcontext()
        with context:
            return sample_one_cell(self.process.model, masked.to(self.device), temperature=self.temperature,
                                   generator=self.generator, on_step=on_step)

    def rollout_from_batch(self, xt_batch, t_value, *, update_mask=None, denoise_steps=None, t_is_time=False):
        if not 0 <= t_value <= (1.0 if t_is_time else self.T):
            raise ValueError("time outside rollout horizon")
        if denoise_steps not in (None, 100):
            raise ValueError("Sokoban must complete all masks with one-cell sampling; omit denoise_steps")
        current = xt_batch.to(self.device)
        if update_mask is not None:
            editable = torch.as_tensor(update_mask, dtype=torch.bool, device=self.device).expand_as(current)
            if torch.any((current == 7) & ~editable):
                raise ValueError("cannot complete a frozen mask token")
        states = [current.clone()]
        self.complete(current, on_step=lambda board, _: states.append(board))
        return states

    def rollout(self, x0):
        return self.rollout_from(x0, 0)

    def rollout_from(self, xt, t):
        return [state[0] for state in self.rollout_from_batch(xt.unsqueeze(0), t)]

    def propose(self, parents, mask_count):
        if not 1 <= mask_count <= 100:
            raise ValueError("mask_count must be in [1,100]")
        parents = parents.to(self.device)
        positions = torch.rand(parents.shape, device=self.device, generator=self.generator).argsort(1)
        masked = parents.clone().scatter(1, positions[:, :mask_count], 7)
        return self.complete(masked)
