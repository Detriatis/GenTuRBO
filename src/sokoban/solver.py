"""Deterministic push BFS and independently replayed solution certificates.

Method: Baghal, *Solvable Sokoban Without a Solver via Diffusion* (2026),
https://arxiv.org/abs/2608.15958. This is an independent implementation.
Walking is free; states identify box positions and the worker's reachable
region. Only static reverse-goal dead cells are pruned. Effort counts states
popped before the first solution is generated, including its parent state.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from sokoban.boards import HEIGHT, SEQ_LEN, WIDTH, structural_errors

# Stable tie order: row-major boxes, then right, left, down, up. The released
# reference solver iterates a set of boxes, so exact effort values can differ.
DIRECTIONS = (("R", 0, 1), ("L", 0, -1), ("D", 1, 0), ("U", -1, 0))
NEIGHBORS = tuple(tuple(
    (r + dr) * WIDTH + c + dc if 0 <= r + dr < HEIGHT and 0 <= c + dc < WIDTH else -1
    for _, dr, dc in DIRECTIONS
) for r, c in (divmod(i, WIDTH) for i in range(SEQ_LEN)))
OPPOSITE = (1, 0, 3, 2)
SOLVER_VERSION = "push-bfs-v1"


@dataclass(frozen=True)
class SolverConfig:
    max_states: int = 200_000
    max_seconds: float | None = 30.0
    max_discovered_states: int = 1_000_000

    def __post_init__(self):
        if self.max_states < 1 or self.max_discovered_states < 1:
            raise ValueError("state limits must be positive")
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite, or None")


@dataclass(frozen=True)
class SolveResult:
    status: str
    reason: str
    structural_errors: tuple[str, ...]
    expanded_states: int
    discovered_states: int
    elapsed_seconds: float
    minimum_pushes: int | None = None
    solution: str | None = None  # lowercase walk, uppercase push; not minimum walking distance

    @property
    def difficulty(self) -> int | None:
        """Paper's solver effort; undefined for every unsuccessful search."""
        return self.expanded_states if self.status == "solved" else None

    def to_dict(self) -> dict:
        return {**asdict(self), "difficulty_expanded_states": self.difficulty,
                "solution_moves": len(self.solution) if self.solution is not None else None,
                "solution_verified": self.status == "solved"}


def _cells(bits: int):
    while bits:
        bit = bits & -bits
        yield bit.bit_length() - 1
        bits ^= bit


def _mask(cells) -> int:
    return sum(1 << int(cell) for cell in cells)


def _geometry(tokens):
    board = np.asarray(tokens).reshape(-1)
    floor = _mask(np.flatnonzero(board != 0))
    boxes = _mask(np.flatnonzero(np.isin(board, (3, 5))))
    goals = _mask(np.flatnonzero(np.isin(board, (4, 5, 6))))
    player = int(np.flatnonzero(np.isin(board, (2, 6)))[0])
    return floor, boxes, goals, player


def _reachable(player: int, walkable: int) -> int:
    # The validated wall border prevents horizontal bit shifts wrapping rows.
    region = 1 << player
    while True:
        grown = (region | region << 1 | region >> 1 | region << WIDTH | region >> WIDTH) & walkable
        if grown == region:
            return region
        region = grown


def _live_cells(floor: int, goals: int) -> int:
    """Reverse a hypothetical single box from goals, ignoring other boxes."""
    live = goals
    queue = deque(_cells(goals))
    while queue:
        cell = queue.popleft()
        for direction in range(4):
            previous = NEIGHBORS[cell][OPPOSITE[direction]]
            if previous < 0 or not floor & (1 << previous):
                continue
            stand = NEIGHBORS[previous][OPPOSITE[direction]]
            if stand >= 0 and floor & (1 << stand) and not live & (1 << previous):
                live |= 1 << previous
                queue.append(previous)
    return live


def replay_solution(tokens, solution: str) -> int:
    """Validate every literal worker step; return pushes only if solved.

    This uses grid coordinates and sets, independently of the solver's region
    reduction and dead-cell pruning. Incorrect case or illegal moves raise.
    """
    errors = structural_errors(tokens)
    if errors:
        raise ValueError("; ".join(errors))
    board = np.asarray(tokens).reshape(HEIGHT, WIDTH)
    boxes = {tuple(p) for p in np.argwhere(np.isin(board, (3, 5)))}
    goals = {tuple(p) for p in np.argwhere(np.isin(board, (4, 5, 6)))}
    player = tuple(np.argwhere(np.isin(board, (2, 6)))[0])
    offsets = {symbol: (dr, dc) for symbol, dr, dc in DIRECTIONS}
    pushes = 0
    for move in solution:
        if move.upper() not in offsets:
            raise ValueError(f"unknown move {move!r}")
        dr, dc = offsets[move.upper()]
        target = (player[0] + dr, player[1] + dc)
        if not (0 <= target[0] < HEIGHT and 0 <= target[1] < WIDTH) or board[target] == 0:
            raise ValueError("move enters a wall or leaves the board")
        pushing = target in boxes
        if pushing != move.isupper():
            raise ValueError("move case does not match walking/pushing")
        if pushing:
            destination = (target[0] + dr, target[1] + dc)
            if not (0 <= destination[0] < HEIGHT and 0 <= destination[1] < WIDTH):
                raise ValueError("push leaves the board")
            if board[destination] == 0 or destination in boxes:
                raise ValueError("push destination is blocked")
            boxes.remove(target)
            boxes.add(destination)
            pushes += 1
        player = target
    if boxes != goals:
        raise ValueError("solution does not put all four boxes on goals")
    return pushes


def _certificate(floor: int, boxes: int, player: int, actions: list[tuple[int, int]]) -> str:
    moves = []
    for source, direction in actions:
        target = NEIGHBORS[source][OPPOSITE[direction]]
        parents = {player: None}
        queue = deque([player])
        while target not in parents and queue:
            current = queue.popleft()
            for d, next_cell in enumerate(NEIGHBORS[current]):
                if next_cell >= 0 and floor & (1 << next_cell) and not boxes & (1 << next_cell):
                    if next_cell not in parents:
                        parents[next_cell] = (current, d)
                        queue.append(next_cell)
        if target not in parents:
            raise RuntimeError("solver certificate has an unreachable push")
        path = []
        current = target
        while parents[current] is not None:
            current, d = parents[current]
            path.append(DIRECTIONS[d][0].lower())
        moves.extend(reversed(path))
        moves.append(DIRECTIONS[direction][0])
        boxes ^= (1 << source) | (1 << NEIGHBORS[source][direction])
        player = source
    return "".join(moves)


def solve(tokens, config: SolverConfig | None = None) -> SolveResult:
    """Complete push BFS unless a declared resource limit is reached.

    A discovered solution is push-optimal. Exhaustion and static dead cells
    prove unsolvability. Limits mean unknown, never proof of unsolvability.
    Malformed samples are invalid under the four-box Boxoban contract.
    """
    config = config or SolverConfig()
    started = time.monotonic()
    errors = tuple(structural_errors(tokens))
    expanded = 0
    discovered = 0

    def finish(status, reason, pushes=None, solution=None):
        return SolveResult(status, reason, errors, expanded, discovered,
                           time.monotonic() - started, pushes, solution)

    if errors:
        return finish("structurally_invalid", "invalid_board")
    floor, initial_boxes, goals, initial_player = _geometry(tokens)
    discovered = 1
    if initial_boxes == goals:
        replay_solution(tokens, "")
        return finish("solved", "already_solved", 0, "")
    live = _live_cells(floor, goals)
    if initial_boxes & ~live:
        return finish("proven_unsolvable", "static_dead_cell")
    region = _reachable(initial_player, floor & ~initial_boxes)
    initial = (initial_boxes, region & -region)
    parents = {initial: None}
    queue = deque([(initial, region)])
    while queue:
        if expanded >= config.max_states:
            return finish("unknown", "state_limit")
        if config.max_seconds is not None and time.monotonic() - started >= config.max_seconds:
            return finish("unknown", "time_limit")
        state, region = queue.popleft()
        boxes = state[0]
        expanded += 1
        for source in _cells(boxes):
            for direction, destination in enumerate(NEIGHBORS[source]):
                stand = NEIGHBORS[source][OPPOSITE[direction]]
                if (destination < 0 or stand < 0 or not region & (1 << stand)
                        or not live & (1 << destination) or boxes & (1 << destination)):
                    continue
                next_boxes = boxes ^ (1 << source) ^ (1 << destination)
                if next_boxes == goals:
                    actions = [(source, direction)]
                    previous = state
                    while parents[previous] is not None:
                        previous, src, d = parents[previous]
                        actions.append((src, d))
                    actions.reverse()
                    certificate = _certificate(floor, initial_boxes, initial_player, actions)
                    if replay_solution(tokens, certificate) != len(actions):
                        raise RuntimeError("solver certificate push count mismatch")
                    discovered += 1
                    return finish("solved", "solution_found", len(actions), certificate)
                next_region = _reachable(source, floor & ~next_boxes)
                next_state = (next_boxes, next_region & -next_region)
                if next_state in parents:
                    continue
                if discovered >= config.max_discovered_states:
                    return finish("unknown", "discovered_state_limit")
                parents[next_state] = (state, source, direction)
                discovered += 1
                queue.append((next_state, next_region))
    return finish("proven_unsolvable", "exhausted_search")
