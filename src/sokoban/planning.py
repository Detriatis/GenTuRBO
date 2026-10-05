"""Exact minimum box/group alternations over all solutions, with bounded search.

Jarušek & Pelánek, Difficulty Rating of Sokoban Puzzle (2010), section 4.2:
https://www.fi.muni.cz/~xpelanek/publications/stairs2010-final.pdf

The first pushed group costs zero; each subsequent change costs one. Walking
and continued pushes within a group cost zero. A solution need not be push-optimal.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import asdict, dataclass

from sokoban.boards import structural_errors
from sokoban.solver import (
    DIRECTIONS,
    NEIGHBORS,
    OPPOSITE,
    _cells,
    _certificate,
    _geometry,
    _live_cells,
    _reachable,
    replay_solution,
)

PLANNING_VERSION = "group-switch-astar-v1"
PARTITIONS = ("AABB", "ABAB", "ABBA")


@dataclass(frozen=True)
class PlanningConfig:
    max_states: int = 200_000
    max_discovered_states: int = 400_000
    max_seconds: float | None = 10.0

    def __post_init__(self):
        if self.max_states < 1 or self.max_discovered_states < 1:
            raise ValueError("state budgets must be positive")
        if self.max_seconds is not None and (not math.isfinite(self.max_seconds) or self.max_seconds <= 0):
            raise ValueError("max_seconds must be positive and finite, or None")


@dataclass
class PlanningResult:
    partition: str
    status: str
    reason: str
    lower_bound: int | None
    upper_bound: int | None
    value: int | None
    solution: str | None
    solution_pushes: int | None
    expanded_states: int
    discovered_states: int
    elapsed_seconds: float

    def to_dict(self):
        return asdict(self)


def _labels(partition):
    if len(partition) != 4:
        raise ValueError("partition must assign a group to each of four initial row-major boxes")
    labels = {name: i for i, name in enumerate(dict.fromkeys(partition))}
    return tuple(labels[x] for x in partition)


def solution_switches(tokens, solution: str, partition: str = "ABCD") -> dict:
    """Independently replay a witness and follow stable box group identities."""
    labels = _labels(partition)
    pushes = replay_solution(tokens, solution)
    _, boxes, _, player = _geometry(tokens)
    groups = dict(zip(_cells(boxes), labels, strict=True))
    offsets = {name: dr * 10 + dc for name, dr, dc in DIRECTIONS}
    sequence = []
    for move in solution:
        target = player + offsets[move.upper()]
        if move.isupper():
            group = groups.pop(target)
            groups[target + offsets[move]] = group
            sequence.append(group)
        player = target
    switches = sum(a != b for a, b in zip(sequence, sequence[1:], strict=False))
    return {"switches": switches, "blocks": switches + bool(sequence),
            "pushes": pushes, "groups": sequence}


def minimum_switches(tokens, partition="ABCD", config=None, *, witness=None):
    """A* on (group-labelled box sets, reachable worker region, last group).

    Groups containing an off-goal box must each be pushed again. Counting the
    groups other than the active one is an admissible lower bound on remaining
    switches. Geometric distances only break priority ties. Immutable search
    nodes retain valid certificates when a cheaper route reopens a state.
    """
    labels = _labels(partition)
    config = config or PlanningConfig()
    started = time.monotonic()
    expanded, discovered = 0, 0
    lower, upper, certificate = None, None, None

    def finish(status, reason):
        exact = lower if status == "exact" else None
        return PlanningResult(partition, status, reason, lower, upper, exact, certificate,
                              replay_solution(tokens, certificate) if certificate is not None else None,
                              expanded, discovered, time.monotonic() - started)

    if structural_errors(tokens):
        return finish("structurally_invalid", "invalid_board")
    floor, initial_boxes, goals, player = _geometry(tokens)
    if witness is not None:
        upper = solution_switches(tokens, witness, partition)["switches"]
        certificate = witness
    if initial_boxes == goals:
        lower = upper = 0
        certificate = ""
        return finish("exact", "already_solved")
    live = _live_cells(floor, goals)
    if initial_boxes & ~live:
        return finish("proven_unsolvable", "static_dead_cell")
    group_count = max(labels) + 1
    initial_groups = [0] * group_count
    for cell, group in zip(_cells(initial_boxes), labels, strict=True):
        initial_groups[group] |= 1 << cell

    def remaining(groups, last):
        needed = [i for i, bits in enumerate(groups) if bits & ~goals]
        return max(0, len(needed) - 1) if last == -1 else sum(i != last for i in needed)

    lower = remaining(initial_groups, -1)
    if upper == lower:
        return finish("exact", "witness_matches_lower_bound")
    goal_cells = list(_cells(goals))
    distances = [min(abs(c // 10 - g // 10) + abs(c % 10 - g % 10) for g in goal_cells)
                 for c in range(100)]
    region = _reachable(player, floor & ~initial_boxes)
    initial = (tuple(initial_groups), region & -region, -1)
    # node = (state, g, parent node id, push action, full worker region, boxes)
    nodes = [(initial, 0, None, None, region, initial_boxes)]
    best = {initial: (0, 0)}
    queue = [(lower, (initial_boxes & ~goals).bit_count(),
              sum(distances[c] for c in _cells(initial_boxes)), 0, 0)]
    discovered = 1

    def save_witness(node_id, action):
        actions = [action]
        while nodes[node_id][2] is not None:
            actions.append(nodes[node_id][3])
            node_id = nodes[node_id][2]
        actions.reverse()
        return _certificate(floor, initial_boxes, player, actions)

    def limit(reason):
        if upper == lower:
            return finish("exact", "bounds_match")
        return finish("bounded" if upper is not None else "unknown", reason)

    while queue:
        estimate, _, _, _, node_id = heapq.heappop(queue)
        state, cost, _, _, region, boxes = nodes[node_id]
        if best.get(state) != (cost, node_id):
            continue
        if upper is not None and estimate >= upper:
            lower = upper
            return finish("exact", "frontier_proves_optimum")
        lower = max(lower, estimate)
        if expanded >= config.max_states:
            return limit("state_limit")
        if config.max_seconds is not None and time.monotonic() - started >= config.max_seconds:
            return limit("time_limit")
        expanded += 1
        groups, _, last = state
        for group, group_boxes in enumerate(groups):
            new_cost = cost + int(last != -1 and group != last)
            if upper is not None and new_cost >= upper:
                continue
            for source in _cells(group_boxes):
                for direction, destination in enumerate(NEIGHBORS[source]):
                    stand = NEIGHBORS[source][OPPOSITE[direction]]
                    if (destination < 0 or stand < 0 or not region & (1 << stand)
                            or not live & (1 << destination) or boxes & (1 << destination)):
                        continue
                    change = (1 << source) | (1 << destination)
                    next_boxes = boxes ^ change
                    if next_boxes == goals:
                        upper = new_cost
                        certificate = save_witness(node_id, (source, direction))
                        if solution_switches(tokens, certificate, partition)["switches"] != upper:
                            raise RuntimeError("switch certificate disagrees with search cost")
                        if upper == lower:
                            return finish("exact", "bounds_match")
                        continue
                    next_groups = list(groups)
                    next_groups[group] ^= change
                    next_groups = tuple(next_groups)
                    next_estimate = new_cost + remaining(next_groups, group)
                    if upper is not None and next_estimate >= upper:
                        continue
                    next_region = _reachable(source, floor & ~next_boxes)
                    next_state = (next_groups, next_region & -next_region, group)
                    if new_cost >= best.get(next_state, (math.inf, None))[0]:
                        continue
                    if discovered >= config.max_discovered_states:
                        return limit("discovered_state_limit")
                    new_id = len(nodes)
                    nodes.append((next_state, new_cost, node_id, (source, direction), next_region, next_boxes))
                    best[next_state] = (new_cost, new_id)
                    discovered += 1
                    heapq.heappush(queue, (next_estimate, (next_boxes & ~goals).bit_count(),
                                          sum(distances[c] for c in _cells(next_boxes)), -new_cost, new_id))
    if upper is not None:
        lower = upper
        return finish("exact", "exhausted_better_solutions")
    lower = None
    return finish("proven_unsolvable", "exhausted_search")


def combine_decomposition(partitions):
    """Bound the minimum over all three pairings, including partial searches."""
    if len(partitions) != 3 or {p.partition for p in partitions} != set(PARTITIONS):
        raise ValueError("all three pair partitions are required")
    if any(p.status == "structurally_invalid" for p in partitions):
        return {"status": "structurally_invalid", "value": None, "lower_bound": None, "upper_bound": None}
    if any(p.status == "proven_unsolvable" for p in partitions):
        return {"status": "proven_unsolvable", "value": None, "lower_bound": None, "upper_bound": None}
    lower = min(p.lower_bound for p in partitions)
    witnesses = [p for p in partitions if p.upper_bound is not None]
    winner = min(witnesses, key=lambda p: p.upper_bound) if witnesses else None
    upper = winner.upper_bound if winner else None
    exact = upper is not None and lower == upper
    return {"status": "exact" if exact else ("bounded" if upper is not None else "unknown"),
            "value": lower if exact else None, "lower_bound": lower, "upper_bound": upper,
            "witness_partition": winner.partition if winner else None,
            "solution": winner.solution if winner else None}


def assess_planning(tokens, config=None, *, witness=None):
    config = config or PlanningConfig()
    box = minimum_switches(tokens, "ABCD", config, witness=witness)
    pair_results = [minimum_switches(tokens, p, config, witness=witness) for p in PARTITIONS]
    return {"version": PLANNING_VERSION, "config": asdict(config),
            "convention": "first group free; count changes between consecutive pushed groups; walking free",
            "box_switches": box.to_dict(),
            "two_decomposition": combine_decomposition(pair_results),
            "pairings": [p.to_dict() for p in pair_results]}
