"""Raw-sample solvability, push-search difficulty, and optional wall repair."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import asdict, replace

import numpy as np

from sokoban.boards import WIDTH, structural_errors
from sokoban.solver import NEIGHBORS, SOLVER_VERSION, SolverConfig, solve

PAPER = "https://arxiv.org/abs/2608.15958"
STATUSES = ("solved", "proven_unsolvable", "unknown", "structurally_invalid")


def one_wall_repair(tokens, config: SolverConfig, *, max_seconds: float = 60.0) -> dict:
    """Try deleting one interior wall, without modifying the input board.

    Only call for a structurally valid, proven-unsolvable original. Every
    candidate gets the same state cap. The total time budget covers all tries;
    any unfinished/capped candidate prevents a negative proof.
    """
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("repair time limit must be positive and finite")
    errors = structural_errors(tokens)
    if errors:
        raise ValueError("; ".join(errors))
    board = np.asarray(tokens).reshape(-1)
    candidates = [cell for cell in range(100)
                  if 0 < cell // WIDTH < 9 and 0 < cell % WIDTH < 9 and board[cell] == 0
                  and any(board[neighbor] != 0 for neighbor in NEIGHBORS[cell])]
    started = time.monotonic()
    attempted = unresolved = 0
    for cell in candidates:
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        repaired = board.copy()
        repaired[cell] = 1
        attempt_config = replace(config, max_seconds=min(remaining, config.max_seconds or remaining))
        result = solve(repaired, attempt_config)
        attempted += 1
        unresolved += result.status == "unknown"
        if result.status == "solved":
            return {"status": "repaired", "removed_wall": list(divmod(cell, WIDTH)),
                    "attempted": attempted, "candidates": len(candidates),
                    "elapsed_seconds": time.monotonic() - started, "solution": result.to_dict()}
    return {"status": "unknown" if unresolved or attempted < len(candidates) else "unrepairable",
            "attempted": attempted, "candidates": len(candidates), "capped_attempts": unresolved,
            "elapsed_seconds": time.monotonic() - started}


def distribution(values) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "min": None, "q25": None, "median": None,
                "q75": None, "max": None}
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {"count": len(values), "mean": float(values.mean()), "min": float(values.min()),
            "q25": float(q25), "median": float(median), "q75": float(q75), "max": float(values.max())}


def summarize(records: list[dict]) -> dict:
    n = len(records)
    counts = Counter(record["status"] for record in records)
    solved = [record for record in records if record["status"] == "solved"]
    def ratio(count):
        return count / n if n else None
    result = {
        "num_samples": n,
        "status_counts": {status: counts[status] for status in STATUSES},
        "structurally_valid_fraction": ratio(n - counts["structurally_invalid"]),
        "raw_solvability_rate": ratio(counts["solved"]),
        "raw_solvability_bounds": [ratio(counts["solved"]), ratio(counts["solved"] + counts["unknown"])],
        "bounds_note": "Unresolved-search bounds, not a statistical confidence interval; denominator is every raw sample.",
        "reason_counts": dict(Counter(record["reason"] for record in records)),
        "difficulty": {
            "scope": "raw solved boards only; failures and capped searches have no difficulty score",
            "expanded_states": distribution([record["difficulty_expanded_states"] for record in solved]),
            "minimum_pushes": distribution([record["minimum_pushes"] for record in solved]),
        },
    }
    if any("one_wall_repair" in record for record in records):
        repairs = Counter(record.get("one_wall_repair", {}).get("status", "not_assessed") for record in records)
        eligible = counts["proven_unsolvable"]
        lower = counts["solved"] + repairs["repaired"]
        result["one_wall_repair"] = {
            "eligibility": "proven-unsolvable raw boards only; original unknown searches remain unresolved",
            "eligible": eligible, "status_counts": dict(repairs),
            "raw_nonsolved": n - counts["solved"],
            "repairable_fraction_of_raw_nonsolved": repairs["repaired"] / (n - counts["solved"])
            if n > counts["solved"] else None,
            "repairable_fraction_of_proven_unsolvable": repairs["repaired"] / eligible if eligible else None,
            "effective_solvability_bounds": [ratio(lower), ratio(lower + counts["unknown"] + repairs["unknown"])],
            "raw_samples_modified": False,
        }
    return result


def assess_boards(boards, config: SolverConfig | None = None, *, ids=None,
                  repair: bool = False, repair_max_seconds: float = 60.0, on_record=None) -> tuple[dict, list[dict]]:
    config = config or SolverConfig()
    records = []
    if ids is None:
        ids = (str(i) for i in range(len(boards)))
    for board_id, board in zip(ids, boards, strict=True):
        result = solve(board, config)
        record = {"id": str(board_id),
                  "board_sha256": hashlib.sha256(np.asarray(board, dtype="<i8").tobytes()).hexdigest(),
                  **result.to_dict()}
        if repair:
            record["one_wall_repair"] = (
                one_wall_repair(board, config, max_seconds=repair_max_seconds)
                if result.status == "proven_unsolvable" else {"status": "not_assessed"}
            )
        records.append(record)
        if on_record is not None:
            on_record(record)
    summary = summarize(records)
    summary["method"] = {
        "paper": PAPER, "solver_version": SOLVER_VERSION, "limits": asdict(config),
        "search": "push BFS, canonical worker region, reverse-goal static dead-cell pruning",
        "effort": "states popped before first solution generated; already solved = 0",
        "tie_order": "row-major boxes; right, left, down, up",
        "comparability": "same metric definition; exact effort depends on solver tie order and pruning",
        "certificates": "every solution replayed independently; minimum pushes, not minimum total moves",
        "repair_max_seconds": repair_max_seconds if repair else None,
    }
    return summary, records


def compare_difficulty(records: list[dict], reference: list[dict]) -> dict:
    """Reference quartiles, never regenerated batch-relative easy/hard labels."""
    efforts = sorted(record["difficulty_expanded_states"] for record in reference if record["status"] == "solved")
    thresholds = np.quantile(efforts, [0.25, 0.5, 0.75]).tolist() if efforts else []
    counts = [0] * 4
    for record in records:
        quartile = None
        if record["status"] == "solved" and thresholds:
            quartile = int(np.searchsorted(thresholds, record["difficulty_expanded_states"], side="left")) + 1
            counts[quartile - 1] += 1
        record["reference_difficulty_quartile"] = quartile
    return {"reference_total": len(reference), "reference_solved": len(efforts),
            "expanded_state_thresholds": thresholds, "generated_solved_quartile_counts": counts,
            "note": "Calibrated on solved reference boards only; ties go to the lower quartile. Retain reference and solver settings across comparisons."}
