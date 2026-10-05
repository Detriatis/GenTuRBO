"""Plan and run the 13-arm L256 Ehrlich baseline cohort."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from genturbo.control_study.runner import (
    _generation_for_search,
    _iterative_config,
    _unconditional_config,
)
from genturbo.control_study.runner import (
    bound_smoke_cells as _bound_control_smoke_cells,
)
from genturbo.kernel_study.config import KernelStudyConfigError
from genturbo.kernel_study.runner import (
    RunStatus,
    StudyCell,
    _output_value,
    cell_state,
    filter_cells,
    run_cells,
)
from genturbo.kernel_study.runner import plan_summary as _plan_summary

from .config import (
    ARMS,
    CONDITION,
    HOLO_ARM,
    PROTOCOL_VERSION,
    SEED_BATCHES,
    STUDY,
    SequenceBaselineStudyConfig,
)

_OPERATOR_KEYS = (
    "proposal_operator",
    "baseline_population_size",
    "baseline_generations",
    "baseline_fitness_threshold",
    "baseline_mutation_edits",
    "baseline_parent_temperature",
    "baseline_ucb_beta",
    "baseline_children_per_selection",
    "fixed_time",
)
_SEED_TO_TRANCHE = {seed: tranche for tranche, seeds in enumerate(SEED_BATCHES, start=1) for seed in seeds}


def _estimate_seconds(arm: str) -> float:
    if arm == "unconditional":
        return 2.0 * 60.0
    if arm == "oracle_upper_bound":
        return 20.0 * 60.0
    if arm in {"no_tr_gp", "tr_gp", "unconditional_gp", "hamming_gp"}:
        return 75.0 * 60.0
    if arm == "adalead_gp_ei":
        return 90.0 * 60.0
    if arm == "bo_evo":
        return 45.0 * 60.0
    if arm == "lambo2":
        return 6.0 * 60.0 * 60.0
    if arm == HOLO_ARM:
        return 15.0 * 60.0
    return 15.0 * 60.0


def _execution_device(study: SequenceBaselineStudyConfig, *, arm: str, seed: int) -> str | None:
    """Return a launch device without rewriting completed historical cells."""

    execution = study.raw["execution"]
    if seed <= int(execution["legacy_completed_seed_max"]) and arm != HOLO_ARM:
        return None
    if seed in execution.get("cpu_generator_seeds", {}).get(arm, ()):
        return "cpu"
    if arm in execution["generator_gpu_arms"] or arm in execution["lambo2_arms"]:
        return str(execution["new_generator_device"])
    if arm in execution["cpu_arms"]:
        return str(execution["new_cpu_device"])
    raise KernelStudyConfigError(f"{arm} is absent from the declared recorded device policy")


def _apply_execution_device(
    resolved: dict[str, Any],
    study: SequenceBaselineStudyConfig,
    *,
    arm: str,
    seed: int,
) -> None:
    device = _execution_device(study, arm=arm, seed=seed)
    if device is None:
        return
    resolved["device"] = device
    if arm == "lambo2":
        resolved["lambo2"] = {**resolved["lambo2"], "device": device}
    elif arm == HOLO_ARM:
        resolved["holo"] = {**resolved["holo"], "device": device}


def _holo_config(
    study: SequenceBaselineStudyConfig,
    *,
    seed: int,
    cell_dir: Path,
    initial_batch_path: Path,
    repo_root: Path,
    status: RunStatus,
) -> dict[str, Any]:
    """Build the manifest-compatible Holo cell without interpolant-only fields."""

    condition = study.condition
    protocol = dict(study.raw["holo"])
    population_size = int(protocol["population_size"])
    generations = int(protocol["generation_count"])
    return {
        "runner": "holo_ga",
        "config": study.path.relative_to(repo_root).as_posix(),
        "device": str(protocol["device"]),
        "reward": {
            "name": "ehrlich",
            "instance_path": condition["instance_path"],
            "quantization_override": int(condition["quantization"]),
        },
        "sequence_length": int(condition["sequence_length"]),
        "batch_size": population_size,
        "initial_oracle_evaluations": int(protocol["initial_archive_size"]),
        "initial_batch_path": _output_value(initial_batch_path, repo_root=repo_root, status=status),
        "initial_batch_out": _output_value(
            cell_dir / "initial_batch.jsonl.gz", repo_root=repo_root, status=status
        ),
        "initial_batch_advance_rng": False,
        "initial_batch_rescore_with_reward": False,
        "num_iterations": generations,
        "max_oracle_evaluations": int(study.raw["generation"]["max_selected_archive_evaluations"]),
        "active_learning": True,
        "al_oracle_batch_size": population_size,
        "al_ehrlich_black_box_feasibility": True,
        "trace_level": "full",
        "trace_record_limit": population_size * generations,
        "compress_jsonl": True,
        "write_proposal_diagnostics": True,
        "proposal_traces_out": _output_value(
            cell_dir / "proposal_traces.jsonl.gz", repo_root=repo_root, status=status
        ),
        "proposal_diagnostics_out": _output_value(
            cell_dir / "proposal_diagnostics.jsonl.gz", repo_root=repo_root, status=status
        ),
        "samples_out": _output_value(cell_dir / "samples.jsonl.gz", repo_root=repo_root, status=status),
        "run_manifest_out": _output_value(
            cell_dir / "run_manifest.json", repo_root=repo_root, status=status
        ),
        "proposal_operator": "holo_discrete_evolution",
        "oracle_budget_mode": "physical",
        "holo": protocol,
        "plot_label": HOLO_ARM,
        "metadata": {"plot_label": HOLO_ARM, "experiment_status": status, "runner": "holo_ga"},
        "seed": int(seed),
    }


def build_sequence_baseline_cells(
    study: SequenceBaselineStudyConfig,
    *,
    repo_root: Path | str,
    run_root: Path | None = None,
    status: RunStatus = "canonical",
) -> list[StudyCell]:
    """Build all 260 cells in seed-major order to preserve paired blocks."""

    root = Path(repo_root).resolve()
    config = study.raw
    base = Path(run_root) if run_root is not None else root / str(config["run_dir"])
    dataset_base = root / str(study.kernel_study.datasets["run_dir"])
    generation = {
        **_generation_for_search(config["generation"]),
        "unconditional_chunk_size": int(config["generation"]["unconditional_chunk_size"]),
        "iteration_cap": int(config["generation"]["iteration_cap"]),
        "random_time": dict(config["generation"]["random_time"]),
    }
    cells: list[StudyCell] = []
    for seed in study.seeds:
        for arm in config["arms"]:
            arm_name = str(arm["name"])
            cell_dir = base / CONDITION / arm_name / f"seed{seed}"
            if arm_name == "unconditional":
                resolved = _unconditional_config(
                    condition=study.condition,
                    generation=generation,
                    seed=seed,
                    cell_dir=cell_dir,
                    repo_root=root,
                    status=status,
                )
            elif arm_name == HOLO_ARM:
                initial = dataset_base / CONDITION / "random" / f"seed{seed}" / "initial_batch.jsonl.gz"
                resolved = _holo_config(
                    study,
                    seed=seed,
                    cell_dir=cell_dir,
                    initial_batch_path=initial,
                    repo_root=root,
                    status=status,
                )
            else:
                initial = dataset_base / CONDITION / "random" / f"seed{seed}" / "initial_batch.jsonl.gz"
                iterative_arm = (
                    {**arm, "surrogate": None, "acquisition": "random"}
                    if arm_name == "lambo2"
                    else arm
                )
                resolved = _iterative_config(
                    condition=study.condition,
                    generation=generation,
                    gp=config["gp"],
                    arm=iterative_arm,
                    seed=seed,
                    cell_dir=cell_dir,
                    initial_batch_path=initial,
                    repo_root=root,
                    status=status,
                )
                resolved.update({key: arm[key] for key in _OPERATOR_KEYS if key in arm})
                if arm_name == "bo_evo":
                    resolved["mixes_per_time"] = int(arm["baseline_children_per_selection"]) * int(
                        generation["acquisition_batch_size"]
                    )
                if arm_name == "lambo2":
                    lambo = config["lambo2"]
                    pretraining = lambo["pretraining"]
                    resolved.update(
                        {
                            "runner": "lambo2",
                            "checkpoint": str(pretraining["checkpoint"]),
                            "config": study.path.relative_to(root).as_posix(),
                            "pretraining_manifest": str(pretraining["manifest"]),
                            "lambo_requirements": str(lambo["requirements"]),
                            "lambo2": dict(lambo["online"]),
                            "mixes_per_time": int(generation["acquisition_batch_size"]),
                            "al_acquisition": "graph_nei",
                            "al_feature_space": "lambo2_partial_deep_ensemble",
                        }
                    )
                    resolved["metadata"] = {**resolved["metadata"], "runner": "lambo2"}
                resolved["al_ehrlich_black_box_feasibility"] = True
                if arm["surrogate"] is not None:
                    resolved["al_gp_hyperparameter_refit_interval"] = int(config["gp"]["hyperparameter_refit_interval"])
            _apply_execution_device(resolved, study, arm=arm_name, seed=seed)
            cells.append(
                StudyCell(
                    workflow="online",
                    condition=CONDITION,
                    seed=seed,
                    arm=arm_name,
                    run_dir=cell_dir,
                    resolved_config=resolved,
                    estimated_seconds=_estimate_seconds(arm_name),
                    status=status,
                    study=STUDY,
                )
            )
    return cells


def bound_smoke_cells(
    cells: list[StudyCell],
    *,
    selected_archive_evaluations: int,
    device: str | None,
) -> list[StudyCell]:
    """Bound smoke cells while retaining whole Holo generations."""

    bounded: list[StudyCell] = []
    for cell in cells:
        if cell.arm != HOLO_ARM:
            bounded.extend(
                _bound_control_smoke_cells(
                    [cell],
                    selected_archive_evaluations=selected_archive_evaluations,
                    device=device,
                )
            )
            continue
        resolved = dict(cell.resolved_config)
        initial = int(resolved["initial_oracle_evaluations"])
        population = int(resolved["al_oracle_batch_size"])
        new_evaluations = int(selected_archive_evaluations) - initial
        generations, remainder = divmod(new_evaluations, population)
        if generations < 1 or remainder:
            raise KernelStudyConfigError(
                f"Holo smoke budget must be 64 plus a positive whole multiple of {population}"
            )
        resolved["max_oracle_evaluations"] = int(selected_archive_evaluations)
        resolved["num_iterations"] = generations
        if device is not None:
            resolved["device"] = str(device)
            resolved["holo"] = {**resolved["holo"], "device": str(device)}
        bounded.append(replace(cell, resolved_config=resolved))
    return bounded


def filter_tranches(cells: Iterable[StudyCell], tranches: Sequence[int] | None) -> list[StudyCell]:
    """Select complete paired five-seed tranches while preserving seed-major order."""

    materialized = list(cells)
    if tranches is None:
        return materialized
    selected = [int(value) for value in tranches]
    if not selected or len(selected) != len(set(selected)):
        raise KernelStudyConfigError("tranche filter must be non-empty and contain no duplicates")
    unknown = sorted(set(selected) - set(range(1, len(SEED_BATCHES) + 1)))
    if unknown:
        raise KernelStudyConfigError(f"unknown tranche filter values: {', '.join(map(str, unknown))}")
    selected_set = set(selected)
    result = [cell for cell in materialized if _SEED_TO_TRANCHE[cell.seed] in selected_set]
    if not result:
        raise KernelStudyConfigError("tranche filter selected no work")
    return result


def plan_summary(cells: Sequence[StudyCell]) -> dict[str, Any]:
    """Return the normal launch plan plus its explicit paired-block matrix."""

    summary = _plan_summary(cells)
    summary["estimate_kind"] = "conservative MPS estimate from optimized smokes plus measured GP scaling"
    seeds = list(dict.fromkeys(cell.seed for cell in cells))
    tranches = sorted({_SEED_TO_TRANCHE[seed] for seed in seeds})
    summary["protocol_version"] = PROTOCOL_VERSION
    summary["condition"] = CONDITION
    summary["sequence_length"] = 256
    summary["iteration_order"] = "seed_major_then_arm"
    summary["arm_order"] = list(ARMS)
    summary["proposal_queries_per_batch"] = {
        "pool_based_arms": 128,
        "bo_evo": 64,
        "lambo2": 8,
        "holo_ga": 484,
    }
    summary["holo_generations"] = 4
    summary["holo_prepaid_initial_archive"] = 64
    summary["selected_seeds"] = seeds
    summary["selected_tranches"] = tranches
    summary["tranche_matrix"] = [
        {
            "tranche": tranche,
            "seeds": list(SEED_BATCHES[tranche - 1]),
            "selected_cell_count": sum(_SEED_TO_TRANCHE[cell.seed] == tranche for cell in cells),
        }
        for tranche in tranches
    ]
    return summary


def progress(cells: Sequence[StudyCell]) -> dict[str, Any]:
    """Report exact cumulative cell, paired-seed, and tranche completeness."""

    states = [(cell, cell_state(cell)) for cell in cells]
    totals = Counter(state for _cell, state in states)
    completed = int(totals.get("completed", 0))
    by_tranche: dict[str, Any] = {}
    complete_tranches: list[int] = []
    for tranche, seeds in enumerate(SEED_BATCHES, start=1):
        group = [(cell, state) for cell, state in states if cell.seed in seeds]
        if not group:
            continue
        tranche_states = Counter(state for _cell, state in group)
        tranche_completed = int(tranche_states.get("completed", 0))
        is_complete = len(group) == len(seeds) * len(ARMS) and tranche_completed == len(group)
        if is_complete:
            complete_tranches.append(tranche)
        by_tranche[str(tranche)] = {
            "seeds": list(seeds),
            "expected_cells": len(group),
            "completed_cells": tranche_completed,
            "remaining_cells": len(group) - tranche_completed,
            "states": dict(sorted(tranche_states.items())),
            "complete": is_complete,
        }
    paired_seed_completion = {
        str(seed): sum(state == "completed" for cell, state in states if cell.seed == seed) == len(ARMS)
        and sum(cell.seed == seed for cell, _state in states) == len(ARMS)
        for seed in sorted({cell.seed for cell, _state in states})
    }
    full_cohort = len(cells) == len(ARMS) * sum(len(batch) for batch in SEED_BATCHES)
    aggregation_ready = full_cohort and completed == len(cells)
    ready_cumulative_seed_counts: list[int] = []
    for tranche in range(1, len(SEED_BATCHES) + 1):
        tranche_progress = by_tranche.get(str(tranche))
        if tranche_progress is None or tranche_progress["complete"] is not True:
            break
        ready_cumulative_seed_counts.append(tranche * len(SEED_BATCHES[0]))
    return {
        "status": "complete" if completed == len(cells) else "incomplete",
        "expected_cells": len(cells),
        "completed_cells": completed,
        "remaining_cells": len(cells) - completed,
        "states": dict(sorted(totals.items())),
        "complete_tranches": complete_tranches,
        "ready_cumulative_seed_counts": ready_cumulative_seed_counts,
        "by_tranche": by_tranche,
        "paired_seed_completion": paired_seed_completion,
        "aggregation_ready": aggregation_ready,
        "results_hook": "aggregate only complete tranche prefixes listed in ready_cumulative_seed_counts",
    }


__all__ = [
    "bound_smoke_cells",
    "build_sequence_baseline_cells",
    "filter_cells",
    "filter_tranches",
    "plan_summary",
    "progress",
    "run_cells",
]
