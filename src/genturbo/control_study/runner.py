"""Plan the six-arm, 2000-call Ehrlich paper-control cohort."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from genturbo.kernel_study.runner import (
    RunStatus,
    StudyCell,
    _search_config,
    filter_cells,
    plan_summary,
    run_cells,
)
from genturbo.serialization import atomic_write_json

from .config import STUDY, ControlStudyConfig

_TR_KEYS = {
    "tr_type",
    "tr_domain",
    "tr_sampling",
    "tr_edits",
    "edits_min",
    "edits_init",
    "edits_max",
    "region_expand_scalar",
    "region_shrink_scalar",
    "tau_success",
    "tau_failure",
    "tr_eps",
    "tr_log_every",
}

_COMPLETION_MARKER = "scientific_completion.json"
RUNTIME_CALIBRATION = {
    "gp_archive_rows": 2000,
    "gp_fit_seconds": 123.25,
    "posterior_candidates": 128,
    "posterior_seconds": 13.41,
    "peak_rss_gib": 1.02,
    "practical_serial_cohort_hours": [40, 60],
}


def _required_count(run: Mapping[str, Any], key: str) -> int:
    value = run.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"control completion requires non-negative integer {key}")
    return int(value)


def _validate_control_summary(
    cell: StudyCell,
    run: Mapping[str, Any],
    *,
    selected_trace_count: int | None,
    eligible_trace_count: int | None,
) -> dict[str, int]:
    """Validate the exact scientific stopping contract for one control cell."""

    budget = int(cell.resolved_config["max_oracle_evaluations"])
    initial = int(cell.resolved_config["initial_oracle_evaluations"])
    if budget <= 0 or initial < 0 or initial > budget:
        raise ValueError("control completion has invalid configured oracle counts")
    if run.get("method") != cell.arm or _required_count(run, "seed") != cell.seed:
        raise ValueError(f"{cell.identity} analysis identity does not match its cell")

    trajectory = run.get("best_so_far_trajectory")
    if not isinstance(trajectory, list) or len(trajectory) != budget:
        raise ValueError(f"{cell.identity} did not produce an exact {budget}-call trajectory")
    if run.get("trajectory_covers_all_oracle_calls") is not True:
        raise ValueError(f"{cell.identity} trajectory does not cover every budgeted oracle call")
    if _required_count(run, "trajectory_oracle_calls") != budget:
        raise ValueError(f"{cell.identity} trajectory oracle count is under budget")

    budgeted = _required_count(run, "oracle_budgeted_evaluations_total")
    physical = _required_count(run, "oracle_physical_evaluations_total")
    privileged = _required_count(run, "oracle_privileged_evaluations_total")
    if budgeted != budget or physical != budgeted + privileged:
        raise ValueError(
            f"{cell.identity} has incomplete oracle accounting: "
            f"budgeted={budgeted}, physical={physical}, privileged={privileged}"
        )

    if cell.arm == "unconditional":
        if (
            _required_count(run, "initial_rows") != 0
            or _required_count(run, "serialized_initial_rows") != budget
            or _required_count(run, "sample_rows") != budget
        ):
            raise ValueError(f"{cell.identity} did not serialize exactly {budget} unconditional samples")
        if (
            _required_count(run, "diagnostic_rows") != 0
            or _required_count(run, "trace_rows") != 0
            or privileged != 0
            or physical != budget
        ):
            raise ValueError(f"{cell.identity} has unexpected adaptive or privileged accounting")
        return {
            "budgeted_oracle_evaluations": budgeted,
            "initial_oracle_evaluations": 0,
            "selected_oracle_evaluations": budget,
            "diagnostic_rows": 0,
            "selected_trace_count": 0,
            "eligible_trace_count": 0,
            "physical_oracle_evaluations": physical,
            "privileged_oracle_evaluations": privileged,
        }

    expected_selected = budget - initial
    batch_size = int(cell.resolved_config["al_oracle_batch_size"])
    expected_batches, remainder = divmod(expected_selected, batch_size)
    if remainder:
        raise ValueError(f"{cell.identity} selected budget is not divisible by its batch size")
    if (
        _required_count(run, "initial_rows") != initial
        or _required_count(run, "sample_rows") != expected_selected
        or _required_count(run, "selected_oracle_evaluations") != expected_selected
    ):
        raise ValueError(
            f"{cell.identity} must contain exactly {initial} initial and "
            f"{expected_selected} selected evaluations"
        )
    diagnostic_rows = _required_count(run, "diagnostic_rows")
    if diagnostic_rows < expected_batches:
        raise ValueError(
            f"{cell.identity} has {diagnostic_rows} diagnostics for {expected_batches} required batches"
        )
    if (
        run.get("has_full_traces") is not True
        or run.get("trace_truncated") is not False
        or selected_trace_count != expected_selected
    ):
        raise ValueError(f"{cell.identity} lacks complete selected-proposal trace accounting")

    eligible = int(eligible_trace_count) if eligible_trace_count is not None else -1
    if cell.arm == "oracle_upper_bound":
        expected_physical = initial + eligible
        expected_privileged = eligible - expected_selected
        if (
            eligible < expected_selected
            or privileged <= 0
            or physical != expected_physical
            or privileged != expected_privileged
        ):
            raise ValueError(f"{cell.identity} has inconsistent privileged proposal-pool accounting")
    elif privileged != 0 or physical != budget:
        raise ValueError(f"{cell.identity} unexpectedly used privileged oracle scores")

    return {
        "budgeted_oracle_evaluations": budgeted,
        "initial_oracle_evaluations": initial,
        "selected_oracle_evaluations": expected_selected,
        "diagnostic_rows": diagnostic_rows,
        "selected_trace_count": int(selected_trace_count),
        "eligible_trace_count": eligible,
        "physical_oracle_evaluations": physical,
        "privileged_oracle_evaluations": privileged,
    }


def finalize_control_completion(cell: StudyCell, *, repo_root: Path) -> None:
    """Write an immutable completion marker only after exact raw-output validation."""

    from genturbo.artifacts import sha256_file
    from genturbo.kernel_study.online_analysis import _analyze_run, select_manifest
    from genturbo.kernel_study.results import TraceSource, read_proposal_traces

    selected = select_manifest(cell.run_dir)
    experiment_status = selected.data.get("experiment_status")
    if experiment_status != cell.status:
        raise ValueError(
            f"{cell.identity} manifest experiment_status={experiment_status!r}; expected {cell.status!r}"
        )
    run = _analyze_run(
        cell.run_dir,
        run_root=cell.run_dir.parent.parent,
        repo_root=repo_root,
        reward_cache={},
    )
    selected_trace_count: int | None = None
    eligible_trace_count: int | None = None
    if cell.arm != "unconditional":
        classified = read_proposal_traces(
            [
                TraceSource(
                    cell.run_dir / "proposal_traces.jsonl.gz",
                    cell.condition,
                    int(cell.seed),
                )
            ]
        )
        selected_trace_count = len(classified.selected_records)
        eligible_trace_count = len(classified.eligible_records)
    counts = _validate_control_summary(
        cell,
        run,
        selected_trace_count=selected_trace_count,
        eligible_trace_count=eligible_trace_count,
    )
    atomic_write_json(
        cell.run_dir / _COMPLETION_MARKER,
        {
            "schema_version": 1,
            "study": cell.study,
            "identity": cell.identity,
            "status": "completed",
            "experiment_status": cell.status,
            "manifest_sha256": sha256_file(cell.manifest_path),
            "resolved_config_sha256": sha256_file(cell.resolved_config_path),
            **counts,
        },
    )


def control_completion_marker_is_valid(cell: StudyCell) -> bool:
    """Return whether a checksum-current cell has passed the scientific gate."""

    from genturbo.artifacts import sha256_file
    from genturbo.kernel_study.online_analysis import EhrlichSweepError, select_manifest

    marker_path = cell.run_dir / _COMPLETION_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        selected = select_manifest(cell.run_dir)
    except (OSError, json.JSONDecodeError, EhrlichSweepError):
        return False
    if not isinstance(marker, dict):
        return False
    budget = int(cell.resolved_config["max_oracle_evaluations"])
    configured_initial = int(cell.resolved_config["initial_oracle_evaluations"])
    expected_initial = 0 if cell.arm == "unconditional" else configured_initial
    expected_selected = budget - expected_initial
    counts_are_exact = (
        marker.get("budgeted_oracle_evaluations") == budget
        and marker.get("initial_oracle_evaluations") == expected_initial
        and marker.get("selected_oracle_evaluations") == expected_selected
        and marker.get("selected_trace_count")
        == (0 if cell.arm == "unconditional" else expected_selected)
    )
    return (
        marker.get("schema_version") == 1
        and marker.get("study") == cell.study
        and marker.get("identity") == cell.identity
        and marker.get("status") == "completed"
        and marker.get("experiment_status") == cell.status
        and selected.path.resolve() == cell.manifest_path.resolve()
        and selected.data.get("status") == "completed"
        and selected.data.get("experiment_status") == cell.status
        and marker.get("manifest_sha256") == sha256_file(cell.manifest_path)
        and marker.get("resolved_config_sha256") == sha256_file(cell.resolved_config_path)
        and counts_are_exact
    )


def _generation_for_search(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "device": config["device"],
        "initial_oracle_evaluations": config["initial_oracle_evaluations"],
        "conditional_iterations": config["iteration_cap"],
        "proposal_batch_size": config["proposal_batch_size"],
        "acquisition_batch_size": config["acquisition_batch_size"],
        "max_oracle_evaluations": config["max_selected_archive_evaluations"],
        "flow_steps": config["flow_steps"],
        "candidate_source": config["candidate_source"],
        "trust_region": config["trust_region"],
    }


def _drop_keys(config: dict[str, Any], keys: set[str]) -> None:
    for key in keys:
        config.pop(key, None)


def _unconditional_config(
    *,
    condition: Mapping[str, Any],
    generation: Mapping[str, Any],
    seed: int,
    cell_dir: Path,
    repo_root: Path,
    status: RunStatus,
) -> dict[str, Any]:
    resolved = _search_config(
        condition=condition,
        generation=generation,
        gp=None,
        acquisition="random",
        kernel=None,
        seed=seed,
        arm="unconditional",
        cell_dir=cell_dir,
        initial_batch_path=None,
        repo_root=repo_root,
        status=status,
    )
    resolved.update(
        {
            "sample_only": True,
            "sample_only_count": int(generation["max_oracle_evaluations"]),
            # The search runner preserves its normal manifest while sampling in
            # bounded chunks; this avoids materialising 2000 L256 chains at once.
            "sample_only_batch_size": int(generation["unconditional_chunk_size"]),
            "batch_size": int(generation["unconditional_chunk_size"]),
            "initial_oracle_evaluations": int(generation["max_oracle_evaluations"]),
            "num_iterations": 0,
            "active_learning": False,
            "candidate_strategy": "random",
            "trace_level": "summary",
            "write_proposal_diagnostics": False,
        }
    )
    _drop_keys(
        resolved,
        {
            key for key in resolved if key.startswith("al_")
        }
        | _TR_KEYS
        | {"proposal_traces_out", "proposal_diagnostics_out", "trace_record_limit"},
    )
    return resolved


def _iterative_config(
    *,
    condition: Mapping[str, Any],
    generation: Mapping[str, Any],
    gp: Mapping[str, Any],
    arm: Mapping[str, Any],
    seed: int,
    cell_dir: Path,
    initial_batch_path: Path,
    repo_root: Path,
    status: RunStatus,
) -> dict[str, Any]:
    acquisition = str(arm["acquisition"])
    uses_gp = arm["surrogate"] is not None
    resolved = _search_config(
        condition=condition,
        generation=generation,
        gp=(gp if uses_gp else None),
        acquisition=acquisition,
        kernel=(str(arm["surrogate"]) if uses_gp else None),
        seed=seed,
        arm=str(arm["name"]),
        cell_dir=cell_dir,
        initial_batch_path=initial_batch_path,
        repo_root=repo_root,
        status=status,
    )
    resolved.update(
        {
            "num_iterations": int(generation["iteration_cap"]),
            "num_times": int(generation["random_time"]["num_times"]),
            "mixes_per_time": int(generation["random_time"]["mixes_per_time"]),
            "al_initial_batch_matches_buffer_size": False,
            "al_buffer_size": int(gp["buffer_size"]),
            "al_buffer_mode": str(gp["buffer_mode"]),
            "trace_record_limit": 300_000,
        }
    )
    if arm["trust_region"]:
        resolved["candidate_strategy"] = "trust_region"
    else:
        resolved["candidate_strategy"] = "random"
        _drop_keys(resolved, _TR_KEYS)
    if acquisition == "oracle_best":
        resolved["al_acquisition"] = "oracle_best"
        resolved["al_oracle_best_budget_mode"] = str(arm["oracle_budget_mode"])
    return resolved


def _estimate_seconds(arm: str) -> float:
    if arm == "unconditional":
        return 2.0 * 60.0
    if arm in {"no_tr_gp", "tr_gp"}:
        # Calibrated with the canonical L256 W32 policy at n=2,000:
        # 123.25 s fit + 13.41 s posterior for one late batch. Integrating
        # over the growing archive gives about 3.8 h/cell; retain headroom.
        return 4.0 * 60.0 * 60.0
    return 45.0 * 60.0


def build_control_cells(
    study: ControlStudyConfig,
    *,
    repo_root: Path | str,
    run_root: Path | None = None,
    status: RunStatus = "canonical",
) -> list[StudyCell]:
    """Expand the sole L256 config into six arms by five seeds."""

    root = Path(repo_root).resolve()
    config = study.raw
    base = Path(run_root) if run_root is not None else root / str(config["run_dir"])
    dataset_base = root / str(study.kernel_study.datasets["run_dir"])
    condition = study.condition
    condition_name = str(config["condition"])
    generation = {
        **_generation_for_search(config["generation"]),
        "unconditional_chunk_size": int(config["generation"]["unconditional_chunk_size"]),
        "iteration_cap": int(config["generation"]["iteration_cap"]),
        "random_time": dict(config["generation"]["random_time"]),
    }
    cells: list[StudyCell] = []
    for arm in config["arms"]:
        arm_name = str(arm["name"])
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            cell_dir = base / condition_name / arm_name / f"seed{seed}"
            if arm_name == "unconditional":
                resolved = _unconditional_config(
                    condition=condition,
                    generation=generation,
                    seed=seed,
                    cell_dir=cell_dir,
                    repo_root=root,
                    status=status,
                )
            else:
                initial = dataset_base / condition_name / "random" / f"seed{seed}" / "initial_batch.jsonl.gz"
                resolved = _iterative_config(
                    condition=condition,
                    generation=generation,
                    gp=config["gp"],
                    arm=arm,
                    seed=seed,
                    cell_dir=cell_dir,
                    initial_batch_path=initial,
                    repo_root=root,
                    status=status,
                )
            cells.append(
                StudyCell(
                    workflow="controls",
                    condition=condition_name,
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
    device: str | None = None,
) -> list[StudyCell]:
    """Reduce isolated smoke cells to an exact, non-publishable oracle budget."""

    bounded: list[StudyCell] = []
    for cell in cells:
        if cell.status != "smoke":
            raise ValueError("control smoke bounds can only be applied to isolated smoke cells")
        resolved = dict(cell.resolved_config)
        if device is not None:
            resolved["device"] = device
        acquisition_batch_size = int(resolved.get("al_oracle_batch_size", 8))
        if cell.arm == "unconditional":
            if selected_archive_evaluations < 1:
                raise ValueError("unconditional smoke budget must be positive")
            resolved.update(
                {
                    "sample_only_count": int(selected_archive_evaluations),
                    "initial_oracle_evaluations": int(selected_archive_evaluations),
                    "max_oracle_evaluations": int(selected_archive_evaluations),
                }
            )
        else:
            initial = int(resolved["initial_oracle_evaluations"])
            remainder = int(selected_archive_evaluations) - initial
            if remainder < acquisition_batch_size or remainder % acquisition_batch_size:
                raise ValueError(
                    "adaptive smoke budget must exceed the initial archive by a positive multiple of K"
                )
            iterations = remainder // acquisition_batch_size
            resolved.update(
                {
                    "max_oracle_evaluations": int(selected_archive_evaluations),
                    "num_iterations": iterations,
                    "al_buffer_size": int(selected_archive_evaluations),
                    "trace_record_limit": iterations * int(resolved["mixes_per_time"]),
                }
            )
        bounded.append(
            replace(
                cell,
                resolved_config=resolved,
                estimated_seconds=min(cell.estimated_seconds, 20.0 * 60.0),
            )
        )
    return bounded


__all__ = [
    "bound_smoke_cells",
    "build_control_cells",
    "control_completion_marker_is_valid",
    "finalize_control_completion",
    "filter_cells",
    "plan_summary",
    "RUNTIME_CALIBRATION",
    "run_cells",
]
