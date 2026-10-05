"""Cell planning and safe execution for dataset and online study workflows."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from genturbo.serialization import atomic_write_json, canonical_json_bytes

from .config import KernelStudyConfigError, StudyConfigs

RunStatus = Literal["canonical", "smoke"]
_EXACT_ONLINE_STUDIES = {
    "ehrlich_controls_v1",
    "ehrlich_sequence_baselines_v2",
}


@dataclass(frozen=True, slots=True)
class StudyCell:
    workflow: str
    condition: str
    seed: int
    arm: str
    run_dir: Path
    resolved_config: dict[str, Any]
    estimated_seconds: float
    status: RunStatus = "canonical"
    study: str = "ehrlich_kernel_v1"

    @property
    def identity(self) -> str:
        return f"{self.workflow}:{self.condition}:{self.arm}:seed{self.seed}"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.run_dir / "resolved_config.json"


def _condition_map(datasets: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["name"]): item for item in datasets["conditions"]}


def _output_value(path: Path, *, repo_root: Path, status: RunStatus) -> str:
    if status == "smoke":
        return str(path)
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:  # pragma: no cover - guarded by canonical root construction
        raise KernelStudyConfigError(f"canonical output path escapes repository: {path}") from exc


def _kernel_overrides(kernel: str | None) -> dict[str, Any]:
    if kernel is None or kernel == "onehot_tanimoto":
        return {"al_feature_space": "ehrlich_onehot"}
    if kernel == "spaced_pair_w32":
        return {
            "al_feature_space": "ehrlich_spaced_pair",
            "al_ehrlich_spaced_ngram_patterns": "local_pair",
            "al_ehrlich_spaced_ngram_window": 32,
        }
    if kernel == "spaced_pair_w48":
        return {
            "al_feature_space": "ehrlich_spaced_pair",
            "al_ehrlich_spaced_ngram_patterns": "local_pair",
            "al_ehrlich_spaced_ngram_window": 48,
        }
    if kernel == "local_spaced_trigram_w16":
        return {
            "al_feature_space": "ehrlich_spaced_trigram",
            "al_ehrlich_spaced_ngram_patterns": "local_trigram",
            "al_ehrlich_spaced_ngram_window": 16,
        }
    if kernel == "pair_w48_trigram_w16_additive":
        return {
            "al_feature_space": "ehrlich_local_spaced_ngram_additive",
            "al_ehrlich_spaced_ngram_orders": [2, 3],
            "al_ehrlich_spaced_ngram_windows": [48, 16],
            "al_ehrlich_spaced_ngram_weights": [0.5, 0.5],
        }
    raise KernelStudyConfigError(f"kernel {kernel!r} is not deployable online")


def _search_config(
    *,
    condition: Mapping[str, Any],
    generation: Mapping[str, Any],
    gp: Mapping[str, Any] | None,
    acquisition: str,
    kernel: str | None,
    seed: int,
    arm: str,
    cell_dir: Path,
    initial_batch_path: Path | None,
    repo_root: Path,
    status: RunStatus,
) -> dict[str, Any]:
    trust = generation["trust_region"]
    samples = cell_dir / "samples.jsonl.gz"
    initial_output = cell_dir / "initial_batch.jsonl.gz"
    resolved: dict[str, Any] = {
        "runner": "interpolant",
        "checkpoint": condition["checkpoint"],
        "config": condition["runtime_config"],
        "device": generation["device"],
        "reward": {
            "name": "ehrlich",
            "instance_path": condition["instance_path"],
            "quantization_override": int(condition["quantization"]),
        },
        "sequence_length": int(condition["sequence_length"]),
        "batch_size": int(generation["acquisition_batch_size"]),
        "initial_oracle_evaluations": int(generation["initial_oracle_evaluations"]),
        "initial_batch_out": _output_value(initial_output, repo_root=repo_root, status=status),
        "initial_batch_advance_rng": initial_batch_path is not None,
        "initial_batch_rescore_with_reward": initial_batch_path is not None,
        "num_iterations": int(generation["conditional_iterations"]),
        "max_oracle_evaluations": int(generation["max_oracle_evaluations"]),
        "flow_steps": int(generation["flow_steps"]),
        "max_time": 1.0,
        "num_times": 1,
        "mixes_per_time": int(generation["proposal_batch_size"]),
        "candidate_strategy": "trust_region",
        "candidate_source": generation["candidate_source"],
        "candidate_k": 1,
        "mixing_mode": "random",
        "temperature": 1.0,
        "stop_when_filled": False,
        "active_learning": True,
        "al_oracle_batch_size": int(generation["acquisition_batch_size"]),
        "al_sequential_selection": False,
        "al_local_penalization": False,
        "al_buffer_size": int(generation["max_oracle_evaluations"]),
        "al_buffer_mode": "recent",
        "al_acquisition": ("expected_improvement" if acquisition == "marginal_expected_improvement" else "random"),
        "tr_type": trust["type"],
        "tr_domain": "edits",
        "tr_sampling": trust["sampling"],
        "tr_edits": int(trust["edits_init"]),
        "edits_min": int(trust["edits_min"]),
        "edits_init": int(trust["edits_init"]),
        "edits_max": int(trust["edits_max"]),
        "region_expand_scalar": float(trust["expand"]),
        "region_shrink_scalar": float(trust["shrink"]),
        "tau_success": int(trust["success_patience"]),
        "tau_failure": int(trust["failure_patience"]),
        "tr_eps": 1e-6,
        "tr_log_every": 1,
        "trace_level": "full",
        "trace_record_limit": 100_000,
        "compress_jsonl": True,
        "write_proposal_diagnostics": True,
        "proposal_traces_out": _output_value(cell_dir / "proposal_traces.jsonl.gz", repo_root=repo_root, status=status),
        "proposal_diagnostics_out": _output_value(
            cell_dir / "proposal_diagnostics.jsonl.gz", repo_root=repo_root, status=status
        ),
        "samples_out": _output_value(samples, repo_root=repo_root, status=status),
        "run_manifest_out": _output_value(cell_dir / "run_manifest.json", repo_root=repo_root, status=status),
        "plot_label": arm,
        # The retained cohort analyzer reads the stable label from metadata;
        # keep the top-level field too for existing search logs.
        "metadata": {"plot_label": arm, "experiment_status": status},
        "seed": int(seed),
    }
    if initial_batch_path is not None:
        resolved["initial_batch_path"] = _output_value(initial_batch_path, repo_root=repo_root, status=status)
    resolved.update(_kernel_overrides(kernel))
    if gp is not None:
        resolved.update(
            {
                "al_gp_device": gp["device"],
                "al_gp_dtype": gp["dtype"],
                "al_gp_jitter": float(gp["jitter"]),
                # JSON null is intentional: it selects fitted likelihood noise.
                "al_gp_noise_variance": None,
                "al_gp_outputscale": 1.0,
                "al_gp_learn_outputscale": False,
                "al_gp_fit_mean_and_noise_only": True,
                "al_gp_learn_lengthscale": False,
            }
        )
    return resolved


def _fallback_estimate(workflow: str, condition: str, arm: str) -> float:
    length_factor = 2.0 if condition == "l256_q4_k8" else 1.0
    if workflow == "dataset":
        return 12.0 * 60.0 * length_factor
    if arm == "random":
        return 35.0 * 60.0 * length_factor
    return 150.0 * 60.0 * length_factor


def build_dataset_cells(
    study: StudyConfigs,
    *,
    repo_root: Path | str,
    run_root: Path | None = None,
    status: RunStatus = "canonical",
) -> list[StudyCell]:
    root = Path(repo_root).resolve()
    config = study.datasets
    base = Path(run_root) if run_root is not None else root / config["run_dir"]
    conditions = _condition_map(config)
    cells: list[StudyCell] = []
    for condition_name in config["conditions"]:
        name = str(condition_name["name"])
        for seed in config["seeds"]:
            cell_dir = base / name / "random" / f"seed{int(seed)}"
            resolved = _search_config(
                condition=conditions[name],
                generation=config["generation"],
                gp=None,
                acquisition="random",
                kernel=None,
                seed=int(seed),
                arm="random",
                cell_dir=cell_dir,
                initial_batch_path=None,
                repo_root=root,
                status=status,
            )
            cells.append(
                StudyCell(
                    "dataset",
                    name,
                    int(seed),
                    "random",
                    cell_dir,
                    resolved,
                    _fallback_estimate("dataset", name, "random"),
                    status,
                )
            )
    return cells


def build_online_cells(
    study: StudyConfigs,
    *,
    repo_root: Path | str,
    run_root: Path | None = None,
    status: RunStatus = "canonical",
) -> list[StudyCell]:
    root = Path(repo_root).resolve()
    config = study.online
    base = Path(run_root) if run_root is not None else root / config["run_dir"]
    dataset_base = root / study.datasets["run_dir"]
    conditions = _condition_map(study.datasets)
    cells: list[StudyCell] = []
    arms = config["arms"]
    for condition_name in config["conditions"]:
        for arm in arms:
            for seed in config["seeds"]:
                name = str(condition_name)
                arm_name = str(arm["name"])
                cell_dir = base / name / arm_name / f"seed{int(seed)}"
                initial = dataset_base / name / "random" / f"seed{int(seed)}" / "initial_batch.jsonl.gz"
                resolved = _search_config(
                    condition=conditions[name],
                    generation=config["generation"],
                    gp=config["gp"] if arm["kernel"] is not None else None,
                    acquisition=str(arm["acquisition"]),
                    kernel=arm["kernel"],
                    seed=int(seed),
                    arm=arm_name,
                    cell_dir=cell_dir,
                    initial_batch_path=initial,
                    repo_root=root,
                    status=status,
                )
                cells.append(
                    StudyCell(
                        "online",
                        name,
                        int(seed),
                        arm_name,
                        cell_dir,
                        resolved,
                        _fallback_estimate("online", name, arm_name),
                        status,
                    )
                )
    return cells


def cell_state(cell: StudyCell) -> str:
    """Return pending, partial, failed, or completed without changing a cell."""

    if (cell.run_dir / "failure.json").is_file():
        return "failed"
    if cell.manifest_path.is_file():
        try:
            manifest = json.loads(cell.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "partial"
        status = str(manifest.get("status", ""))
        if status == "completed" and cell.study in _EXACT_ONLINE_STUDIES:
            try:
                cell_record = json.loads((cell.run_dir / "cell.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return "incomplete"
            expected_config_hash = hashlib.sha256(canonical_json_bytes(cell.resolved_config)).hexdigest()
            if not isinstance(cell_record, dict) or cell_record.get("resolved_config_sha256") != expected_config_hash:
                return "incomplete"
            from genturbo.control_study.runner import control_completion_marker_is_valid

            return "completed" if control_completion_marker_is_valid(cell) else "incomplete"
        if status in {"completed", "failed"}:
            return status
        return "partial"
    if cell.run_dir.exists() and any(cell.run_dir.iterdir()):
        return "partial"
    return "pending"


def filter_cells(
    cells: Iterable[StudyCell],
    *,
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    arms: Sequence[str] | None = None,
    max_cells: int | None = None,
) -> list[StudyCell]:
    def validate_filter(
        requested: Sequence[Any] | None,
        available: set[Any],
        *,
        label: str,
        convert: Any,
    ) -> set[Any] | None:
        if requested is None:
            return None
        normalized = [convert(value) for value in requested]
        if not normalized:
            raise KernelStudyConfigError(f"{label} filter must not be empty")
        if len(normalized) != len(set(normalized)):
            raise KernelStudyConfigError(f"{label} filter contains duplicates")
        unknown = sorted(set(normalized) - available)
        if unknown:
            raise KernelStudyConfigError(f"unknown {label} filter values: {', '.join(map(str, unknown))}")
        return set(normalized)

    condition_set = validate_filter(
        conditions,
        {cell.condition for cell in cells},
        label="condition",
        convert=str,
    )
    seed_set = validate_filter(
        seeds,
        {cell.seed for cell in cells},
        label="seed",
        convert=int,
    )
    arm_set = validate_filter(
        arms,
        {cell.arm for cell in cells},
        label="arm",
        convert=str,
    )
    selected = [
        cell
        for cell in cells
        if (condition_set is None or cell.condition in condition_set)
        and (seed_set is None or cell.seed in seed_set)
        and (arm_set is None or cell.arm in arm_set)
    ]
    if max_cells is not None:
        if max_cells < 1:
            raise KernelStudyConfigError("max_cells must be positive")
        selected = selected[:max_cells]
    if not selected:
        raise KernelStudyConfigError("cell filters selected no work")
    return selected


def plan_summary(cells: Sequence[StudyCell]) -> dict[str, Any]:
    cell_states = [(cell, cell_state(cell)) for cell in cells]
    states = Counter(state for _cell, state in cell_states)
    estimates = sum(cell.estimated_seconds for cell, state in cell_states if state != "completed")
    by_condition = Counter(cell.condition for cell in cells)
    by_arm = Counter(cell.arm for cell in cells)
    return {
        "status": cells[0].status if cells else "canonical",
        "cell_count": len(cells),
        "states": dict(sorted(states.items())),
        "cells_by_condition": dict(sorted(by_condition.items())),
        "cells_by_arm": dict(sorted(by_arm.items())),
        "estimated_remaining_seconds": estimates,
        "estimated_remaining_hours": estimates / 3600.0,
        "estimate_kind": "conservative fallback; review before launch",
        "cells": [
            {
                "identity": cell.identity,
                "state": state,
                "run_dir": str(cell.run_dir),
                "resolved_config": str(cell.resolved_config_path),
                "estimated_seconds": cell.estimated_seconds,
            }
            for cell, state in cell_states
        ],
    }


def _validate_initial_archive(cell: StudyCell) -> None:
    path_value = cell.resolved_config.get("initial_batch_path")
    if path_value is None:
        return
    path = Path(str(path_value))
    if not path.is_absolute():
        # Canonical paths are repository-relative and the parent of runs/ is the repo root.
        parents = [parent for parent in cell.run_dir.parents if parent.name == "runs"]
        repo_root = parents[-1].parent if parents else Path.cwd()
        path = repo_root / path
    if not path.is_file():
        raise KernelStudyConfigError(
            f"{cell.identity} requires dataset initial archive {path}; run the dataset workflow first"
        )


def _verify_completed_cell(cell: StudyCell, *, repo_root: Path | None = None) -> None:
    """Prove that a zero-exit subprocess produced one checksum-valid completed manifest."""

    from genturbo.artifacts import sha256_file
    from genturbo.kernel_study.online_analysis import EhrlichSweepError, select_manifest

    try:
        selected = select_manifest(cell.run_dir)
    except EhrlichSweepError as exc:
        raise KernelStudyConfigError(f"{cell.identity} did not produce a verifiable run manifest: {exc}") from exc
    if selected.path.resolve() != cell.manifest_path.resolve() or selected.data.get("status") != "completed":
        raise KernelStudyConfigError(f"{cell.identity} did not produce its canonical completed manifest")
    input_hashes = selected.data.get("input_hashes")
    search_hash = input_hashes.get("search_config") if isinstance(input_hashes, Mapping) else None
    if search_hash != sha256_file(cell.resolved_config_path):
        raise KernelStudyConfigError(f"{cell.identity} manifest does not verify its resolved search config")
    if cell.study in _EXACT_ONLINE_STUDIES:
        from genturbo.control_study.runner import finalize_control_completion

        finalize_control_completion(cell, repo_root=(repo_root or Path.cwd()).resolve())


def _record_failure(cell: StudyCell, *, reason: str, returncode: int) -> None:
    atomic_write_json(
        cell.run_dir / "failure.json",
        {
            "schema_version": 1,
            "study": cell.study,
            "identity": cell.identity,
            "status": "failed",
            "returncode": int(returncode),
            "reason": reason,
        },
    )


def _quarantine_retryable_control_cell(cell: StudyCell, *, state: str) -> Path:
    """Move a finished failed/incomplete control attempt aside without deleting it."""

    if cell.study not in _EXACT_ONLINE_STUDIES or state not in {"failed", "incomplete"}:
        raise KernelStudyConfigError(f"{cell.identity} is not a retryable completed control attempt")
    if cell.run_dir.is_symlink() or not cell.run_dir.is_dir():
        raise KernelStudyConfigError(f"cannot preserve retryable cell directory {cell.run_dir}")
    attempts = cell.run_dir.parent / "_attempts"
    if attempts.is_symlink() or (attempts.exists() and not attempts.is_dir()):
        raise KernelStudyConfigError(f"control attempt archive must be a regular directory: {attempts}")
    attempts.mkdir(parents=False, exist_ok=True)
    attempt = 1
    while True:
        destination = attempts / f"{cell.run_dir.name}-{state}-{attempt:03d}"
        if not destination.exists():
            break
        attempt += 1
    cell.run_dir.rename(destination)
    return destination


def run_cells(
    cells: Sequence[StudyCell],
    *,
    repo_root: Path | str,
    python: Path | str | None = None,
) -> dict[str, Any]:
    """Run selected fresh cells; completed cells are immutable and skipped."""

    root = Path(repo_root).resolve()
    outcomes: list[dict[str, Any]] = []
    for cell in cells:
        state = cell_state(cell)
        if state == "completed":
            outcomes.append({"identity": cell.identity, "outcome": "skipped_completed"})
            continue
        if cell.study in _EXACT_ONLINE_STUDIES and state in {"failed", "incomplete"}:
            preserved = _quarantine_retryable_control_cell(cell, state=state)
            outcomes.append(
                {
                    "identity": cell.identity,
                    "outcome": "preserved_retryable_attempt",
                    "previous_state": state,
                    "preserved_at": str(preserved),
                }
            )
            state = "pending"
        if state != "pending":
            raise KernelStudyConfigError(f"refusing to overwrite {state} cell {cell.identity} at {cell.run_dir}")
        _validate_initial_archive(cell)
        cell.run_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(cell.resolved_config_path, cell.resolved_config)
        cell_record = {
            "schema_version": 1,
            "study": cell.study,
            "identity": cell.identity,
            "status": cell.status,
            "resolved_config_sha256": hashlib.sha256(canonical_json_bytes(cell.resolved_config)).hexdigest(),
        }
        atomic_write_json(cell.run_dir / "cell.json", cell_record)
        runner_name = str(cell.resolved_config.get("runner", "interpolant"))
        is_lambo2 = runner_name == "lambo2"
        executable = str(python or (root / ".venv-lambo2/bin/python" if is_lambo2 else sys.executable))
        if is_lambo2 and not Path(executable).is_file():
            raise KernelStudyConfigError(
                "LaMBO-2 requires the isolated .venv-lambo2 environment; see environments/lambo2/uv.lock"
            )
        module = {
            "interpolant": "runners.run_ctmc_interpolant_search",
            "lambo2": "genturbo.sequence_baseline_study.lambo2",
            "holo_ga": "genturbo.sequence_baseline_study.holo",
        }.get(runner_name)
        if module is None:
            raise KernelStudyConfigError(f"unsupported cell runner {runner_name!r} for {cell.identity}")
        command = [executable, "-m", module]
        if is_lambo2:
            command.extend(["--repo-root", str(root), "online"])
        command.extend(["--search-config", str(cell.resolved_config_path)])
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root / "src"), environment.get("PYTHONPATH"))))
        if is_lambo2:
            assert environment is not None
            source_path = str(root / "src")
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (source_path, environment.get("PYTHONPATH")))
            )
            environment.setdefault("GENTURBO_DEPENDENCY_PROFILE", "lambo2-py310")
        with (cell.run_dir / "runner.log").open("wb") as log:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            reason = f"subprocess exited with code {completed.returncode}"
            _record_failure(cell, reason=reason, returncode=completed.returncode)
            if cell.study in _EXACT_ONLINE_STUDIES:
                failure_dir = _quarantine_retryable_control_cell(cell, state="failed")
                detail = f"preserved at {failure_dir}"
            else:
                detail = f"see {cell.run_dir / 'runner.log'}"
            outcomes.append({"identity": cell.identity, "outcome": "failed", "returncode": completed.returncode})
            raise KernelStudyConfigError(f"{cell.identity} failed with exit code {completed.returncode}; {detail}")
        try:
            _verify_completed_cell(cell, repo_root=root)
        except KernelStudyConfigError as exc:
            _record_failure(cell, reason=str(exc), returncode=completed.returncode)
            if cell.study in _EXACT_ONLINE_STUDIES:
                failure_dir = _quarantine_retryable_control_cell(cell, state="incomplete")
            outcomes.append({"identity": cell.identity, "outcome": "failed", "returncode": completed.returncode})
            if cell.study in _EXACT_ONLINE_STUDIES:
                raise KernelStudyConfigError(f"{exc}; preserved at {failure_dir}") from exc
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            if cell.study not in _EXACT_ONLINE_STUDIES:
                raise
            reason = f"scientific completion validation failed: {exc}"
            _record_failure(cell, reason=reason, returncode=completed.returncode)
            failure_dir = _quarantine_retryable_control_cell(cell, state="incomplete")
            outcomes.append({"identity": cell.identity, "outcome": "failed", "returncode": completed.returncode})
            raise KernelStudyConfigError(
                f"{cell.identity} did not satisfy scientific completion: {exc}; preserved at {failure_dir}"
            ) from exc
        outcomes.append({"identity": cell.identity, "outcome": "completed", "returncode": completed.returncode})
    return {"cells": outcomes, "completed": sum(row["outcome"] == "completed" for row in outcomes)}


__all__ = [
    "StudyCell",
    "build_dataset_cells",
    "build_online_cells",
    "cell_state",
    "filter_cells",
    "plan_summary",
    "run_cells",
]
