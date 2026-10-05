"""Exact cell planning, state accounting, and execution for protein_design_v1."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from genturbo.artifacts import ArtifactError, sha256_file
from genturbo.io import iter_jsonl
from genturbo.kernel_study.runner import RunStatus, StudyCell, cell_state, run_cells
from genturbo.serialization import atomic_write_json, canonical_json_bytes

from ..configuration import load_config
from .compatibility import ProteinCompatibilityError, validate_retained_compatibility
from .config import (
    DIAGNOSTIC_METHODS,
    LEGACY_PRIMARY_METHODS,
    NEW_CELL_COUNT,
    PRIMARY_METHODS,
    STUDY,
    TASKS,
    TOTAL_CELL_COUNT,
    ProteinStudyConfig,
    ProteinStudyConfigError,
)

_NATIVE_ADAPTIVE_ARMS = frozenset({"rt_random", "rt_gp", "tr_random", "tr_gp"})
_LAMBO_ONLINE_INITIALIZATION = "online_archive_random"
_LAMBO_COMPLETE_HISTORY = "complete_paid_history"
_HARD5_V2_PROTOCOL = "hard5_online_full_history_v2"
_HARD5_V2_GP_ARMS = frozenset({"rt_gp", "tr_gp", "adalead", "bo_evo"})
_ADALEAD_EVODIFF_ARM = "adalead_evodiff"
_ADALEAD_EVODIFF_PROTOCOL = "hard5_adalead_evodiff_full_history_v1"
_LAMBO_PRETRAINING_TOP_LEVEL_KEYS = frozenset(
    {
        "checkpoint",
        "pretraining_manifest",
        "checkpoint_validation_samples",
        "checkpoint_validation_manifest",
    }
)
_LAMBO_PRETRAINING_NESTED_KEYS = frozenset(
    {
        "checkpoint",
        "pretraining_manifest",
        "checkpoint_validation_samples",
        "checkpoint_validation_manifest",
        "corpus",
        "expected_transferred_tensors",
        "pretraining_scores",
        "pretraining_attestation",
    }
)


@dataclass(frozen=True, slots=True)
class ProteinStudyCell(StudyCell):
    """One declared protein cell, including retained-artifact provenance."""

    cohort: str = "primary"
    runtime_class: str = "evodiff"
    launch_required: bool = True
    launch_tranche: int | None = None
    legacy_source_dir: Path | None = None
    legacy_import_manifest: Path | None = None

    @property
    def identity(self) -> str:
        return f"{self.condition}:{self.arm}:seed{self.seed}"

    @property
    def completion_marker_path(self) -> Path:
        return Path(self.run_dir) / "scientific_completion.json"


def _output(path: Path, *, repo_root: Path, status: RunStatus) -> str:
    if status == "smoke":
        return str(path.resolve())
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ProteinStudyConfigError(f"canonical protein output escapes repository: {path}") from exc


def _declared_path(study: ProteinStudyConfig, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else study.repo_root / candidate).resolve()
    try:
        path.relative_to(study.repo_root)
    except ValueError as exc:
        raise ProteinStudyConfigError(f"protein declaration escapes repository: {value}") from exc
    return path


def _checkpoint_path(task: Mapping[str, Any]) -> str:
    """Translate the tracked checkpoint metadata path to the external weight path."""

    metadata = Path(str(task["checkpoint"]))
    if metadata.suffix != ".json" or metadata.with_suffix("").suffix != ".pt":
        raise ProteinStudyConfigError(f"protein checkpoint metadata must end in .pt.json: {metadata}")
    return metadata.with_suffix("").as_posix()


def _load_template(study: ProteinStudyConfig, *, task: str, method: str) -> dict[str, Any]:
    lookup_method = method if method in (*LEGACY_PRIMARY_METHODS, *DIAGNOSTIC_METHODS) else "rt_gp"
    condition = str(study.raw["legacy"]["condition_map"][task][lookup_method])
    path = study.repo_root / f"configs/search/protein/gfp_aav_baselines_2000/{condition}.json"
    try:
        value = load_config(path)
    except (OSError, ValueError, TypeError) as exc:
        raise ProteinStudyConfigError(f"cannot read retained protein template {path}: {exc}") from exc
    return value


def _reward_spec(study: ProteinStudyConfig, task: str, *, device: str) -> dict[str, Any]:
    declared = study.tasks[task]
    length = int(declared["sequence_length"])
    return {
        "name": "gfp_aav_cnn_oracle",
        "protein": str(declared["protein"]),
        "benchmark_root": str(study.raw["benchmark_root"]),
        "normalize": True,
        "min_len": length,
        "max_len": length,
        "sequence_length": length,
        "device": str(device),
    }


def _reward_adapter(study: ProteinStudyConfig, task: str, *, device: str) -> dict[str, Any]:
    reward = _reward_spec(study, task, device=device)
    kwargs = {key: value for key, value in reward.items() if key != "name"}
    return {
        "module": "adapters.protein.reward",
        "class": "ProteinGFPAVCNNOracleReward",
        "kwargs": kwargs,
    }


def _common_outputs(
    resolved: dict[str, Any],
    *,
    cell_dir: Path,
    repo_root: Path,
    status: RunStatus,
    arm: str,
    seed: int,
) -> None:
    resolved.update(
        {
            "initial_batch_out": _output(cell_dir / "initial_batch.jsonl.gz", repo_root=repo_root, status=status),
            "samples_out": _output(cell_dir / "samples.jsonl.gz", repo_root=repo_root, status=status),
            "proposal_diagnostics_out": _output(
                cell_dir / "proposal_diagnostics.jsonl.gz", repo_root=repo_root, status=status
            ),
            "proposal_traces_out": _output(cell_dir / "proposal_traces.jsonl.gz", repo_root=repo_root, status=status),
            "run_manifest_out": _output(cell_dir / "run_manifest.json", repo_root=repo_root, status=status),
            "trace_level": "full",
            "trace_record_limit": 100_000,
            "compress_jsonl": True,
            "write_proposal_diagnostics": True,
            "plot_label": arm,
            "metadata": {
                "runner": str(resolved.get("runner", "interpolant")),
                "plot_label": arm,
                "experiment_status": status,
            },
            "seed": int(seed),
        }
    )


def _gp_policy(study: ProteinStudyConfig) -> dict[str, Any]:
    gp = study.raw["gp"]
    return {
        "al_feature_space": "onehot_tanimoto",
        "al_gp_device": str(gp["device"]),
        "al_gp_dtype": str(gp["dtype"]),
        "al_gp_jitter": float(gp["jitter"]),
        "al_gp_noise_variance": None,
        "al_gp_outputscale": 1.0,
        "al_gp_learn_outputscale": False,
        "al_gp_fit_mean_and_noise_only": True,
        "al_gp_learn_lengthscale": False,
        "al_gp_hyperparameter_refit_interval": 1,
        "al_buffer_size": int(gp["buffer_size"]),
        "al_buffer_mode": str(gp["buffer_mode"]),
    }


def _interpolant_config(
    study: ProteinStudyConfig,
    *,
    task: str,
    method: str,
    seed: int,
    cell_dir: Path,
    repo_root: Path,
    status: RunStatus,
) -> dict[str, Any]:
    declared_task = study.tasks[task]
    method_config = study.methods[method]
    budget = study.raw["budget"]
    resolved = copy.deepcopy(_load_template(study, task=task, method=method))
    resolved.update(
        {
            "runner": "interpolant",
            "checkpoint": _checkpoint_path(declared_task),
            "config": str(declared_task["runtime_config"]),
            "benchmark_task": task,
            "protein_objective": task,
            "protein_min_len": int(declared_task["sequence_length"]),
            "protein_max_len": int(declared_task["sequence_length"]),
            "reward": _reward_spec(
                study,
                task,
                device=("cpu" if method in {"adalead", "bo_evo"} else "cuda"),
            ),
            "max_oracle_evaluations": int(budget["total_oracle_calls"]),
        }
    )
    if method == "unconditional":
        resolved.update(
            {
                "batch_size": 32,
                "initial_oracle_evaluations": 0,
                "num_iterations": 0,
                "active_learning": False,
                "sample_only": True,
                "sample_only_count": int(budget["unconditional_samples"]),
                "candidate_strategy": "random",
            }
        )
        for key in (
            "initial_batch_path",
            "initial_batch_advance_rng",
            "initial_batch_rescore_with_reward",
            "proposal_operator",
        ):
            resolved.pop(key, None)
    else:
        resolved.update(
            {
                "batch_size": int(budget["acquisition_batch_size"]),
                "initial_oracle_evaluations": int(budget["initial_archive_calls"]),
                "initial_batch_path": str(declared_task["initial_archive"]),
                "initial_batch_advance_rng": method in LEGACY_PRIMARY_METHODS,
                "initial_batch_rescore_with_reward": False,
                "num_iterations": int(budget["post_initial_batches"]),
                "active_learning": True,
                "al_oracle_batch_size": int(budget["acquisition_batch_size"]),
            }
        )
    if method in {"rt_gp", "tr_gp", "myopic_oracle", "adalead", "bo_evo"}:
        resolved.update(_gp_policy(study))
        resolved["al_onehot_max_len"] = int(declared_task["sequence_length"])
    if method in {"rt_random", "tr_random"}:
        resolved["al_acquisition"] = "random"
    if method in {"rt_random", "rt_gp"}:
        resolved["candidate_strategy"] = "random"
        for key in (
            "tr_type",
            "tr_sampling",
            "tr_domain",
            "time_min",
            "time_max",
            "tr_mean",
            "region_expand_scalar",
            "region_shrink_scalar",
            "tau_success",
            "tau_failure",
        ):
            resolved.pop(key, None)
    if method in {"tr_random", "tr_gp", "myopic_oracle"}:
        trust = declared_task["trust_region"]
        resolved.update(
            {
                "candidate_strategy": "trust_region",
                "tr_type": "ScalarTrustRegion",
                "tr_sampling": "fixed_budget",
                "tr_domain": "time",
                "time_min": float(trust["time_min"]),
                "time_max": float(trust["time_max"]),
                "tr_mean": float(trust["time_init"]),
                "region_expand_scalar": float(trust["expand"]),
                "region_shrink_scalar": float(trust["shrink"]),
                "tau_success": int(trust["success_patience"]),
                "tau_failure": int(trust["failure_patience"]),
            }
        )
    if method in {"rt_gp", "tr_gp"}:
        resolved["al_acquisition"] = "batched_expected_improvement"
    if method == "myopic_oracle":
        resolved["al_acquisition"] = "oracle_best"
        resolved["al_oracle_best_budget_mode"] = "selected_archive"
    if method == "adalead":
        resolved.update(
            {
                "device": "cpu",
                "proposal_operator": "adalead_official",
                "candidate_strategy": "random",
                "num_times": 1,
                "mixes_per_time": int(method_config["paid_batch_size"]),
                "baseline_fitness_threshold": float(method_config["threshold"]),
                "baseline_adalead_surrogate_queries": int(
                    method_config["surrogate_queries_per_batch"]
                ),
                "baseline_adalead_paid_batch_size": int(method_config["paid_batch_size"]),
                "baseline_adalead_mu": float(method_config["mu"]),
                "baseline_adalead_eval_batch_size": int(method_config["eval_batch_size"]),
                "baseline_adalead_rho": int(method_config["rho"]),
                "baseline_adalead_recombination_rate": float(method_config["recombination_rate"]),
                "baseline_adalead_alphabet": str(method_config["alphabet"]),
                "baseline_adalead_rng_backend": str(method_config["rng_backend"]),
                # The official rollout already returns its top predicted
                # sequences.  Random selection of the complete eight-row output
                # prevents a second GP pass from exceeding the 128-query cap.
                "al_acquisition": "random",
            }
        )
    if method == "bo_evo":
        resolved.update(
            {
                "device": "cpu",
                "proposal_operator": "bo_evo",
                "candidate_strategy": "random",
                "mixes_per_time": int(method_config["proposal_pool_size"]),
                "baseline_parent_temperature": float(method_config["parent_temperature"]),
                "baseline_ucb_beta": float(method_config["ucb_coefficient"]),
                "baseline_children_per_selection": int(method_config["children_per_selection"]),
                "al_acquisition": "random",
            }
        )
    _common_outputs(
        resolved,
        cell_dir=cell_dir,
        repo_root=repo_root,
        status=status,
        arm=method,
        seed=seed,
    )
    if method in LEGACY_PRIMARY_METHODS:
        try:
            provenance = validate_retained_compatibility(study, resolved, task=task, method=method)
        except ProteinCompatibilityError as exc:
            raise ProteinStudyConfigError(str(exc)) from exc
        resolved["metadata"].update(provenance)
    return resolved


def _holo_config(
    study: ProteinStudyConfig,
    *,
    task: str,
    seed: int,
    cell_dir: Path,
    repo_root: Path,
    status: RunStatus,
) -> dict[str, Any]:
    declared_task = study.tasks[task]
    method = study.methods["holo_ga"]
    budget = study.raw["budget"]
    length = int(declared_task["sequence_length"])
    resolved: dict[str, Any] = {
        "runner": "holo_ga",
        "config": study.path.relative_to(repo_root).as_posix(),
        "device": "cpu",
        "sequence_length": length,
        "reward": _reward_spec(study, task, device="cpu"),
        "reward_adapter": _reward_adapter(study, task, device="cpu"),
        "batch_size": int(method["generation_size"]),
        "initial_oracle_evaluations": int(budget["initial_archive_calls"]),
        "initial_batch_path": str(declared_task["initial_archive"]),
        "initial_batch_advance_rng": False,
        "initial_batch_rescore_with_reward": False,
        "num_iterations": int(method["generations"]),
        "max_oracle_evaluations": int(budget["total_oracle_calls"]),
        "active_learning": True,
        "al_oracle_batch_size": int(method["generation_size"]),
        "proposal_operator": "holo_discrete_evolution",
        "oracle_budget_mode": "physical",
        "holo": {
            "task_name": task,
            "alphabet": str(study.raw["lambo2"]["alphabet"]),
            "population_size": int(method["generation_size"]),
            "generation_count": int(method["generations"]),
            "mutation_prob": 1.0 / length,
            "recombine_prob": 1.0 / length,
            "survival_quantile": float(method["survival_quantile"]),
            "vocab_size": 20,
            "prepaid_initial_archive": True,
            "device": "cpu",
            "upstream": dict(method["upstream"]),
        },
    }
    _common_outputs(
        resolved,
        cell_dir=cell_dir,
        repo_root=repo_root,
        status=status,
        arm="holo_ga",
        seed=seed,
    )
    return resolved


def _lambo_config(
    study: ProteinStudyConfig,
    *,
    task: str,
    seed: int,
    cell_dir: Path,
    repo_root: Path,
    status: RunStatus,
) -> dict[str, Any]:
    declared_task = study.tasks[task]
    method = study.methods["lambo2"]
    lambo = study.raw["lambo2"]
    online = lambo["online"]
    task_paths = lambo["tasks"][task]
    budget = study.raw["budget"]
    length = int(declared_task["sequence_length"])
    corpus_dir = Path(str(task_paths["corpus_dir"]))
    resolved: dict[str, Any] = {
        "runner": "lambo2",
        "config": study.path.relative_to(repo_root).as_posix(),
        "checkpoint": str(task_paths["checkpoint"]),
        "pretraining_manifest": str(task_paths["manifest"]),
        "checkpoint_validation_samples": str(task_paths["validation_samples"]),
        "checkpoint_validation_manifest": str(task_paths["validation_manifest"]),
        "lambo_requirements": str(lambo["requirements"]),
        "device": "cuda",
        "sequence_length": length,
        "reward": _reward_spec(study, task, device="cuda"),
        "batch_size": int(budget["acquisition_batch_size"]),
        "initial_oracle_evaluations": int(budget["initial_archive_calls"]),
        "initial_batch_path": str(declared_task["initial_archive"]),
        "initial_batch_advance_rng": False,
        "initial_batch_rescore_with_reward": False,
        "num_iterations": int(budget["post_initial_batches"]),
        "max_oracle_evaluations": int(budget["total_oracle_calls"]),
        "active_learning": True,
        "al_oracle_batch_size": int(budget["acquisition_batch_size"]),
        "al_acquisition": "graph_nei",
        "al_feature_space": "lambo2_partial_deep_ensemble",
        "proposal_operator": "guided_discrete_diffusion",
        "oracle_budget_mode": "physical",
        "lambo2": {
            "alphabet": str(lambo["alphabet"]),
            "task_name": task,
            "corpus": {
                "train": (corpus_dir / "train.csv").as_posix(),
                "validation": (corpus_dir / "validation.csv").as_posix(),
                "manifest": (corpus_dir / "corpus_manifest.json").as_posix(),
            },
            "checkpoint": str(task_paths["checkpoint"]),
            "pretraining_manifest": str(task_paths["manifest"]),
            "checkpoint_validation_samples": str(task_paths["validation_samples"]),
            "checkpoint_validation_manifest": str(task_paths["validation_manifest"]),
            "reward_adapter": _reward_adapter(study, task, device="cuda"),
            "direct_designs_per_batch": int(method["direct_designs_per_batch"]),
            "objective_refit_epochs_per_round": int(method["objective_refit_epochs_per_round"]),
            "pretraining_scores": "forbidden",
            "pretraining_attestation": {
                "oracle_labels_seen": 0,
                "fitness_columns_visible": False,
            },
            "device": "cuda",
            "training_batch_size": int(online["training_batch_size"]),
            "initial_training_epochs": int(online["initial_training_epochs"]),
            "retraining_epochs": int(online["retraining_epochs"]),
            "num_steps": int(online["num_steps"]),
            "mutations_per_step": int(online["mutations_per_step"]),
            "max_guidance_updates": int(online["max_guidance_updates"]),
            "expected_transferred_tensors": int(online["expected_transferred_tensors"]),
        },
    }
    _common_outputs(
        resolved,
        cell_dir=cell_dir,
        repo_root=repo_root,
        status=status,
        arm="lambo2",
        seed=seed,
    )
    return resolved


def _estimate_seconds(task: str, method: str) -> float:
    length_factor = 4.0 if task.startswith("gfp_") else 1.0
    if method == "unconditional":
        return 45.0 * 60.0 * length_factor
    if method in {"rt_random", "tr_random", "holo_ga"}:
        return 90.0 * 60.0 * length_factor
    if method in {"adalead", "bo_evo"}:
        return 4.0 * 60.0 * 60.0 * length_factor
    if method == "lambo2":
        return 8.0 * 60.0 * 60.0 * length_factor
    return 8.0 * 60.0 * 60.0 * length_factor


def _launch_mapping(study: ProteinStudyConfig) -> dict[str, int]:
    known_missing = [str(value) for value in study.raw["legacy"]["known_missing_primary_cells"]]
    mapping: dict[str, int] = {}
    for declaration in study.raw["launch_tranches"]:
        tranche = int(declaration["tranche"])
        identities = [
            f"{task}:{method}:seed{int(seed)}"
            for task in TASKS
            for method in declaration["methods"]
            for seed in declaration["seeds"]
        ]
        if declaration.get("include_known_missing_legacy_cells") is True:
            identities.extend(known_missing)
        for identity in identities:
            if identity in mapping:
                raise ProteinStudyConfigError(f"duplicate launch identity {identity}")
            mapping[identity] = tranche
    if len(mapping) != NEW_CELL_COUNT:
        raise ProteinStudyConfigError(f"launch map has {len(mapping)} cells; expected {NEW_CELL_COUNT}")
    return mapping


def build_protein_cells(
    study: ProteinStudyConfig,
    *,
    repo_root: Path | str,
    run_root: Path | str | None = None,
    status: RunStatus = "canonical",
) -> list[ProteinStudyCell]:
    """Build all 760 declared cells in task/method/seed order."""

    root = Path(repo_root).expanduser().resolve()
    if root != study.repo_root:
        raise ProteinStudyConfigError(f"repo_root {root} differs from loaded study root {study.repo_root}")
    if status not in {"canonical", "smoke"}:
        raise ProteinStudyConfigError(f"unknown protein run status: {status}")
    base = Path(run_root).expanduser() if run_root is not None else root / str(study.raw["run_dir"])
    if not base.is_absolute():
        base = root / base
    base = base.resolve()
    if status == "canonical":
        try:
            base.relative_to(root)
        except ValueError as exc:
            raise ProteinStudyConfigError("canonical protein run_root must be inside the repository") from exc
    launch = _launch_mapping(study)
    legacy_root = _declared_path(study, str(study.raw["legacy"]["source_run_dir"]))
    legacy_import = _declared_path(study, str(study.raw["legacy"]["import_manifest"]))
    cells: list[ProteinStudyCell] = []
    for task in TASKS:
        for method in PRIMARY_METHODS:
            for seed in study.seeds:
                identity = f"{task}:{method}:seed{seed}"
                cell_dir = base / task / method / f"seed{seed}"
                if method == "holo_ga":
                    resolved = _holo_config(
                        study, task=task, seed=seed, cell_dir=cell_dir, repo_root=root, status=status
                    )
                elif method == "lambo2":
                    resolved = _lambo_config(
                        study, task=task, seed=seed, cell_dir=cell_dir, repo_root=root, status=status
                    )
                else:
                    resolved = _interpolant_config(
                        study,
                        task=task,
                        method=method,
                        seed=seed,
                        cell_dir=cell_dir,
                        repo_root=root,
                        status=status,
                    )
                condition = study.raw["legacy"]["condition_map"][task].get(method)
                legacy_source = (
                    legacy_root / str(condition) / f"seed{seed}"
                    if condition is not None and seed < 10 and identity not in launch
                    else None
                )
                cells.append(
                    ProteinStudyCell(
                        workflow="online",
                        condition=task,
                        seed=seed,
                        arm=method,
                        run_dir=cell_dir,
                        resolved_config=resolved,
                        estimated_seconds=_estimate_seconds(task, method),
                        status=status,
                        study=STUDY,
                        cohort="primary",
                        runtime_class=str(study.methods[method]["runtime_class"]),
                        launch_required=identity in launch,
                        launch_tranche=launch.get(identity),
                        legacy_source_dir=legacy_source,
                        legacy_import_manifest=(legacy_import if legacy_source is not None else None),
                    )
                )
        for method in DIAGNOSTIC_METHODS:
            for seed in range(10):
                cell_dir = base / task / method / f"seed{seed}"
                resolved = _interpolant_config(
                    study,
                    task=task,
                    method=method,
                    seed=seed,
                    cell_dir=cell_dir,
                    repo_root=root,
                    status=status,
                )
                condition = str(study.raw["legacy"]["condition_map"][task][method])
                cells.append(
                    ProteinStudyCell(
                        workflow="online",
                        condition=task,
                        seed=seed,
                        arm=method,
                        run_dir=cell_dir,
                        resolved_config=resolved,
                        estimated_seconds=_estimate_seconds(task, method),
                        status=status,
                        study=STUDY,
                        cohort="diagnostic",
                        runtime_class=str(study.methods[method]["runtime_class"]),
                        launch_required=False,
                        launch_tranche=None,
                        legacy_source_dir=legacy_root / condition / f"seed{seed}",
                        legacy_import_manifest=legacy_import,
                    )
                )
    if len(cells) != TOTAL_CELL_COUNT or len({cell.identity for cell in cells}) != TOTAL_CELL_COUNT:
        raise AssertionError(f"protein matrix expanded to {len(cells)} non-unique cells")
    return cells
