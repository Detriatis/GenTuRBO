"""Fail-closed semantic projection for retained protein methods.

The fixture below is a compact transcription of the consequential fields in
the 20 GFP/AAV deployable configs at commit ``24aca235``.  It deliberately
normalizes inert iteration limits and the content-addressed archive migration,
so equality means configuration-level semantic compatibility.  It is not a
claim of numerical trajectory parity.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from genturbo.artifacts import ArtifactError, sha256_file
from genturbo.io import iter_jsonl
from genturbo.serialization import atomic_write_json, canonical_json_bytes

from .config import LEGACY_PRIMARY_METHODS, ProteinStudyConfig

COMPATIBILITY_COMMIT = "24aca235"
COMPATIBILITY_MODE = "protein_gfp_aav_24aca235"
COMPATIBILITY_STATUS = (
    "configuration_projection_verified_numerical_parity_unverified_"
    "canonical_reuse_and_legacy_retirement_blocked"
)
NUMERICAL_COMPATIBILITY_CONTRACT = "canonical_paid_trajectory_float32_exact_v1"
NUMERICAL_COMPATIBILITY_FILENAME = "legacy_numerical_compatibility_attestation.json"
NUMERICAL_COMPATIBILITY_REGRESSION_SEED = 9
NUMERICAL_COMPATIBILITY_STATUS_UNVERIFIED = "numerical_parity_unverified"
NUMERICAL_COMPATIBILITY_STATUS_VERIFIED = "numerical_parity_verified"


class ProteinCompatibilityError(RuntimeError):
    """Raised when a retained method no longer matches the locked projection."""


_COMMON_GENERATOR = {
    "runner": "interpolant",
    "protein_mode": True,
    "flow_steps": 32,
    "max_time": 1.0,
    "device": "cuda",
    "temperature": 1.0,
    "candidate_source": "best",
    "candidate_k": 1,
    "num_times": 1,
    "mixing_mode": "random",
    "rollout_use_corrupt_mask": False,
}

_TASK_FIXTURES: dict[str, dict[str, Any]] = {
    "gfp_medium": {
        "protein": "GFP",
        "sequence_length": 237,
        "checkpoint": "configs/checkpoints/protein_evodiff_d3pm_38M_uniform_gfp237.pt",
        "runtime_config": "configs/sample/protein_evodiff_d3pm_38M_uniform_gfp237.json",
        "initial_archive_content_sha256": "deccef8745dd17aa1b0148664d7c12c06f6d6d133469df4a06e2e861db1458c5",
        "trust_region": {
            "type": "ScalarTrustRegion",
            "sampling": "fixed_budget",
            "domain": "time",
            "time_min": 0.0,
            "time_max": 0.99,
            "time_init": 0.75,
            "expand": 1.5,
            "shrink": 0.75,
            "success_patience": 1,
            "failure_patience": 1,
        },
    },
    "gfp_hard": {
        "protein": "GFP",
        "sequence_length": 237,
        "checkpoint": "configs/checkpoints/protein_evodiff_d3pm_38M_uniform_gfp237.pt",
        "runtime_config": "configs/sample/protein_evodiff_d3pm_38M_uniform_gfp237.json",
        "initial_archive_content_sha256": "c71047a3ccddde458ecbf8c638f1c63a90582282688ce110f7211acf2ef53730",
        "trust_region": {
            "type": "ScalarTrustRegion",
            "sampling": "fixed_budget",
            "domain": "time",
            "time_min": 0.0,
            "time_max": 0.99,
            "time_init": 0.75,
            "expand": 1.5,
            "shrink": 0.75,
            "success_patience": 1,
            "failure_patience": 1,
        },
    },
    "aav_medium": {
        "protein": "AAV",
        "sequence_length": 28,
        "checkpoint": "configs/checkpoints/protein_evodiff_d3pm_38M_uniform_aav28.pt",
        "runtime_config": "configs/sample/protein_evodiff_d3pm_38M_uniform_aav28.json",
        "initial_archive_content_sha256": "94f03398b8707a73ee82e76c12be915fff21fc47e011657e96c8487d42cf6599",
        "trust_region": {
            "type": "ScalarTrustRegion",
            "sampling": "fixed_budget",
            "domain": "time",
            "time_min": 0.0,
            "time_max": 0.95,
            "time_init": 0.5,
            "expand": 1.5,
            "shrink": 0.75,
            "success_patience": 1,
            "failure_patience": 1,
        },
    },
    "aav_hard": {
        "protein": "AAV",
        "sequence_length": 28,
        "checkpoint": "configs/checkpoints/protein_evodiff_d3pm_38M_uniform_aav28.pt",
        "runtime_config": "configs/sample/protein_evodiff_d3pm_38M_uniform_aav28.json",
        "initial_archive_content_sha256": "9d84562d9a85b52275380b02f2400be6bce318c9e3df11d9a887ec74e3a5b13b",
        "trust_region": {
            "type": "ScalarTrustRegion",
            "sampling": "fixed_budget",
            "domain": "time",
            "time_min": 0.0,
            "time_max": 0.95,
            "time_init": 0.5,
            "expand": 1.5,
            "shrink": 0.75,
            "success_patience": 1,
            "failure_patience": 1,
        },
    },
}

_METHOD_FIXTURES = {
    "unconditional": {
        "proposal_pool_size": 24,
        "candidate_strategy": "random",
        "acquisition": None,
        "active_learning": False,
        "sample_only": True,
    },
    "rt_random": {
        "proposal_pool_size": 24,
        "candidate_strategy": "random",
        "acquisition": "random",
        "active_learning": True,
        "sample_only": False,
    },
    "rt_gp": {
        "proposal_pool_size": 256,
        "candidate_strategy": "random",
        "acquisition": "batched_expected_improvement",
        "active_learning": True,
        "sample_only": False,
    },
    "tr_random": {
        "proposal_pool_size": 24,
        "candidate_strategy": "trust_region",
        "acquisition": "random",
        "active_learning": True,
        "sample_only": False,
    },
    "tr_gp": {
        "proposal_pool_size": 256,
        "candidate_strategy": "trust_region",
        "acquisition": "batched_expected_improvement",
        "active_learning": True,
        "sample_only": False,
    },
}


def _effective_batches(raw: Mapping[str, Any]) -> int:
    if not bool(raw.get("active_learning")):
        return 0
    initial = int(raw.get("initial_oracle_evaluations", raw.get("batch_size", 0)))
    paid_batch = int(raw.get("al_oracle_batch_size", raw.get("batch_size", 0)))
    available = int(raw.get("max_oracle_evaluations", 0)) - initial
    if paid_batch <= 0 or available < 0 or available % paid_batch:
        return -1
    return min(int(raw.get("num_iterations", 0)), available // paid_batch)


def project_retained_config(
    raw: Mapping[str, Any],
    *,
    task: str,
    method: str,
    initial_archive_content_sha256: str,
) -> dict[str, Any]:
    """Project one legacy or modern flat config onto consequential semantics."""

    if task not in _TASK_FIXTURES or method not in LEGACY_PRIMARY_METHODS:
        raise ProteinCompatibilityError(f"no retained compatibility fixture for {task}:{method}")
    active = bool(raw.get("active_learning"))
    sample_only = bool(raw.get("sample_only"))
    sequence_length = int(raw.get("protein_max_len", -1))
    projection: dict[str, Any] = {
        "schema_version": 1,
        "source_commit": COMPATIBILITY_COMMIT,
        "task": task,
        "method": method,
        "generator": {
            "runner": raw.get("runner"),
            "protein_mode": raw.get("protein_mode"),
            "flow_steps": int(raw.get("flow_steps", -1)),
            "max_time": float(raw.get("max_time", -1.0)),
            "device": raw.get("device"),
            "temperature": float(raw.get("temperature", -1.0)),
            "candidate_source": raw.get("candidate_source"),
            "candidate_k": int(raw.get("candidate_k", -1)),
            "num_times": int(raw.get("num_times", -1)),
            "mixing_mode": raw.get("mixing_mode"),
            "rollout_use_corrupt_mask": raw.get("rollout_use_corrupt_mask"),
            "checkpoint": raw.get("checkpoint"),
            "runtime_config": raw.get("config"),
            "benchmark_task": raw.get("benchmark_task"),
            "protein_objective": raw.get("protein_objective"),
            "sequence_length": sequence_length,
        },
        "oracle": {
            "name": raw.get("reward", {}).get("name") if isinstance(raw.get("reward"), Mapping) else None,
            "protein": raw.get("reward", {}).get("protein") if isinstance(raw.get("reward"), Mapping) else None,
            "benchmark_root": (
                raw.get("reward", {}).get("benchmark_root") if isinstance(raw.get("reward"), Mapping) else None
            ),
            "normalize": raw.get("reward", {}).get("normalize") if isinstance(raw.get("reward"), Mapping) else None,
            "sequence_length": (
                raw.get("reward", {}).get("sequence_length") if isinstance(raw.get("reward"), Mapping) else None
            ),
            "device": raw.get("reward", {}).get("device") if isinstance(raw.get("reward"), Mapping) else None,
        },
        "budget": {
            "max_oracle_evaluations": raw.get("max_oracle_evaluations"),
            "active_learning": active,
            "sample_only": sample_only,
            "sample_only_count": raw.get("sample_only_count") if sample_only else None,
            "sample_generation_chunk_size": (
                raw.get("sample_only_batch_size", raw.get("sample_only_count")) if sample_only else None
            ),
            "initial_oracle_evaluations": int(raw.get("initial_oracle_evaluations", 0)) if active else 0,
            "paid_batch_size": int(raw.get("al_oracle_batch_size", 0)) if active else None,
            "effective_post_initial_batches": _effective_batches(raw),
            "initial_archive_content_sha256": initial_archive_content_sha256 if active else None,
            "initial_batch_advance_rng": (
                bool(raw.get("initial_batch_advance_rng"))
                if raw.get("initial_batch_advance_rng") is not None
                else (True if active else None)
            ),
            "initial_batch_rescore_with_reward": bool(raw.get("initial_batch_rescore_with_reward", False)),
        },
        "proposal": {
            "configured_pool_size": raw.get("mixes_per_time"),
            "candidate_strategy": raw.get("candidate_strategy"),
            "acquisition": raw.get("al_acquisition") if active else None,
            "sequential_selection": bool(raw.get("al_sequential_selection", False)) if active else None,
            "local_penalization": bool(raw.get("al_local_penalization", False)) if active else None,
        },
        "trust_region": None,
        "surrogate": None,
    }
    if method.startswith("tr_"):
        projection["trust_region"] = {
            "type": raw.get("tr_type"),
            "sampling": raw.get("tr_sampling"),
            "domain": raw.get("tr_domain"),
            "time_min": raw.get("time_min"),
            "time_max": raw.get("time_max"),
            "time_init": raw.get("tr_mean"),
            "expand": raw.get("region_expand_scalar"),
            "shrink": raw.get("region_shrink_scalar"),
            "success_patience": raw.get("tau_success"),
            "failure_patience": raw.get("tau_failure"),
        }
    if method.endswith("_gp"):
        fixed_scale = raw.get("al_gp_outputscale") == 1.0 and raw.get("al_gp_learn_outputscale") is False
        fitted_noise = raw.get("al_gp_noise_variance") is None
        legacy_equivalent_fit = fixed_scale and fitted_noise
        projection["surrogate"] = {
            "feature_space": raw.get("al_feature_space"),
            "onehot_max_len": raw.get("al_onehot_max_len"),
            "buffer_size": raw.get("al_buffer_size"),
            "buffer_mode": raw.get("al_buffer_mode"),
            "device": raw.get("al_gp_device"),
            "dtype": raw.get("al_gp_dtype"),
            "jitter": raw.get("al_gp_jitter"),
            "outputscale": raw.get("al_gp_outputscale"),
            "learn_outputscale": raw.get("al_gp_learn_outputscale"),
            "noise_variance": raw.get("al_gp_noise_variance"),
            "fit_mean_and_noise_only": bool(raw.get("al_gp_fit_mean_and_noise_only", legacy_equivalent_fit)),
            "learn_lengthscale": bool(raw.get("al_gp_learn_lengthscale", False)),
            "hyperparameter_refit_interval": int(raw.get("al_gp_hyperparameter_refit_interval", 1)),
            "qei_mc_samples": raw.get("al_qei_mc_samples"),
        }
    return projection


def expected_retained_projection(*, task: str, method: str) -> dict[str, Any]:
    """Construct the immutable compact fixture transcribed from ``24aca235``."""

    task_fixture = _TASK_FIXTURES[task]
    method_fixture = _METHOD_FIXTURES[method]
    length = int(task_fixture["sequence_length"])
    active = bool(method_fixture["active_learning"])
    raw: dict[str, Any] = {
        **_COMMON_GENERATOR,
        "checkpoint": task_fixture["checkpoint"],
        "config": task_fixture["runtime_config"],
        "benchmark_task": task,
        "protein_objective": task,
        "protein_min_len": length,
        "protein_max_len": length,
        "reward": {
            "name": "gfp_aav_cnn_oracle",
            "protein": task_fixture["protein"],
            "benchmark_root": "benchmarks/gfp_aav",
            "normalize": True,
            "sequence_length": length,
            "device": "cuda",
        },
        "mixes_per_time": method_fixture["proposal_pool_size"],
        "candidate_strategy": method_fixture["candidate_strategy"],
        "active_learning": active,
        "sample_only": method_fixture["sample_only"],
        "max_oracle_evaluations": 2000,
        "batch_size": 32 if method == "unconditional" else 8,
        "num_iterations": 2000 if method == "unconditional" else 2200,
    }
    if method == "unconditional":
        raw["sample_only_count"] = 2000
    else:
        raw.update(
            {
                "initial_oracle_evaluations": 128,
                "al_oracle_batch_size": 8,
                "al_acquisition": method_fixture["acquisition"],
                "al_sequential_selection": False,
                "al_local_penalization": False,
            }
        )
    if method.startswith("tr_"):
        trust = task_fixture["trust_region"]
        raw.update(
            {
                "tr_type": trust["type"],
                "tr_sampling": trust["sampling"],
                "tr_domain": trust["domain"],
                "time_min": trust["time_min"],
                "time_max": trust["time_max"],
                "tr_mean": trust["time_init"],
                "region_expand_scalar": trust["expand"],
                "region_shrink_scalar": trust["shrink"],
                "tau_success": trust["success_patience"],
                "tau_failure": trust["failure_patience"],
            }
        )
    if method.endswith("_gp"):
        raw.update(
            {
                "al_feature_space": "onehot_tanimoto",
                "al_onehot_max_len": length,
                "al_buffer_size": 128,
                "al_buffer_mode": "recent",
                "al_gp_device": "cpu",
                "al_gp_dtype": "double",
                "al_gp_jitter": 0.0001,
                "al_gp_outputscale": 1.0,
                "al_gp_learn_outputscale": False,
                "al_qei_mc_samples": 128,
            }
        )
    return project_retained_config(
        raw,
        task=task,
        method=method,
        initial_archive_content_sha256=str(task_fixture["initial_archive_content_sha256"]),
    )


def validate_retained_compatibility(
    study: ProteinStudyConfig,
    resolved: Mapping[str, Any],
    *,
    task: str,
    method: str,
) -> dict[str, str]:
    """Fail closed if a modern resolved retained cell changes the projection."""

    if (
        study.raw["legacy"].get("compatibility_commit") != COMPATIBILITY_COMMIT
        or study.raw["legacy"].get("compatibility_mode") != COMPATIBILITY_MODE
    ):
        raise ProteinCompatibilityError("retained protein compatibility declaration drifted; rerun_required=true")
    if method != "unconditional" and resolved.get("initial_batch_path") != study.tasks[task]["initial_archive"]:
        raise ProteinCompatibilityError(
            f"{task}:{method} initial archive path changed; configuration parity failed; rerun_required=true"
        )
    expected = expected_retained_projection(task=task, method=method)
    observed = project_retained_config(
        resolved,
        task=task,
        method=method,
        initial_archive_content_sha256=str(study.tasks[task]["initial_archive_sha256"]),
    )
    if observed != expected:
        differences = sorted(key for key in expected if observed.get(key) != expected[key])
        raise ProteinCompatibilityError(
            f"{task}:{method} compatibility projection changed in {differences}; rerun_required=true"
        )
    identity = hashlib.sha256(canonical_json_bytes(observed)).hexdigest()
    return {
        "compatibility_mode": COMPATIBILITY_MODE,
        "compatibility_commit": COMPATIBILITY_COMMIT,
        "compatibility_projection_sha256": identity,
        "compatibility_status": COMPATIBILITY_STATUS,
    }
