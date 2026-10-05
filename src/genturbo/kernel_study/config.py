"""Strict contracts for the four ``ehrlich_kernel_v1`` configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STUDY = "ehrlich_kernel_v1"
SCHEMA_VERSION = 1
CONFIG_FILENAMES = ("datasets.json", "static.json", "timing.json", "online.json")
CONDITIONS = ("l128_q4_k4", "l256_q4_k8")
STATIC_KERNELS = (
    "onehot_tanimoto",
    "spaced_pair_w32",
    "spaced_pair_w48",
    "local_spaced_trigram_w16",
    "pair_w48_trigram_w16_additive",
    "task_aware_spaced_ngram",
)
ONLINE_ARMS = (
    "onehot_ei",
    "spaced_pair_w32_ei",
    "spaced_pair_w48_ei",
    "random",
)
KERNEL_LABELS = {
    "onehot_tanimoto": "Positional one-hot",
    "spaced_pair_w32": "Spaced pair (W=32)",
    "spaced_pair_w48": "Spaced pair (W=48)",
    "local_spaced_trigram_w16": "Spaced trigram (W=16)",
    "pair_w48_trigram_w16_additive": "Pair+trigram additive",
    "task_aware_spaced_ngram": "Task-aware (privileged)",
}
ONLINE_ARM_LABELS = {
    "onehot_ei": "One-hot EI",
    "spaced_pair_w32_ei": "Spaced pair (W=32) EI",
    "spaced_pair_w48_ei": "Spaced pair (W=48) EI",
    "random": "Random selection",
}
CONDITION_LABELS = {
    "l128_q4_k4": "L=128, q=4, K=4",
    "l256_q4_k8": "L=256, q=4, K=8",
}


def canonical_kernel_name(value: str) -> str:
    """Normalize the low-level task-aware kernel name used by legacy runners."""

    return "task_aware_spaced_ngram" if value == "spaced_ngram" else value

_CONFIG_PREFIX = f"configs/experiments/{STUDY}"
_GENERATOR_TRAINING_CONDITION = {
    "sequence_length": 128,
    "quantization": 4,
    "motif_length": 4,
}
_DATASET_CONDITIONS = [
    {
        "name": "l128_q4_k4",
        "sequence_length": 128,
        "vocab_size": 20,
        "quantization": 4,
        "motif_length": 4,
        "generator_training_condition": _GENERATOR_TRAINING_CONDITION,
        "generation_length_override": None,
        "checkpoint": "runs/train/trained_ctmc/checkpoints/model_step18000.pt",
        "runtime_config": "runs/train/trained_ctmc/config.json",
        "instance_path": "runs/train/trained_ctmc/dataset_instance.pt",
    },
    {
        "name": "l256_q4_k8",
        "sequence_length": 256,
        "vocab_size": 20,
        "quantization": 4,
        "motif_length": 8,
        "generator_training_condition": _GENERATOR_TRAINING_CONDITION,
        "generation_length_override": 256,
        "checkpoint": "runs/train/trained_ctmc512/checkpoints/model_step18000.pt",
        "runtime_config": "configs/train/ctmc_trainer_conv512.json",
        "instance_path": "runs/search/ehrlich_grid/Q4K8/dataset_instance.pt",
    },
]
_TRUST_REGION = {
    "type": "ScalarTrustRegion",
    "domain": "absolute_edits",
    "sampling": "fixed_budget",
    "edits_min": 2,
    "edits_init": 64,
    "edits_max": 128,
    "expand": 1.5,
    "shrink": 0.75,
    "success_patience": 1,
    "failure_patience": 1,
}
_DATASET_GENERATION = {
    "device": "mps",
    "initial_generation": "unconditional",
    "initial_oracle_evaluations": 64,
    "conditional_iterations": 24,
    "proposal_batch_size": 128,
    "acquisition_batch_size": 8,
    "max_oracle_evaluations": 256,
    "flow_steps": 20,
    "acquisition": "random",
    "candidate_source": "best",
    "trace_level": "full",
    "trust_region": _TRUST_REGION,
    "validity_policy": "reward_adapter",
    "duplicate_policy": "exclude_archive_and_batch_duplicates",
    "cache_policy": "reuse_scored_sequences",
}
_DATASET_PROCESSING = {
    "freeze_population": "all_valid_acquisition_eligible",
    "rescore_with_condition_reward": True,
    "deduplicate": True,
    "distance_metric": "hamming",
    "locality_populations": ["all", "eligible", "selected"],
    "publication_formats": ["pdf", "png", "svg"],
}
_STATIC_SPLIT = {
    "strategy": "leave_one_source_seed_out",
    "heldout_seeds": list(range(20)),
    "train_size": 512,
    "test_size": 256,
    "top_k": 8,
    "random_seed": 20260716,
}
_GP_POLICY = {
    "device": "cpu",
    "dtype": "double",
    "jitter": 0.0001,
    "noise_mode": "fitted",
    "outputscale": 1.0,
    "learn_outputscale": False,
    "trainable_parameters": ["constant_mean", "likelihood_noise"],
}
_STATIC_KERNEL_DECLARATIONS = [
    {"name": "onehot_tanimoto", "type": "onehot_tanimoto", "parameters": {}, "privileged": False},
    {"name": "spaced_pair_w32", "type": "spaced_pair", "parameters": {"window": 32}, "privileged": False},
    {"name": "spaced_pair_w48", "type": "spaced_pair", "parameters": {"window": 48}, "privileged": False},
    {
        "name": "local_spaced_trigram_w16",
        "type": "local_spaced_trigram",
        "parameters": {"window": 16},
        "privileged": False,
    },
    {
        "name": "pair_w48_trigram_w16_additive",
        "type": "mixed_window_spaced_ngram_additive",
        "parameters": {"orders": [2, 3], "windows": [48, 16], "weights": [0.5, 0.5]},
        "privileged": False,
    },
    {
        "name": "task_aware_spaced_ngram",
        "type": "spaced_ngram",
        "parameters": {"patterns": "instance"},
        "privileged": True,
    },
]
_STATIC_METRICS = [
    "simple_regret",
    "top_k_recall",
    "rmse",
    "spearman",
    "nlpd",
    "coverage_90",
    "fit_seconds",
    "inference_seconds",
    "failure_count",
]
_STATIC_COMPARISON = {
    "baseline_kernel": "onehot_tanimoto",
    "cross_condition_summary": "within_condition_rank_and_paired_delta",
    "do_not_pool_raw_metrics": ["rmse", "simple_regret"],
}
_ONLINE_GENERATION = {
    "device": "mps",
    "initial_generation": "unconditional",
    "initial_oracle_evaluations": 64,
    "conditional_iterations": 56,
    "proposal_batch_size": 128,
    "acquisition_batch_size": 8,
    "max_oracle_evaluations": 512,
    "flow_steps": 20,
    "candidate_source": "best",
    "trust_region": _TRUST_REGION,
    "validity_policy": "reward_adapter",
    "duplicate_policy": "exclude_archive_and_batch_duplicates",
    "cache_policy": "reuse_scored_sequences",
}
_ONLINE_ARM_DECLARATIONS = [
    {"name": "onehot_ei", "kernel": "onehot_tanimoto", "acquisition": "marginal_expected_improvement"},
    {"name": "spaced_pair_w32_ei", "kernel": "spaced_pair_w32", "acquisition": "marginal_expected_improvement"},
    {"name": "spaced_pair_w48_ei", "kernel": "spaced_pair_w48", "acquisition": "marginal_expected_improvement"},
    {"name": "random", "kernel": None, "acquisition": "random"},
]
_ONLINE_METRICS = [
    "best_so_far",
    "normalized_post_initial_auc",
    "final_best",
    "improvement_over_initial",
    "paired_seed_difference",
    "batch_diversity_hamming",
    "runtime_seconds",
    "validity_rate",
    "uniqueness_rate",
    "failure_count",
]

_TOP_LEVEL_KEYS = {
    "dataset": {
        "schema_version",
        "study",
        "workflow",
        "run_dir",
        "results_dir",
        "seeds",
        "conditions",
        "generation",
        "processing",
    },
    "static": {
        "schema_version",
        "study",
        "workflow",
        "datasets_config",
        "run_dir",
        "results_dir",
        "conditions",
        "split",
        "gp",
        "kernels",
        "metrics",
        "comparison",
    },
    "timing": {
        "schema_version",
        "study",
        "workflow",
        "static_config",
        "run_dir",
        "results_dir",
        "conditions",
        "heldout_seed",
        "fit_repeats",
        "inference_repeats",
        "device",
        "dtype",
        "garbage_collection",
        "metrics",
    },
    "online": {
        "schema_version",
        "study",
        "workflow",
        "datasets_config",
        "static_config",
        "run_dir",
        "results_dir",
        "conditions",
        "seeds",
        "generation",
        "gp",
        "shared_initial_archive",
        "arms",
        "metrics",
    },
}


class KernelStudyConfigError(ValueError):
    """Raised when the four-file study contract is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class StudyConfigs:
    directory: Path
    datasets: dict[str, Any]
    static: dict[str, Any]
    timing: dict[str, Any]
    online: dict[str, Any]

    def for_workflow(self, workflow: str) -> dict[str, Any]:
        normalized = "dataset" if workflow == "datasets" else workflow
        if normalized == "dataset":
            return self.datasets
        if normalized in {"static", "timing", "online"}:
            return getattr(self, normalized)
        raise KernelStudyConfigError(f"unknown kernel-study workflow {workflow!r}")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KernelStudyConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise KernelStudyConfigError(f"{path} must contain a JSON object")
    return value


def _integers(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise KernelStudyConfigError(f"{label} must be an integer array")
    return [int(item) for item in value]


def _validate_gp(gp: Any, *, label: str) -> None:
    if not isinstance(gp, dict):
        raise KernelStudyConfigError(f"{label}.gp must be an object")
    if gp != _GP_POLICY:
        raise KernelStudyConfigError(
            f"{label}.gp must fit only the constant mean and likelihood noise with outputscale fixed at 1.0"
        )


def _validate_trust_region(value: Any, *, label: str) -> None:
    if value != _TRUST_REGION:
        raise KernelStudyConfigError(f"{label} must use the fixed absolute edit-domain policy")


def _validate_common(raw: dict[str, Any], *, path: Path, workflow: str) -> None:
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS[workflow])
    missing = sorted(_TOP_LEVEL_KEYS[workflow] - set(raw))
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        raise KernelStudyConfigError(f"invalid {workflow} config {path}: {'; '.join(details)}")
    if raw["schema_version"] != SCHEMA_VERSION or raw["study"] != STUDY or raw["workflow"] != workflow:
        raise KernelStudyConfigError(f"{path} does not identify {STUDY}/{workflow} schema 1")
    suffix = "datasets" if workflow == "dataset" else workflow
    if raw["run_dir"] != f"runs/{STUDY}/{suffix}":
        raise KernelStudyConfigError(f"{path} has a non-canonical run_dir")
    if raw["results_dir"] != f"results/{STUDY}/{suffix}":
        raise KernelStudyConfigError(f"{path} has a non-canonical results_dir")


def _validate_dataset(raw: dict[str, Any]) -> None:
    if _integers(raw["seeds"], label="datasets.seeds") != list(range(20)):
        raise KernelStudyConfigError("datasets.seeds must be exactly 0..19")
    if raw["conditions"] != _DATASET_CONDITIONS:
        raise KernelStudyConfigError("datasets.conditions must be the two pinned generator/task declarations")
    if raw["generation"] != _DATASET_GENERATION:
        raise KernelStudyConfigError("datasets.generation must be the exact 64 + 24x8 full-trace protocol")
    _validate_trust_region(raw["generation"].get("trust_region"), label="datasets.generation.trust_region")
    if raw["processing"] != _DATASET_PROCESSING:
        raise KernelStudyConfigError("datasets.processing must be the exact frozen-population and locality policy")


def _validate_static(raw: dict[str, Any]) -> None:
    if raw["datasets_config"] != f"{_CONFIG_PREFIX}/datasets.json":
        raise KernelStudyConfigError("static.datasets_config must reference the study datasets config")
    if tuple(raw["conditions"]) != CONDITIONS:
        raise KernelStudyConfigError("static.conditions must match the dataset conditions")
    if raw["split"] != _STATIC_SPLIT:
        raise KernelStudyConfigError("static.split must be the fixed 20-split 512/256/top-8 protocol")
    _validate_gp(raw["gp"], label="static")
    if raw["kernels"] != _STATIC_KERNEL_DECLARATIONS:
        raise KernelStudyConfigError("static.kernels must be the six exact decision-relevant declarations")
    if raw["metrics"] != _STATIC_METRICS or raw["comparison"] != _STATIC_COMPARISON:
        raise KernelStudyConfigError("static reporting policy must remain fixed")


def _validate_timing(raw: dict[str, Any]) -> None:
    if raw["static_config"] != f"{_CONFIG_PREFIX}/static.json":
        raise KernelStudyConfigError("timing.static_config must reference the study static config")
    if tuple(raw["conditions"]) != CONDITIONS:
        raise KernelStudyConfigError("timing.conditions must match the dataset conditions")
    expected = (raw["heldout_seed"], raw["fit_repeats"], raw["inference_repeats"])
    if expected != (0, 1, 3) or raw["device"] != "cpu" or raw["dtype"] != "double":
        raise KernelStudyConfigError("timing must use seed 0, 1 fit, 3 inference repeats, CPU double")
    if raw["garbage_collection"] != "outside_measurement":
        raise KernelStudyConfigError("timing garbage collection must be outside measured intervals")
    if raw["metrics"] != ["fit_seconds", "inference_seconds"]:
        raise KernelStudyConfigError("timing metrics must contain only fit and inference cost")


def _validate_online(raw: dict[str, Any], *, static: dict[str, Any] | None = None) -> None:
    if raw["datasets_config"] != f"{_CONFIG_PREFIX}/datasets.json" or raw["static_config"] != (
        f"{_CONFIG_PREFIX}/static.json"
    ):
        raise KernelStudyConfigError("online must reference this study's datasets and static configs")
    if tuple(raw["conditions"]) != CONDITIONS or _integers(raw["seeds"], label="online.seeds") != list(range(5)):
        raise KernelStudyConfigError("online must use both conditions and seeds 0..4")
    _validate_gp(raw["gp"], label="online")
    if static is not None and raw["gp"] != static["gp"]:
        raise KernelStudyConfigError("online and static GP policies must be byte-equivalent")
    if raw["generation"] != _ONLINE_GENERATION:
        raise KernelStudyConfigError("online.generation must be the exact 64 + 56x8 absolute-edit protocol")
    _validate_trust_region(raw["generation"].get("trust_region"), label="online.generation.trust_region")
    if raw["shared_initial_archive"] != {"enabled": True, "byte_identical_across_arms": True}:
        raise KernelStudyConfigError("online must use byte-identical shared initial archives")
    if raw["arms"] != _ONLINE_ARM_DECLARATIONS:
        raise KernelStudyConfigError("online arms must be the exact three marginal-EI kernels plus random")
    if raw["metrics"] != _ONLINE_METRICS:
        raise KernelStudyConfigError("online reporting metrics must remain fixed")


def load_config(
    path: Path | str,
    *,
    expected_workflow: str | None = None,
    static: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one strict study config without accepting experiment drift."""

    config_path = Path(path).expanduser().resolve()
    raw = _read_object(config_path)
    workflow = str(raw.get("workflow", ""))
    if expected_workflow is not None and workflow != expected_workflow:
        raise KernelStudyConfigError(f"expected {expected_workflow} config, got {workflow or 'unknown'}")
    if workflow not in _TOP_LEVEL_KEYS:
        raise KernelStudyConfigError(f"unsupported kernel-study workflow {workflow!r}")
    _validate_common(raw, path=config_path, workflow=workflow)
    if workflow == "dataset":
        _validate_dataset(raw)
    elif workflow == "static":
        _validate_static(raw)
    elif workflow == "timing":
        _validate_timing(raw)
    else:
        _validate_online(raw, static=static)
    return raw


def _resolve_reference(repo_root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise KernelStudyConfigError(f"{label} must be a repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise KernelStudyConfigError(f"{label} must stay inside the repository")
    resolved = (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise KernelStudyConfigError(f"{label} escapes the repository") from exc
    if not resolved.is_file():
        raise KernelStudyConfigError(f"{label} does not resolve: {value}")
    return resolved


def load_study(directory: Path | str, *, repo_root: Path | str, verify_inputs: bool = True) -> StudyConfigs:
    """Load all four configs and prove their paths and cross-references agree."""

    root = Path(repo_root).expanduser().resolve()
    config_dir = Path(directory).expanduser()
    if not config_dir.is_absolute():
        config_dir = root / config_dir
    config_dir = config_dir.resolve()
    actual = sorted(path.name for path in config_dir.glob("*.json"))
    if actual != sorted(CONFIG_FILENAMES):
        raise KernelStudyConfigError(
            f"{config_dir} must contain exactly {', '.join(CONFIG_FILENAMES)}; found {', '.join(actual) or 'none'}"
        )
    datasets = load_config(config_dir / "datasets.json", expected_workflow="dataset")
    static = load_config(config_dir / "static.json", expected_workflow="static")
    timing = load_config(config_dir / "timing.json", expected_workflow="timing")
    online = load_config(config_dir / "online.json", expected_workflow="online", static=static)
    expected_refs = {
        _resolve_reference(root, static["datasets_config"], label="static.datasets_config"): config_dir
        / "datasets.json",
        _resolve_reference(root, timing["static_config"], label="timing.static_config"): config_dir / "static.json",
        _resolve_reference(root, online["datasets_config"], label="online.datasets_config"): config_dir
        / "datasets.json",
        _resolve_reference(root, online["static_config"], label="online.static_config"): config_dir / "static.json",
    }
    for observed, expected in expected_refs.items():
        if observed != expected.resolve():
            raise KernelStudyConfigError(f"cross-reference {observed} must resolve to {expected}")
    if verify_inputs:
        for condition in datasets["conditions"]:
            for key in ("checkpoint", "runtime_config", "instance_path"):
                _resolve_reference(root, condition[key], label=f"{condition['name']}.{key}")
    return StudyConfigs(config_dir, datasets, static, timing, online)


__all__ = [
    "CONFIG_FILENAMES",
    "CONDITION_LABELS",
    "CONDITIONS",
    "KERNEL_LABELS",
    "KernelStudyConfigError",
    "ONLINE_ARM_LABELS",
    "ONLINE_ARMS",
    "STATIC_KERNELS",
    "STUDY",
    "StudyConfigs",
    "canonical_kernel_name",
    "load_config",
    "load_study",
]
