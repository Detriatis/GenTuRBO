"""Strict contract for the 13-arm L256 Ehrlich baseline cohort."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genturbo.kernel_study.config import StudyConfigs, load_study
from genturbo.serialization import canonical_json_bytes

STUDY = "ehrlich_sequence_baselines_v2"
PROTOCOL_VERSION = 6
CONFIG_FILENAME = "baselines.json"
CONDITION = "l256_q4_k8"
LEGACY_ARMS = (
    "unconditional",
    "no_tr_no_gp",
    "tr_no_gp",
    "no_tr_gp",
    "tr_gp",
    "oracle_upper_bound",
)
ADDITIVE_ARMS = (
    "unconditional_gp",
    "hamming_random",
    "hamming_gp",
    "adalead_gp_ei",
    "bo_evo",
    "lambo2",
)
PREEXISTING_ARMS = (*LEGACY_ARMS, *ADDITIVE_ARMS)
HOLO_ARM = "holo_ga"
ARMS = (*PREEXISTING_ARMS, HOLO_ARM)
SEED_BATCHES = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9),
    (10, 11, 12, 13, 14),
    (15, 16, 17, 18, 19),
)
_BASE_DATASETS_CONFIG = "configs/experiments/ehrlich_kernel_v1/datasets.json"
_EXPECTED_CONFIG_SHA256 = "00300ab731b339e5327e7af5ad5dbdfcc2c74eea9874a2bfb2982e4f0b24ffe9"


class SequenceBaselineConfigError(ValueError):
    """Raised when the sequence-baseline protocol has drifted."""


@dataclass(frozen=True, slots=True)
class SequenceBaselineStudyConfig:
    path: Path
    raw: dict[str, Any]
    kernel_study: StudyConfigs

    @property
    def condition(self) -> dict[str, Any]:
        for condition in self.kernel_study.datasets["conditions"]:
            if condition["name"] == CONDITION:
                return condition
        raise SequenceBaselineConfigError(f"base dataset config is missing {CONDITION}")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(seed for batch in SEED_BATCHES for seed in batch)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SequenceBaselineConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SequenceBaselineConfigError(f"{path} must contain a JSON object")
    return value


def _validate(raw: dict[str, Any], path: Path) -> None:
    identity = (
        raw.get("schema_version"),
        raw.get("protocol_version"),
        raw.get("study"),
        raw.get("workflow"),
    )
    if identity != (1, PROTOCOL_VERSION, STUDY, "online"):
        raise SequenceBaselineConfigError(f"{path} does not identify the active {STUDY} protocol")
    if raw.get("base_datasets_config") != _BASE_DATASETS_CONFIG or raw.get("condition") != CONDITION:
        raise SequenceBaselineConfigError("sequence baselines must use only the pinned L256 condition")
    if raw.get("run_dir") != f"runs/{STUDY}/online" or raw.get("results_dir") != f"results/{STUDY}/online":
        raise SequenceBaselineConfigError("v2 sequence-baseline paths must use the fresh v2 namespace")
    if raw.get("seed_batches") != [list(batch) for batch in SEED_BATCHES]:
        raise SequenceBaselineConfigError("seed_batches must be the four paired five-seed tranches")
    if tuple(arm.get("name") for arm in raw.get("arms", ())) != ARMS:
        raise SequenceBaselineConfigError("the corrected cohort must contain the fixed 13-arm order")
    if hashlib.sha256(canonical_json_bytes(raw)).hexdigest() != _EXPECTED_CONFIG_SHA256:
        raise SequenceBaselineConfigError(f"{path} differs from the versioned {STUDY} protocol")


def load_sequence_baseline_study(
    path: Path | str,
    *,
    repo_root: Path | str,
    verify_inputs: bool = True,
) -> SequenceBaselineStudyConfig:
    """Load the sole active config and its pinned L256 dataset declaration."""

    root = Path(repo_root).expanduser().resolve()
    config_path = Path(path).expanduser()
    config_path = (config_path if config_path.is_absolute() else root / config_path).resolve()
    if config_path.name != CONFIG_FILENAME:
        raise SequenceBaselineConfigError(f"expected {CONFIG_FILENAME}, got {config_path.name}")
    if sorted(item.name for item in config_path.parent.glob("*.json")) != [CONFIG_FILENAME]:
        raise SequenceBaselineConfigError(f"{config_path.parent} must contain exactly {CONFIG_FILENAME}")
    raw = _read_object(config_path)
    _validate(raw, config_path)
    base_path = (root / str(raw["base_datasets_config"])).resolve()
    if base_path != (root / _BASE_DATASETS_CONFIG).resolve() or not base_path.is_file():
        raise SequenceBaselineConfigError("base_datasets_config does not resolve to the pinned dataset config")
    try:
        kernel_study = load_study(base_path.parent, repo_root=root, verify_inputs=verify_inputs)
    except ValueError as exc:
        raise SequenceBaselineConfigError(str(exc)) from exc
    study = SequenceBaselineStudyConfig(config_path, raw, kernel_study)
    condition = study.condition
    task = (
        int(condition["sequence_length"]),
        int(condition["quantization"]),
        int(condition["motif_length"]),
    )
    if task != (256, 4, 8):
        raise SequenceBaselineConfigError(f"{CONDITION} drifted from L256/q4/K8: got {task}")
    if study.seeds != tuple(range(20)):
        raise SequenceBaselineConfigError("seed_batches must cover seeds 0..19 exactly once")
    generation = raw["generation"]
    if int(generation["initial_oracle_evaluations"]) + int(generation["expected_full_batches"]) * int(
        generation["acquisition_batch_size"]
    ) != int(generation["max_selected_archive_evaluations"]) or int(generation["iteration_cap"]) != int(
        generation["expected_full_batches"]
    ):
        raise SequenceBaselineConfigError("the corrected selected-archive budget is not exact")
    if generation.get("validity_policy") != "black_box_paid_oracle" or generation.get("invalid_gp_target") != 0.0:
        raise SequenceBaselineConfigError("iterative arms must learn paid black-box invalidity with GP target zero")
    execution = raw.get("execution")
    if not isinstance(execution, dict):
        raise SequenceBaselineConfigError("the local execution and device policy must be explicit")
    generator_arms = tuple(execution.get("generator_gpu_arms", ()))
    cpu_arms = tuple(execution.get("cpu_arms", ()))
    lambo_arms = tuple(execution.get("lambo2_arms", ()))
    if (
        execution.get("backend") != "local"
        or int(execution.get("legacy_completed_seed_max", -1)) != 9
        or execution.get("legacy_completed_device") != "mps"
        or execution.get("new_generator_device") != "cuda"
        or execution.get("new_cpu_device") != "cpu"
        or set(generator_arms) | set(cpu_arms) | set(lambo_arms) != set(ARMS)
        or set(generator_arms) & set(cpu_arms)
        or set(generator_arms) & set(lambo_arms)
        or set(cpu_arms) & set(lambo_arms)
        or lambo_arms != ("lambo2",)
        or HOLO_ARM not in cpu_arms
    ):
        raise SequenceBaselineConfigError("the recorded device policy must partition all 13 arms exactly")
    cpu_generator_seeds = execution.get("cpu_generator_seeds", {})
    if not isinstance(cpu_generator_seeds, dict) or any(
        arm not in generator_arms
        or not isinstance(seeds, list)
        or any(type(seed) is not int or seed not in range(10, 20) for seed in seeds)
        or len(seeds) != len(set(seeds))
        for arm, seeds in cpu_generator_seeds.items()
    ):
        raise SequenceBaselineConfigError(
            "cpu_generator_seeds must list unique seeds 10..19 for frozen-generator arms only"
        )
    if int(raw["gp"].get("hyperparameter_refit_interval", 0)) != 8:
        raise SequenceBaselineConfigError("the L256 exact GP must refit mean and noise every eight archive updates")
    bo_evo = next(arm for arm in raw["arms"] if arm["name"] == "bo_evo")
    if int(bo_evo.get("baseline_children_per_selection", 0)) != 8:
        raise SequenceBaselineConfigError("BO-EVO must screen its published eight children per paid selection")
    lambo = raw.get("lambo2")
    if not isinstance(lambo, dict) or not isinstance(lambo.get("pretraining"), dict):
        raise SequenceBaselineConfigError("LaMBO-2 must declare reward-free generator pretraining")
    pretraining = lambo["pretraining"]
    if pretraining.get("oracle_labels_seen") != 0 or pretraining.get("task_motifs_seen") is not False:
        raise SequenceBaselineConfigError("LaMBO-2 pretraining must not see oracle labels or task motifs")
    sampled_corpus_size = int(pretraining["sampled_corpus_size"])
    train_fraction = float(pretraining["adapter_train_fraction"])
    training_corpus_size = int(pretraining["training_corpus_size"])
    if int(sampled_corpus_size * train_fraction) != training_corpus_size:
        raise SequenceBaselineConfigError("LaMBO-2 must account for Cortex's fixed training split")
    token_exposures = training_corpus_size * int(pretraining["epochs"]) * int(pretraining["sequence_length"])
    reference_exposures = (
        int(pretraining["reference_updates"])
        * int(pretraining["reference_batch_size"])
        * int(pretraining["reference_sequence_length"])
    )
    if token_exposures != reference_exposures or token_exposures != int(pretraining["matched_token_exposures"]):
        raise SequenceBaselineConfigError("LaMBO-2 pretraining must match the CTMC token-position exposure")
    online = lambo.get("online", {})
    if int(online.get("initial_training_epochs", 0)) != 1 or int(online.get("retraining_epochs", 0)) != 1:
        raise SequenceBaselineConfigError("LaMBO-2 must persist one initial scored epoch and refit one epoch per round")
    lambo_arm = next(arm for arm in raw["arms"] if arm["name"] == "lambo2")
    if (
        lambo_arm.get("runner") != "lambo2"
        or lambo_arm.get("pretraining_scores") != "forbidden"
        or int(lambo_arm.get("online_initial_archive", 0)) != 64
    ):
        raise SequenceBaselineConfigError("LaMBO-2 must use reward-blind pretraining and the shared 64-point archive")
    holo = raw.get("holo")
    holo_arm = next(arm for arm in raw["arms"] if arm["name"] == HOLO_ARM)
    if not isinstance(holo, dict) or (
        holo_arm.get("runner") != "holo_ga"
        or holo_arm.get("proposal_operator") != "holo_discrete_evolution"
        or int(holo_arm.get("online_initial_archive", 0)) != 64
        or holo.get("initialization") != "tile_prepaid_archive_then_released_survive_recombine_mutate"
        or int(holo.get("initial_archive_size", 0)) != 64
        or int(holo.get("population_size", 0)) != 484
        or int(holo.get("generation_count", 0)) != 4
        or float(holo.get("mutation_prob", 0.0)) != 1.0 / 256.0
        or float(holo.get("recombine_prob", 0.0)) != 1.0 / 256.0
        or float(holo.get("survival_quantile", 0.0)) != 0.01
        or int(holo.get("vocab_size", 0)) != int(condition["vocab_size"])
        or holo.get("alphabet") != "ACDEFGHIKLMNPQRSTVWY"
        or holo.get("device") != "cpu"
        or holo.get("duplicate_policy") != "score_all_like_released_holo"
        or holo.get("feasibility_policy") != "black_box_score_all"
    ):
        raise SequenceBaselineConfigError("Holo GA must use the locked prepaid-archive DiscreteEvolution protocol")
    if int(holo["initial_archive_size"]) + int(holo["generation_count"]) * int(
        holo["population_size"]
    ) != int(generation["max_selected_archive_evaluations"]):
        raise SequenceBaselineConfigError("Holo GA must consume exactly 64 + 4*484 = 2,000 charged calls")
    return study


__all__ = [
    "ADDITIVE_ARMS",
    "ARMS",
    "CONDITION",
    "CONFIG_FILENAME",
    "HOLO_ARM",
    "LEGACY_ARMS",
    "PREEXISTING_ARMS",
    "PROTOCOL_VERSION",
    "SEED_BATCHES",
    "STUDY",
    "SequenceBaselineConfigError",
    "SequenceBaselineStudyConfig",
    "load_sequence_baseline_study",
]
