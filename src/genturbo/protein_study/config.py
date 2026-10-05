"""Validated declaration for the GFP/AAV paper cohort."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

STUDY = "protein_design_v1"
PROTOCOL_VERSION = 1
CONFIG_FILENAME = "online.json"
TASKS = ("gfp_medium", "gfp_hard", "aav_medium", "aav_hard")
PRIMARY_METHODS = (
    "unconditional",
    "rt_random",
    "rt_gp",
    "tr_random",
    "tr_gp",
    "holo_ga",
    "adalead",
    "bo_evo",
    "lambo2",
)
LEGACY_PRIMARY_METHODS = PRIMARY_METHODS[:5]
ADDITIVE_METHODS = PRIMARY_METHODS[5:]
DIAGNOSTIC_METHODS = ("myopic_oracle",)
ALL_METHODS = (*PRIMARY_METHODS, *DIAGNOSTIC_METHODS)
SEED_BATCHES = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9),
    (10, 11, 12, 13, 14),
    (15, 16, 17, 18, 19),
)
PRIMARY_CELL_COUNT = 720
DIAGNOSTIC_CELL_COUNT = 40
TOTAL_CELL_COUNT = PRIMARY_CELL_COUNT + DIAGNOSTIC_CELL_COUNT
NEW_CELL_COUNT = 524


class ProteinStudyConfigError(ValueError):
    """Raised when the active protein declaration violates the locked protocol."""


@dataclass(frozen=True, slots=True)
class ProteinStudyConfig:
    path: Path
    raw: dict[str, Any]
    repo_root: Path

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(seed for batch in SEED_BATCHES for seed in batch)

    @property
    def methods(self) -> dict[str, dict[str, Any]]:
        return {str(method["name"]): method for method in self.raw["methods"]}

    @property
    def tasks(self) -> dict[str, dict[str, Any]]:
        return cast(dict[str, dict[str, Any]], self.raw["tasks"])


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProteinStudyConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProteinStudyConfigError(f"{path} must contain a JSON object")
    return value


def _identity(task: str, method: str, seed: int) -> str:
    return f"{task}:{method}:seed{seed}"


def _expected_new_cells(raw: dict[str, Any]) -> set[str]:
    cells = {_identity(task, method, seed) for task in TASKS for method in ADDITIVE_METHODS for seed in range(20)}
    cells.update(
        _identity(task, method, seed) for task in TASKS for method in LEGACY_PRIMARY_METHODS for seed in range(10, 20)
    )
    cells.update(str(value) for value in raw["legacy"]["known_missing_primary_cells"])
    return cells


def _declared_launch_cells(raw: dict[str, Any]) -> list[str]:
    known_missing = [str(value) for value in raw["legacy"]["known_missing_primary_cells"]]
    cells: list[str] = []
    for tranche in raw["launch_tranches"]:
        tranche_cells = [
            _identity(task, str(method), int(seed))
            for task in TASKS
            for method in tranche["methods"]
            for seed in tranche["seeds"]
        ]
        if tranche.get("include_known_missing_legacy_cells") is True:
            tranche_cells.extend(known_missing)
        if len(tranche_cells) != int(tranche["expected_cells"]):
            raise ProteinStudyConfigError(
                f"launch tranche {tranche.get('tranche')} expands to {len(tranche_cells)} cells, "
                f"expected {tranche.get('expected_cells')}"
            )
        cells.extend(tranche_cells)
    return cells


def _validate(raw: dict[str, Any], path: Path) -> None:
    identity = (
        raw.get("schema_version"),
        raw.get("protocol_version"),
        raw.get("study"),
        raw.get("workflow"),
    )
    if identity != (1, PROTOCOL_VERSION, STUDY, "online"):
        raise ProteinStudyConfigError(f"{path} does not identify the active {STUDY} protocol")
    if raw.get("run_dir") != f"runs/{STUDY}/online":
        raise ProteinStudyConfigError("protein run_dir must mirror the active study name")
    if raw.get("pretraining_dir") != f"runs/{STUDY}/pretraining":
        raise ProteinStudyConfigError("protein pretraining_dir must mirror the active study name")
    if raw.get("results_dir") != f"results/{STUDY}/online":
        raise ProteinStudyConfigError("protein results_dir must mirror the active study name")
    if tuple(raw.get("task_order", ())) != TASKS or tuple(raw.get("tasks", {})) != TASKS:
        raise ProteinStudyConfigError("protein tasks must be the locked four-task order")
    if raw.get("seed_batches") != [list(batch) for batch in SEED_BATCHES]:
        raise ProteinStudyConfigError("seed_batches must be four five-seed tranches covering 0..19")

    methods = raw.get("methods")
    if not isinstance(methods, list) or tuple(method.get("name") for method in methods) != ALL_METHODS:
        raise ProteinStudyConfigError("methods must contain the nine primary systems and privileged diagnostic")
    cohorts = {str(method["name"]): method.get("cohort") for method in methods}
    if any(cohorts[name] != "primary" for name in PRIMARY_METHODS) or cohorts["myopic_oracle"] != "diagnostic":
        raise ProteinStudyConfigError("method cohort labels do not separate primary and diagnostic rows")

    budget = raw.get("budget", {})
    initial = int(budget.get("initial_archive_calls", -1))
    batches = int(budget.get("post_initial_batches", -1))
    batch_size = int(budget.get("acquisition_batch_size", -1))
    total = int(budget.get("total_oracle_calls", -1))
    if (initial, batches, batch_size, total) != (128, 234, 8, 2000) or initial + batches * batch_size != total:
        raise ProteinStudyConfigError("protein oracle budget must be exactly 128 + 234x8 = 2,000")
    if int(budget.get("unconditional_samples", -1)) != total:
        raise ProteinStudyConfigError("unconditional must score exactly 2,000 cold-start samples")

    gp = raw.get("gp", {})
    if (
        gp.get("feature_space") != "onehot_tanimoto"
        or gp.get("outputscale") != 1.0
        or gp.get("learn_outputscale") is not False
        or gp.get("noise_variance", "missing") is not None
        or gp.get("fit_mean_and_noise_only") is not True
        or gp.get("trainable_parameters") != ["constant_mean", "likelihood_noise"]
        or int(gp.get("buffer_size", -1)) != 128
        or gp.get("buffer_mode") != "recent"
        or int(gp.get("hyperparameter_refit_interval", -1)) != 1
    ):
        raise ProteinStudyConfigError("protein GP must fix scale and refit only constant mean/noise every batch")

    by_method = {str(method["name"]): method for method in methods}
    pools = {name: by_method[name].get("proposal_pool_size") for name in PRIMARY_METHODS}
    expected_pools = {
        "unconditional": 2000,
        "rt_random": 24,
        "rt_gp": 256,
        "tr_random": 24,
        "tr_gp": 256,
        "holo_ga": None,
        "adalead": None,
        "bo_evo": 64,
        "lambo2": None,
    }
    if pools != expected_pools:
        raise ProteinStudyConfigError("protein proposal-pool sizes differ from the retained end-to-end protocols")
    holo = by_method["holo_ga"]
    if (
        int(holo.get("generations", -1)) != 4
        or int(holo.get("generation_size", -1)) != 468
        or holo.get("mutation_probability") != "inverse_sequence_length"
        or holo.get("recombination_probability") != "inverse_sequence_length"
        or float(holo.get("survival_quantile", -1.0)) != 0.01
        or holo.get("prepaid_initial_archive") is not True
        or holo.get("upstream")
        != {
            "repository": "https://github.com/prescient-design/holo-bench",
            "commit": "4d94731017e63a73008c94a99a1bee87585bf9b0",
            "source_url": (
                "https://github.com/prescient-design/holo-bench/blob/"
                "4d94731017e63a73008c94a99a1bee87585bf9b0/holo/optim/_discrete_evolution.py"
            ),
        }
        or initial + int(holo["generations"]) * int(holo["generation_size"]) != total
    ):
        raise ProteinStudyConfigError("protein Holo must use 128 prepaid labels plus four generations of 468")
    adalead = by_method["adalead"]
    if (
        adalead.get("proposal_operator") != "adalead_official"
        or int(adalead.get("surrogate_queries_per_batch", -1)) != 128
        or int(adalead.get("paid_batch_size", -1)) != batch_size
        or int(adalead.get("sequences_batch_size", -1)) != 8
        or int(adalead.get("eval_batch_size", -1)) != 20
        or int(adalead.get("mu", -1)) != 1
        or float(adalead.get("threshold", -1.0)) != 0.05
        or int(adalead.get("rho", -1)) != 0
        or adalead.get("alphabet") != "ACDEFGHIKLMNPQRSTVWY"
        or adalead.get("rng_backend") != "torch_global_seeded"
        or float(adalead.get("recombination_rate", -1.0)) != 0.0
        or float(adalead.get("crossover_rate", -1.0)) != 0.0
        or adalead.get("selection") != "top_surrogate_mean"
        or adalead.get("upstream")
        != {
            "repository": "https://github.com/samsinai/FLEXS",
            "commit": "dd409167b5575b51967d325b4e48aef3577a505f",
            "adalead_sha256": "d34958ae2035c2346c0bce769f2d0d89c63a62820003a5672d5295d85ac799b5",
            "sequence_utils_sha256": "f2cb4829531dbae974788d4aa0a96cdd9a2873470a214e20d344d94c1a82e8c4",
        }
    ):
        raise ProteinStudyConfigError("AdaLead mechanics differ from the locked adaptation")
    bo_evo = by_method["bo_evo"]
    if (
        bo_evo.get("proposal_operator") != "bo_evo"
        or int(bo_evo.get("children_per_selection", -1)) != 8
        or float(bo_evo.get("ucb_coefficient", -1.0)) != 0.2
        or float(bo_evo.get("parent_temperature", -1.0)) != 10.0
        or float(bo_evo.get("crossover_rate", -1.0)) != 0.0
        or bo_evo.get("selection") != "direct_ucb_screening"
    ):
        raise ProteinStudyConfigError("BO-EVO mechanics differ from the locked adaptation")
    lambo_method = by_method["lambo2"]
    if (
        int(lambo_method.get("direct_designs_per_batch", -1)) != 8
        or int(lambo_method.get("objective_refit_epochs_per_round", -1)) != 1
        or lambo_method.get("pretraining_scores") != "forbidden"
    ):
        raise ProteinStudyConfigError("LaMBO-2 online mechanics differ from the locked adaptation")

    expected_lengths = {"gfp_medium": 237, "gfp_hard": 237, "aav_medium": 28, "aav_hard": 28}
    published_sizes = {"gfp_medium": 2828, "gfp_hard": 2426, "aav_medium": 2139, "aav_hard": 3448}
    accepted_sizes = {"gfp_medium": 2828, "gfp_hard": 2426, "aav_medium": 2145, "aav_hard": 3453}
    for task_name, task in raw["tasks"].items():
        if int(task.get("sequence_length", -1)) != expected_lengths[task_name]:
            raise ProteinStudyConfigError(f"{task_name} sequence length drifted")
        if int(task.get("expected_training_size", -1)) != published_sizes[task_name]:
            raise ProteinStudyConfigError(f"{task_name} published reference training size drifted")
        if int(task.get("accepted_observed_training_size", -1)) != accepted_sizes[task_name]:
            raise ProteinStudyConfigError(f"{task_name} accepted observed training size drifted")
        archive = str(task.get("initial_archive", ""))
        digest = str(task.get("initial_archive_sha256", ""))
        if len(digest) != 64 or Path(archive).name != f"{digest}.jsonl.gz":
            raise ProteinStudyConfigError(f"{task_name} initial archive is not content addressed")

    lambo = raw.get("lambo2", {})
    lambo_online = lambo.get("online", {})
    checkpoint_validation = lambo.get("checkpoint_validation", {})
    partition = lambo.get("partition", {})
    if (
        lambo.get("description") != "label-free pretraining on the predeclared task training split"
        or lambo.get("alphabet") != "ACDEFGHIKLMNPQRSTVWY"
        or int(lambo.get("generative_training_epochs", -1)) != 1
        or int(lambo.get("batch_size", -1)) != 128
        or lambo.get("pretraining_accelerator") != "gpu"
        or lambo.get("pretraining_device") != "cuda"
        or lambo.get("requirements") != "environments/lambo2/uv.lock"
        or set(lambo.get("tasks", {})) != set(TASKS)
    ):
        raise ProteinStudyConfigError("LaMBO-2 pretraining declaration is incomplete")
    if partition != {
        "algorithm": "sha256_sequence_modulo",
        "validation_modulus": 10,
        "validation_residue": 0,
        "seed": 0,
    }:
        raise ProteinStudyConfigError("LaMBO-2 sequence-hash partition declaration has drifted")
    if (
        int(checkpoint_validation.get("sample_count_per_seed", -1)) != 64
        or checkpoint_validation.get("seeds") != [0, 1]
        or int(checkpoint_validation.get("determinism_repeats", -1)) != 2
        or float(checkpoint_validation.get("minimum_unique_fraction", -1.0)) != 0.5
    ):
        raise ProteinStudyConfigError("LaMBO-2 checkpoint sampling declaration has drifted")
    if (
        int(lambo_online.get("training_batch_size", -1)) != 128
        or int(lambo_online.get("initial_training_epochs", -1)) != 1
        or int(lambo_online.get("retraining_epochs", -1)) != 1
        or int(lambo_online.get("num_steps", -1)) != 1
        or int(lambo_online.get("mutations_per_step", -1)) != 2
        or int(lambo_online.get("max_guidance_updates", -1)) != 8
        or int(lambo_online.get("expected_transferred_tensors", -1)) != 17
    ):
        raise ProteinStudyConfigError("LaMBO-2 online adapter declaration has drifted")
    for task_name, task_pretraining in lambo.get("tasks", {}).items():
        expected_root = f"runs/{STUDY}/pretraining/{task_name}"
        expected_paths = {
            "corpus_dir": f"{expected_root}/corpus",
            "checkpoint": f"{expected_root}/generator.ckpt",
            "manifest": f"{expected_root}/pretraining_manifest.json",
            "validation_samples": f"{expected_root}/checkpoint_samples.jsonl",
            "validation_manifest": f"{expected_root}/checkpoint_validation.json",
        }
        if task_pretraining != expected_paths:
            raise ProteinStudyConfigError(f"{task_name} LaMBO-2 pretraining paths have drifted")

    legacy = raw.get("legacy", {})
    if (
        legacy.get("compatibility_commit") != "24aca235"
        or int(legacy.get("expected_primary_cells", -1)) != 196
        or int(legacy.get("expected_diagnostic_cells", -1)) != DIAGNOSTIC_CELL_COUNT
        or set(legacy.get("condition_map", {})) != set(TASKS)
    ):
        raise ProteinStudyConfigError("legacy compatibility/import declaration has drifted")
    expected_legacy_methods = {*LEGACY_PRIMARY_METHODS, *DIAGNOSTIC_METHODS}
    if any(set(mapping) != expected_legacy_methods for mapping in legacy["condition_map"].values()):
        raise ProteinStudyConfigError("legacy condition map must cover five deployable arms plus oracle diagnostic")
    missing = legacy.get("known_missing_primary_cells", [])
    if len(missing) != 4 or len(set(missing)) != 4:
        raise ProteinStudyConfigError("legacy import must retain the four known missing primary cells")

    execution = raw.get("execution", {})
    if execution != {
        "backend": "local",
        "gpu_runtime_classes": ["evodiff", "lambo2"],
        "cpu_runtime_classes": ["gp_evolution", "holo"],
    }:
        raise ProteinStudyConfigError("local runtime declaration has drifted")

    declared = _declared_launch_cells(raw)
    expected = _expected_new_cells(raw)
    if len(declared) != NEW_CELL_COUNT or len(set(declared)) != NEW_CELL_COUNT or set(declared) != expected:
        raise ProteinStudyConfigError("launch tranches must partition exactly the 524 new protein cells")
    if PRIMARY_CELL_COUNT != len(TASKS) * len(PRIMARY_METHODS) * 20:
        raise AssertionError("internal primary cell count is inconsistent")
    if DIAGNOSTIC_CELL_COUNT != len(TASKS) * 10:
        raise AssertionError("internal diagnostic cell count is inconsistent")
    metrics = raw.get("metrics", {})
    if (
        metrics.get("checkpoints") != [128, 250, 500, 1000, 2000]
        or metrics.get("carry_incumbent_forward") is not True
        or metrics.get("paired_reference") != "tr_gp"
        or int(metrics.get("bootstrap_resamples", -1)) != 10_000
        or int(metrics.get("bootstrap_rng_seed", -1)) != 20_260_824
        or float(metrics.get("win_tie_loss_tolerance", -1.0)) != 1e-12
        or metrics.get("checkpoint_boundary_rule") != "last_completed_batch_carry_forward"
    ):
        raise ProteinStudyConfigError("protein metric/checkpoint/bootstrap protocol has drifted")


def load_protein_study(path: Path | str, *, repo_root: Path | str) -> ProteinStudyConfig:
    """Load the sole active protein declaration and validate its locked matrix."""

    root = Path(repo_root).expanduser().resolve()
    config_path = Path(path).expanduser()
    config_path = (config_path if config_path.is_absolute() else root / config_path).resolve()
    if config_path.name != CONFIG_FILENAME:
        raise ProteinStudyConfigError(f"expected {CONFIG_FILENAME}, got {config_path.name}")
    if sorted(item.name for item in config_path.parent.glob("*.json")) != [CONFIG_FILENAME]:
        raise ProteinStudyConfigError(f"{config_path.parent} must contain exactly {CONFIG_FILENAME}")
    raw = _read_object(config_path)
    _validate(raw, config_path)
    return ProteinStudyConfig(config_path, raw, root)


__all__ = [
    "ADDITIVE_METHODS",
    "ALL_METHODS",
    "CONFIG_FILENAME",
    "DIAGNOSTIC_CELL_COUNT",
    "DIAGNOSTIC_METHODS",
    "LEGACY_PRIMARY_METHODS",
    "NEW_CELL_COUNT",
    "PRIMARY_CELL_COUNT",
    "PRIMARY_METHODS",
    "PROTOCOL_VERSION",
    "SEED_BATCHES",
    "STUDY",
    "TASKS",
    "TOTAL_CELL_COUNT",
    "ProteinStudyConfig",
    "ProteinStudyConfigError",
    "load_protein_study",
]
