"""Frozen-archive evaluation for the fixed ``ehrlich_kernel_v1`` kernels."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata

from ehrlich.ehrlich_instance import EhrlichInstance
from surrogate.features import (
    AdditiveSpacedNGramGPRegressor,
    SpacedNGramGPRegressor,
    TanimotoGPRegressor,
)
from surrogate.model import make_local_spaced_ngram_offsets
from genturbo.artifacts import sha256_file
from genturbo.io import iter_jsonl
from genturbo.kernel_study.online_analysis import analyze_sweep
from genturbo.serialization import atomic_write, canonical_json_bytes, csv_bytes

SCHEMA_VERSION = 1
_KERNELS = (
    "onehot_tanimoto",
    "spaced_pair",
    "local_spaced_trigram",
    "mixed_window_spaced_ngram_additive",
    "spaced_ngram",
)
_CONFIG_KEYS = {
    "schema_version",
    "run_root",
    "output_dir",
    "random_seed",
    "repeats_per_seed",
    "train_sizes",
    "test_pool_size",
    "top_k",
    "score_batch_size",
    "kernels",
    "spaced_pair_window",
    "spaced_pair_window_grid",
    "local_spaced_trigram_window_grid",
    "mixed_window_spaced_ngram_orders",
    "mixed_window_spaced_ngram_windows",
    "mixed_window_spaced_ngram_weights",
    "gp",
}
_GP_KEYS = {"device", "dtype", "jitter", "outputscale", "learn_outputscale"}


class EhrlichKernelBenchmarkError(RuntimeError):
    """Raised when the fixed frozen-archive benchmark is invalid."""


@dataclass(frozen=True)
class FrozenPool:
    sequences: np.ndarray
    scores: np.ndarray
    source_seeds: tuple[frozenset[int], ...]
    source_methods: tuple[frozenset[str], ...]
    trace_hashes: Mapping[str, str]
    instance_path: Path
    instance_sha256: str


@dataclass(frozen=True)
class FrozenSplit:
    split_id: str
    heldout_seed: int
    repeat: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def _normalize_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise EhrlichKernelBenchmarkError(f"unknown benchmark config keys: {', '.join(unknown)}")
    gp_raw = raw.get("gp", {})
    if not isinstance(gp_raw, Mapping):
        raise EhrlichKernelBenchmarkError("gp must be a JSON object")
    unknown_gp = sorted(set(gp_raw) - _GP_KEYS)
    if unknown_gp:
        raise EhrlichKernelBenchmarkError(f"unknown gp config keys: {', '.join(unknown_gp)}")

    config: dict[str, Any] = {
        "schema_version": int(raw.get("schema_version", SCHEMA_VERSION)),
        "run_root": str(raw.get("run_root", "runs/ehrlich_kernel_v1/datasets/l128_q4_k4")),
        "output_dir": str(raw.get("output_dir", "results/ehrlich_kernel_v1/static/l128_q4_k4")),
        "random_seed": int(raw.get("random_seed", 20260716)),
        "repeats_per_seed": int(raw.get("repeats_per_seed", 1)),
        "train_sizes": [int(value) for value in raw.get("train_sizes", [512])],
        "test_pool_size": int(raw.get("test_pool_size", 256)),
        "top_k": int(raw.get("top_k", 8)),
        "score_batch_size": int(raw.get("score_batch_size", 512)),
        "kernels": [str(value) for value in raw.get("kernels", _KERNELS)],
        "spaced_pair_window": int(raw.get("spaced_pair_window", 32)),
        "spaced_pair_window_grid": [int(value) for value in raw.get("spaced_pair_window_grid", [32, 48])],
        "local_spaced_trigram_window_grid": [int(value) for value in raw.get("local_spaced_trigram_window_grid", [16])],
        "mixed_window_spaced_ngram_orders": [
            int(value) for value in raw.get("mixed_window_spaced_ngram_orders", [2, 3])
        ],
        "mixed_window_spaced_ngram_windows": [
            int(value) for value in raw.get("mixed_window_spaced_ngram_windows", [48, 16])
        ],
        "mixed_window_spaced_ngram_weights": [
            float(value) for value in raw.get("mixed_window_spaced_ngram_weights", [0.5, 0.5])
        ],
        "gp": {
            "device": str(gp_raw.get("device", "cpu")),
            "dtype": str(gp_raw.get("dtype", "double")),
            "jitter": float(gp_raw.get("jitter", 1e-4)),
            "outputscale": float(gp_raw.get("outputscale", 1.0)),
            "learn_outputscale": bool(gp_raw.get("learn_outputscale", False)),
        },
    }
    if config["schema_version"] != SCHEMA_VERSION:
        raise EhrlichKernelBenchmarkError(f"unsupported schema_version {config['schema_version']}")
    if config["kernels"] != list(_KERNELS):
        raise EhrlichKernelBenchmarkError("kernels must be the fixed ehrlich_kernel_v1 kernel families")
    if config["spaced_pair_window_grid"] != [32, 48]:
        raise EhrlichKernelBenchmarkError("spaced_pair_window_grid must be [32, 48]")
    if config["local_spaced_trigram_window_grid"] != [16]:
        raise EhrlichKernelBenchmarkError("local_spaced_trigram_window_grid must be [16]")
    additive = (
        config["mixed_window_spaced_ngram_orders"],
        config["mixed_window_spaced_ngram_windows"],
        config["mixed_window_spaced_ngram_weights"],
    )
    if additive != ([2, 3], [48, 16], [0.5, 0.5]):
        raise EhrlichKernelBenchmarkError("additive kernel must be fixed pair-W48 plus trigram-W16 with equal weights")
    if config["gp"]["outputscale"] != 1.0 or config["gp"]["learn_outputscale"]:
        raise EhrlichKernelBenchmarkError("outputscale=1.0 must remain fixed")
    if config["gp"]["device"] != "cpu" or config["gp"]["dtype"] != "double":
        raise EhrlichKernelBenchmarkError("the fixed study uses CPU double precision")
    if len(config["train_sizes"]) != 1 or config["train_sizes"][0] < 2:
        raise EhrlichKernelBenchmarkError("the fixed study requires one training size of at least two")
    if config["repeats_per_seed"] < 1:
        raise EhrlichKernelBenchmarkError("repeats_per_seed must be positive")
    if config["test_pool_size"] < 2 or not 1 <= config["top_k"] <= config["test_pool_size"]:
        raise EhrlichKernelBenchmarkError("test_pool_size must be >= 2 and top_k must fit within it")
    if config["score_batch_size"] < 1:
        raise EhrlichKernelBenchmarkError("score_batch_size must be positive")
    return config


def load_config(path: Path | str) -> dict[str, Any]:
    """Load the internal resolved benchmark configuration used by timing."""
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EhrlichKernelBenchmarkError(f"cannot read benchmark config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EhrlichKernelBenchmarkError("benchmark config must be a JSON object")
    return _normalize_config(raw)


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def build_frozen_pool(
    run_root: Path | str,
    *,
    repo_root: Path | str,
    score_batch_size: int = 512,
) -> FrozenPool:
    """Load the dataset-finalisation table after checking its complete provenance."""
    repository = Path(repo_root).expanduser().resolve()
    root = _resolve_repo_path(repository, run_root)
    if score_batch_size < 1:
        raise EhrlichKernelBenchmarkError("score_batch_size must be positive")
    analysis = analyze_sweep(root, repo_root=repository)
    if not analysis["runs"]:
        raise EhrlichKernelBenchmarkError(f"no completed runs found below {root}")

    trace_hashes: dict[str, str] = {}
    instance_path: Path | None = None
    instance_hashes: set[str] = set()
    run_methods: set[str] = set()
    source_seed_ids: set[int] = set()
    for run in analysis["runs"]:
        run_dir = root / str(run["run_directory"])
        manifest_path = run_dir / str(run["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reward = manifest.get("resolved_config", {}).get("reward", {})
        candidate_instance = _resolve_repo_path(repository, reward.get("instance_path", ""))
        if not candidate_instance.is_file():
            raise EhrlichKernelBenchmarkError(f"reward instance is missing: {candidate_instance}")
        instance_path = candidate_instance if instance_path is None else instance_path
        if candidate_instance != instance_path:
            raise EhrlichKernelBenchmarkError("completed cells do not share one reward instance path")
        instance_hashes.add(str(manifest.get("input_hashes", {}).get("reward_instance", "")))

        trace_path = run_dir / "proposal_traces.jsonl.gz"
        if not trace_path.is_file():
            raise EhrlichKernelBenchmarkError(f"full proposal trace is missing: {trace_path}")
        trace_hashes[_portable_path(trace_path, repository)] = sha256_file(trace_path)
        source_seed_ids.add(int(run["seed"]))
        run_methods.add(str(run["method"]))

    if instance_path is None or len(instance_hashes) != 1 or "" in instance_hashes:
        raise EhrlichKernelBenchmarkError("completed cells do not provide one pinned reward-instance hash")
    expected_instance_hash = next(iter(instance_hashes))
    if sha256_file(instance_path) != expected_instance_hash:
        raise EhrlichKernelBenchmarkError("reward instance checksum no longer matches completed manifests")
    instance = torch.load(instance_path, map_location="cpu", weights_only=False)
    if not isinstance(instance, EhrlichInstance):
        raise EhrlichKernelBenchmarkError("reward instance file did not contain an EhrlichInstance")

    provenance_path = root.parent / "provenance.json"
    if not provenance_path.is_file():
        provenance_path = repository / "results/ehrlich_kernel_v1/datasets/provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EhrlichKernelBenchmarkError("dataset finalisation is required before static or timing runs") from exc
    if (
        provenance.get("study") != "ehrlich_kernel_v1"
        or provenance.get("workflow") != "datasets"
        or provenance.get("status") != "completed"
    ):
        raise EhrlichKernelBenchmarkError("dataset provenance is not a completed ehrlich_kernel_v1 finalisation")
    condition = root.name
    condition_records = [
        record
        for record in provenance.get("conditions", [])
        if isinstance(record, Mapping) and record.get("condition") == condition
    ]
    if len(condition_records) != 1:
        raise EhrlichKernelBenchmarkError(f"dataset provenance does not identify exactly one {condition} cohort")
    condition_record = condition_records[0]
    recorded_trace_hashes = condition_record.get("trace_sha256")
    if recorded_trace_hashes != dict(sorted(trace_hashes.items())):
        raise EhrlichKernelBenchmarkError("proposal traces no longer match dataset-finalisation provenance")

    frozen_path = _resolve_repo_path(repository, str(condition_record.get("frozen_candidates", "")))
    expected_frozen_path = root / "frozen_candidates.jsonl.gz"
    if frozen_path != expected_frozen_path or not frozen_path.is_file():
        raise EhrlichKernelBenchmarkError(f"canonical frozen candidate table is missing: {expected_frozen_path}")
    if sha256_file(frozen_path) != str(condition_record.get("frozen_candidates_sha256", "")):
        raise EhrlichKernelBenchmarkError("frozen candidate table no longer matches dataset provenance")

    ordered: list[tuple[int, ...]] = []
    scores: list[float] = []
    memberships: list[frozenset[int]] = []
    seen_sequences: set[tuple[int, ...]] = set()
    sequence_length: int | None = None
    for row_number, record in enumerate(iter_jsonl(frozen_path), start=1):
        raw_sequence = record.get("sequence")
        raw_memberships = record.get("source_seed_memberships")
        score = record.get("score")
        if (
            record.get("condition") != condition
            or not isinstance(raw_sequence, list)
            or not raw_sequence
            or any(isinstance(token, bool) or not isinstance(token, int) for token in raw_sequence)
        ):
            raise EhrlichKernelBenchmarkError(f"frozen candidate row {row_number} has an invalid sequence")
        sequence = tuple(int(token) for token in raw_sequence)
        if sequence in seen_sequences:
            raise EhrlichKernelBenchmarkError(f"frozen candidate row {row_number} duplicates a sequence")
        seen_sequences.add(sequence)
        sequence_length = len(sequence) if sequence_length is None else sequence_length
        if len(sequence) != sequence_length or int(record.get("sequence_length", -1)) != sequence_length:
            raise EhrlichKernelBenchmarkError("frozen candidate table mixes sequence lengths")
        if (
            not isinstance(raw_memberships, list)
            or not raw_memberships
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_memberships)
        ):
            raise EhrlichKernelBenchmarkError(f"frozen candidate row {row_number} has invalid source seeds")
        source_seeds = frozenset(int(seed) for seed in raw_memberships)
        if len(source_seeds) != len(raw_memberships) or not source_seeds.issubset(source_seed_ids):
            raise EhrlichKernelBenchmarkError(f"frozen candidate row {row_number} has unknown source seeds")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise EhrlichKernelBenchmarkError(f"frozen candidate row {row_number} has an invalid score")
        ordered.append(sequence)
        scores.append(float(score))
        memberships.append(source_seeds)

    recorded_count = condition_record.get("frozen_candidate_count")
    if len(ordered) != recorded_count or len(ordered) < 2:
        raise EhrlichKernelBenchmarkError(
            f"frozen candidate count is {len(ordered)}; provenance records {recorded_count}"
        )
    order = sorted(range(len(ordered)), key=ordered.__getitem__)
    sequence_array = np.asarray([ordered[index] for index in order], dtype=np.int64)
    score_array = np.asarray([scores[index] for index in order], dtype=np.float64)

    return FrozenPool(
        sequences=sequence_array,
        scores=score_array,
        source_seeds=tuple(memberships[index] for index in order),
        source_methods=tuple(frozenset(run_methods) for _ in order),
        trace_hashes=dict(sorted(trace_hashes.items())),
        instance_path=instance_path,
        instance_sha256=expected_instance_hash,
    )


def make_frozen_splits(
    pool: FrozenPool,
    *,
    train_size: int,
    test_pool_size: int,
    repeats_per_seed: int,
    random_seed: int,
) -> list[FrozenSplit]:
    """Create deterministic, leakage-free leave-one-source-seed-out splits."""
    all_seeds = sorted({seed for source in pool.source_seeds for seed in source})
    splits: list[FrozenSplit] = []
    for heldout_seed in all_seeds:
        train_population = np.asarray(
            [index for index, source in enumerate(pool.source_seeds) if heldout_seed not in source], dtype=np.int64
        )
        test_population = np.asarray(
            [index for index, source in enumerate(pool.source_seeds) if heldout_seed in source], dtype=np.int64
        )
        if train_population.size < train_size or test_population.size < test_pool_size:
            raise EhrlichKernelBenchmarkError(
                f"seed {heldout_seed} has {train_population.size} leakage-free training and "
                f"{test_population.size} held-out candidates; requested {train_size} and {test_pool_size}"
            )
        for repeat in range(repeats_per_seed):
            rng = np.random.default_rng(np.random.SeedSequence([random_seed, heldout_seed, repeat]))
            train_indices = rng.permutation(train_population)[:train_size]
            test_indices = rng.permutation(test_population)[:test_pool_size]
            if set(train_indices.tolist()) & set(test_indices.tolist()):
                raise AssertionError("frozen split leaked an exact sequence")
            splits.append(
                FrozenSplit(
                    split_id=f"seed{heldout_seed}_repeat{repeat}",
                    heldout_seed=heldout_seed,
                    repeat=repeat,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            )
    return splits


def _onehot(sequences: np.ndarray, vocab_size: int) -> np.ndarray:
    features = np.eye(vocab_size, dtype=np.float64)[sequences]
    return np.asarray(features.reshape(sequences.shape[0], -1), dtype=np.float64)


def _likelihood_noise_variance(regressor: Any) -> float:
    model = getattr(regressor, "model_", None)
    noise = getattr(getattr(model, "likelihood", None), "noise", None)
    if noise is None:
        raise EhrlichKernelBenchmarkError("fitted GP does not expose likelihood noise")
    value = float(torch.as_tensor(noise).detach().cpu().mean().item())
    if not math.isfinite(value) or value <= 0.0:
        raise EhrlichKernelBenchmarkError(f"fitted likelihood noise is invalid: {value}")
    return value


def _gp_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fixed-scale mean-and-noise fitting policy used by the study."""
    gp = config["gp"]
    return {
        "device": str(gp["device"]),
        "dtype": str(gp["dtype"]),
        "jitter": float(gp["jitter"]),
        "noise_variance": None,
        "outputscale": 1.0,
        "learn_outputscale": False,
        "fit_mean_and_noise_only": True,
    }


def _fit_regressor(
    name: str,
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    instance: EhrlichInstance,
    config: Mapping[str, Any],
    fit_seed: int,
) -> tuple[Any, None, str]:
    """Fit one of the five families that expand to the six study variants."""
    kwargs = _gp_kwargs(config)
    if name == "onehot_tanimoto":
        regressor = TanimotoGPRegressor(**kwargs)
        features = _onehot(train_x, int(instance.cfg.vocab_size))
    elif name == "spaced_ngram":
        regressor = SpacedNGramGPRegressor(
            instance=instance,
            seq_len=int(instance.cfg.seq_len),
            vocab_size=int(instance.cfg.vocab_size),
            offsets=instance.spacings,
            **kwargs,
        )
        features = train_x
    elif name == "mixed_window_spaced_ngram_additive":
        orders = [int(value) for value in config["mixed_window_spaced_ngram_orders"]]
        windows = [int(value) for value in config["mixed_window_spaced_ngram_windows"]]
        groups = [make_local_spaced_ngram_offsets(order, window) for order, window in zip(orders, windows, strict=True)]
        if any(int(offsets.max().item()) >= int(instance.cfg.seq_len) for offsets in groups):
            raise EhrlichKernelBenchmarkError("additive offsets do not fit the Ehrlich sequence length")
        regressor = AdditiveSpacedNGramGPRegressor(
            instance=instance,
            seq_len=int(instance.cfg.seq_len),
            vocab_size=int(instance.cfg.vocab_size),
            offset_groups=groups,
            weights=config["mixed_window_spaced_ngram_weights"],
            **kwargs,
        )
        features = train_x
    elif name in {"spaced_pair", "local_spaced_trigram"}:
        order = 2 if name == "spaced_pair" else 3
        window = int(config.get("local_window", config["spaced_pair_window"]))
        offsets = make_local_spaced_ngram_offsets(order, window)
        if int(offsets.max().item()) >= int(instance.cfg.seq_len):
            raise EhrlichKernelBenchmarkError("kernel window must not exceed the Ehrlich sequence length")
        regressor = SpacedNGramGPRegressor(
            instance=instance,
            seq_len=int(instance.cfg.seq_len),
            vocab_size=int(instance.cfg.vocab_size),
            offsets=offsets,
            **kwargs,
        )
        features = train_x
    else:
        raise ValueError(f"unknown kernel {name}")
    torch.manual_seed(int(fit_seed))
    regressor.fit(features, train_y)
    return regressor, None, ""


def _predict(regressor: Any, name: str, test_x: np.ndarray, instance: EhrlichInstance) -> tuple[np.ndarray, np.ndarray]:
    features = _onehot(test_x, int(instance.cfg.vocab_size)) if name == "onehot_tanimoto" else test_x
    return regressor.predict(features), regressor.predict_std(features)


def _expected_improvement(mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
    mean_t = torch.as_tensor(mean, dtype=torch.double)
    std_t = torch.as_tensor(std, dtype=torch.double).clamp_min(0.0)
    improvement = mean_t - float(best)
    safe_std = torch.where(std_t > 0.0, std_t, torch.ones_like(std_t))
    z = improvement / safe_std
    density = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    expected = improvement * cdf + std_t * density
    return torch.where(std_t > 0.0, expected, improvement.clamp_min(0.0)).numpy()


def _metrics(
    *,
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    train_best: float,
    top_k: int,
) -> dict[str, float | int]:
    errors = mean - truth
    rank_truth = rankdata(truth, method="average")
    rank_mean = rankdata(mean, method="average")
    spearman = (
        0.0
        if np.std(rank_truth) == 0.0 or np.std(rank_mean) == 0.0
        else float(np.corrcoef(rank_truth, rank_mean)[0, 1])
    )
    variance = np.square(np.maximum(std, 1e-9))
    nlpd = 0.5 * np.log(2.0 * math.pi * variance) + 0.5 * np.square(errors) / variance
    ei = _expected_improvement(mean, std, train_best)
    indices = np.arange(truth.shape[0])
    selected = np.lexsort((indices, -ei))[:top_k]
    true_top = np.lexsort((indices, -truth))[:top_k]
    pool_best = float(np.max(truth))
    selected_best = float(np.max(truth[selected]))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "spearman": spearman,
        "latent_gaussian_nlpd": float(np.mean(nlpd)),
        "latent_90_coverage": float(np.mean(np.abs(errors) <= 1.6448536269514722 * np.sqrt(variance))),
        "pool_best": pool_best,
        "selected_best": selected_best,
        "simple_regret": pool_best - selected_best,
        "top_k_recall": float(len(set(selected.tolist()) & set(true_top.tolist())) / top_k),
        "pool_best_selected": int(int(np.argmax(truth)) in set(selected.tolist())),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["kernel"]), int(row["train_size"]))].append(row)
    metrics = (
        "rmse",
        "mae",
        "spearman",
        "latent_gaussian_nlpd",
        "latent_90_coverage",
        "simple_regret",
        "top_k_recall",
        "pool_best_selected",
        "likelihood_noise_variance_standardized",
    )
    output: list[dict[str, Any]] = []
    for (kernel, train_size), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "kernel": kernel,
            "train_size": train_size,
            "split_count": len(group),
            "independent_seed_count": len({int(row["heldout_seed"]) for row in group}),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        output.append(record)
    return output


def _paired_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        (str(row["split_id"]), int(row["train_size"])): row
        for row in rows
        if str(row.get("base_kernel", row["kernel"])) == "onehot_tanimoto"
    }
    comparisons: list[dict[str, Any]] = []
    for candidate in sorted(rows, key=lambda row: (str(row["split_id"]), str(row["kernel"]))):
        if str(candidate.get("base_kernel", candidate["kernel"])) == "onehot_tanimoto":
            continue
        key = (str(candidate["split_id"]), int(candidate["train_size"]))
        baseline = baselines.get(key)
        if baseline is None:
            raise EhrlichKernelBenchmarkError(f"missing one-hot baseline for {key[0]}/{key[1]}")
        comparison = {
            "split_id": key[0],
            "heldout_seed": int(candidate["heldout_seed"]),
            "repeat": int(candidate["repeat"]),
            "train_size": key[1],
            "kernel": str(candidate["kernel"]),
            "simple_regret_delta_vs_onehot": float(candidate["simple_regret"]) - float(baseline["simple_regret"]),
            "spearman_delta_vs_onehot": float(candidate["spearman"]) - float(baseline["spearman"]),
            "rmse_delta_vs_onehot": float(candidate["rmse"]) - float(baseline["rmse"]),
            "regret_win_vs_onehot": int(float(candidate["simple_regret"]) < float(baseline["simple_regret"])),
        }
        comparisons.append(comparison)
    return comparisons


def _sign_flip_p_value(values: np.ndarray) -> tuple[float, str]:
    differences = np.asarray(values, dtype=np.float64).reshape(-1)
    observed = abs(float(differences.mean()))
    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * 8.0
    if differences.size <= 16:
        sums = np.asarray([0.0], dtype=np.float64)
        for value in differences:
            sums = np.concatenate((sums + value, sums - value))
        return float(np.mean(np.abs(sums / differences.size) >= observed - tolerance)), "exact"
    rng = np.random.default_rng(0)
    trials = 100_000
    exceedances = 0
    for start in range(0, trials, 10_000):
        count = min(10_000, trials - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, differences.size))
        exceedances += int(np.count_nonzero(np.abs((signs * differences).mean(axis=1)) >= observed - tolerance))
    return float((exceedances + 1) / (trials + 1)), "monte_carlo_100000"


def _paired_aggregates(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in comparisons:
        groups[(str(row["kernel"]), int(row["train_size"]))].append(row)
    metrics = (
        ("simple_regret", "simple_regret_delta_vs_onehot", "lower"),
        ("rmse", "rmse_delta_vs_onehot", "lower"),
        ("spearman", "spearman_delta_vs_onehot", "higher"),
    )
    output: list[dict[str, Any]] = []
    for (kernel, train_size), group in sorted(groups.items()):
        for metric, column, preferred in metrics:
            raw = np.asarray([float(row[column]) for row in group], dtype=np.float64)
            improvement = -raw if preferred == "lower" else raw
            by_seed: dict[int, list[float]] = defaultdict(list)
            for row, value in zip(group, improvement.tolist(), strict=True):
                by_seed[int(row["heldout_seed"])].append(value)
            independent = np.asarray([np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=np.float64)
            p_value, method = _sign_flip_p_value(independent)
            output.append(
                {
                    "kernel": kernel,
                    "train_size": train_size,
                    "metric": metric,
                    "preferred": preferred,
                    "split_count": len(group),
                    "independent_seed_count": len(independent),
                    "raw_delta_mean_vs_onehot": float(raw.mean()),
                    "improvement_mean_vs_onehot": float(independent.mean()),
                    "improvement_std_vs_onehot": float(independent.std(ddof=1)) if len(independent) > 1 else 0.0,
                    "better_seed_fraction_vs_onehot": float(np.mean(independent > 0.0)),
                    "tie_seed_fraction_vs_onehot": float(np.mean(independent == 0.0)),
                    "two_sided_sign_flip_p": p_value,
                    "sign_flip_method": method,
                }
            )
    return output


def _kernel_window_variants(kernel: str, config: Mapping[str, Any]) -> tuple[int | None, ...]:
    """Expand the two pair windows and the single trigram window."""
    if kernel == "spaced_pair":
        return tuple(int(value) for value in config["spaced_pair_window_grid"])
    if kernel == "local_spaced_trigram":
        return tuple(int(value) for value in config["local_spaced_trigram_window_grid"])
    return (None,)


def _kernel_result_name(kernel: str, config: Mapping[str, Any], local_window: int | None) -> str:
    if kernel == "mixed_window_spaced_ngram_additive":
        return "pair_w48_trigram_w16_additive"
    return f"{kernel}_w{local_window}" if local_window is not None else kernel


def _fit_config(config: Mapping[str, Any], local_window: int | None) -> Mapping[str, Any]:
    if local_window is None:
        return config
    return {**dict(config), "local_window": int(local_window), "spaced_pair_window": int(local_window)}


def run_benchmark(
    config: Mapping[str, Any],
    *,
    repo_root: Path | str,
    max_splits: int | None = None,
    train_sizes: Sequence[int] | None = None,
    only_kernels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the six fixed kernels over leakage-free frozen splits."""
    normalized = _normalize_config(config)
    repository = Path(repo_root).expanduser().resolve()
    selected_train_sizes = list(normalized["train_sizes"] if train_sizes is None else train_sizes)
    if selected_train_sizes != normalized["train_sizes"]:
        raise EhrlichKernelBenchmarkError("the fixed study supports only its configured training size")
    selected_kernels = list(normalized["kernels"] if only_kernels is None else only_kernels)
    if not selected_kernels or any(kernel not in normalized["kernels"] for kernel in selected_kernels):
        raise EhrlichKernelBenchmarkError("selected kernels must be a non-empty configured subset")

    train_size = selected_train_sizes[0]
    pool = build_frozen_pool(
        normalized["run_root"], repo_root=repository, score_batch_size=normalized["score_batch_size"]
    )
    instance = torch.load(pool.instance_path, map_location="cpu", weights_only=False)
    splits = make_frozen_splits(
        pool,
        train_size=train_size,
        test_pool_size=normalized["test_pool_size"],
        repeats_per_seed=normalized["repeats_per_seed"],
        random_seed=normalized["random_seed"],
    )
    if max_splits is not None:
        if max_splits < 1:
            raise EhrlichKernelBenchmarkError("max_splits must be positive")
        splits = splits[:max_splits]

    seed_offsets = {
        "onehot_tanimoto": 0,
        "spaced_ngram": 2,
        "spaced_pair": 3,
        "local_spaced_trigram": 5,
        "mixed_window_spaced_ngram_additive": 8,
    }
    variant_count = sum(len(_kernel_window_variants(kernel, normalized)) for kernel in selected_kernels)
    total_models = len(splits) * variant_count
    rows: list[dict[str, Any]] = []
    completed = 0
    for split_number, split in enumerate(splits):
        train_x = pool.sequences[split.train_indices[:train_size]]
        train_y = pool.scores[split.train_indices[:train_size]]
        test_x = pool.sequences[split.test_indices]
        test_y = pool.scores[split.test_indices]
        for kernel in selected_kernels:
            for local_window in _kernel_window_variants(kernel, normalized):
                base_name = _kernel_result_name(kernel, normalized, local_window)
                kernel_label = base_name
                fit_seed = (
                    normalized["random_seed"]
                    + split_number * 1000
                    + train_size * 10
                    + seed_offsets[kernel]
                    + (0 if local_window is None else local_window * 100_000)
                )
                print(f"benchmarking model {completed + 1}/{total_models}: {split.split_id} {kernel_label}", flush=True)
                fit_started = time.perf_counter()
                regressor, _, _ = _fit_regressor(
                    kernel,
                    train_x=train_x,
                    train_y=train_y,
                    instance=instance,
                    config=_fit_config(normalized, local_window),
                    fit_seed=fit_seed,
                )
                fit_seconds = time.perf_counter() - fit_started
                inference_started = time.perf_counter()
                mean, std = _predict(regressor, kernel, test_x, instance)
                inference_seconds = time.perf_counter() - inference_started
                rows.append(
                    {
                        "split_id": split.split_id,
                        "heldout_seed": split.heldout_seed,
                        "repeat": split.repeat,
                        "train_size": train_size,
                        "test_pool_size": int(test_y.shape[0]),
                        "kernel": kernel_label,
                        "window_family": kernel if local_window is not None else "",
                        "local_window": "" if local_window is None else local_window,
                        "base_kernel": base_name,
                        "likelihood_noise_variance_standardized": _likelihood_noise_variance(regressor),
                        "fit_seconds": fit_seconds,
                        "inference_seconds": inference_seconds,
                        **_metrics(
                            truth=test_y,
                            mean=mean,
                            std=std,
                            train_best=float(np.max(train_y)),
                            top_k=normalized["top_k"],
                        ),
                    }
                )
                completed += 1
        print(f"benchmarked split {split_number + 1}/{len(splits)}", flush=True)

    dataset_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_root": normalized["run_root"],
        "candidate_count": int(pool.sequences.shape[0]),
        "sequence_sha256": hashlib.sha256(pool.sequences.astype("<i8", copy=False).tobytes()).hexdigest(),
        "score_sha256": hashlib.sha256(pool.scores.astype("<f8", copy=False).tobytes()).hexdigest(),
        "instance_path": _portable_path(pool.instance_path, repository),
        "instance_sha256": pool.instance_sha256,
        "trace_hashes": pool.trace_hashes,
        "independent_source_seed_ids": sorted({split.heldout_seed for split in splits}),
        "independent_source_seed_count": len({split.heldout_seed for split in splits}),
        "resampled_split_count": len(splits),
        "repeats_per_source_seed": normalized["repeats_per_seed"],
        "split_ids": [split.split_id for split in splits],
    }
    paired = _paired_comparisons(rows) if "onehot_tanimoto" in selected_kernels else []
    return {
        "schema_version": SCHEMA_VERSION,
        "resolved_config": normalized,
        "execution": {
            "max_splits": max_splits,
            "only_kernels": None if only_kernels is None else selected_kernels,
        },
        "dataset_manifest": dataset_manifest,
        "split_definitions": [
            {
                "split_id": split.split_id,
                "heldout_seed": split.heldout_seed,
                "repeat": split.repeat,
                "train_indices": split.train_indices.tolist(),
                "test_indices": split.test_indices.tolist(),
            }
            for split in splits
        ],
        "rows": rows,
        "aggregates": _aggregate(rows),
        "paired_comparisons": paired,
        "paired_aggregates": _paired_aggregates(paired),
    }


def _benchmark_payloads(result: Mapping[str, Any]) -> dict[str, bytes]:
    """Build every deterministic compact benchmark payload without writing it."""

    summary = {
        "schema_version": result["schema_version"],
        "resolved_config": result["resolved_config"],
        "execution": result["execution"],
        "dataset_manifest": result["dataset_manifest"],
        "aggregates": result["aggregates"],
        "paired_aggregates": result["paired_aggregates"],
    }
    return {
        "summary.json": canonical_json_bytes(summary),
        "splits.csv": csv_bytes(result["rows"]),
        "aggregates.csv": csv_bytes(result["aggregates"]),
        "paired_comparisons.csv": csv_bytes(result["paired_comparisons"]),
        "paired_aggregates.csv": csv_bytes(result["paired_aggregates"]),
        "dataset_manifest.json": canonical_json_bytes(result["dataset_manifest"]),
    }


def materialize_benchmark(
    result: Mapping[str, Any],
    output_dir: Path | str,
) -> dict[str, str]:
    """Atomically regenerate the known compact benchmark artifacts."""

    destination = Path(output_dir).expanduser().resolve()
    payloads = _benchmark_payloads(result)
    for name, payload in payloads.items():
        atomic_write(destination / name, payload)
    return {name: str(destination / name) for name in payloads}


def write_benchmark(
    result: Mapping[str, Any],
    output_dir: Path | str,
) -> dict[str, str]:
    """Write a fresh compact benchmark result without replacing prior output."""

    destination = Path(output_dir).expanduser().resolve()
    filenames = tuple(_benchmark_payloads(result))
    existing = [name for name in filenames if (destination / name).exists()]
    if existing:
        raise EhrlichKernelBenchmarkError(f"benchmark outputs already exist below {destination}")
    return materialize_benchmark(result, destination)


__all__ = [
    "EhrlichKernelBenchmarkError",
    "FrozenPool",
    "FrozenSplit",
    "build_frozen_pool",
    "load_config",
    "materialize_benchmark",
    "make_frozen_splits",
    "run_benchmark",
    "write_benchmark",
]
