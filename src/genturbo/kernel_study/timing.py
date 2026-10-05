"""CPU fit and inference timing for the fixed Ehrlich kernel matrix."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from surrogate.model import make_local_spaced_ngram_offsets
from genturbo.artifacts import sha256_file
from genturbo.kernel_study.benchmark import (
    EhrlichKernelBenchmarkError,
    _fit_regressor,
    _kernel_window_variants,
    _likelihood_noise_variance,
    _predict,
    build_frozen_pool,
    make_frozen_splits,
)
from genturbo.kernel_study.benchmark import (
    load_config as load_performance_config,
)
from genturbo.kernel_study.config import CONDITION_LABELS, KERNEL_LABELS, STATIC_KERNELS, canonical_kernel_name
from genturbo.serialization import atomic_save_figure, atomic_write, canonical_json_bytes, csv_bytes

_MPL_CACHE = Path(tempfile.gettempdir()) / "genturbo-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

SCHEMA_VERSION = 1
_CONFIG_KEYS = {
    "benchmark_config",
    "heldout_seed",
    "inference_repeats",
    "output_dir",
    "plot_formats",
    "schema_version",
    "timing_repeats",
}
_PLOT_FORMATS = {"png", "svg"}
_KERNEL_SEED_OFFSETS = {
    "onehot_tanimoto": 0,
    "spaced_ngram": 2,
    "spaced_pair": 3,
    "local_spaced_trigram": 5,
    "mixed_window_spaced_ngram_additive": 8,
}
_T = TypeVar("_T")


class EhrlichKernelRuntimeError(RuntimeError):
    """Raised when the fixed timing profile is invalid."""


def load_runtime_config(path: Path | str) -> dict[str, Any]:
    """Load and validate an internal resolved timing configuration."""
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EhrlichKernelRuntimeError(f"cannot read runtime config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EhrlichKernelRuntimeError("runtime config must be a JSON object")
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise EhrlichKernelRuntimeError(f"unknown runtime config keys: {', '.join(unknown)}")
    benchmark_config = raw.get("benchmark_config")
    if not isinstance(benchmark_config, str) or not benchmark_config:
        raise EhrlichKernelRuntimeError("benchmark_config must be a JSON path")
    formats = raw.get("plot_formats", ["png", "svg"])
    if not isinstance(formats, list) or not formats or not all(isinstance(value, str) for value in formats):
        raise EhrlichKernelRuntimeError("plot_formats must be a non-empty JSON array")
    normalized_formats = [value.strip().lower() for value in formats]
    if set(normalized_formats) - _PLOT_FORMATS or len(normalized_formats) != len(set(normalized_formats)):
        raise EhrlichKernelRuntimeError("plot_formats must contain unique png/svg values")
    config = {
        "schema_version": int(raw.get("schema_version", SCHEMA_VERSION)),
        "benchmark_config": benchmark_config,
        "output_dir": str(raw.get("output_dir", "results/ehrlich_kernel_v1/timing")),
        "heldout_seed": int(raw.get("heldout_seed", 0)),
        "timing_repeats": int(raw.get("timing_repeats", 1)),
        "inference_repeats": int(raw.get("inference_repeats", 3)),
        "plot_formats": normalized_formats,
    }
    if config["schema_version"] != SCHEMA_VERSION:
        raise EhrlichKernelRuntimeError(f"unsupported schema_version {config['schema_version']}")
    if config["heldout_seed"] < 0:
        raise EhrlichKernelRuntimeError("heldout_seed must be non-negative")
    if config["timing_repeats"] != 1 or config["inference_repeats"] != 3:
        raise EhrlichKernelRuntimeError("the fixed timing study requires one fit and three inference measurements")
    return config


def _variant_name(
    kernel: str,
    performance: Mapping[str, Any],
    local_window: int | None,
) -> str:
    if kernel == "mixed_window_spaced_ngram_additive":
        name = "pair_w48_trigram_w16_additive"
    else:
        name = kernel if local_window is None else f"{kernel}_w{local_window}"
    return name


def _fit_config(performance: Mapping[str, Any], local_window: int | None) -> Mapping[str, Any]:
    if local_window is None:
        return performance
    return {**dict(performance), "local_window": int(local_window), "spaced_pair_window": int(local_window)}


def _offset_complexity(
    kernel: str,
    *,
    performance: Mapping[str, Any],
    local_window: int | None,
    instance: Any,
) -> dict[str, int | str]:
    """Describe the exact fixed representation without materialising its features."""
    seq_len = int(instance.cfg.seq_len)
    vocab_size = int(instance.cfg.vocab_size)
    if kernel == "onehot_tanimoto":
        return {
            "representation": "positional_onehot",
            "offset_pattern_count": seq_len,
            "nominal_feature_dimension": seq_len * vocab_size,
            "motif_occurrences_per_sequence": seq_len,
        }

    if kernel == "spaced_ngram":
        offsets = torch.as_tensor(instance.spacings, dtype=torch.long)
        if offsets.ndim == 1:
            offsets = offsets.view(1, -1)
        groups = [offsets]
        representation = "task_aware_spaced_ngram"
    elif kernel in {"spaced_pair", "local_spaced_trigram"}:
        order = 2 if kernel == "spaced_pair" else 3
        if local_window is None:
            raise EhrlichKernelRuntimeError(f"{kernel} requires an explicit window")
        groups = [make_local_spaced_ngram_offsets(order, local_window)]
        representation = "local_spaced_pair" if order == 2 else "local_spaced_trigram"
    elif kernel == "mixed_window_spaced_ngram_additive":
        groups = [
            make_local_spaced_ngram_offsets(int(order), int(window))
            for order, window in zip(
                performance["mixed_window_spaced_ngram_orders"],
                performance["mixed_window_spaced_ngram_windows"],
                strict=True,
            )
        ]
        representation = "mixed_window_spaced_ngram_additive"
    else:
        raise EhrlichKernelRuntimeError(f"unknown kernel {kernel}")

    pattern_count = sum(int(offsets.shape[0]) for offsets in groups)
    feature_dimension = sum(int(offsets.shape[0]) * vocab_size ** int(offsets.shape[1]) for offsets in groups)
    occurrences = sum(seq_len - int(last_offset) for offsets in groups for last_offset in offsets[:, -1].tolist())
    return {
        "representation": representation,
        "offset_pattern_count": pattern_count,
        "nominal_feature_dimension": feature_dimension,
        "motif_occurrences_per_sequence": occurrences,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["kernel"])].append(row)
    output: list[dict[str, Any]] = []
    ordered_groups = sorted(groups.items(), key=lambda item: STATIC_KERNELS.index(canonical_kernel_name(item[0])))
    for kernel, group in ordered_groups:
        template = group[0]
        fit = [float(row["fit_seconds"]) for row in group]
        first = [float(row["inference_first_seconds"]) for row in group]
        warm = [float(row["inference_warm_median_seconds"]) for row in group]
        noise = [float(row["likelihood_noise_variance_standardized"]) for row in group]
        output.append(
            {
                "kernel": kernel,
                "base_kernel": str(template["base_kernel"]),
                "local_window": template["local_window"],
                "timing_repeat_count": len(group),
                "fit_seconds_median": statistics.median(fit),
                "fit_seconds_mean": statistics.fmean(fit),
                "fit_seconds_min": min(fit),
                "fit_seconds_max": max(fit),
                "inference_first_seconds_median": statistics.median(first),
                "inference_first_seconds_mean": statistics.fmean(first),
                "inference_warm_seconds_median": statistics.median(warm),
                "likelihood_noise_variance_standardized_median": statistics.median(noise),
                "representation": str(template["representation"]),
                "offset_pattern_count": int(template["offset_pattern_count"]),
                "nominal_feature_dimension": int(template["nominal_feature_dimension"]),
                "motif_occurrences_per_sequence": int(template["motif_occurrences_per_sequence"]),
            }
        )
    return output


def _measure(call: Callable[[], _T]) -> tuple[_T, float]:
    """Run garbage collection before, and disable it during, a measured interval."""
    gc.collect()
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        started = time.perf_counter()
        value = call()
        elapsed = time.perf_counter() - started
    finally:
        if was_enabled:
            gc.enable()
    return value, elapsed


def run_runtime_profile(
    runtime_config: Mapping[str, Any],
    *,
    repo_root: Path | str,
    max_kernels: int | None = None,
    only_kernels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Measure one fit and three posterior calls for the six fixed variants."""
    repository = Path(repo_root).expanduser().resolve()
    benchmark_path = Path(str(runtime_config["benchmark_config"])).expanduser()
    if not benchmark_path.is_absolute():
        benchmark_path = repository / benchmark_path
    performance = load_performance_config(benchmark_path)
    train_size = int(performance["train_sizes"][0])

    preparation_started = time.perf_counter()
    try:
        pool = build_frozen_pool(
            performance["run_root"],
            repo_root=repository,
            score_batch_size=int(performance["score_batch_size"]),
        )
        instance = torch.load(pool.instance_path, map_location="cpu", weights_only=False)
        splits = make_frozen_splits(
            pool,
            train_size=train_size,
            test_pool_size=int(performance["test_pool_size"]),
            repeats_per_seed=int(performance["repeats_per_seed"]),
            random_seed=int(performance["random_seed"]),
        )
    except EhrlichKernelBenchmarkError as exc:
        raise EhrlichKernelRuntimeError(str(exc)) from exc
    preparation_seconds = time.perf_counter() - preparation_started

    matches = [
        split for split in splits if split.heldout_seed == int(runtime_config["heldout_seed"]) and split.repeat == 0
    ]
    if len(matches) != 1:
        raise EhrlichKernelRuntimeError("heldout_seed does not identify exactly one repeat-0 split")
    split = matches[0]
    train_x = pool.sequences[split.train_indices[:train_size]]
    train_y = pool.scores[split.train_indices[:train_size]]
    test_x = pool.sequences[split.test_indices]

    selected = list(performance["kernels"] if only_kernels is None else only_kernels)
    if not selected or any(kernel not in performance["kernels"] for kernel in selected):
        raise EhrlichKernelRuntimeError("selected kernels must be a non-empty configured subset")
    if max_kernels is not None and only_kernels is not None:
        raise EhrlichKernelRuntimeError("max_kernels and only_kernels cannot be combined")
    variants = [(kernel, window) for kernel in selected for window in _kernel_window_variants(kernel, performance)]
    if max_kernels is not None:
        if max_kernels < 1:
            raise EhrlichKernelRuntimeError("max_kernels must be positive")
        variants = variants[:max_kernels]

    rows: list[dict[str, Any]] = []
    for variant_index, (kernel, local_window) in enumerate(variants):
        name = _variant_name(kernel, performance, local_window)
        print(f"profiling {variant_index + 1}/{len(variants)} {name}...", flush=True)
        fit_seed = (
            int(performance["random_seed"])
            + int(runtime_config["heldout_seed"]) * 1000
            + train_size * 10
            + _KERNEL_SEED_OFFSETS[kernel]
            + (0 if local_window is None else int(local_window) * 100_000)
        )

        def fit(
            kernel_name: str = kernel,
            window: int | None = local_window,
            seed: int = fit_seed,
        ) -> Any:
            model, _, _ = _fit_regressor(
                kernel_name,
                train_x=train_x,
                train_y=train_y,
                instance=instance,
                config=_fit_config(performance, window),
                fit_seed=seed,
            )
            return model

        fitted_regressor, fit_seconds = _measure(fit)
        inference_seconds: list[float] = []
        for _ in range(int(runtime_config["inference_repeats"])):
            (mean, std), elapsed = _measure(
                lambda model=fitted_regressor, kernel_name=kernel: _predict(model, kernel_name, test_x, instance)
            )
            inference_seconds.append(elapsed)
            if mean.shape != (test_x.shape[0],) or std.shape != (test_x.shape[0],):
                raise EhrlichKernelRuntimeError(f"{name} returned an invalid posterior shape")
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise EhrlichKernelRuntimeError(f"{name} returned non-finite posterior values")
        warm_values = inference_seconds[1:] or inference_seconds
        rows.append(
            {
                "kernel": name,
                "base_kernel": kernel,
                "local_window": "" if local_window is None else int(local_window),
                "heldout_seed": int(runtime_config["heldout_seed"]),
                "split_id": split.split_id,
                "train_size": train_size,
                "test_pool_size": int(test_x.shape[0]),
                "timing_repeat": 0,
                "fit_seconds": fit_seconds,
                "inference_first_seconds": inference_seconds[0],
                "inference_warm_median_seconds": statistics.median(warm_values),
                "inference_repeat_count": len(inference_seconds),
                **_offset_complexity(
                    kernel,
                    performance=performance,
                    local_window=local_window,
                    instance=instance,
                ),
                "likelihood_noise_variance_standardized": _likelihood_noise_variance(fitted_regressor),
            }
        )
        del fitted_regressor

    return {
        "schema_version": SCHEMA_VERSION,
        "resolved_config": dict(runtime_config),
        "benchmark_config": performance,
        "execution": {
            "max_kernels": max_kernels,
            "only_kernels": None if only_kernels is None else selected,
            "archive_preparation_seconds": preparation_seconds,
        },
        "dataset_identity": {
            "candidate_count": int(pool.sequences.shape[0]),
            "sequence_sha256": hashlib.sha256(pool.sequences.astype("<i8", copy=False).tobytes()).hexdigest(),
            "score_sha256": hashlib.sha256(pool.scores.astype("<f8", copy=False).tobytes()).hexdigest(),
            "instance_path": pool.instance_path.relative_to(repository).as_posix(),
            "instance_sha256": pool.instance_sha256,
            "split_id": split.split_id,
        },
        "rows": rows,
        "aggregates": _aggregate(rows),
        "environment": {
            "machine": platform.machine(),
            "mac_ver": platform.mac_ver()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "input_hashes": {
            "benchmark_config": sha256_file(benchmark_path),
            "reward_instance": pool.instance_sha256,
        },
    }


def _write_runtime_plot(
    aggregates: Sequence[Mapping[str, Any]],
    destination: Path,
    formats: Sequence[str],
    *,
    condition: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with matplotlib.rc_context({"svg.hashsalt": "ehrlich_kernel_v1"}):
        labels = [KERNEL_LABELS[canonical_kernel_name(str(row["kernel"]))] for row in aggregates]
        fit = [float(row["fit_seconds_median"]) for row in aggregates]
        first_inference = [float(row["inference_first_seconds_median"]) for row in aggregates]
        warm_inference = [float(row["inference_warm_seconds_median"]) for row in aggregates]
        positions = np.arange(len(labels))
        figure, axes = plt.subplots(1, 3, figsize=(14.8, max(4.5, 0.42 * len(labels) + 1.5)), sharey=True)
        figure.subplots_adjust(left=0.19, right=0.99, bottom=0.12, top=0.84, wspace=0.20)
        for axis, values, title, color in zip(
            axes,
            (fit, first_inference, warm_inference),
            ("GP fit", "First posterior", "Warm posterior median"),
            ("#0072B2", "#D55E00", "#009E73"),
            strict=True,
        ):
            bars = axis.barh(positions, values, color=color)
            axis.set_yticks(positions, labels=labels)
            axis.invert_yaxis()
            axis.set_xlabel("Wall time (seconds)")
            axis.set_title(title)
            axis.grid(axis="x", alpha=0.25)
            positive = [value for value in values if value > 0.0]
            if positive and max(positive) / min(positive) >= 20.0:
                axis.set_xscale("log")
            axis.bar_label(bars, labels=[f"{value:.3g}s" for value in values], padding=3, fontsize=8)
            axis.margins(x=0.16)
        condition_label = CONDITION_LABELS.get(condition, condition)
        figure.suptitle(
            f"Ehrlich kernel runtime — {condition_label}; fixed 512-train / 256-test split",
            y=0.96,
        )
        for suffix in formats:
            path = destination / f"kernel_runtime_bars.{suffix}"
            metadata = (
                {"Creator": "genturbo kernel-study", "Date": None}
                if suffix == "svg"
                else {"Creator": "genturbo kernel-study"}
            )
            atomic_save_figure(
                figure,
                path,
                format=suffix,
                dpi=180 if suffix == "png" else None,
                bbox_inches="tight",
                pad_inches=0.22,
                metadata=metadata,
            )
        plt.close(figure)


def materialize_runtime_profile(
    result: Mapping[str, Any],
    output_dir: Path | str,
) -> dict[str, str]:
    """Atomically regenerate all known runtime tables, provenance, and plots."""

    destination = Path(output_dir).expanduser().resolve()
    rows = list(result["rows"])
    aggregates = _aggregate(rows)
    formats = [str(value) for value in result["resolved_config"]["plot_formats"]]
    filenames = (
        "runtime_rows.csv",
        "runtime_aggregates.csv",
        "runtime_manifest.json",
        *(f"kernel_runtime_bars.{suffix}" for suffix in formats),
    )
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write(destination / "runtime_rows.csv", csv_bytes(rows))
    atomic_write(destination / "runtime_aggregates.csv", csv_bytes(aggregates))
    manifest = {key: value for key, value in result.items() if key not in {"rows", "aggregates"}}
    manifest["aggregates"] = aggregates
    atomic_write(destination / "runtime_manifest.json", canonical_json_bytes(manifest))
    _write_runtime_plot(aggregates, destination, formats, condition=destination.name)
    return {name: str(destination / name) for name in filenames}


def write_runtime_profile(
    result: Mapping[str, Any],
    output_dir: Path | str,
) -> dict[str, str]:
    """Write runtime rows, aggregates, machine provenance, and bar plots."""
    destination = Path(output_dir).expanduser().resolve()
    formats = [str(value) for value in result["resolved_config"]["plot_formats"]]
    filenames = (
        "runtime_rows.csv",
        "runtime_aggregates.csv",
        "runtime_manifest.json",
        *(f"kernel_runtime_bars.{suffix}" for suffix in formats),
    )
    existing = [name for name in filenames if (destination / name).exists()]
    if existing:
        raise EhrlichKernelRuntimeError(f"runtime outputs already exist below {destination}")
    return materialize_runtime_profile(result, destination)


__all__ = [
    "EhrlichKernelRuntimeError",
    "load_runtime_config",
    "materialize_runtime_profile",
    "run_runtime_profile",
    "write_runtime_profile",
]
