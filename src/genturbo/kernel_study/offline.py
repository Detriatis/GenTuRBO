"""Static-ablation and timing adapters for the fixed Ehrlich kernel matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from genturbo.artifacts import sha256_file
from genturbo.kernel_study.benchmark import materialize_benchmark, run_benchmark
from genturbo.kernel_study.timing import materialize_runtime_profile, run_runtime_profile
from genturbo.serialization import atomic_write_json, canonical_json_bytes

from .config import KernelStudyConfigError, StudyConfigs

OfflineWorkflow = Literal["static", "timing"]
STATIC_RAW_FILENAME = "benchmark_raw.json"
TIMING_RAW_FILENAME = "runtime_raw.json"


def _repo_value(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolved_kernel_core(study: StudyConfigs) -> dict[str, Any]:
    """Translate the sole checked-in kernel declarations into core parameters."""

    declarations = study.static["kernels"]
    families: list[str] = []
    pair_windows: list[int] = []
    trigram_windows: list[int] = []
    additive: dict[str, Any] | None = None
    family_for_type = {
        "onehot_tanimoto": "onehot_tanimoto",
        "spaced_pair": "spaced_pair",
        "local_spaced_trigram": "local_spaced_trigram",
        "mixed_window_spaced_ngram_additive": "mixed_window_spaced_ngram_additive",
        "spaced_ngram": "spaced_ngram",
    }
    for declaration in declarations:
        kernel_type = str(declaration["type"])
        try:
            family = family_for_type[kernel_type]
        except KeyError as exc:  # pragma: no cover - strict study loading guards this
            raise KernelStudyConfigError(f"unsupported static kernel type {kernel_type!r}") from exc
        if family not in families:
            families.append(family)
        parameters = declaration["parameters"]
        if kernel_type == "spaced_pair":
            pair_windows.append(int(parameters["window"]))
        elif kernel_type == "local_spaced_trigram":
            trigram_windows.append(int(parameters["window"]))
        elif kernel_type == "mixed_window_spaced_ngram_additive":
            additive = parameters
    if not pair_windows or not trigram_windows or additive is None:
        raise KernelStudyConfigError("static kernel declarations are incomplete")
    return {
        "kernels": families,
        "spaced_pair_window": pair_windows[0],
        "spaced_pair_window_grid": pair_windows,
        "local_spaced_trigram_window_grid": trigram_windows,
        "mixed_window_spaced_ngram_orders": [int(value) for value in additive["orders"]],
        "mixed_window_spaced_ngram_windows": [int(value) for value in additive["windows"]],
        "mixed_window_spaced_ngram_weights": [float(value) for value in additive["weights"]],
    }


def benchmark_config_for_condition(
    study: StudyConfigs,
    *,
    condition: str,
    repo_root: Path | str,
) -> dict[str, Any]:
    """Translate the fixed six-kernel declaration into the proven benchmark core."""

    root = Path(repo_root).resolve()
    if condition not in study.static["conditions"]:
        raise KernelStudyConfigError(f"unknown static condition {condition!r}")
    split = study.static["split"]
    gp = study.static["gp"]
    kernels = _resolved_kernel_core(study)
    return {
        "schema_version": 1,
        "run_root": f"{study.datasets['run_dir']}/{condition}",
        "output_dir": f"{study.static['results_dir']}/{condition}",
        "random_seed": int(split["random_seed"]),
        "repeats_per_seed": 1,
        "train_sizes": [int(split["train_size"])],
        "test_pool_size": int(split["test_size"]),
        "top_k": int(split["top_k"]),
        "score_batch_size": 512,
        **kernels,
        "gp": {
            "device": gp["device"],
            "dtype": gp["dtype"],
            "jitter": gp["jitter"],
            "outputscale": 1.0,
            "learn_outputscale": False,
        },
        "_repo_root": str(root),
    }


def _benchmark_core_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _resolved_timing_config(
    study: StudyConfigs,
    *,
    benchmark_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_config": _repo_value(benchmark_path, repo_root=repo_root),
        "output_dir": _repo_value(output_dir, repo_root=repo_root),
        "heldout_seed": int(study.timing["heldout_seed"]),
        "timing_repeats": int(study.timing["fit_repeats"]),
        "inference_repeats": int(study.timing["inference_repeats"]),
        "plot_formats": ["png", "svg"],
    }


def offline_plan(
    study: StudyConfigs,
    workflow: OfflineWorkflow,
    *,
    conditions: list[str] | None = None,
    max_splits: int | None = None,
    max_kernels: int | None = None,
    run_root: Path | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    configured = list(study.for_workflow(workflow)["conditions"])
    conditions = list(configured if conditions is None else conditions)
    unknown = sorted(set(conditions) - set(configured))
    if unknown or not conditions or len(conditions) != len(set(conditions)):
        raise KernelStudyConfigError(
            "offline conditions must be a non-empty unique subset of the configured conditions"
            + (f"; unknown: {', '.join(unknown)}" if unknown else "")
        )
    if workflow == "static":
        if max_kernels is not None:
            raise KernelStudyConfigError("max_kernels applies only to timing")
        split_count = len(study.static["split"]["heldout_seeds"])
        if max_splits is not None:
            if max_splits < 1:
                raise KernelStudyConfigError("max_splits must be positive")
            split_count = min(split_count, max_splits)
        models = len(study.static["kernels"])
        cell_count = len(conditions) * split_count * models
        # The first canonical v1 cohort took 54.7 minutes for 240 fits on the
        # reference CPU. Keep a machine-tolerant margin without reverting to
        # the original six-hour placeholder.
        estimated_seconds = cell_count * 20.0
        unit = "kernel fits"
    else:
        if max_splits is not None:
            raise KernelStudyConfigError("max_splits applies only to static")
        models = len(study.static["kernels"])
        if max_kernels is not None:
            if max_kernels < 1:
                raise KernelStudyConfigError("max_kernels must be positive")
            models = min(models, max_kernels)
        cell_count = len(conditions) * models
        estimated_seconds = cell_count * 120.0
        unit = "kernel timing profiles"
    run_base = Path(run_root) if run_root is not None else Path(study.for_workflow(workflow)["run_dir"])
    result_base = Path(results_root) if results_root is not None else Path(study.for_workflow(workflow)["results_dir"])
    return {
        "workflow": workflow,
        "condition_count": len(conditions),
        "conditions": conditions,
        "cell_count": cell_count,
        "cell_unit": unit,
        "expected_run_paths": [str(run_base / condition) for condition in conditions],
        "expected_result_paths": [str(result_base / condition) for condition in conditions],
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_seconds / 3600.0,
        "estimate_kind": "conservative fallback; review before launch",
    }


def _prepare_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise KernelStudyConfigError(f"outputs already exist below {path}; refusing to overwrite them")


def _write_study_manifest(
    destination: Path,
    *,
    workflow: OfflineWorkflow,
    condition: str,
    status: str,
    source_config: dict[str, Any],
    raw_files: list[Path],
) -> dict[str, Any]:
    raw_hashes = {path.name: sha256_file(path) for path in sorted(raw_files)}
    manifest = {
        "schema_version": 1,
        "study": "ehrlich_kernel_v1",
        "workflow": workflow,
        "condition": condition,
        "status": status,
        "source_config_sha256": hashlib.sha256(canonical_json_bytes(source_config)).hexdigest(),
        "raw_files_sha256": raw_hashes,
    }
    atomic_write_json(
        destination / "study_manifest.json",
        manifest,
    )
    return manifest


def run_static(
    study: StudyConfigs,
    *,
    repo_root: Path | str,
    conditions: list[str] | None = None,
    max_splits: int | None = None,
    run_root: Path | None = None,
    results_root: Path | None = None,
    status: str = "canonical",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    selected = list(conditions or study.static["conditions"])
    run_base = Path(run_root) if run_root is not None else root / study.static["run_dir"]
    result_base = Path(results_root) if results_root is not None else root / study.static["results_dir"]
    summaries: list[dict[str, Any]] = []
    for condition in selected:
        performance = benchmark_config_for_condition(study, condition=condition, repo_root=root)
        core = _benchmark_core_config(performance)
        raw_dir = run_base / condition
        output_dir = result_base / condition
        _prepare_destination(raw_dir)
        if status == "smoke":
            _prepare_destination(output_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        benchmark_path = raw_dir / "resolved_benchmark_config.json"
        atomic_write_json(benchmark_path, core)
        result = run_benchmark(core, repo_root=root, max_splits=max_splits)
        raw_result_path = raw_dir / STATIC_RAW_FILENAME
        atomic_write_json(raw_result_path, result)
        manifest = _write_study_manifest(
            raw_dir,
            workflow="static",
            condition=condition,
            status=status,
            source_config=study.static,
            raw_files=[benchmark_path, raw_result_path],
        )
        smoke_outputs = materialize_benchmark(result, output_dir) if status == "smoke" else {}
        summaries.append(
            {
                "condition": condition,
                "row_count": len(result["rows"]),
                "raw_manifest": str(raw_dir / "study_manifest.json"),
                "raw_files_sha256": manifest["raw_files_sha256"],
                "smoke_outputs": smoke_outputs,
            }
        )
    return {"workflow": "static", "status": status, "conditions": summaries}


def run_timing(
    study: StudyConfigs,
    *,
    repo_root: Path | str,
    conditions: list[str] | None = None,
    max_kernels: int | None = None,
    run_root: Path | None = None,
    results_root: Path | None = None,
    status: str = "canonical",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    selected = list(conditions or study.timing["conditions"])
    run_base = Path(run_root) if run_root is not None else root / study.timing["run_dir"]
    result_base = Path(results_root) if results_root is not None else root / study.timing["results_dir"]
    summaries: list[dict[str, Any]] = []
    for condition in selected:
        performance = _benchmark_core_config(benchmark_config_for_condition(study, condition=condition, repo_root=root))
        raw_dir = run_base / condition
        output_dir = result_base / condition
        _prepare_destination(raw_dir)
        if status == "smoke":
            _prepare_destination(output_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        benchmark_path = raw_dir / "resolved_benchmark_config.json"
        atomic_write_json(benchmark_path, performance)
        runtime = _resolved_timing_config(
            study,
            benchmark_path=benchmark_path,
            output_dir=output_dir,
            repo_root=root,
        )
        timing_config_path = raw_dir / "resolved_timing_config.json"
        atomic_write_json(timing_config_path, runtime)
        result = run_runtime_profile(runtime, repo_root=root, max_kernels=max_kernels)
        raw_result_path = raw_dir / TIMING_RAW_FILENAME
        atomic_write_json(raw_result_path, result)
        manifest = _write_study_manifest(
            raw_dir,
            workflow="timing",
            condition=condition,
            status=status,
            source_config=study.timing,
            raw_files=[benchmark_path, timing_config_path, raw_result_path],
        )
        smoke_outputs = materialize_runtime_profile(result, output_dir) if status == "smoke" else {}
        summaries.append(
            {
                "condition": condition,
                "row_count": len(result["rows"]),
                "raw_manifest": str(raw_dir / "study_manifest.json"),
                "raw_files_sha256": manifest["raw_files_sha256"],
                "smoke_outputs": smoke_outputs,
            }
        )
    return {"workflow": "timing", "status": status, "conditions": summaries}


__all__ = [
    "benchmark_config_for_condition",
    "offline_plan",
    "run_static",
    "run_timing",
    "STATIC_RAW_FILENAME",
    "TIMING_RAW_FILENAME",
]
