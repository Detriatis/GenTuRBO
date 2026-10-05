"""Local, explicit recipes for the current publication experiments.

A plan is read-only. Execution writes resolved configurations and immutable run
outputs to the selected directory; numerical settings come from the frozen
recipe builders, not from defaults in this dispatcher.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from genturbo.configuration import load_config
from genturbo.io import iter_jsonl
from genturbo.kernel_study.runner import StudyCell, run_cells
from genturbo.serialization import atomic_write_json, canonical_json_bytes

TASKS = ("gfp_medium", "gfp_hard", "aav_medium", "aav_hard")
PROTEIN_METHODS = ("kermut", "tanimoto", "hamming_gp", "holo_ga", "adalead", "bo_evo", "lambo2")
COMPONENT_METHODS = ("rt_random", "tr_random", "kermut_rt_gp", "tanimoto_rt_gp")
SUPPORT_STAGES = ("protein-components", "protein-surrogate", "protein-calibration", "kernel-datasets", "kernel-static", "kernel-timing", "kernel-online", "visitation", "counts")
OUTPUT_FILES = {"samples_out": "samples.jsonl.gz", "initial_batch_out": "initial_batch.jsonl.gz", "run_manifest_out": "run_manifest.json"}
INPUT_KEYS = {"checkpoint", "config", "instance_path", "initial_batch_path", "al_kermut_resources", "lambo_requirements", "pretraining_manifest", "benchmark_root"}


@dataclass(frozen=True)
class Recipe:
    suite: str
    task: str
    method: str
    seed: int
    config: dict[str, Any]
    source: str
    runner: str = "interpolant"

    @property
    def identity(self) -> str:
        return f"{self.suite}:{self.task}:{self.method}:seed{self.seed}"


def _protein_recipes(root: Path, *, components: bool = False, surrogate: bool = False) -> list[Recipe]:
    from genturbo.protein_study.config import load_protein_study
    from genturbo.protein_study.runner import build_protein_cells
    from genturbo.protein_study.recipes import _v2_resolved_config, _marginal_ei_resolved_config

    declaration = "configs/experiments/protein_design_v1/online.json"
    study = load_protein_study(declaration, repo_root=root)
    # The v2 builders intentionally use fresh-output cells. 'smoke' is their
    # historical metadata name, not a reduction in budget (still 2,000).
    cells = build_protein_cells(study, repo_root=root, status="smoke")
    by_key = {(c.condition, c.arm, c.seed): c for c in cells}
    recipes: list[Recipe] = []
    methods = ("tanimoto_learned",) if surrogate else COMPONENT_METHODS if components else PROTEIN_METHODS
    for task in (("gfp_medium",) if surrogate else TASKS):
        for method in methods:
            source = declaration
            for seed in range(20):
                if method in {"holo_ga", "adalead", "bo_evo", "lambo2"}:
                    cfg = _v2_resolved_config(by_key[task, method, seed], root=root)
                elif method in {"tanimoto", "tanimoto_rt_gp"} and task.endswith("hard"):
                    arm = "tr_gp" if method == "tanimoto" else "rt_gp"
                    cfg = _marginal_ei_resolved_config(by_key[task, arm, seed], root=root)
                elif method in {"rt_random", "tr_random"} and task.endswith("hard"):
                    cfg = _v2_resolved_config(by_key[task, method, seed], root=root)
                else:
                    if method == "kermut":
                        relative = f"kermut_v1/{task}.json"
                    elif method == "kermut_rt_gp":
                        relative = f"kermut_rt_gp_v1/{task}.json"
                    elif method == "tanimoto_rt_gp":
                        relative = f"tanimoto_rt_gp_medium_v1/{task}.json"
                    elif method == "tanimoto_learned":
                        relative = "tanimoto_matched_v1/gfp_medium.json"
                    elif method == "tanimoto":
                        group = "tanimoto_fixed_noise_v1" if task == "gfp_medium" else "tanimoto_matched_v1"
                        relative = f"{group}/{task}.json"
                    elif method == "hamming_gp":
                        relative = f"hamming_tanimoto_v1/{task}.json"
                    else:
                        relative = f"medium_random_v1/{task}_{method}.json"
                    source = f"configs/search/protein/{relative}"
                    cfg = load_config(root / source)
                    # Recorded Medium RT+GP experiments used shared CUDA inference.
                    if method == "tanimoto_rt_gp":
                        cfg["device"] = "cuda"
                        cfg["reward"]["device"] = "cuda"
                        protein = "gfp237" if task == "gfp_medium" else "aav28"
                        cfg["config"] = f"configs/sample/protein_evodiff_d3pm_38M_uniform_{protein}.json"
                cfg = copy.deepcopy(cfg)
                cfg["seed"] = seed
                recipes.append(Recipe("supporting" if components or surrogate else "protein", task, method, cfg["seed"], cfg, source, cfg.get("runner", "interpolant")))
    return recipes


def _ehrlich_recipes(root: Path) -> list[Recipe]:
    from genturbo.sequence_baseline_study.config import load_sequence_baseline_study
    from genturbo.sequence_baseline_study.runner import build_sequence_baseline_cells
    from genturbo.ehrlich_adalead import build_cells
    source = "configs/experiments/ehrlich_sequence_baselines_v2/baselines.json"
    study = load_sequence_baseline_study(source, repo_root=root, verify_inputs=False)
    cells = build_sequence_baseline_cells(study, repo_root=root)
    cells = [c for c in cells if c.arm != "adalead_gp_ei"]
    official = build_cells(root, verify_inputs=False)
    return [Recipe("ehrlich", c.condition, c.arm, c.seed, c.resolved_config,
                   "configs/experiments/ehrlich_adalead_official_v1/online.json" if c.arm == "adalead_official" else source,
                   c.resolved_config.get("runner", "interpolant")) for c in [*cells, *official]]


def _kernel_recipes(root: Path, stage: str) -> list[Recipe]:
    from genturbo.kernel_study.config import load_study
    from genturbo.kernel_study.runner import build_dataset_cells, build_online_cells
    study = load_study("configs/experiments/ehrlich_kernel_v1", repo_root=root, verify_inputs=False)
    builder = build_dataset_cells if stage == "kernel-datasets" else build_online_cells
    return [Recipe(f"supporting/{stage}", c.condition, c.arm, c.seed, c.resolved_config,
                   f"configs/experiments/ehrlich_kernel_v1/{'datasets' if stage == 'kernel-datasets' else 'online'}.json")
            for c in builder(study, repo_root=root)]


def _sokoban_recipes(root: Path) -> list[Recipe]:
    from sokoban.optimise import load_config as load_sokoban
    manifest = json.loads((root / "configs/search/sokoban/main.json").read_text())
    recipes = []
    for method, filename in manifest["recipes"].items():
        path = root / "configs/search/sokoban" / filename
        config = load_sokoban(path)
        for seed in config["seeds"]:
            cfg = copy.deepcopy(config)
            cfg["seeds"] = [seed]
            recipes.append(Recipe("sokoban", "box_switches", method, seed, cfg, path.relative_to(root).as_posix(), "sokoban"))
    return recipes


def build_recipes(root: Path, suite: str, *, analyses: list[str] | None = None) -> list[Recipe]:
    recipes = []
    if suite in {"ehrlich", "all"}:
        recipes.extend(_ehrlich_recipes(root))
    if suite in {"protein", "all"}:
        recipes.extend(_protein_recipes(root))
    if suite in {"sokoban", "all"}:
        recipes.extend(_sokoban_recipes(root))
    if suite in {"supporting", "all"}:
        stages = analyses or list(SUPPORT_STAGES)
        if "protein-components" in stages:
            recipes.extend(_protein_recipes(root, components=True))
        if "protein-surrogate" in stages:
            recipes.extend(_protein_recipes(root, surrogate=True))
        if "protein-calibration" in stages:
            if "protein-surrogate" not in stages:
                recipes.extend(c for c in _protein_recipes(root, surrogate=True) if c.seed < 5)
            if suite != "all":
                recipes.extend(c for c in _protein_recipes(root) if c.task == "gfp_medium" and c.method == "tanimoto" and c.seed < 5)
        for stage in ("kernel-datasets", "kernel-online"):
            if stage in stages:
                recipes.extend(_kernel_recipes(root, stage))
    return recipes


def _input_path(value: str, root: Path, artifacts: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            return str(path)
    if path.parts and path.parts[0] in {"runs", "benchmarks", "results", "reference"}:
        return str(artifacts / path)
    if path.parts and path.parts[0] in {"configs", "environments", "external"}:
        return str(root / path)
    return value


def relocate(value: Any, *, root: Path, artifacts: Path) -> Any:
    if isinstance(value, dict):
        return {key: relocate(nested, root=root, artifacts=artifacts) for key, nested in value.items()}
    if isinstance(value, list):
        return [relocate(item, root=root, artifacts=artifacts) for item in value]
    if isinstance(value, str):
        return _input_path(value, root, artifacts)
    return value


def resolve_recipe(recipe: Recipe, *, root: Path, artifacts: Path, output: Path, device: str | None = None) -> Recipe:
    cfg = relocate(copy.deepcopy(recipe.config), root=root, artifacts=artifacts)
    destination = output / recipe.suite / recipe.task / recipe.method / f"seed{recipe.seed}"
    if recipe.runner == "sokoban":
        cfg["output"] = str(destination)
    else:
        for key, value in tuple(cfg.items()):
            if key.endswith("_out") and isinstance(value, str):
                cfg[key] = str(destination / Path(value).name)
        cfg.update({key: str(destination / filename) for key, filename in OUTPUT_FILES.items()})
    if device is not None:
        cfg["device"] = device
        for key in ("reward", "lambo2", "holo"):
            if isinstance(cfg.get(key), dict):
                if "device" in cfg[key]:
                    cfg[key]["device"] = device
                if isinstance(cfg[key].get("reward_adapter"), dict) and "device" in cfg[key]["reward_adapter"].get("kwargs", {}):
                    cfg[key]["reward_adapter"].get("kwargs", {})["device"] = device
        if isinstance(cfg.get("reward_adapter"), dict) and "device" in cfg["reward_adapter"].get("kwargs", {}):
            cfg["reward_adapter"].get("kwargs", {})["device"] = device
        if recipe.task in TASKS and recipe.runner == "interpolant":
            runtime = Path(str(cfg.get("config", "")))
            if device == "cpu" and runtime.with_name(runtime.stem + "_cpu.json").is_file():
                cfg["config"] = str(runtime.with_name(runtime.stem + "_cpu.json"))
            elif device != "cpu" and runtime.stem.endswith("_cpu"):
                cfg["config"] = str(runtime.with_name(runtime.stem[:-4] + ".json"))
    cfg.setdefault("metadata", {})["plot_label"] = recipe.method
    return replace(recipe, config=cfg)


def _check_device(device: str) -> None:
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Required device {device} is unavailable; choose --device explicitly to run a new hardware profile")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Required device mps is unavailable; choose --device explicitly to run a new hardware profile")
    if device != "cpu" and device != "mps" and not device.startswith("cuda"):
        raise ValueError(f"Unsupported device {device!r}")


def missing_inputs(recipe: Recipe) -> list[str]:
    missing = []
    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for name, item in value.items():
                walk(item, name)
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif key in INPUT_KEYS and isinstance(value, str):
            path = Path(value)
            # EvoDiff checkpoints are metadata descriptors; weights are obtained
            # through the pinned adapter. Both loader formats remain supported.
            if key == "checkpoint" and path.with_suffix(path.suffix + ".json").is_file():
                return
            if not path.exists():
                missing.append(str(path))
    walk(recipe.config)
    return sorted(set(missing))


def _paid_rows(path: Path) -> list[dict[str, Any]]:
    return [row for row in iter_jsonl(path) if row.get("oracle_evaluated", True)] if path.is_file() else []


def verify_search_output(recipe: Recipe) -> dict[str, Any]:
    cfg = recipe.config
    initial = _paid_rows(Path(cfg["initial_batch_out"]))
    samples = _paid_rows(Path(cfg["samples_out"]))
    # Cold unconditional search serializes its samples in both streams.
    rows = samples if cfg.get("sample_only") else initial + samples
    expected = int(cfg["max_oracle_evaluations"])
    if len(rows) != expected:
        raise ValueError(f"{recipe.identity}: completed {len(rows)} charged labels; expected {expected}")
    manifest = json.loads(Path(cfg["run_manifest_out"]).read_text())
    if manifest.get("status") != "completed":
        raise ValueError(f"{recipe.identity}: runner did not write completed status")
    if recipe.task in TASKS and cfg.get("initial_batch_path"):
        import struct
        saved = _paid_rows(Path(cfg["initial_batch_path"]))
        f32 = lambda number: struct.unpack("f", struct.pack("f", float(number)))[0]
        project = lambda records: [(record["seq"], f32(record["score"])) for record in records]
        if project(initial) != project(saved):
            raise ValueError(f"{recipe.identity}: initial protein sequences/scores changed")
    return {"identity": recipe.identity, "charged_labels": len(rows), "status": "completed"}


def execute_search(recipe: Recipe, *, root: Path) -> dict[str, Any]:
    cfg = recipe.config
    _check_device(str(cfg.get("device", "cpu")))
    for key in ("reward", "lambo2", "holo"):
        if isinstance(cfg.get(key), dict) and cfg[key].get("device"):
            _check_device(str(cfg[key]["device"]))
    missing = missing_inputs(recipe)
    if missing:
        raise FileNotFoundError("Missing scientific inputs:\n" + "\n".join(missing))
    destination = Path(cfg["run_manifest_out"]).parent
    config_path = destination / "resolved_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError(f"Refusing changed configuration in existing output {destination}")
    # Local output semantics intentionally replace the old imported-run/Git
    # ancestry gates. Runner manifests still hash every concrete input/output.
    cell = StudyCell("online", recipe.task, recipe.seed, recipe.method, destination, cfg,
                     0.0, "canonical", "genturbo_reproduction")
    python = None
    if recipe.runner == "lambo2":
        python = Path(os.environ.get("GENTURBO_LAMBO2_PYTHON", str(root / ".venv-lambo2/bin/python")))
    run_cells([cell], repo_root=root, python=python)
    from genturbo.kernel_study.runner import _verify_completed_cell
    _verify_completed_cell(cell, repo_root=root)
    checked = verify_search_output(recipe)
    if recipe.suite == "ehrlich" and recipe.method == "adalead_official":
        from genturbo.ehrlich_adalead import verify_cell
        verify_cell(cell, repo_root=root)
    atomic_write_json(destination / "reproduction_complete.json", checked)
    return checked


def execute_sokoban(recipe: Recipe, *, root: Path, artifacts: Path, threads: int) -> dict[str, Any]:
    cfg = dict(recipe.config)
    cfg.pop("metadata", None)  # PilotConfig accepts scientific fields only.
    _check_device(str(cfg["device"]))
    destination = Path(cfg["output"])
    checkpoint = artifacts / "runs/train/sokoban_mask_transformer/checkpoints/model.pt"
    starts = artifacts / f"runs/data/sokoban_initial/run{recipe.seed}"
    if not checkpoint.is_file() or not (starts / f"seed{recipe.seed}/initial.npy").is_file():
        raise FileNotFoundError(f"Sokoban requires frozen checkpoint {checkpoint} and saved starts {starts}")
    plan_dir = destination.parent / ".configs"
    plan_dir.mkdir(parents=True, exist_ok=True)
    config_path = plan_dir / f"seed{recipe.seed}.json"
    if config_path.is_file() and json.loads(config_path.read_text()) != cfg:
        raise ValueError(f"Sokoban output already belongs to a different configuration: {destination}")
    atomic_write_json(config_path, cfg)
    command = [sys.executable, "-m", "sokoban.optimise", "--config", str(config_path), "--checkpoint", str(checkpoint),
               "--output", str(destination), "--mode", "search", "--threads", str(threads), "--initial-from", str(starts)]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(command, cwd=root, env=environment, check=True)
    reports = list(destination.rglob("report.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected one Sokoban report for {recipe.identity}, found {len(reports)}")
    report = json.loads(reports[0].read_text())
    if report.get("status") != "completed" or report.get("queries") != cfg["budget"]:
        raise ValueError(f"Sokoban did not complete its exact budget: {recipe.identity}")
    return {"identity": recipe.identity, "status": "completed", "charged_labels": report["queries"]}


def _seeds(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    result = []
    for value in values:
        for part in value.split(","):
            if "-" in part:
                left, right = map(int, part.split("-", 1)); result.extend(range(left, right + 1))
            else:
                result.append(int(part))
    if not result or len(result) != len(set(result)) or not set(result) <= set(range(20)):
        raise ValueError("seeds must be unique values in 0..19, for example --seeds 0-4")
    return result


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("reproduce", help="Plan or execute the local publication recipes")
    parser.add_argument("suite", choices=("ehrlich", "protein", "sokoban", "supporting", "all"))
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--device", help="Explicit generation/oracle device override; recorded devices are otherwise preserved")
    parser.add_argument("--artifacts", type=Path, help="Root containing the verified runs/ and benchmarks/ inputs")
    parser.add_argument("--output", type=Path, default=Path("runs/reproduction"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--analysis", nargs="+", choices=SUPPORT_STAGES, help="Supporting study stages (default: all)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, choices=(1,), default=1, help="Sequential local execution")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Execute; otherwise print a read-only plan")
    mode.add_argument("--dry-run", action="store_true", help="Print the plan without creating files (default)")
    parser.set_defaults(func=handle_reproduce)


def handle_reproduce(args: argparse.Namespace) -> int:
    root = args.repo_root.expanduser().resolve()
    artifacts = (args.artifacts or root).expanduser().resolve()
    output = args.output.expanduser().resolve()
    seeds = _seeds(args.seeds)
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    recipes = build_recipes(root, args.suite, analyses=args.analysis)
    for label, selected, available in (("tasks", args.tasks, {r.task for r in recipes}), ("methods", args.methods, {r.method for r in recipes})):
        if selected and set(selected) - available:
            raise ValueError(f"Unknown {label}: {sorted(set(selected) - available)}; available: {sorted(available)}")
    recipes = [r for r in recipes if (not args.tasks or r.task in args.tasks) and (not args.methods or r.method in args.methods) and (seeds is None or r.seed in seeds)]
    recipes = [resolve_recipe(r, root=root, artifacts=artifacts, output=output, device=args.device) for r in recipes]
    jobs = supporting_jobs(root, artifacts, output, args) if args.suite in {"supporting", "all"} else []
    if not recipes and not jobs:
        raise ValueError("Filters selected no experiments")
    plan = {"suite": args.suite, "execute": bool(args.run), "workers": 1, "artifact_root": str(artifacts), "output": str(output),
            "cell_count": len(recipes), "analysis_jobs": [j[0] for j in jobs], "cells": [
                {"identity": r.identity, "source": r.source, "device": r.config.get("device"),
                 "oracle_budget": r.config.get("max_oracle_evaluations", r.config.get("budget")),
                 "resolved_config_sha256": hashlib.sha256(canonical_json_bytes(r.config)).hexdigest(),
                 "missing_inputs": missing_inputs(r)} for r in recipes]}
    print(json.dumps(plan, indent=2))
    if not args.run:
        return 0
    os.environ.update({key: str(args.threads) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "reproduction_plan.json", plan)
    for recipe in recipes:
        print(json.dumps(execute_sokoban(recipe, root=root, artifacts=artifacts, threads=args.threads) if recipe.runner == "sokoban" else execute_search(recipe, root=root)), flush=True)
    for name, action in jobs:
        action()
        print(json.dumps({"analysis": name, "status": "completed"}), flush=True)
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    from genturbo.serialization import csv_bytes, atomic_write
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, csv_bytes(rows))


def _freeze_kernel_datasets(root: Path, artifacts: Path, output: Path) -> Path:
    """Freeze the complete new candidate cohort using the original classifier."""
    from genturbo.artifacts import sha256_file
    from genturbo.kernel_study.config import load_study
    from genturbo.kernel_study.reporting import _score_frozen_candidates, _atomic_write_jsonl
    from genturbo.kernel_study.results import TraceSource, read_proposal_traces, locality_calibration_rows, freeze_eligible_candidates
    from genturbo.kernel_study.benchmark import _portable_path

    base = output / "supporting/kernel-datasets"
    if not base.is_dir():
        raise FileNotFoundError(f"Generate kernel-datasets into {output} before running the offline analyses")
    study = load_study("configs/experiments/ehrlich_kernel_v1", repo_root=root, verify_inputs=False)
    conditions = []
    calibration = []
    for original in study.datasets["conditions"]:
        condition = relocate(original, root=root, artifacts=artifacts)
        task = condition["name"]
        sources = [TraceSource(base / task / "random" / f"seed{seed}/proposal_traces.jsonl.gz", task, seed) for seed in range(20)]
        for source in sources:
            directory = source.path.parent
            manifest = json.loads((directory / "run_manifest.json").read_text())
            if manifest.get("status") != "completed":
                raise ValueError(f"Incomplete kernel dataset: {directory}")
            if not source.path.is_file():
                raise FileNotFoundError(source.path)
        classified = read_proposal_traces(sources)
        expected = int(study.datasets["generation"]["conditional_iterations"]) * int(study.datasets["generation"]["proposal_batch_size"])
        for seed in range(20):
            if sum(int(row["source_seed"]) == seed for row in classified.all_records) != expected:
                raise ValueError(f"Incomplete proposal trace: {task}/seed{seed}")
        calibration.extend(locality_calibration_rows(classified))
        frozen = freeze_eligible_candidates(classified)
        _score_frozen_candidates(frozen, condition=condition, repo_root=root)
        frozen_path = base / task / "frozen_candidates.jsonl.gz"
        _atomic_write_jsonl(frozen_path, frozen)
        conditions.append({"condition": task, "trace_count": len(sources),
                           "trace_sha256": {_portable_path(s.path, root): sha256_file(s.path) for s in sources},
                           "frozen_candidate_count": len(frozen), "frozen_candidates": str(frozen_path),
                           "frozen_candidates_sha256": sha256_file(frozen_path)})
    atomic_write_json(base / "provenance.json", {"schema_version": 1, "study": "ehrlich_kernel_v1", "workflow": "datasets", "status": "completed", "conditions": conditions})
    _write_csv(output / "supporting/figure_inputs/ehrlich_locality_calibration.csv", calibration)
    return base


def _kernel_offline(root: Path, artifacts: Path, output: Path, workflow: str) -> None:
    from genturbo.kernel_study.config import load_study
    from genturbo.kernel_study.offline import run_static, run_timing
    base = _freeze_kernel_datasets(root, artifacts, output)
    study = load_study("configs/experiments/ehrlich_kernel_v1", repo_root=root, verify_inputs=False)
    study.datasets["run_dir"] = str(base)
    function = run_static if workflow == "static" else run_timing
    function(study, repo_root=root, run_root=output / f"supporting/kernel-{workflow}",
             results_root=output / f"supporting/kernel-{workflow}/tables", status="smoke")


def _kernel_figure_inputs(root: Path, artifacts: Path, output: Path) -> None:
    from genturbo.kernel_study.online_analysis import analyze_sweep
    base = output / "supporting/kernel-online"
    if not base.is_dir():
        raise FileNotFoundError(f"Generate kernel-online into {output} before building trajectories")
    rows = []
    for task in ("l128_q4_k4", "l256_q4_k8"):
        analysis = analyze_sweep(base / task, repo_root=root)
        for run in analysis["runs"]:
            values = run["best_so_far_trajectory"]
            if len(values) != 512:
                raise ValueError(f"Incomplete kernel online trajectory: {task}/{run['method']}/seed{run['seed']}")
            rows.extend({"condition": task, "arm": run["method"], "seed": run["seed"], "oracle_evaluations": i + 1, "best_so_far": value} for i, value in enumerate(values))
    if len(rows) != 40 * 512:
        raise ValueError("Kernel online reporting requires both conditions, four arms, five seeds")
    _write_csv(output / "supporting/figure_inputs/ehrlich_kernel_trajectories.csv", rows)


def _visitation(root: Path, artifacts: Path, output: Path, device: str | None) -> None:
    directory = output / "supporting/visitation"
    directory.mkdir(parents=True, exist_ok=True)
    source = root / "configs/experiments/ehrlich_radial_search_v1/generator_one_edit.json"
    config = relocate(load_config(source), root=root, artifacts=artifacts)
    if device:
        config["device"] = device
    _check_device(str(config.get("device", "mps")))
    path = directory / "resolved_config.json"
    atomic_write_json(path, config)
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    command = [sys.executable, str(root / "scripts/plot_ehrlich_hamming_ball_visitation.py"), "--search-config", str(path),
               "--num-samples", "100000", "--seed", "20260901", "--output-dir", str(directory)]
    if device:
        command.extend(["--device", device])
    subprocess.run(command, cwd=root, env=env, check=True)
    subprocess.run([sys.executable, str(root / "scripts/plot_ehrlich_infeasible_cone.py"), "--input-dir", str(directory),
                    "--clean", "--no-colourbars", "--no-infeasible-bars", "--cutoff-multiple", "0.5", "--paper-width", "6.075",
                    "--output-prefix", str(directory / "ehrlich_infeasible_cone_half_uniform")], cwd=root, env=env, check=True)


def supporting_jobs(root: Path, artifacts: Path, output: Path, args: argparse.Namespace) -> list[tuple[str, Any]]:
    stages = args.analysis or list(SUPPORT_STAGES)
    # Filters indicate a partial experiment; full-cohort aggregation is explicit.
    if args.tasks or args.methods or args.seeds:
        return []
    jobs = []
    if "kernel-datasets" in stages and "kernel-static" not in stages and "kernel-timing" not in stages:
        jobs.append(("kernel-freeze", lambda: _freeze_kernel_datasets(root, artifacts, output)))
    for workflow in ("static", "timing"):
        if f"kernel-{workflow}" in stages:
            jobs.append((f"kernel-{workflow}", lambda workflow=workflow: _kernel_offline(root, artifacts, output, workflow)))
    if "kernel-online" in stages:
        jobs.append(("kernel-trajectory-tables", lambda: _kernel_figure_inputs(root, artifacts, output)))
    if "visitation" in stages:
        jobs.append(("visitation", lambda: _visitation(root, artifacts, output, args.device)))
    if "protein-calibration" in stages:
        jobs.append(("protein-calibration", lambda: _protein_calibration(root, artifacts, output, args.threads)))
    if "counts" in stages:
        jobs.append(("counts", lambda: _counting_inputs(artifacts, output)))
    return jobs


def _counting_inputs(artifacts: Path, output: Path) -> None:
    import shutil
    from genturbo.figures import neighbourhood
    source = artifacts / "reference/paper/figures/data/ehrlich_neighbourhood_source.json"
    destination = output / "supporting/figure_inputs/ehrlich_neighbourhood_source.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    # The standalone mathematical computation uses only the recorded directed
    # support and fixed reference; it does not need a model or new oracle calls.
    figure_root = output / "supporting/neighbourhood"
    (figure_root / "figures/data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, figure_root / "figures/data/ehrlich_neighbourhood_source.json")
    neighbourhood.ROOT = figure_root
    neighbourhood.main(["--check"])


def _protein_calibration(root: Path, artifacts: Path, output: Path, threads: int) -> None:
    """Replay the frozen GFP-M validation/test protocol after fresh searches."""
    config = load_config(root / "configs/experiments/gfp_medium_calibration_v1.json")
    config.update(learned_history_root=str(output / "supporting/gfp_medium/tanimoto_learned"),
                  fixed_history_root=str(output / "protein/gfp_medium/tanimoto"),
                  benchmark_root=str(artifacts / "benchmarks/gfp_aav"), threads=threads)
    destination = output / "supporting/protein-calibration"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "resolved_config.json"
    atomic_write_json(path, config)
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    command = [sys.executable, str(root / "scripts/diagnose_gfp_medium_gp_calibration.py")]
    shared = ["--output", str(destination), "--config", str(path), "--workers", "1", "--threads", str(threads)]
    subprocess.run([*command, "prepare", *shared], cwd=root, env=env, check=True)
    subprocess.run([*command, "run", *shared, "--split", "validation"], cwd=root, env=env, check=True)
    subprocess.run([*command, "summarize", *shared, "--split", "validation"], cwd=root, env=env, check=True)
    # The published decision is frozen: test only its selected setting and the
    # original learned-noise baseline, without selecting again on fresh data.
    subprocess.run([*command, "run", *shared, "--split", "test", "--variants", "learned_unit", "fixed_0.001"], cwd=root, env=env, check=True)
    subprocess.run([*command, "summarize", *shared, "--split", "test"], cwd=root, env=env, check=True)
