"""Explicit local generator-training recipes; frozen search assets stay separate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from genturbo.configuration import load_config
from genturbo.reproduce import _check_device, relocate
from genturbo.serialization import atomic_write_json

MODELS = ("ehrlich", "ehrlich-short", "sokoban", "lambo2-ehrlich")


def training_plan(model: str, *, root: Path, artifacts: Path, output: Path,
                  device: str, data_root: Path | None = None) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f"Unknown training recipe {model}")
    sources = {
        "ehrlich": root / "configs/train/ctmc_trainer_conv512.json",
        "ehrlich-short": artifacts / "runs/train/trained_ctmc/config.json",
        "sokoban": root / "configs/train/sokoban_mask_transformer.json",
        "lambo2-ehrlich": root / "configs/experiments/ehrlich_sequence_baselines_v2/baselines.json",
    }
    source = sources[model]
    config = relocate(load_config(source), root=root, artifacts=artifacts)
    config_path = output / "resolved_training_config.json"
    inputs = [source]
    if model == "lambo2-ehrlich":
        spec = config["lambo2"]["pretraining"]
        spec.update(checkpoint=str(output / "generator.ckpt"),
                    manifest=str(output / "pretraining_manifest.json"), device=device)
        inputs.extend(Path(spec[key]) for key in ("markov_training_config", "reference_checkpoint"))
        executable = os.environ.get("GENTURBO_LAMBO2_PYTHON", str(root / ".venv-lambo2/bin/python"))
        command = [executable, "-m", "genturbo.sequence_baseline_study.lambo2", "--repo-root", str(root),
                   "pretrain", "--study-config", str(config_path)]
        training_output = output
    else:
        config.update(device=device, run_root=str(output), run_name="model")
        training_output = output / "model"
        if model.startswith("ehrlich"):
            config.update(steps=18000, checkpoint_path=None, resume_from=None,
                          resume_if_checkpoint_exists=False, wipe_run_dir=False)
            command = [sys.executable, "-m", "runners.train", "--config", str(config_path)]
        else:
            config["dataset_cfg"]["root"] = str(data_root or artifacts / "runs/data/boxoban")
            if device in {"cpu", "mps"}:
                config["precision"] = "float32"
            inputs.append(Path(config["dataset_cfg"]["root"]))
            command = [sys.executable, "-m", "sokoban.train_transformer", "--config", str(config_path)]
    return {"model": model, "source": str(source), "output": str(output),
            "training_output": str(training_output), "device": device,
            "config_path": str(config_path), "resolved_config": config, "command": command,
            "missing_inputs": [str(path) for path in inputs if not path.exists()]}


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("train", help="Plan or train generators locally with explicit output and device")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--data-root", type=Path, help="Boxoban training data root (Sokoban only)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="Print the complete plan without writing files (default)")
    parser.set_defaults(func=handle_train)


def handle_train(args: argparse.Namespace) -> int:
    root = args.repo_root.expanduser().resolve()
    artifacts = (args.artifacts or root).expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.data_root is not None and args.model != "sokoban":
        raise ValueError("--data-root is only used by the Sokoban recipe")
    plan = training_plan(args.model, root=root, artifacts=artifacts, output=output, device=args.device,
                         data_root=args.data_root.expanduser().resolve() if args.data_root else None)
    print(json.dumps({"execute": bool(args.run), **plan}, indent=2))
    if not args.run:
        return 0
    _check_device(args.device)
    if plan["missing_inputs"]:
        raise FileNotFoundError("Missing training inputs: " + ", ".join(plan["missing_inputs"]))
    executable = Path(plan["command"][0])
    if not executable.is_file():
        raise FileNotFoundError(f"Training environment is missing: {executable}; run scripts/setup.py --lambo2")
    # Every public training command creates a fresh directory, so it cannot
    # resume from or replace one of the imported publication checkpoints.
    if output.exists():
        raise ValueError(f"Training requires a fresh --output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(Path(plan["config_path"]), plan["resolved_config"])
    atomic_write_json(output / "training_plan.json", plan)
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    if args.model == "lambo2-ehrlich":
        environment.setdefault("GENTURBO_DEPENDENCY_PROFILE", "lambo2-py310")
    subprocess.run(plan["command"], cwd=root, env=environment, check=True)
    return 0
