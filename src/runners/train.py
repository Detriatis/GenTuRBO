"""Minimal training harness for GenTuRBO experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

from core.optimisers import make_optimiser
from core.registry import build as build_from_registry
from core.utils import maybe_load_checkpoint, resolve_device, seed_everything, write_checkpoint_metadata
from data.base import make_dataloader
from eval.seq_eval import summarise_transition_errors
from runners.common import build_model, build_process


@dataclass
class TrainConfig:
    process: str
    model: Optional[str]
    dataset: str
    process_cfg: Dict[str, Any] = field(default_factory=dict)
    model_cfg: Dict[str, Any] = field(default_factory=dict)
    dataset_cfg: Dict[str, Any] = field(default_factory=dict)
    steps: int = 1000
    steps_per_epoch: Optional[int] = None
    avg_loss_window: int = 0
    eval_every_steps: int = 0
    eval_every_epochs: int = 0
    eval_num_samples: int = 256
    eval_steps: Optional[int] = None
    lr: float = 1e-3
    optimiser: str = "adam"
    optimiser_cfg: Dict[str, Any] = field(default_factory=dict)
    device: Optional[str] = None
    log_every: int = 1
    seed: int = 0
    checkpoint_path: Optional[str] = None
    checkpoint_every: int = 0
    resume_from: Optional[str] = None
    resume_if_checkpoint_exists: bool = True
    wipe_run_dir: bool = False
    run_root: str = "runs"
    run_name: Optional[str] = None
    tensorboard: bool = True
    profile: bool = False
    profile_every: int = 0


_RUN_MARKER = ".genturbo-run.json"


def _safe_run_directory(config: TrainConfig) -> tuple[Path, Path]:
    """Resolve a run path and prove it remains below its configured root."""

    configured_root = Path(config.run_root).expanduser()
    if configured_root.is_symlink():
        raise ValueError(f"run_root must not be a symbolic link: {configured_root}")
    base = configured_root.resolve()
    raw_name = config.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    name = Path(raw_name)
    if name.is_absolute() or not name.parts or any(part in {"", ".", ".."} for part in name.parts):
        raise ValueError("run_name must be a non-empty relative path without '.' or '..' components")
    run_dir = (base / name).resolve()
    try:
        run_dir.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"run directory escapes run_root: {run_dir}") from exc
    if run_dir == base:
        raise ValueError("run directory must be a child of run_root")
    return base, run_dir


def _is_managed_run_directory(run_dir: Path) -> bool:
    marker = run_dir / _RUN_MARKER
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("kind") == "genturbo-run"


def _prepare_run_dirs(config: TrainConfig) -> tuple[Path, Path, Path]:
    _base, run_dir = _safe_run_directory(config)
    if config.wipe_run_dir and run_dir.exists():
        if run_dir.is_symlink() or not run_dir.is_dir() or not _is_managed_run_directory(run_dir):
            raise ValueError(
                f"refusing to wipe an unmanaged run path: {run_dir}; "
                f"only directories containing {_RUN_MARKER} may be replaced"
            )
        shutil.rmtree(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / _RUN_MARKER
    if marker.is_symlink():
        raise ValueError(f"run marker must not be a symbolic link: {marker}")
    marker.write_text(
        json.dumps({"kind": "genturbo-run", "schema_version": 1}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))
    except TypeError:
        (run_dir / "config.txt").write_text(str(config))
    return run_dir, checkpoint_dir, tb_dir


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    step: int,
    *,
    extra_metadata: Optional[dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {"step": step, "config": asdict(config)}
    if extra_metadata:
        metadata.update(extra_metadata)
    torch.save(model.state_dict(), path)
    write_checkpoint_metadata(path, metadata)


def train(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)

    run_dir, checkpoint_dir, tb_dir = _prepare_run_dirs(config)
    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path).expanduser()
    else:
        checkpoint_path = checkpoint_dir / "model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_from_registry("dataset", config.dataset, **config.dataset_cfg)
    instance = getattr(dataset, "instance", None)
    if instance is not None:
        instance_path = run_dir / "dataset_instance.pt"
        try:
            torch.save(instance, instance_path)
        except Exception as exc:
            print(f"[warn] Failed to save dataset instance: {exc}")
    dataloader = make_dataloader(dataset, num_workers=0)
    data_iter = iter(dataloader)
    transition_mask = None
    instance = getattr(dataset, "instance", None)
    if instance is not None:
        transition_mask = getattr(instance, "transition_mask", None)
    steps_per_epoch = config.steps_per_epoch
    if steps_per_epoch is None:
        try:
            steps_per_epoch = len(dataloader)  # type: ignore[arg-type]
        except TypeError:
            steps_per_epoch = None

    model = build_model(config.model, config.model_cfg)
    process = build_process(config.process, model, {**config.process_cfg, "device": device})
    if transition_mask is None:
        transition_mask = getattr(process, "transition_mask", None)
    model_for_optim = process.model if hasattr(process, "model") else model
    if model_for_optim is None:
        raise ValueError("Process did not expose a model to optimise")

    optim_cfg = {"lr": config.lr, **config.optimiser_cfg}
    optimizer = make_optimiser(config.optimiser, model_for_optim.parameters(), **optim_cfg)
    start_step = 0
    resume_path: Optional[Path] = None
    if config.resume_from:
        resume_path = Path(config.resume_from).expanduser()
    elif config.resume_if_checkpoint_exists and checkpoint_path.exists():
        resume_path = checkpoint_path
    elif config.resume_if_checkpoint_exists and not checkpoint_path.exists():
        step_paths = list(checkpoint_dir.glob("model_step*.pt"))
        if step_paths:
            def _step_from_name(path: Path) -> int:
                name = path.stem
                suffix = name.replace("model_step", "")
                return int(suffix) if suffix.isdigit() else -1

            step_paths.sort(key=_step_from_name, reverse=True)
            resume_path = step_paths[0]

    if resume_path and resume_path.exists():
        checkpoint = maybe_load_checkpoint(str(resume_path), model_for_optim)
        if checkpoint and "step" in checkpoint.metadata:
            start_step = int(checkpoint.metadata["step"])

    history: dict[str, list[float]] = {"loss": []}
    loss_window: list[float] = []
    writer: Optional[SummaryWriter] = SummaryWriter(log_dir=str(tb_dir)) if config.tensorboard else None

    data_shape: Optional[tuple[int, ...]] = None
    start_time = time.perf_counter()
    warned_no_transition_mask = False
    use_profile = bool(config.profile)
    profile_every = int(config.profile_every) if config.profile_every else config.log_every

    def _sync_device() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    eval_every_steps = int(config.eval_every_steps)
    if config.eval_every_epochs > 0:
        if steps_per_epoch is None or steps_per_epoch <= 0:
            print("[warn] eval_every_epochs set but steps_per_epoch is undefined; skipping eval cadence.")
        else:
            eval_every_steps = int(config.eval_every_epochs) * int(steps_per_epoch)

    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours:d}h{mins:02d}m{secs:02d}s"
        if mins > 0:
            return f"{mins:d}m{secs:02d}s"
        return f"{secs:d}s"

    def _valid_fraction(samples: Tensor, mask: Tensor) -> float:
        prev = samples[:, :-1]
        nxt = samples[:, 1:]
        allowed = mask[prev, nxt]
        return float(allowed.all(dim=1).float().mean().item())

    for step in range(start_step, config.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        x_batch = batch.get("x")
        if x_batch is not None:
            data_shape = tuple(x_batch.shape[1:])
        optimizer.zero_grad()
        if use_profile:
            t_fwd_start = time.perf_counter()
            _sync_device()
            loss, metrics = process.training_step(batch)
            _sync_device()
            t_fwd = time.perf_counter() - t_fwd_start
            t_bwd_start = time.perf_counter()
            loss.backward()
            _sync_device()
            t_bwd = time.perf_counter() - t_bwd_start
            t_opt_start = time.perf_counter()
            optimizer.step()
            _sync_device()
            t_opt = time.perf_counter() - t_opt_start
        else:
            loss, metrics = process.training_step(batch)
            loss.backward()
            optimizer.step()
        loss_value = float(metrics["loss"].item())
        history["loss"].append(loss_value)
        if config.avg_loss_window > 0:
            loss_window.append(loss_value)
            if len(loss_window) > config.avg_loss_window:
                loss_window.pop(0)
        global_step = step + 1
        if writer is not None:
            writer.add_scalar("train/loss", loss_value, global_step)
            for key, value in metrics.items():
                if key == "loss":
                    continue
                writer.add_scalar(f"train/{key}", float(value.item()), global_step)
            if use_profile and global_step % profile_every == 0:
                writer.add_scalar("perf/fwd_ms", t_fwd * 1000.0, global_step)
                writer.add_scalar("perf/bwd_ms", t_bwd * 1000.0, global_step)
                writer.add_scalar("perf/opt_ms", t_opt * 1000.0, global_step)
        if config.checkpoint_every > 0 and global_step % config.checkpoint_every == 0:
            periodic_path = checkpoint_dir / f"model_step{global_step}.pt"
            extra = {
                "data_shape": list(data_shape) if data_shape is not None else None,
                "run_dir": str(run_dir),
                "loss": loss_value,
            }
            _save_checkpoint(periodic_path, model_for_optim, config, global_step, extra_metadata=extra)

        if eval_every_steps > 0 and global_step % eval_every_steps == 0:
            if transition_mask is None:
                if not warned_no_transition_mask:
                    print("[warn] No transition_mask available; skipping validity eval.")
                    warned_no_transition_mask = True
            else:
                model_ref = process.model if hasattr(process, "model") else model_for_optim
                was_training = model_ref.training if model_ref is not None else False
                samples = process.sample(
                    config.eval_num_samples,
                    steps=config.eval_steps,
                    device=device,
                    show_progress=False,
                )
                if model_ref is not None:
                    model_ref.train(was_training)
                valid_frac = _valid_fraction(samples, transition_mask.to(samples.device))
                if writer is not None:
                    writer.add_scalar("eval/valid_fraction", valid_frac, global_step)
                    errors = summarise_transition_errors(samples, transition_mask)
                    writer.add_scalar("eval/avg_incorrect_transitions", errors["avg_incorrect_transitions"], global_step)
                    writer.add_scalar("eval/max_incorrect_transitions", errors["max_incorrect_transitions"], global_step)

        if global_step % config.log_every == 0:
            elapsed = time.perf_counter() - start_time
            steps_done = max(1, global_step - start_step)
            avg_step = elapsed / float(steps_done)
            eta_total = _format_duration(avg_step * (config.steps - global_step))
            avg_loss = None
            if config.avg_loss_window > 0 and loss_window:
                avg_loss = sum(loss_window) / float(len(loss_window))
            perf_str = ""
            if use_profile and global_step % profile_every == 0:
                perf_str = (
                    f" fwd={t_fwd*1000:.1f}ms"
                    f" bwd={t_bwd*1000:.1f}ms"
                    f" opt={t_opt*1000:.1f}ms"
                )
            if steps_per_epoch is not None and steps_per_epoch > 0:
                steps_into_epoch = steps_done % steps_per_epoch
                steps_left_epoch = steps_per_epoch - steps_into_epoch if steps_into_epoch else 0
                eta_epoch = _format_duration(avg_step * steps_left_epoch)
                total_epochs = (config.steps + steps_per_epoch - 1) // steps_per_epoch
                epoch_idx = steps_done // steps_per_epoch + (1 if steps_into_epoch else 0)
                avg_loss_str = f" avg_loss={avg_loss:.4f}" if avg_loss is not None else ""
                print(
                    f"[step {global_step}] loss={loss_value:.4f} "
                    f"eta_total={eta_total} epoch={epoch_idx}/{total_epochs} eta_epoch={eta_epoch}"
                    f"{avg_loss_str}{perf_str}"
                )
            else:
                avg_loss_str = f" avg_loss={avg_loss:.4f}" if avg_loss is not None else ""
                print(f"[step {global_step}] loss={loss_value:.4f}{avg_loss_str} eta_total={eta_total}{perf_str}")

    if writer is not None:
        writer.flush()
        writer.close()

    extra = {
        "data_shape": list(data_shape) if data_shape is not None else None,
        "run_dir": str(run_dir),
        "loss": history["loss"][-1] if history["loss"] else None,
    }
    _save_checkpoint(checkpoint_path, model_for_optim, config, config.steps, extra_metadata=extra)

    return {"history": history, "checkpoint": str(checkpoint_path), "run_dir": str(run_dir)}


def _load_config(path: Optional[str]) -> TrainConfig:
    if path is None:
        raise ValueError("an explicit training configuration is required")
    cfg_path = Path(path)
    data = json.loads(cfg_path.read_text())
    return TrainConfig(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GenTuRBO experiment")
    parser.add_argument("--config", type=str, required=True, help="Named training configuration JSON")
    parser.add_argument("--process", type=str, default=None, help="Override process name")
    parser.add_argument("--steps", type=int, default=None, help="Override number of steps")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), help="Override the configured local training device")
    parser.add_argument("--run-root", help="Root for locally generated training outputs")
    parser.add_argument("--run-name", help="Training run directory name below --run-root")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.process:
        cfg.process = args.process
    if args.steps is not None:
        cfg.steps = args.steps
    if args.device is not None:
        cfg.device = args.device
    if args.run_root is not None:
        cfg.run_root = args.run_root
    if args.run_name is not None:
        cfg.run_name = args.run_name

    train(cfg)


if __name__ == "__main__":
    main()
