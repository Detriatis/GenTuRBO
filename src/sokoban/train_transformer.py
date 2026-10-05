"""Resumable paper-style Sokoban training and GPU throughput benchmark.

Run with --mode benchmark before the long train run. Optimizer updates, masks,
data cursor, gradient scaling and scheduler position are retained on resume.
This dedicated runner leaves the existing Ehrlich/CNN training path intact.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import signal
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from core.utils import resolve_device, seed_everything
from data.boxoban import BoxobanConfig, BoxobanIterable
from runners.common import build_model, build_process
from runners.train import _safe_run_directory
from sokoban.boards import decode_board
from sokoban.metrics import assess_boards, summarize
from sokoban.prepare import sha256
from sokoban.solver import SolverConfig

DEFAULT_CONFIG = "configs/train/sokoban_mask_transformer.json"


@dataclass
class TransformerTrainConfig:
    model: str = "sokoban_transformer"
    model_cfg: dict = field(default_factory=dict)
    process: str = "sokoban_masked_diffusion"
    process_cfg: dict = field(default_factory=dict)
    dataset: str = "boxoban"
    dataset_cfg: dict = field(default_factory=dict)
    steps: int = 292000
    micro_batch_size: int = 1536
    lr: float = 0.000245
    weight_decay: float = 0.01
    warmup_steps: int = 125
    min_lr_ratio: float = 0.0
    gradient_clip: float = 1.0
    precision: str = "float16"
    device: str = "cuda"
    seed: int = 0
    checkpoint_every: int = 5000
    validation_every: int = 1000
    validation_size: int = 4096
    evaluation_every: int = 50000
    evaluation_samples: int = 5000
    evaluation_batch_size: int = 128
    evaluation_seed: int = 42
    solver_workers: int = 4
    solver_max_states: int = 200000
    solver_max_seconds: float = 30.0
    log_every: int = 100
    run_root: str = "runs/train"
    run_name: str = "sokoban_mask_transformer"

    def __post_init__(self):
        if (self.model, self.process, self.dataset) != ("sokoban_transformer", "sokoban_masked_diffusion", "boxoban"):
            raise ValueError("expected the paper-style Boxoban Transformer configuration")
        if self.dataset_cfg.get("split", "train") != "train":
            raise ValueError("training must use the train split")
        if not self.dataset_cfg.get("shuffle", True):
            raise ValueError("training must shuffle each epoch")
        if self.dataset_cfg.get("seed", self.seed) != self.seed:
            raise ValueError("dataset seed must equal training seed")
        for name in ("steps", "micro_batch_size", "validation_size", "evaluation_samples",
                     "evaluation_batch_size", "solver_workers", "solver_max_states", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("checkpoint_every", "validation_every", "evaluation_every", "warmup_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.warmup_steps >= self.steps or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("warmup must be shorter than the run; min_lr_ratio must lie in [0,1]")
        for name in ("lr", "gradient_clip", "solver_max_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unknown precision")
        BoxobanConfig(**self.dataset_cfg)
        _safe_run_directory(self)


class BoardStream:
    """Full effective batches across epoch boundaries, with no dropped boards."""

    def __init__(self, dataset: BoxobanIterable):
        self.dataset = dataset
        self.epoch = self.offset = 0
        self.rng = np.random.default_rng(dataset.cfg.seed)
        self._permutation()

    def _permutation(self):
        self.order = np.random.default_rng(np.random.SeedSequence(
            [self.dataset.cfg.seed, self.epoch]
        )).permutation(self.dataset.num_levels)

    def next(self):
        remaining = self.dataset.cfg.batch_size
        chunks = []
        while remaining:
            take = min(remaining, self.dataset.num_levels - self.offset)
            chunks.append(self.dataset.boards[self.order[self.offset:self.offset + take]])
            self.offset += take
            remaining -= take
            if self.offset == self.dataset.num_levels:
                self.epoch += 1
                self.offset = 0
                self._permutation()
        boards = np.concatenate(chunks).astype(np.int64)
        if self.dataset.cfg.augment:
            symmetries = self.dataset.symmetries[self.rng.integers(0, 8, size=len(boards))]
            boards = np.take_along_axis(boards, symmetries, axis=1)
        return torch.from_numpy(np.ascontiguousarray(boards))

    def state_dict(self):
        return {"epoch": self.epoch, "offset": self.offset, "augmentation_rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.epoch, self.offset = state["epoch"], state["offset"]
        if self.epoch < 0 or not 0 <= self.offset < self.dataset.num_levels:
            raise ValueError("invalid data cursor")
        self.rng.bit_generator.state = state["augmentation_rng"]
        self._permutation()


def learning_rate(config, completed_steps):
    """LR for the next update: warmup peaks at W, last scheduled update is zero."""
    update = completed_steps + 1
    if config.warmup_steps and update <= config.warmup_steps:
        factor = update / config.warmup_steps
    else:
        progress = (update - max(config.warmup_steps, 1)) / max(config.steps - max(config.warmup_steps, 1), 1)
        factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return config.lr * factor


def autocast(config, device):
    return torch.autocast(device_type=device.type, dtype=getattr(torch, config.precision),
                          enabled=config.precision != "float32")


def capture_rng():
    np_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]]}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gauss))


@contextmanager
def isolated_rng(seed):
    state = capture_rng()
    try:
        seed_everything(seed)
        yield
    finally:
        restore_rng(state)


class TrainingState:
    def __init__(self, config):
        self.config = config
        seed_everything(config.seed)
        self.device = resolve_device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run within a GPU allocation")
        if self.device.type != "cuda" and config.precision != "float32":
            raise ValueError("non-CUDA smoke tests require precision=float32")
        self.dataset = BoxobanIterable(BoxobanConfig(**config.dataset_cfg))
        self.stream = BoardStream(self.dataset)
        self.model = build_model(config.model, config.model_cfg)
        self.process = build_process(config.process, self.model, {**config.process_cfg, "device": self.device})
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda" and config.precision == "float16")
        self.step = self.attempts = self.examples = 0
        self.last_validation = self.last_evaluation = 0

    def update(self):
        self.model.train()
        batch = self.stream.next()
        lr = learning_rate(self.config, self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for chunk in batch.split(self.config.micro_batch_size):
            with autocast(self.config, self.device):
                loss, _ = self.process.training_step({"x": chunk})
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite training loss")
            weight = len(chunk) / len(batch)
            self.scaler.scale(loss * weight).backward()
            total_loss += float(loss.detach()) * weight
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        old_scale = self.scaler.get_scale()
        if not self.scaler.is_enabled() and not bool(torch.isfinite(norm)):
            raise RuntimeError("non-finite gradients")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        updated = self.scaler.get_scale() >= old_scale
        self.step += int(updated)
        self.attempts += 1
        self.examples += len(batch)
        return {"step": self.step, "attempt": self.attempts, "examples": self.examples, "loss": total_loss,
                "lr": lr, "gradient_norm": float(norm) if torch.isfinite(norm) else None,
                "optimizer_updated": updated, "scale": self.scaler.get_scale()}

    def metadata(self):
        return {"config": asdict(self.config), "step": self.step, "data_shape": [100],
                "sampler": "uniform-random-order-one-cell", "examples": self.examples}

    def checkpoint(self, path, fingerprint, *, full=True):
        payload = {"model_state": self.model.state_dict(), "metadata": self.metadata()}
        if full:
            payload["training_state"] = {
                "version": 1, "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                "cursor": self.stream.state_dict(), "rng": capture_rng(), "attempts": self.attempts,
                "last_validation": self.last_validation, "last_evaluation": self.last_evaluation,
                "data_fingerprint": fingerprint,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def resume(self, path, fingerprint):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        saved = payload.get("training_state")
        if saved is None or saved.get("version") != 1:
            raise ValueError("resume requires a full training_state.pt checkpoint, not weights only")
        if payload["metadata"]["config"] != asdict(self.config):
            raise ValueError("resume configuration must match exactly, including schedule and batch sizes")
        if saved["data_fingerprint"] != fingerprint:
            raise ValueError("dataset changed since the checkpoint")
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.scaler.load_state_dict(saved["scaler"])
        self.stream.load_state_dict(saved["cursor"])
        self.step = payload["metadata"]["step"]
        self.examples = payload["metadata"]["examples"]
        self.attempts = saved["attempts"]
        self.last_validation = saved["last_validation"]
        self.last_evaluation = saved["last_evaluation"]
        restore_rng(saved["rng"])


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.no_grad()
def validation_loss(state, validation):
    config = state.config
    size = min(config.validation_size, len(validation))
    indices = np.random.default_rng(config.evaluation_seed).choice(len(validation), size=size, replace=False)
    total = 0.0
    state.model.eval()
    with isolated_rng(config.evaluation_seed):
        for start in range(0, size, config.evaluation_batch_size):
            batch = torch.from_numpy(validation[indices[start:start + config.evaluation_batch_size]].astype(np.int64))
            with autocast(config, state.device):
                loss, _ = state.process.training_step({"x": batch})
            total += float(loss) * len(batch)
    return {"step": state.step, "weighted_masked_ce": total / size, "boards": size, "seed": config.evaluation_seed}


def _solve_record(arguments):
    index, board, config = arguments
    return assess_boards([board], config, ids=[str(index)])[1][0]


def _initialize_solver_worker():
    # The parent handles the checkpoint signal, cancels queued searches and
    # waits for the currently bounded searches to finish before saving state.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


@torch.no_grad()
def evaluate_generated(state, output, *, should_stop=lambda: False):
    config = state.config
    output.mkdir(parents=True, exist_ok=True)
    boards = []
    with isolated_rng(config.evaluation_seed):
        for start in range(0, config.evaluation_samples, config.evaluation_batch_size):
            if should_stop():
                raise InterruptedError("evaluation interrupted; checkpoint retained")
            with autocast(config, state.device):
                sample = state.process.sample(min(config.evaluation_batch_size, config.evaluation_samples - start), steps=100)
            boards.extend(sample.cpu().numpy())
    boards = np.asarray(boards, dtype=np.uint8)
    np.save(output / "boards.npy", boards)
    (output / "boards.txt").write_text("\n\n".join(f"; {i}\n{decode_board(b)}" for i, b in enumerate(boards)) + "\n")
    solver_config = SolverConfig(max_states=config.solver_max_states, max_seconds=config.solver_max_seconds)
    work = ((i, b, solver_config) for i, b in enumerate(boards))
    pool = ProcessPoolExecutor(max_workers=config.solver_workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_initialize_solver_worker) \
        if config.solver_workers > 1 else None
    records = []
    try:
        results = pool.map(_solve_record, work, chunksize=1) if pool else map(_solve_record, work)
        with (output / "records.jsonl").open("w") as handle:
            for record in results:
                if should_stop():
                    raise InterruptedError("solver evaluation interrupted; checkpoint retained")
                records.append(record)
                handle.write(json.dumps(record) + "\n")
                if len(records) % 100 == 0:
                    handle.flush()
                    print(f"Solved/checking {len(records)}/{len(boards)} samples at step {state.step}", flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    report, _ = assess_boards([], solver_config)
    report.update(summarize(records))
    report.update(step=state.step, sampling_seed=config.evaluation_seed, sampling_steps=100,
                  sampling_precision=config.precision,
                  sampler="uniform-random-order-one-cell", sample_sha256=sha256(output / "boards.npy"),
                  unique_raw_boards=len({b.tobytes() for b in boards}), filtering="none", repair="not applied")
    atomic_json(output / "report.json", report)
    return report


def benchmark(state, *, warmup_steps, timing_steps):
    if warmup_steps < 1 or timing_steps < 1:
        raise ValueError("benchmark lengths must be positive")
    durations, skipped = [], 0
    if state.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(state.device)
    for attempt in range(warmup_steps + timing_steps):
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = state.update()
        if state.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if attempt >= warmup_steps:
            durations.append(elapsed)
            skipped += not metrics["optimizer_updated"]
        if (attempt + 1) % 10 == 0:
            print(json.dumps({"benchmark_attempt": attempt + 1, "seconds": elapsed, **metrics}), flush=True)
    report = {"warmup_steps": warmup_steps, "timed_attempts": timing_steps, "timed_skipped_updates": skipped,
              "effective_batch_size": state.dataset.cfg.batch_size, "micro_batch_size": state.config.micro_batch_size,
              "seconds_per_update_mean": statistics.mean(durations),
              "projected_training_hours": statistics.mean(durations) * state.config.steps / 3600 if not skipped else None,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if state.device.type == "cuda" else None,
              "precision": state.config.precision, "parameters": sum(p.numel() for p in state.model.parameters()),
              "checkpoint_saved": False, "exclusions": "queue, setup, validation, solver evaluation and checkpoint I/O"}
    if skipped:
        report["warning"] = "FP16 overflow during timed updates; throughput is not a valid training projection"
    return report


def run(config, *, mode="train", resume=None, warmup_steps=25, timing_steps=100, stop_after=None):
    _, output = _safe_run_directory(config)
    if output.exists() and resume is None:
        raise FileExistsError(f"run already exists: {output}; use explicit --resume")
    if resume is not None and (mode != "train" or Path(resume).resolve() != (output / "checkpoints/training_state.pt").resolve()):
        raise ValueError("resume must use this run's checkpoints/training_state.pt")
    state = TrainingState(config)
    validation = BoxobanIterable(BoxobanConfig(root=config.dataset_cfg["root"], split="valid", augment=False)).boards
    fingerprint = {split: sha256(Path(config.dataset_cfg["root"]) / f"{split}.npy") for split in ("train", "valid")}
    if resume is not None:
        state.resume(resume, fingerprint)
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", asdict(config))
        atomic_json(output / "data_fingerprint.json", fingerprint)
        source_root = Path(__file__).resolve().parents[1]
        atomic_json(output / "source_fingerprint.json", {
            name: sha256(source_root / name) for name in (
                "models/sokoban_transformer.py", "processes/sokoban_masked.py", "sokoban/train_transformer.py",
                "data/boxoban.py", "sokoban/boards.py", "sokoban/solver.py", "sokoban/metrics.py",
            )
        })
        atomic_json(output / "hardware.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "device": str(state.device), "gpu": torch.cuda.get_device_name() if state.device.type == "cuda" else None,
                    "precision": config.precision})
    if mode == "benchmark":
        report = benchmark(state, warmup_steps=warmup_steps, timing_steps=timing_steps)
        atomic_json(output / "timing.json", report)
        return report
    interrupted = False

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)
    original_handlers = {sig: signal.signal(sig, request_stop) for sig in signals}
    latest = output / "checkpoints/training_state.pt"
    start_step = state.step
    started = time.perf_counter()

    def save():
        state.checkpoint(latest, fingerprint)

    def diagnostics():
        final = state.step == config.steps
        if state.step and config.validation_every and state.last_validation != state.step and (
                state.step % config.validation_every == 0 or final):
            result = validation_loss(state, validation)
            atomic_json(output / f"validation_step{state.step}.json", result)
            print(json.dumps({"validation": result}), flush=True)
            state.last_validation = state.step
        if state.step and config.evaluation_every and state.last_evaluation != state.step and (
                state.step % config.evaluation_every == 0 or final):
            save()
            result = evaluate_generated(state, output / f"evaluation/step{state.step}", should_stop=lambda: interrupted)
            print(json.dumps({"evaluation_step": state.step, "raw_solvability": result["raw_solvability_rate"]}), flush=True)
            state.last_evaluation = state.step
            save()

    try:
        diagnostics()  # Complete an evaluation interrupted after its pre-evaluation checkpoint.
        consecutive_skips = 0
        with (output / "training.jsonl").open("a") as log:
            while state.step < config.steps and not interrupted:
                metrics = state.update()
                consecutive_skips = 0 if metrics["optimizer_updated"] else consecutive_skips + 1
                if consecutive_skips > 50:
                    raise RuntimeError("50 consecutive AMP overflows; stopping instead of consuming the budget")
                if state.step % config.log_every == 0 or not metrics["optimizer_updated"]:
                    log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps(metrics, allow_nan=False), flush=True)
                if not metrics["optimizer_updated"]:
                    continue
                if config.checkpoint_every and state.step % config.checkpoint_every == 0:
                    save()
                    state.checkpoint(output / f"checkpoints/model_step{state.step}.pt", fingerprint, full=False)
                if stop_after is not None and state.step - start_step >= stop_after:
                    interrupted = True
                if not interrupted:
                    diagnostics()
        save()
        status = "interrupted" if interrupted else "completed"
        if not interrupted:
            state.checkpoint(output / "checkpoints/model.pt", fingerprint, full=False)
            (output / "interrupted.json").unlink(missing_ok=True)
        report = {"status": status, "steps": state.step, "examples": state.examples,
                  "attempts": state.attempts, "session_seconds": time.perf_counter() - started,
                  "resume_checkpoint": str(latest)}
        atomic_json(output / ("training_complete.json" if not interrupted else "interrupted.json"), report)
        return report
    except InterruptedError:
        save()
        report = {"status": "interrupted", "steps": state.step, "resume_checkpoint": str(latest)}
        atomic_json(output / "interrupted.json", report)
        return report
    finally:
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("train", "benchmark"), default="train")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-root")
    parser.add_argument("--run-name")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--warmup-steps", type=int, default=25, help="benchmark warmup; does not change optimizer warmup")
    parser.add_argument("--timing-steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    values = json.loads(Path(args.config).read_text())
    for key in ("run_root", "run_name", "device", "precision", "micro_batch_size"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.data_root:
        values["dataset_cfg"]["root"] = args.data_root
    config = TransformerTrainConfig(**values)
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "config": asdict(config),
                          "scheduled_examples": config.steps * config.dataset_cfg["batch_size"]}, indent=2))
        return
    report = run(config, mode=args.mode, resume=args.resume, warmup_steps=args.warmup_steps, timing_steps=args.timing_steps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
