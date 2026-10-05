"""Oracle accounting and deterministic utilities used by the Holo baseline."""
from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def now_seconds() -> float:
    return float(time.perf_counter())


def elapsed_since(start: float) -> float:
    return float(time.perf_counter() - float(start))


def _pack_samples(samples: list[Any]) -> Any:
    if not samples:
        return []
    if all(isinstance(sample, torch.Tensor) for sample in samples):
        shapes = {tuple(sample.shape) for sample in samples}
        if len(shapes) == 1:
            return torch.stack([sample.detach().cpu() for sample in samples], dim=0)
    if all(isinstance(sample, np.ndarray) for sample in samples):
        shapes = {tuple(np.asarray(sample).shape) for sample in samples}
        if len(shapes) == 1:
            return np.stack([np.asarray(sample) for sample in samples], axis=0)
    return samples


def _coerce_scores(raw: Any, expected: int) -> np.ndarray:
    if isinstance(raw, torch.Tensor):
        arr = raw.detach().cpu().numpy()
    else:
        arr = np.asarray(raw)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size != int(expected):
        raise ValueError(f"score output has {arr.size} values, expected {expected}")
    return arr


def evaluate_oracle(oracle: Any, samples: list[Any]) -> tuple[np.ndarray, int]:
    """Evaluate each requested sample exactly once.

    Batch-capable interfaces are explicit.  An exception from an oracle is a
    systemic failure and is never used as control flow to retry the same
    candidates through a different call shape.
    """

    n = len(samples)
    if n == 0:
        return np.zeros((0,), dtype=float), 0
    packed = _pack_samples(samples)

    score_batch = getattr(oracle, "score_batch", None)
    if callable(score_batch):
        try:
            return _coerce_scores(score_batch(packed), n), n
        except Exception as exc:
            raise RuntimeError(f"oracle.score_batch failed for {n} candidates") from exc

    score = getattr(oracle, "score", None)
    if callable(score):
        if bool(getattr(oracle, "score_is_vectorized", False)):
            try:
                return _coerce_scores(score(packed), n), n
            except Exception as exc:
                raise RuntimeError(f"vectorized oracle.score failed for {n} candidates") from exc
        values: list[float] = []
        for index, sample in enumerate(samples):
            try:
                values.append(float(score(sample)))
            except Exception as exc:
                raise RuntimeError(f"oracle.score failed for candidate {index}") from exc
        return np.asarray(values, dtype=float), n

    if callable(oracle):
        if bool(getattr(oracle, "oracle_is_vectorized", False)):
            try:
                return _coerce_scores(oracle(packed), n), n
            except Exception as exc:
                raise RuntimeError(f"vectorized oracle callable failed for {n} candidates") from exc
        values = []
        for index, sample in enumerate(samples):
            try:
                values.append(float(oracle(sample)))
            except Exception as exc:
                raise RuntimeError(f"oracle callable failed for candidate {index}") from exc
        return np.asarray(values, dtype=float), n

    raise TypeError("oracle must expose score_batch, score, or be callable")


def top_k_samples(samples: list[Any], scores: np.ndarray, *, k: int) -> tuple[list[Any], np.ndarray]:
    scores_np = np.asarray(scores, dtype=float).reshape(-1)
    if scores_np.size != len(samples):
        raise ValueError("scores must align with samples")
    if scores_np.size == 0:
        return [], np.zeros((0,), dtype=float)
    order = np.argsort(np.where(np.isfinite(scores_np), scores_np, -np.inf))[::-1]
    order = order[: max(0, min(int(k), len(order)))]
    return [samples[int(idx)] for idx in order], scores_np[order]
