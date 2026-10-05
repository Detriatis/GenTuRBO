"""Structural interfaces used by the interpolant search runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class RolloutAdapter(Protocol):
    T: int
    device: torch.device

    def rollout_from_batch(
        self,
        xt_batch: torch.Tensor,
        t_value: float,
        *,
        update_mask: torch.Tensor | None = None,
        denoise_steps: int | None = None,
        t_is_time: bool = False,
    ) -> Sequence[torch.Tensor]: ...


@runtime_checkable
class BatchDecoder(Protocol):
    def decode_batch(self, batch: torch.Tensor) -> Sequence[str | None]: ...


@runtime_checkable
class BatchReward(Protocol):
    def score_batch(self, batch: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class ScoreCache(Protocol):
    def cached_scores(self, batch: torch.Tensor) -> Sequence[float | None]: ...


@runtime_checkable
class OracleObserver(Protocol):
    def observe_oracle_batch(self, batch: torch.Tensor, scores: torch.Tensor) -> Any: ...


__all__ = ["BatchDecoder", "BatchReward", "OracleObserver", "RolloutAdapter", "ScoreCache"]
