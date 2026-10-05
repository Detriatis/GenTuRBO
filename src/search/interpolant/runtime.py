"""Typed runtime state extracted from the legacy search engine façade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class SearchRuntime:
    cfg: dict[str, Any]
    process_cfg: dict[str, Any]
    process: Any
    model_for_state: torch.nn.Module | None
    device: torch.device
    adapter: Any
    flow_steps: int
    max_time: float
    time_grid_norm: torch.Tensor
    seq_len: int
    vocab_size: int
    reward_fn: Any
    transition_mask: torch.Tensor | None
    temperature: float


__all__ = ["SearchRuntime"]
