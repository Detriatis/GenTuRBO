"""Simple Markov-chain sequence generator for discrete experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from core.registry import register
from data.base import Compose, Transform, TransformsLike, make_dataloader, to_compose



@dataclass
class SeqMarkovConfig:
    vocab_size: int
    seq_len: int
    batch_size: int = 256
    seed: int = 0
    transforms: TransformsLike = None
    start_probs: Optional[Sequence[float]] = None
    transition_matrix: Optional[Sequence[Sequence[float]]] = None
    self_transition_bias: float = 0.7
    offdiag_smoothing: float = 0.05
    emit_start_token: bool = False


def _make_generator(base_seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(base_seed)
    return g


def _build_start_probs(cfg: SeqMarkovConfig) -> Tensor:
    if cfg.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if cfg.start_probs is None:
        probs = torch.ones(cfg.vocab_size, dtype=torch.float)
        return probs / probs.sum()
    probs = torch.tensor(cfg.start_probs, dtype=torch.float)
    if probs.numel() != cfg.vocab_size:
        raise ValueError("start_probs length must equal vocab_size")
    if (probs < 0).any():
        raise ValueError("start_probs must be non-negative")
    total = probs.sum()
    if total <= 0:
        raise ValueError("start_probs must sum to a positive value")
    return probs / total


def _build_transition(cfg: SeqMarkovConfig) -> Tensor:
    V = cfg.vocab_size
    if V <= 0:
        raise ValueError("vocab_size must be positive")
    if cfg.transition_matrix is not None:
        mat = torch.tensor(cfg.transition_matrix, dtype=torch.float)
        if mat.shape != (V, V):
            raise ValueError(f"transition_matrix must have shape ({V}, {V})")
        if (mat < 0).any():
            raise ValueError("transition_matrix entries must be non-negative")
        row_sums = mat.sum(dim=1, keepdim=True)
        if (row_sums <= 0).any():
            raise ValueError("each row of transition_matrix must sum to a positive value")
        return mat / row_sums

    if V == 1:
        return torch.ones((1, 1), dtype=torch.float)

    base = torch.full((V, V), cfg.offdiag_smoothing / max(1, V - 1), dtype=torch.float)
    base.fill_diagonal_(cfg.self_transition_bias + cfg.offdiag_smoothing)
    return base / base.sum(dim=1, keepdim=True)


def _sample_markov(
    cfg: SeqMarkovConfig,
    start: Tensor,
    trans: Tensor,
    g: torch.Generator,
) -> Tuple[Tensor, Tensor]:
    x = torch.empty((cfg.batch_size, cfg.seq_len), dtype=torch.long)
    start_tokens = torch.multinomial(start, cfg.batch_size, replacement=True, generator=g)
    x[:, 0] = start_tokens
    for t in range(1, cfg.seq_len):
        prev = x[:, t - 1]
        probs = trans[prev]  # [B, V]
        next_tokens = torch.multinomial(probs, 1, generator=g).squeeze(1)
        x[:, t] = next_tokens
    return x, start_tokens


class SeqMarkovIterable(IterableDataset):
    def __init__(self, cfg: SeqMarkovConfig):
        super().__init__()
        self.cfg = cfg
        self.transforms = to_compose(cfg.transforms)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        seed = (self.cfg.seed + 0x9E3779B97F4A7C15 + worker_id) % (2**63 - 1)
        g = _make_generator(seed)
        start = _build_start_probs(self.cfg)
        trans = _build_transition(self.cfg)

        while True:
            x, start_tokens = _sample_markov(self.cfg, start, trans, g)
            x, c = self.transforms(x, start_tokens if self.cfg.emit_start_token else None)
            out = {"x": x.long()}
            if c is not None:
                out["c"] = c.long()
            yield out


def make_dataset(**cfg: object) -> SeqMarkovIterable:  # type: ignore[override]
    return SeqMarkovIterable(SeqMarkovConfig(**cfg))


@register("dataset", "seq_markov")
def build_dataset(**cfg: object) -> SeqMarkovIterable:  # type: ignore
    return make_dataset(**cfg)


__all__ = ["SeqMarkovConfig", "SeqMarkovIterable", "make_dataset", "make_dataloader"]
