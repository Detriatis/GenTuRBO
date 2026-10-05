"""Scoring utilities for Ehrlich benchmark instances."""

from __future__ import annotations

import torch
from torch import Tensor

from .ehrlich_instance import EhrlichInstance


def _require_integer_tensor(x: Tensor, *, name: str) -> None:
    try:
        torch.iinfo(x.dtype)
    except TypeError as exc:
        raise TypeError(f"{name} must have an integer dtype, got {x.dtype}") from exc


def _validate_transition_mask(transition_mask: Tensor, *, vocab_size: int | None = None) -> int:
    if transition_mask.ndim != 2 or transition_mask.shape[0] != transition_mask.shape[1]:
        raise ValueError("transition_mask must have shape [V, V]")
    if transition_mask.dtype != torch.bool:
        raise TypeError("transition_mask must have dtype torch.bool")
    size = int(transition_mask.shape[0])
    if vocab_size is not None and size != int(vocab_size):
        raise ValueError(
            f"transition_mask has vocabulary size {size}, expected {int(vocab_size)}"
        )
    return size


def _validate_sequences(
    x: Tensor,
    *,
    vocab_size: int,
    expected_seq_len: int | None,
    name: str,
) -> None:
    if x.ndim not in {1, 2}:
        raise ValueError(f"{name} expects [B, L] or [L], got {tuple(x.shape)}")
    _require_integer_tensor(x, name=name)
    seq_len = int(x.shape[-1])
    if expected_seq_len is not None and seq_len != int(expected_seq_len):
        raise ValueError(
            f"{name} expected sequence length {int(expected_seq_len)}, got {seq_len}"
        )
    if seq_len < 1:
        raise ValueError(f"{name} requires a non-empty sequence")
    if x.numel() == 0:
        return
    token_min = int(x.min().item())
    token_max = int(x.max().item())
    if token_min < 0 or token_max >= int(vocab_size):
        raise ValueError(
            f"{name} token ids must be in [0, {int(vocab_size) - 1}], "
            f"got range [{token_min}, {token_max}]"
        )


def is_feasible(
    x: Tensor,
    transition_mask: Tensor,
    *,
    expected_seq_len: int | None = None,
) -> Tensor | bool:
    """Check Markov feasibility for a batch or single sequence."""
    vocab_size = _validate_transition_mask(transition_mask)
    _validate_sequences(
        x,
        vocab_size=vocab_size,
        expected_seq_len=expected_seq_len,
        name="is_feasible",
    )
    if x.ndim == 1:
        x_batch = x.view(1, -1)
        feasible = _is_feasible_batch(x_batch, transition_mask)[0].item()
        return bool(feasible)
    return _is_feasible_batch(x, transition_mask)


def _is_feasible_batch(x: Tensor, transition_mask: Tensor) -> Tensor:
    x_long = x.to(dtype=torch.long)
    prev = x_long[:, :-1]
    nxt = x_long[:, 1:]
    allowed = transition_mask.to(device=x.device)[prev, nxt]
    return allowed.all(dim=1)


def hq_batch(
    x: Tensor,
    motif: Tensor,
    spacing: Tensor,
    q: int,
    *,
    indices: Tensor | None = None,
) -> Tensor:
    """
    Quantized motif satisfaction for a batch of sequences.
    """
    if x.ndim == 1:
        x = x.view(1, -1)
    if x.ndim != 2:
        raise ValueError(f"hq_batch expects [B, L] or [L], got {tuple(x.shape)}")
    _require_integer_tensor(x, name="hq_batch input")
    B, L = x.shape
    k = int(motif.numel())
    if k == 0:
        raise ValueError("motif must be non-empty")
    _require_integer_tensor(motif, name="motif")
    _require_integer_tensor(spacing, name="spacing")
    if motif.ndim != 1 or spacing.ndim != 1 or int(spacing.numel()) != k:
        raise ValueError("motif and spacing must be 1D tensors with equal length")
    if q <= 0:
        raise ValueError("q must be positive")
    if q > k or k % int(q) != 0:
        raise ValueError("q must be <= motif length and evenly divide it")

    if indices is None:
        max_offset = int(spacing.max().item())
        num_starts = L - max_offset
        if num_starts <= 0:
            raise ValueError("spacing exceeds sequence length")
        starts = torch.arange(num_starts, device=x.device).view(-1, 1)
        indices = starts + spacing.to(x.device).view(1, -1)

    gathered = x[:, indices]  # [B, num_starts, k]
    motif_dev = motif.to(x.device).view(1, 1, -1)
    matches = (gathered == motif_dev).sum(dim=-1)
    bucket = k // int(q)
    hq_all = (matches // bucket).float() / float(q)
    return hq_all.max(dim=1).values


def ehrlich_score_batch(x: Tensor, instance: EhrlichInstance) -> Tensor:
    """Compute Ehrlich scores for a batch of sequences."""
    instance.cfg.validate()
    vocab_size = int(instance.cfg.vocab_size)
    _validate_transition_mask(instance.transition_mask, vocab_size=vocab_size)
    _validate_sequences(
        x,
        vocab_size=vocab_size,
        expected_seq_len=int(instance.cfg.seq_len),
        name="ehrlich_score_batch",
    )
    if x.ndim == 1:
        x = x.view(1, -1)
    device = x.device
    transition_mask = instance.transition_mask.to(device)
    feasible = _is_feasible_batch(x, transition_mask)
    indices = instance.indices_for_device(device)
    q = int(instance.cfg.quantization)
    scores = torch.ones((x.shape[0],), dtype=torch.float, device=device)
    for motif, spacing, idx in zip(instance.motifs, instance.spacings, indices, strict=True):
        hq = hq_batch(x, motif, spacing, q, indices=idx)
        scores = scores * hq
    scores = torch.where(feasible, scores, torch.tensor(float("-inf"), device=device))
    return scores


__all__ = ["ehrlich_score_batch", "hq_batch", "is_feasible"]
