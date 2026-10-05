"""Sampling utilities for Ehrlich Markov chains."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def sample_markov_sequences(
    transition_probs: Tensor,
    seq_len: int,
    batch_size: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Sample sequences from a discrete Markov process with transition matrix A.

    Args:
        transition_probs: Tensor [V, V] with row-stochastic transitions.
        seq_len: length of each sequence.
        batch_size: number of sequences.
    """
    if transition_probs.ndim != 2 or transition_probs.shape[0] != transition_probs.shape[1]:
        raise ValueError("transition_probs must have shape [V, V]")
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    V = transition_probs.shape[0]
    x = torch.empty((batch_size, seq_len), dtype=torch.long)
    x[:, 0] = torch.randint(0, V, (batch_size,), generator=generator)
    for t in range(1, seq_len):
        prev = x[:, t - 1]
        probs = transition_probs[prev]
        next_tokens = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        x[:, t] = next_tokens
    return x


__all__ = ["sample_markov_sequences"]
