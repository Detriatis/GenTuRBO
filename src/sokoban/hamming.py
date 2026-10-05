"""Uniform exact-distance proposals over the seven public Sokoban tiles."""

from __future__ import annotations

import torch


def mutate_exact(parents: torch.Tensor, radius: int, *, generator: torch.Generator) -> torch.Tensor:
    """Change exactly radius distinct raw cells; do not repair or canonicalise.

    Each cell subset is uniform, and each replacement is uniform among the six
    public tile values other than its parent's value. Token 7 is the model mask
    and is never a candidate tile. Invalid boards remain in the proposal pool.
    """
    if parents.device.type != "cpu" or parents.dtype != torch.long or parents.ndim != 2 or parents.shape[1] != 100:
        raise ValueError("expected CPU long parents of shape [B,100]")
    if not isinstance(radius, int) or isinstance(radius, bool) or not 1 <= radius <= 100:
        raise ValueError("radius must be an integer in [1,100]")
    if torch.any((parents < 0) | (parents >= 7)):
        raise ValueError("parents must use the public tile alphabet 0..6")
    children = parents.clone()
    for parent, child in zip(parents, children, strict=True):
        positions = torch.randperm(100, generator=generator)[:radius]
        values = torch.randint(6, (radius,), generator=generator)
        child[positions] = values + (values >= parent[positions]).long()
    return children
