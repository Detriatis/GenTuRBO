"""Utilities for sampling diffusion timesteps with configurable bias."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from core.registry import available as _registry_available
from core.registry import build as _registry_build
from core.registry import register

TimeSampler = Callable[[int, torch.device], Tensor]

__all__ = ["TimeSampler", "make_time_sampler", "list_time_samplers"]


def make_time_sampler(name: str, /, **cfg: object) -> TimeSampler:
    """Instantiate a registered time sampler."""
    sampler = _registry_build("time_sampler", name, **cfg)
    if not callable(sampler):
        raise TypeError(f"Time sampler '{name}' is not callable (got {type(sampler).__name__})")
    return sampler  # type: ignore[return-value]


def list_time_samplers() -> list[str]:
    """Return the list of registered time sampler names."""
    return _registry_available("time_sampler")


@register("time_sampler", "uniform")
def _build_uniform_sampler() -> TimeSampler:
    """Uniform t ~ U(0, 1)."""

    def _sample(batch_size: int, device: torch.device) -> Tensor:
        return torch.rand(batch_size, device=device)

    return _sample


@register("time_sampler", "beta")
def _build_beta_sampler(alpha: float = 2.0, beta: float = 2.0) -> TimeSampler:
    """Beta(alpha, beta) sampler that can emphasise mid-range or edge timesteps."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("beta time sampler requires alpha,beta > 0")

    def _sample(batch_size: int, device: torch.device) -> Tensor:
        dist = torch.distributions.Beta(alpha, beta)
        samples = dist.sample((batch_size,))
        return samples.to(device)

    return _sample


@register("time_sampler", "power")
def _build_power_sampler(exponent: float = 1.0) -> TimeSampler:
    """Sample t = U(0,1)^(1/exponent). exponent > 1 favours small t, < 1 favours large t."""
    exponent = float(exponent)
    if exponent <= 0:
        raise ValueError("power time sampler requires exponent > 0")

    inv = 1.0 / exponent

    def _sample(batch_size: int, device: torch.device) -> Tensor:
        u = torch.rand(batch_size, device=device)
        return torch.pow(u, inv)

    return _sample
