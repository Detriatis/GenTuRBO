"""Loss utilities for diffusion and flow-matching experiments.

This module exposes a small registry-backed factory for commonly used
objectives so experiments can request them by name (e.g. ``build('loss',
'diffusion_simple')``).  The provided losses cover both discrete and continuous
flow matching as well as the standard diffusion noise/velocity prediction
objectives.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from core.registry import available as _registry_available
from core.registry import build as _registry_build
from core.registry import register
from core.types import Batch

LossFn = Callable[[Tensor, Batch], Tensor]
Reduction = str


def make_loss(name: str, /, **cfg: object) -> LossFn:
    """Instantiate a registered loss."""
    loss = _registry_build("loss", name, **cfg)
    if not callable(loss):
        raise TypeError(f"Loss '{name}' is not callable (got {type(loss).__name__})")
    return loss  # type: ignore[return-value]


def list_losses() -> list[str]:
    """Return the list of registered loss names."""
    return _registry_available("loss")


# Helper utilities -------------------------------------------------------------


def _resolve_tensor(batch: Batch, keys: Sequence[str]) -> Tensor:
    for key in keys:
        if key in batch:
            return batch[key]
    joined = ", ".join(keys)
    raise KeyError(f"Batch does not contain any of the required keys: {joined}")


def _optional_tensor(batch: Batch, key: Optional[str]) -> Optional[Tensor]:
    if key is None:
        return None
    value = batch.get(key)
    if value is None:
        raise KeyError(f"Batch does not contain optional key '{key}'")
    return value


def _apply_reduction(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unsupported reduction '{reduction}' (expected 'mean', 'sum', or 'none')")


def _weighted(values: Tensor, weight: Optional[Tensor]) -> Tensor:
    if weight is None:
        return values
    return values * weight


def _mse(prediction: Tensor, target: Tensor, *, weight: Optional[Tensor], reduction: Reduction) -> Tensor:
    losses = (prediction - target) ** 2
    losses = _weighted(losses, weight)
    return _apply_reduction(losses, reduction)


def _cross_entropy(
    logits: Tensor,
    target: Tensor,
    *,
    reduction: Reduction,
    weight: Optional[Tensor],
) -> Tensor:
    if target.shape == logits.shape:
        log_probs = F.log_softmax(logits, dim=-1)
        losses = torch.sum(-target * log_probs, dim=-1)
    else:
        losses = F.cross_entropy(logits, target.long(), reduction="none")
    losses = _weighted(losses, weight)
    return _apply_reduction(losses, reduction)


# Registered losses ------------------------------------------------------------


@register("loss", "diffusion_simple")
def _build_diffusion_simple_loss(
    *,
    target_keys: Iterable[str] = ("target_noise", "eps", "noise"),
    weight_key: Optional[str] = None,
    reduction: Reduction = "mean",
) -> LossFn:
    target_keys = tuple(target_keys)

    def loss_fn(prediction: Tensor, batch: Batch) -> Tensor:
        target = _resolve_tensor(batch, target_keys)
        weight = _optional_tensor(batch, weight_key)
        return _mse(prediction, target, weight=weight, reduction=reduction)

    return loss_fn


@register("loss", "diffusion_velocity")
def _build_diffusion_velocity_loss(
    *,
    x_key: str = "x0",
    eps_key: str = "eps",
    alpha_key: str = "alpha",
    sigma_key: str = "sigma",
    weight_key: Optional[str] = None,
    reduction: Reduction = "mean",
) -> LossFn:
    def loss_fn(prediction: Tensor, batch: Batch) -> Tensor:
        x0 = _resolve_tensor(batch, (x_key,))
        eps = _resolve_tensor(batch, (eps_key,))
        alpha = _resolve_tensor(batch, (alpha_key,))
        sigma = _resolve_tensor(batch, (sigma_key,))
        target = alpha * eps - sigma * x0
        weight = _optional_tensor(batch, weight_key)
        return _mse(prediction, target, weight=weight, reduction=reduction)

    return loss_fn


@register("loss", "flow_matching_continuous")
def _build_flow_matching_continuous_loss(
    *,
    velocity_keys: Iterable[str] = ("target_velocity", "v_target"),
    endpoints: tuple[str, str] = ("x0", "x1"),
    weight_key: Optional[str] = None,
    reduction: Reduction = "mean",
) -> LossFn:
    velocity_keys = tuple(velocity_keys)

    def _target(batch: Batch) -> Tensor:
        try:
            return _resolve_tensor(batch, velocity_keys)
        except KeyError:
            x0_key, x1_key = endpoints
            x0 = _resolve_tensor(batch, (x0_key,))
            x1 = _resolve_tensor(batch, (x1_key,))
            return x1 - x0

    def loss_fn(prediction: Tensor, batch: Batch) -> Tensor:
        target = _target(batch)
        weight = _optional_tensor(batch, weight_key)
        return _mse(prediction, target, weight=weight, reduction=reduction)

    return loss_fn


@register("loss", "flow_matching_discrete")
def _build_flow_matching_discrete_loss(
    *,
    target_keys: Iterable[str] = ("target_logits", "target_probs", "target_class"),
    weight_key: Optional[str] = None,
    reduction: Reduction = "mean",
) -> LossFn:
    target_keys = tuple(target_keys)

    def loss_fn(logits: Tensor, batch: Batch) -> Tensor:
        target = _resolve_tensor(batch, target_keys)
        weight = _optional_tensor(batch, weight_key)
        return _cross_entropy(logits, target, reduction=reduction, weight=weight)

    return loss_fn


__all__ = ["LossFn", "list_losses", "make_loss"]
