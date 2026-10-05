"""Noise schedule builders and utilities.

Schedules map the normalised diffusion time ``t ∈ [0, 1]`` to quantities such
as beta, alphā, or log-SNR that parameterise diffusion or flow-matching
processes.  This module provides a small collection of commonly used schedules
and exposes a registry-backed factory so experiments can request them by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch
from torch import Tensor

from core.registry import available as _registry_available
from core.registry import build as _registry_build
from core.registry import register
from core.types import ScheduleFn

# Small numerical helpers -------------------------------------------------------

_EPS = 1e-8
_ALPHA_EPS = 1e-6
_BETA_EPS = 1e-8


def _ensure_tensor(t: Tensor | float | Iterable[float]) -> Tensor:
    if isinstance(t, Tensor):
        return t
    return torch.as_tensor(t, dtype=torch.float32)


def _clamp_alpha_bar(alpha_bar: Tensor) -> Tensor:
    return torch.clamp(alpha_bar, min=_ALPHA_EPS, max=1.0 - _ALPHA_EPS)


# Noise schedule container ------------------------------------------------------


@dataclass
class NoiseSchedule:
    """Container exposing the standard schedule parameterisations."""

    name: str
    alpha_bar_fn: ScheduleFn
    beta_fn: Optional[ScheduleFn] = None
    log_snr_fn: Optional[ScheduleFn] = None

    def alpha_bar(self, timesteps: Tensor | float | Iterable[float]) -> Tensor:
        t = _ensure_tensor(timesteps)
        return _clamp_alpha_bar(self.alpha_bar_fn(t))

    def alpha(self, timesteps: Tensor | float | Iterable[float]) -> Tensor:
        return torch.sqrt(self.alpha_bar(timesteps))

    def sigma(self, timesteps: Tensor | float | Iterable[float]) -> Tensor:
        alpha_bar = self.alpha_bar(timesteps)
        return torch.sqrt(torch.clamp(1.0 - alpha_bar, min=_ALPHA_EPS))

    def log_snr(self, timesteps: Tensor | float | Iterable[float]) -> Tensor:
        if self.log_snr_fn is not None:
            return self.log_snr_fn(_ensure_tensor(timesteps))
        alpha_bar = self.alpha_bar(timesteps)
        alpha_bar = _clamp_alpha_bar(alpha_bar)
        return torch.log(alpha_bar) - torch.log1p(-alpha_bar)

    def beta(self, timesteps: Tensor | float | Iterable[float]) -> Tensor:
        if self.beta_fn is not None:
            return torch.clamp(self.beta_fn(_ensure_tensor(timesteps)), min=_BETA_EPS)
        # Fallback: approximate derivative of log alphā with autograd.
        t = _ensure_tensor(timesteps)
        needs_grad = t.requires_grad
        t_req = t if needs_grad else t.detach().requires_grad_(True)
        with torch.enable_grad():
            alpha_bar = _clamp_alpha_bar(self.alpha_bar_fn(t_req))
            log_alpha = torch.log(alpha_bar)
            grad = torch.autograd.grad(
                outputs=log_alpha,
                inputs=t_req,
                grad_outputs=torch.ones_like(log_alpha),
                create_graph=needs_grad,
                retain_graph=needs_grad,
            )[0]
        beta = -grad
        if not needs_grad:
            beta = beta.detach()
        return torch.clamp(beta, min=_BETA_EPS)


# Registry-backed helpers -------------------------------------------------------


def make_schedule(kind: str, /, **cfg: object) -> NoiseSchedule:
    """Instantiate a registered noise schedule."""
    schedule = _registry_build("schedule", kind, **cfg)
    if not isinstance(schedule, NoiseSchedule):
        raise TypeError(f"Schedule '{kind}' did not return NoiseSchedule, got {type(schedule).__name__}")
    return schedule


def list_schedules() -> list[str]:
    """Return the list of registered schedule names."""
    return _registry_available("schedule")


# Concrete schedules ------------------------------------------------------------


@register("schedule", "linear_beta")
def _linear_beta_schedule(beta_min: float = 1e-4, beta_max: float = 0.02) -> NoiseSchedule:
    beta_min = float(beta_min)
    beta_max = float(beta_max)
    if beta_min <= 0 or beta_max <= 0:
        raise ValueError("linear_beta requires beta_min and beta_max > 0")
    if beta_max < beta_min:
        raise ValueError("linear_beta expects beta_max >= beta_min")
    delta = beta_max - beta_min

    def beta_fn(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        return beta_min + delta * t

    def alpha_bar_fn(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        integral = beta_min * t + 0.5 * delta * t * t
        return torch.exp(-integral)

    return NoiseSchedule(
        name="linear_beta",
        alpha_bar_fn=alpha_bar_fn,
        beta_fn=beta_fn,
    )


@register("schedule", "cosine")
def _cosine_schedule(s: float = 0.008, max_beta: float = 0.999) -> NoiseSchedule:
    if s < 0:
        raise ValueError("cosine schedule requires s >= 0")
    s = float(s)
    max_beta = float(max_beta)
    base = math.cos(s / (1 + s) * math.pi / 2) ** 2
    theta_scale = math.pi / (2 * (1 + s))
    cap = math.pi / 2 - 1e-4

    def alpha_bar_fn(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        theta = torch.clamp((t + s) * theta_scale, max=cap)
        return torch.cos(theta) ** 2 / base

    def beta_fn(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        theta = torch.clamp((t + s) * theta_scale, max=cap)
        tan_theta = torch.tan(theta)
        beta = (math.pi / (1 + s)) * tan_theta
        return torch.clamp(beta, min=_BETA_EPS, max=max_beta)

    return NoiseSchedule(
        name="cosine",
        alpha_bar_fn=alpha_bar_fn,
        beta_fn=beta_fn,
    )


@register("schedule", "sigmoid")
def _sigmoid_schedule(
    start_logsnr: float = 15.0,
    end_logsnr: float = -15.0,
    temperature: float = 0.25,
    center: float = 0.5,
    max_beta: float = 25.0,
) -> NoiseSchedule:
    if temperature <= 0:
        raise ValueError("sigmoid schedule requires temperature > 0")
    start_logsnr = float(start_logsnr)
    end_logsnr = float(end_logsnr)
    temperature = float(temperature)
    center = float(center)
    span = end_logsnr - start_logsnr

    def _sigmoid_interp(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        return torch.sigmoid((t - center) / temperature)

    def logsnr_fn(t: Tensor) -> Tensor:
        s = _sigmoid_interp(t)
        return start_logsnr + span * s

    def alpha_bar_fn(t: Tensor) -> Tensor:
        return torch.sigmoid(logsnr_fn(t))

    def beta_fn(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        s = _sigmoid_interp(t)
        alpha_bar = torch.sigmoid(logsnr_fn(t))
        d_logsnr = span * s * (1 - s) / temperature
        beta = -(1.0 - alpha_bar) * d_logsnr
        return torch.clamp(beta, min=_BETA_EPS, max=max_beta)

    return NoiseSchedule(
        name="sigmoid",
        alpha_bar_fn=alpha_bar_fn,
        beta_fn=beta_fn,
        log_snr_fn=logsnr_fn,
    )


@register("schedule", "power")
def _power_schedule(
    exponent: float = 2.0,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
) -> NoiseSchedule:
    exponent = float(exponent)
    beta_min = float(beta_min)
    beta_max = float(beta_max)
    if exponent <= -1:
        raise ValueError("power schedule requires exponent > -1")
    if beta_min <= 0 or beta_max <= 0:
        raise ValueError("power schedule requires positive betas")
    if beta_max < beta_min:
        raise ValueError("power schedule expects beta_max >= beta_min")
    delta = beta_max - beta_min
    inv = 1.0 / (exponent + 1.0)

    def beta_fn(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        return beta_min + delta * torch.pow(t, exponent)

    def alpha_bar_fn(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        integral = beta_min * t + delta * torch.pow(t, exponent + 1.0) * inv
        return torch.exp(-integral)

    return NoiseSchedule(
        name="power",
        alpha_bar_fn=alpha_bar_fn,
        beta_fn=beta_fn,
    )


@register("schedule", "log_linear")
def _log_linear_schedule(
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
) -> NoiseSchedule:
    beta_min = float(beta_min)
    beta_max = float(beta_max)
    if beta_min <= 0 or beta_max <= 0:
        raise ValueError("log_linear schedule requires positive betas")
    ratio = beta_max / beta_min
    if ratio <= 0:
        raise ValueError("beta_max / beta_min must be > 0 for log_linear schedule")
    log_ratio = math.log(ratio)

    def beta_fn(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        if abs(log_ratio) < 1e-8:
            return torch.full_like(t, beta_min)
        return beta_min * torch.exp(log_ratio * t)

    def alpha_bar_fn(t: Tensor) -> Tensor:
        t = _ensure_tensor(t)
        if abs(log_ratio) < 1e-8:
            integral = beta_min * t
        else:
            integral = beta_min * (torch.exp(log_ratio * t) - 1.0) / log_ratio
        return torch.exp(-integral)

    return NoiseSchedule(
        name="log_linear",
        alpha_bar_fn=alpha_bar_fn,
        beta_fn=beta_fn,
    )


__all__ = [
    "NoiseSchedule",
    "list_schedules",
    "make_schedule",
]
