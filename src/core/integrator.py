"""Integrator builders for deterministic ODEs and stochastic SDEs.

This module mirrors the lightweight registry helpers used throughout the core
package and exposes a small catalogue of fast solvers that can be selected by
name (e.g. ``make_integrator("ode_euler")``).  Integrators consume a drift
function ``f(x, t)`` together with an optional diffusion function ``g(x, t)``
and step the state between two times using either ODE (Heun / Euler) or
Euler–Maruyama updates for SDEs.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Protocol

import torch
from torch import Tensor

from core.registry import available as _registry_available
from core.registry import build as _registry_build
from core.registry import register
from core.types import GeneratorLike

DriftFn = Callable[[Tensor, Tensor], Tensor]
DiffusionFn = Callable[[Tensor, Tensor], Tensor]


class Integrator(Protocol):
    """Callable signature shared by all integrators."""

    def __call__(
        self,
        drift_fn: DriftFn,
        x0: Tensor,
        /,
        *,
        t0: float = 0.0,
        t1: float = 1.0,
        steps: Optional[int] = None,
        diffusion_fn: Optional[DiffusionFn] = None,
        generator: Optional[GeneratorLike] = None,
    ) -> Tensor:
        ...


def make_integrator(kind: str, /, **cfg: object) -> Integrator:
    """Instantiate a registered integrator."""
    integrator = _registry_build("integrator", kind, **cfg)
    if not callable(integrator):
        raise TypeError(f"Integrator '{kind}' is not callable (got {type(integrator).__name__})")
    return integrator  # type: ignore[return-value]


def list_integrators() -> list[str]:
    """Return the available integrator names."""
    return _registry_available("integrator")


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #


def _time_like(reference: Tensor, value: float) -> Tensor:
    return reference.new_tensor(float(value))


def _resolve_steps(default_steps: int, override: Optional[int]) -> int:
    steps = default_steps if override is None else int(override)
    if steps <= 0:
        raise ValueError("Integrator requires steps > 0")
    return steps


def _resolve_generator(gen_like: Optional[GeneratorLike], device: torch.device) -> Optional[torch.Generator]:
    if gen_like is None:
        return None
    if isinstance(gen_like, torch.Generator):
        gen = gen_like
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(gen_like))
    gen_device = getattr(gen, "device", None)
    if gen_device is not None and gen_device != device:
        raise ValueError(f"Generator device ({gen_device}) does not match tensor device ({device})")
    return gen


# --------------------------------------------------------------------------- #
# Deterministic integrators                                                   #
# --------------------------------------------------------------------------- #


@register("integrator", "ode_euler")
def _build_ode_euler(default_steps: int = 100) -> Integrator:
    """Explicit Euler integrator for fast, coarse ODE solves."""

    def integrate(
        drift_fn: DriftFn,
        x0: Tensor,
        /,
        *,
        t0: float = 0.0,
        t1: float = 1.0,
        steps: Optional[int] = None,
        **unused_kwargs: object,
    ) -> Tensor:
        n_steps = _resolve_steps(default_steps, steps)
        dt = (t1 - t0) / n_steps
        x = x0
        t = float(t0)
        for _ in range(n_steps):
            t_tensor = _time_like(x, t)
            drift = drift_fn(x, t_tensor)
            x = x + dt * drift
            t += dt
        return x

    return integrate


@register("integrator", "ode_heun")
def _build_ode_heun(default_steps: int = 100) -> Integrator:
    """Two-stage Heun (RK2) integrator for improved stability."""

    def integrate(
        drift_fn: DriftFn,
        x0: Tensor,
        /,
        *,
        t0: float = 0.0,
        t1: float = 1.0,
        steps: Optional[int] = None,
        **unused_kwargs: object,
    ) -> Tensor:
        n_steps = _resolve_steps(default_steps, steps)
        dt = (t1 - t0) / n_steps
        x = x0
        t = float(t0)
        for _ in range(n_steps):
            t_tensor = _time_like(x, t)
            drift = drift_fn(x, t_tensor)
            x_pred = x + dt * drift
            t_next = t + dt
            t_next_tensor = _time_like(x, t_next)
            drift_next = drift_fn(x_pred, t_next_tensor)
            x = x + 0.5 * dt * (drift + drift_next)
            t = t_next
        return x

    return integrate


# --------------------------------------------------------------------------- #
# Stochastic integrator                                                       #
# --------------------------------------------------------------------------- #


@register("integrator", "sde_euler_maruyama")
def _build_sde_euler_maruyama(default_steps: int = 1000, noise_scale: float = 1.0) -> Integrator:
    """Euler-Maruyama integrator for additive or multiplicative noise SDEs."""

    noise_scale = float(noise_scale)

    def integrate(
        drift_fn: DriftFn,
        x0: Tensor,
        /,
        *,
        t0: float = 0.0,
        t1: float = 1.0,
        steps: Optional[int] = None,
        diffusion_fn: Optional[DiffusionFn] = None,
        generator: Optional[GeneratorLike] = None,
    ) -> Tensor:
        if diffusion_fn is None:
            raise ValueError("sde_euler_maruyama requires a diffusion_fn callable")
        n_steps = _resolve_steps(default_steps, steps)
        dt = (t1 - t0) / n_steps
        sqrt_dt = math.sqrt(abs(dt))
        x = x0
        t = float(t0)
        gen = _resolve_generator(generator, x.device)
        for _ in range(n_steps):
            t_tensor = _time_like(x, t)
            drift = drift_fn(x, t_tensor)
            diffusion = diffusion_fn(x, t_tensor)
            if gen is None:
                noise = torch.randn_like(x)
            else:
                noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
            x = x + dt * drift + noise_scale * sqrt_dt * diffusion * noise
            t += dt
        return x

    return integrate


# --------------------------------------------------------------------------- #
# Discrete CTMC integrators (categorical Euler)                               #
# --------------------------------------------------------------------------- #

from typing import Tuple

RateRowFn = Callable[[Tensor, Tensor], Tensor]
"""
Given (x, t_tensor) returns off-diagonal rates R_row with shape:
  - Single site:          [S]
  - Batch single site:    [B, S]
  - Sequences:            [B, D, S]
Conventions:
  * R_row[..., i] == 0 for the current state i (off-diagonal only).
  * All entries are >= 0 (rates).
  * We'll set the diagonal as R_ii = -sum_{j!=i} R(i->j).
"""

def _as_time_tensor(x: Tensor, t: float) -> Tensor:
    return x.new_tensor(float(t))


def _sample_categorical(probs: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    """
    probs: [..., S], sums to 1 along last dim.
    Returns indices with same leading shape.
    """
    shape = probs.shape[:-1]
    S = probs.shape[-1]
    flat = probs.reshape(-1, S)
    idx = torch.multinomial(flat, num_samples=1, replacement=True, generator=generator)  # [N,1]
    return idx.reshape(*shape)


def _build_euler_probs_from_rates(x: Tensor, R_row: Tensor, dt: float) -> Tensor:
    """
    x: [...], int64 indices
    R_row: [..., S] off-diagonal rates for current state x (R_row[..., x]=0)
    Returns probs: [..., S] for Euler step:
        P(next=j) = dt * R(i->j) for j != i
        P(next=i) = 1 + dt * R_ii  with R_ii = -sum_{j!=i} R(i->j)
    Clamps negatives and renormalizes defensively.
    """
    S = R_row.shape[-1]
    # sum over off-diagonal
    outflow = R_row.sum(dim=-1, keepdim=True)  # [..., 1]
    p = dt * R_row  # off-diagonal part

    # add diagonal mass
    stay = 1.0 - dt * outflow  # [...,1]
    # scatter 'stay' into the index positions
    p = p.clone()
    index = x.unsqueeze(-1)  # [...,1]
    p.scatter_add_(-1, index, stay)

    # numerical guards: clip then renorm to simplex
    p = p.clamp_min_(0.0)
    z = p.sum(dim=-1, keepdim=True)
    # avoid division by zero: if zero, enforce stay with prob 1
    bad = (z <= 0)
    if bad.any():
        p[bad.expand_as(p)] = 0.0
        p.scatter_(-1, index, torch.where(bad, torch.ones_like(stay), torch.zeros_like(stay)))
        z = p.sum(dim=-1, keepdim=True)
    p = p / z
    return p


@register("integrator", "ctmc_factorized")
def _build_ctmc_factorized(default_steps: int = 100, dt_clip: float = 0.5) -> Integrator:
    """
    First-order Euler integrator for discrete CTMCs (factorized across sites).
    - rate_row_fn(x, t_tensor) -> R_row with shape matching x and last dim S.
    - Updates each site independently in parallel (fast; first-order accurate).
    - dt_clip caps dt * max(outflow) heuristically to keep probs stable.

    Example:
        >>> def denoiser_p_theta(x: Tensor, t_tensor: Tensor) -> Tensor:
        ...     return model_posterior(x, t_tensor)  # normalized over last dim
        >>> S = vocab_size
        >>> M = mask_token_index
        >>> rate_row_fn = make_mask_rate_row_fn(M, denoiser_p_theta)   # or make_uniform_rate_row_fn(...)
        >>> integrate = make_integrator("ctmc_factorized", default_steps=200)  # or "ctmc_single_jump"
        >>> x1 = integrate(rate_row_fn, x0, t0=0.0, t1=0.98, steps=200, generator=42)
    """
    def integrate(
        rate_row_fn: RateRowFn,  # NOTE: different signature vs ODE
        x0: Tensor, /, *,
        t0: float = 0.0,
        t1: float = 1.0,
        steps: Optional[int] = None,
        generator: Optional[GeneratorLike] = None,
        **unused_kwargs: object,
    ) -> Tensor:
        n_steps = _resolve_steps(default_steps, steps)
        dt = (t1 - t0) / n_steps
        x = x0
        t = float(t0)
        gen = _resolve_generator(generator, x.device)

        for _ in range(n_steps):
            t_tensor = _as_time_tensor(x, t)
            R_row = rate_row_fn(x, t_tensor)  # [..., S], off-diagonal
            # Optional conservative dt guard (keeps Euler probs well-behaved):
            with torch.no_grad():
                outflow = R_row.sum(dim=-1, keepdim=True)  # [...,1]
                max_out = outflow.max().item()
                if max_out > 0:
                    # Limit dt so that 1 - dt*outflow >= (1 - dt_clip) approximately
                    # (heuristic; dt_clip in [0,1], default 0.5).
                    safe_dt = min(dt, dt_clip / max_out)
                else:
                    safe_dt = dt
            probs = _build_euler_probs_from_rates(x, R_row, safe_dt)  # [..., S]
            x = _sample_categorical(probs, gen)
            t += dt
        return x

    return integrate


@register("integrator", "ctmc_single_jump")
def _build_ctmc_single_jump(default_steps: int = 100) -> Integrator:
    """
    Single-jump Euler integrator: per global step, at most one site changes.
    Good match to 'exact' CTMC semantics when you want one event per step.
    """
    def integrate(
        rate_row_fn: RateRowFn,
        x0: Tensor, /, *,
        t0: float = 0.0,
        t1: float = 1.0,
        steps: Optional[int] = None,
        generator: Optional[GeneratorLike] = None,
        **unused_kwargs: object,
    ) -> Tensor:
        n_steps = _resolve_steps(default_steps, steps)
        dt = (t1 - t0) / n_steps
        x = x0
        t = float(t0)
        gen = _resolve_generator(generator, x.device)

        for _ in range(n_steps):
            t_tensor = _as_time_tensor(x, t)
            R_row = rate_row_fn(x, t_tensor)  # [..., S]

            # Off-diagonal jump masses per site: dt * sum_j R(i->j)
            lambda_site = dt * R_row.sum(dim=-1)  # [...]

            # Decide whether *any* jump happens:
            # p(no jump) = clamp(1 - sum(lambda_site), 0, 1)
            total_lambda = lambda_site.flatten(0).sum()
            p_jump = total_lambda.clamp(min=0.0, max=1.0).item()
            rand_val = torch.rand((), device=x.device, generator=gen).item()
            if rand_val >= p_jump or total_lambda.item() <= 0:
                t += dt
                continue

            # Choose which site jumps, proportional to lambda_site
            site_probs = (lambda_site / (total_lambda + 1e-12)).flatten(0)
            site_idx_flat = torch.multinomial(site_probs[None, :], 1, generator=gen)[0]
            # unravel to index in x
            unraveled = torch.unravel_index(site_idx_flat, lambda_site.shape)
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[unraveled] = True

            # Build per-site transition distribution for the chosen site
            R_row_sel = R_row[mask]                   # [1, S]
            i_sel = x[mask]                           # [1]
            probs_sel = dt * R_row_sel                # off-diagonal masses
            # put the diagonal stay prob so row sums to 1
            stay = 1.0 - probs_sel.sum(dim=-1, keepdim=True)  # [1,1]
            probs_sel = probs_sel.scatter_add(-1, i_sel.view(1, 1), stay)
            probs_sel = probs_sel.clamp_min(0.0)
            probs_sel = probs_sel / probs_sel.sum(dim=-1, keepdim=True)

            j_sel = _sample_categorical(probs_sel, gen).squeeze(0)  # []

            # Apply the jump only at the chosen site
            x = x.clone()
            x[mask] = j_sel
            t += dt
        return x

    return integrate


# --------------------------------------------------------------------------- #
# Convenience: unconditional rate builders for MASK/UNIFORM interpolants      #
# --------------------------------------------------------------------------- #

DenoiserFn = Callable[[Tensor, Tensor], Tensor]
"""
Given (x, t_tensor) returns p_theta(x1 | x_t) as probabilities:
  - Single site:          [S]
  - Batch single site:    [B, S]
  - Sequences:            [B, D, S]
"""

def make_mask_rate_row_fn(M: int, denoiser: DenoiserFn) -> RateRowFn:
    """
    R(i->j) = 1/(1 - t) * 1{i=M} * p_theta(j | x_t)
    (off-diagonal; we'll zero the diagonal implicitly by construction)
    """
    def rate_row(x: Tensor, t_tensor: Tensor) -> Tensor:
        p_theta = denoiser(x, t_tensor)  # [..., S], sums to 1
        R = p_theta / (1.0 - t_tensor)   # broadcast along last dim
        # zero rows where i != M
        mask = (x != M).unsqueeze(-1)
        R = torch.where(mask, torch.zeros_like(R), R)
        # zero the diagonal (off-diagonal only)
        eye_idx = x.unsqueeze(-1)
        R = R.scatter(-1, eye_idx, torch.zeros_like(R[..., :1]))
        return R
    return rate_row


def make_uniform_rate_row_fn(denoiser: DenoiserFn) -> RateRowFn:
    """
    R(i->j) = 1/(1 - t) * 1{j != i} * p_theta(j | x_t)
    """
    def rate_row(x: Tensor, t_tensor: Tensor) -> Tensor:
        p_theta = denoiser(x, t_tensor)  # [..., S]
        R = p_theta / (1.0 - t_tensor)
        # zero diagonal
        eye_idx = x.unsqueeze(-1)
        R = R.scatter(-1, eye_idx, torch.zeros_like(R[..., :1]))
        return R
    return rate_row


__all__ = [
    "DiffusionFn",
    "DriftFn",
    "Integrator",
    "list_integrators",
    "make_integrator",
]
