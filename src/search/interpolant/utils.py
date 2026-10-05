"""Shared helper utilities for interpolant CTMC search."""

from __future__ import annotations

import math
from typing import Optional

import torch


def sample_truncated_normal(
    num_samples: int,
    mean: float,
    std: float,
    *,
    low: float,
    high: float,
    device: torch.device,
) -> torch.Tensor:
    if num_samples <= 0:
        return torch.empty((0,), device=device, dtype=torch.float)
    lo = float(low)
    hi = float(high)
    if lo > hi:
        lo, hi = hi, lo
    if std <= 0:
        clipped = min(max(float(mean), lo), hi)
        return torch.full((num_samples,), clipped, device=device, dtype=torch.float)
    mu = float(mean)
    sigma = float(std)
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    alpha = torch.tensor(a, device=device)
    beta = torch.tensor(b, device=device)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    cdf_a = 0.5 * (1.0 + torch.erf(alpha * inv_sqrt2))
    cdf_b = 0.5 * (1.0 + torch.erf(beta * inv_sqrt2))
    diff = cdf_b - cdf_a
    if diff <= 0:
        clipped = min(max(mu, lo), hi)
        return torch.full((num_samples,), clipped, device=device, dtype=torch.float)
    u = torch.rand((num_samples,), device=device) * diff + cdf_a
    eps = 1e-6
    u = u.clamp(min=eps, max=1.0 - eps)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    samples = z * sigma + mu
    samples = samples.clamp(min=lo, max=hi).to(dtype=torch.float)
    return samples


def stochastic_round_time(t_samples: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    if t_samples.ndim != 1:
        t_samples = t_samples.view(-1)
    t_min = float(t_grid[0].item())
    t_max = float(t_grid[-1].item())
    t = t_samples
    if (t < t_min).any() or (t > t_max).any():
        raise ValueError("t_samples must lie within [t_min, t_max] for stochastic rounding")
    max_mask = t >= t_max
    idx_right = torch.searchsorted(t_grid, t, right=True) - 1
    idx_right = idx_right.clamp(min=0, max=t_grid.numel() - 2)
    t_left = t_grid[idx_right]
    t_right = t_grid[idx_right + 1]
    denom = (t_right - t_left).clamp_min(1e-12)
    alpha = ((t - t_left) / denom).clamp(0.0, 1.0)
    choose_right = torch.rand_like(alpha) < alpha
    idx = torch.where(choose_right, idx_right + 1, idx_right)
    idx = torch.where(max_mask, torch.full_like(idx, t_grid.numel() - 1), idx)
    return idx


def step_to_time(step_idx: float, *, t_grid_norm: Optional[torch.Tensor], total_steps: int) -> float:
    step = min(max(float(step_idx), 0.0), float(max(0, total_steps)))
    if t_grid_norm is None:
        return step / float(max(1, total_steps))
    lo = int(math.floor(step))
    hi = min(int(total_steps), lo + 1)
    if hi == lo:
        return float(t_grid_norm[lo].item())
    alpha = step - float(lo)
    t_lo = float(t_grid_norm[lo].item())
    t_hi = float(t_grid_norm[hi].item())
    return (1.0 - alpha) * t_lo + alpha * t_hi


def edits_to_time(edits: float, *, seq_len: int) -> float:
    l = float(max(1, seq_len))
    e = min(max(float(edits), 0.0), l)
    return 1.0 - (e / l)


def time_to_edits(t_value: float, *, seq_len: int) -> float:
    t = min(max(float(t_value), 0.0), 1.0)
    return (1.0 - t) * float(max(1, seq_len))


def time_to_step(t_value: float, *, t_grid_norm: Optional[torch.Tensor], total_steps: int) -> float:
    t = min(max(float(t_value), 0.0), 1.0)
    if t_grid_norm is None:
        return t * float(max(1, total_steps))
    if t <= float(t_grid_norm[0].item()):
        return 0.0
    if t >= float(t_grid_norm[int(total_steps)].item()):
        return float(total_steps)
    t_tensor = torch.tensor(t, device=t_grid_norm.device)
    idx_hi = int(torch.searchsorted(t_grid_norm, t_tensor, right=False).item())
    idx_hi = max(1, min(idx_hi, int(total_steps)))
    idx_lo = idx_hi - 1
    t_lo = float(t_grid_norm[idx_lo].item())
    t_hi = float(t_grid_norm[idx_hi].item())
    denom = max(t_hi - t_lo, 1e-12)
    alpha = (t - t_lo) / denom
    alpha = min(max(alpha, 0.0), 1.0)
    return float(idx_lo) + alpha


def nearest_time_index(t_value: float, t_grid_norm: torch.Tensor) -> int:
    t = torch.tensor(float(t_value), device=t_grid_norm.device)
    idx = torch.argmin(torch.abs(t_grid_norm - t)).item()
    return int(idx)


def categorical_interpolant_sample(x1: torch.Tensor, t: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    if x1.ndim != 2:
        raise ValueError("categorical_interpolant_sample expects x1 with shape [batch, seq_len]")
    if t.ndim != 1 or t.shape[0] != x1.shape[0]:
        raise ValueError("t must have shape [batch] to match x1")
    if probs.ndim != 2 or probs.shape[0] != x1.shape[1]:
        raise ValueError("probs must have shape [seq_len, vocab]")
    if not torch.is_floating_point(t):
        t = t.float()
    probs = probs.to(x1.device)
    noise = torch.multinomial(probs, num_samples=x1.shape[0], replacement=True).T
    keep = torch.bernoulli(t[:, None].expand_as(x1)).bool()
    xt = torch.where(keep, x1, noise)
    return xt


def update_elite_pool(
    elite_pool: list[torch.Tensor],
    elite_scores: list[float],
    batch: torch.Tensor,
    rewards: torch.Tensor,
    *,
    k: int,
) -> tuple[list[torch.Tensor], list[float]]:
    merged: dict[bytes, tuple[float, torch.Tensor]] = {}
    for seq, score in zip(elite_pool, elite_scores):
        merged[seq.contiguous().numpy().tobytes()] = (float(score), seq)
    batch_cpu = batch.detach().cpu()
    rewards_cpu = rewards.detach().cpu()
    for seq, score in zip(batch_cpu, rewards_cpu):
        s = float(score.item())
        key = seq.contiguous().numpy().tobytes()
        if key not in merged or s > merged[key][0]:
            merged[key] = (s, seq.clone())
    ranked = sorted(merged.values(), key=lambda x: x[0], reverse=True)
    top = ranked[:k]
    new_pool = [seq for _, seq in top]
    new_scores = [score for score, _ in top]
    return new_pool, new_scores


def elite_position_probs(
    elite_pool: list[torch.Tensor],
    vocab_size: int,
    alpha: float,
    device: torch.device,
    *,
    weights: Optional[list[float]] = None,
    mix: float = 1.0,
) -> torch.Tensor:
    pool = torch.stack([seq.to(device) for seq in elite_pool], dim=0)
    one_hot = torch.nn.functional.one_hot(pool, num_classes=vocab_size).float()
    if weights is None:
        counts = one_hot.sum(dim=0)
    else:
        w = torch.tensor(weights, device=device, dtype=torch.float)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        if w.numel() != one_hot.shape[0]:
            raise ValueError("weights must match elite_pool length")
        w_min = float(w.min().item()) if w.numel() else 0.0
        if w_min < 0.0:
            w = w - w_min
        if float(w.sum().item()) <= 0.0:
            w = torch.ones_like(w)
        w = w / w.sum()
        counts = (one_hot * w[:, None, None]).sum(dim=0)
    counts = counts + float(alpha)
    denom = counts.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    probs = counts / denom
    if mix < 1.0:
        if mix < 0.0 or mix > 1.0:
            raise ValueError("elite_prior_mix must be in [0, 1]")
        uniform = torch.full_like(probs, 1.0 / float(vocab_size))
        probs = mix * probs + (1.0 - mix) * uniform
    return probs


def select_topk_unique(
    rewards: torch.Tensor,
    batch: torch.Tensor,
    k: int,
    *,
    prefer_recent: bool = False,
) -> list[int]:
    rewards_cpu = torch.nan_to_num(rewards.detach().cpu(), nan=float("-inf"))
    batch_cpu = batch.detach().cpu()
    if prefer_recent:
        order = sorted(
            range(int(rewards_cpu.numel())),
            key=lambda idx: (float(rewards_cpu[idx].item()), int(idx)),
            reverse=True,
        )
    else:
        order = torch.argsort(rewards_cpu, descending=True).tolist()
    seen: set[bytes] = set()
    selected: list[int] = []
    for idx in order:
        key = batch_cpu[idx].contiguous().numpy().tobytes()
        if key in seen:
            continue
        seen.add(key)
        selected.append(idx)
        if len(selected) >= k:
            break
    return selected


__all__ = [
    "categorical_interpolant_sample",
    "edits_to_time",
    "elite_position_probs",
    "nearest_time_index",
    "sample_truncated_normal",
    "select_topk_unique",
    "step_to_time",
    "time_to_edits",
    "time_to_step",
    "stochastic_round_time",
    "update_elite_pool",
]
