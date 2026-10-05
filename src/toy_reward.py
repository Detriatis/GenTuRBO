"""Toy reward utilities for discrete sequence experiments."""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import torch

from core.registry import build as build_from_registry, register, register_lazy
from torch import Tensor


class ToyReward:
    """
    Hamming-distance based reward relative to a fixed target sequence.
    """

    def __init__(self, target: torch.Tensor) -> None:
        if target.ndim != 1:
            raise ValueError("target must be a 1D tensor.")
        if target.dtype != torch.long:
            raise ValueError("target must be of dtype torch.long.")
        self.target = target.clone()

    def __call__(self, x: torch.Tensor) -> float:
        if x.ndim != 1:
            raise ValueError("ToyReward expects a 1D sequence.")
        score = self.score_batch(x)
        return float(score.item())

    def score_batch(self, x: torch.Tensor) -> Tensor:
        if x.shape != self.target.shape:
            if x.ndim != 2 or x.shape[1] != self.target.shape[0]:
                raise ValueError(f"Input shape {tuple(x.shape)} does not match target {tuple(self.target.shape)}.")
        target = self.target.to(device=x.device)
        if x.ndim == 1:
            diff = (x != target).sum().float()
            return -diff
        diff = (x != target.unsqueeze(0)).sum(dim=1).float()
        return -diff


class ProxyDrugReward:
    """
    Multi-term toy reward:
      reward = target_weight * count(target_token)
             - forbidden_penalty * count(forbidden_tokens)
             - budget_penalty * max(0, count(penalized_tokens) - budget)
    """

    def __init__(
        self,
        target_token: int,
        *,
        target_weight: float = 1.0,
        forbidden_tokens: Optional[Sequence[int]] = None,
        forbidden_penalty: float = 1.0,
        penalized_tokens: Optional[Sequence[int]] = None,
        budget: int = 0,
        budget_penalty: float = 0.5,
    ) -> None:
        self.target_token = int(target_token)
        self.target_weight = float(target_weight)
        self.forbidden = set(int(t) for t in (forbidden_tokens or []))
        self.forbidden_penalty = float(forbidden_penalty)
        self.penalized = set(int(t) for t in (penalized_tokens or []))
        self.budget = int(budget)
        self.budget_penalty = float(budget_penalty)

    def __call__(self, x: torch.Tensor) -> float:
        if x.ndim != 1:
            raise ValueError("ProxyDrugReward expects a 1D sequence.")
        score = self.score_batch(x)
        return float(score.item())

    def score_batch(self, x: Tensor) -> Tensor:
        if x.ndim != 1:
            if x.ndim != 2:
                raise ValueError("ProxyDrugReward expects a [L] or [B, L] tensor.")
            batch = x
        else:
            batch = x.view(1, -1)
        if batch.dtype != torch.long:
            batch = batch.to(torch.long)

        target_count = (batch == self.target_token).sum(dim=1).float()

        forbidden_count = torch.zeros_like(target_count)
        if self.forbidden:
            forbidden = torch.tensor(sorted(self.forbidden), device=batch.device, dtype=batch.dtype)
            forbidden_count = (batch.unsqueeze(-1) == forbidden.view(1, 1, -1)).any(dim=-1).sum(dim=1).float()

        penalized_count = torch.zeros_like(target_count)
        if self.penalized:
            penalized = torch.tensor(sorted(self.penalized), device=batch.device, dtype=batch.dtype)
            penalized_count = (batch.unsqueeze(-1) == penalized.view(1, 1, -1)).any(dim=-1).sum(dim=1).float()

        reward = self.target_weight * target_count
        reward -= self.forbidden_penalty * forbidden_count
        if self.budget > 0:
            reward -= self.budget_penalty * torch.clamp(penalized_count - float(self.budget), min=0.0)
        if x.ndim == 1:
            return reward[0]
        return reward


RewardFn = Callable[[Tensor], float]
RewardSpec = Mapping[str, object] | RewardFn


register_lazy("reward", "ehrlich", "ehrlich.ehrlich_reward:EhrlichReward")
register_lazy("reward", "sokoban_difficulty", "sokoban.search:SokobanReward")
register_lazy("reward", "sokoban_box_switches", "sokoban.search:SokobanSwitchReward")
register_lazy(
    "reward",
    "gfp_aav_cnn_oracle",
    "adapters.protein.reward:build_gfp_aav_cnn_oracle_reward",
    aliases=("kirjner_gfp_aav_cnn", "gfp_aav_oracle"),
)


def _to_long_tensor(target: Sequence[int] | Tensor) -> Tensor:
    if isinstance(target, Tensor):
        out = target
    else:
        out = torch.tensor(list(target), dtype=torch.long)
    if out.ndim != 1:
        raise ValueError("Reward targets must be 1D sequences.")
    if out.dtype != torch.long:
        out = out.to(torch.long)
    return out


def _to_long_tensor_2d(targets: Sequence[Sequence[int]] | Tensor) -> Tensor:
    if isinstance(targets, Tensor):
        out = targets
    else:
        out = torch.tensor([list(t) for t in targets], dtype=torch.long)
    if out.ndim != 2:
        raise ValueError("Reward targets must be a 2D tensor or a sequence of sequences.")
    if out.numel() == 0:
        raise ValueError("Reward targets must contain at least one sequence.")
    if out.dtype != torch.long:
        out = out.to(torch.long)
    return out


@register("reward", "toy_hamming")
def build_toy_reward(target: Sequence[int] | Tensor) -> ToyReward:
    """Factory for Hamming-distance reward."""
    return ToyReward(target=_to_long_tensor(target))


@register("reward", "proxy_drug")
def build_proxy_drug_reward(
    target_token: int,
    *,
    target_weight: float = 1.0,
    forbidden_tokens: Optional[Sequence[int]] = None,
    forbidden_penalty: float = 1.0,
    penalized_tokens: Optional[Sequence[int]] = None,
    budget: int = 0,
    budget_penalty: float = 0.5,
) -> ProxyDrugReward:
    """Factory for the multi-term proxy-drug reward."""
    return ProxyDrugReward(
        target_token=target_token,
        target_weight=target_weight,
        forbidden_tokens=forbidden_tokens,
        forbidden_penalty=forbidden_penalty,
        penalized_tokens=penalized_tokens,
        budget=budget,
        budget_penalty=budget_penalty,
    )


class SetHammingReward:
    """Reward = negative min Hamming distance to any target sequence."""

    def __init__(self, targets: Sequence[Sequence[int]] | Tensor) -> None:
        self.targets = _to_long_tensor_2d(targets)

    def __call__(self, x: Tensor) -> float:
        if x.ndim != 1:
            raise ValueError("SetHammingReward expects a 1D sequence.")
        score = self.score_batch(x)
        return float(score.item())

    def score_batch(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            batch = x.view(1, -1)
        elif x.ndim == 2:
            batch = x
        else:
            raise ValueError("SetHammingReward expects a [L] or [B, L] tensor.")
        if batch.dtype != torch.long:
            batch = batch.to(torch.long)
        if batch.shape[1] != self.targets.shape[1]:
            raise ValueError(
                f"Input length {batch.shape[1]} does not match target length {self.targets.shape[1]}."
            )
        targets = self.targets.to(device=batch.device)
        diff = (batch.unsqueeze(1) != targets.unsqueeze(0)).sum(dim=2)
        best = -diff.min(dim=1).values.float()
        if x.ndim == 1:
            return best[0]
        return best


@register("reward", "set_hamming")
def build_set_hamming_reward(targets: Sequence[Sequence[int]] | Tensor) -> SetHammingReward:
    """Factory for a min-Hamming reward over a set of target sequences."""
    return SetHammingReward(targets=_to_long_tensor_2d(targets))


def resolve_reward(spec: RewardSpec) -> RewardFn:
    """Resolve a reward spec into a callable."""
    if callable(spec):
        return spec  # already a reward fn
    if not isinstance(spec, Mapping):
        raise TypeError("Reward spec must be a callable or mapping with a 'name' field")
    name = spec.get("name")
    if not isinstance(name, str):
        raise ValueError("Reward mapping must include a string 'name'")
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    return build_from_registry("reward", name, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "ToyReward",
    "ProxyDrugReward",
    "RewardFn",
    "RewardSpec",
    "resolve_reward",
    "build_toy_reward",
    "build_proxy_drug_reward",
    "SetHammingReward",
    "build_set_hamming_reward",
]
