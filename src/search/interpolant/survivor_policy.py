"""Survivor-selection policy for interpolant search."""

from __future__ import annotations

from typing import Optional

import torch


class BestSurvivorSelectionPolicy:
    """Keep best-so-far leaf and feed candidate leaves into the next batch."""

    def __init__(self) -> None:
        self.best_overall: Optional[torch.Tensor] = None
        self.best_reward: float = float("-inf")
        self.leaves_log: list[torch.Tensor] = []

    def update(
        self,
        *,
        candidate_leaves: list[torch.Tensor],
        best_candidate_state: torch.Tensor,
        best_candidate_reward: float,
    ) -> tuple[torch.Tensor, float]:
        prev_best = self.best_reward
        if best_candidate_reward >= self.best_reward:
            # On score ties, track the most recent best sequence.
            self.best_reward = best_candidate_reward
            self.best_overall = best_candidate_state.clone()

        next_batch: list[torch.Tensor] = []
        if self.best_overall is not None:
            # Keep best-overall first; tie-breaking logic can then prefer newer candidates when desired.
            next_batch.append(self.best_overall)
        next_batch.extend(candidate_leaves)
        current_batch = torch.cat(next_batch, dim=0) if next_batch else best_candidate_state
        self.leaves_log.append(best_candidate_state.detach().cpu().clone())
        return current_batch, prev_best


__all__ = ["BestSurvivorSelectionPolicy"]
