"""Time-selection and trust-region updates for interpolant proposals."""

from __future__ import annotations

from typing import Any, Optional

import torch

from search.interpolant.trust_region import make_trust_region_strategy
from search.interpolant.types import CandidateConfig, TrustRegionConfig


class CandidateTimePolicy:
    def __init__(
        self,
        *,
        runtime: Any,
        candidate: CandidateConfig,
        trust_region: TrustRegionConfig,
    ) -> None:
        self.runtime = runtime
        self.candidate = candidate
        self.trust_region = trust_region
        self.tr_strategy = (
            make_trust_region_strategy(
                config=trust_region,
                seq_len=int(runtime.seq_len),
                device=runtime.device,
            )
            if candidate.strategy == "trust_region"
            else None
        )

    def editable_seq_len_for_sample(self, sample: Optional[torch.Tensor]) -> int:
        if sample is None:
            return int(max(1, self.runtime.seq_len))
        sample_tensor = torch.as_tensor(sample, device=self.runtime.device, dtype=torch.long)
        if sample_tensor.ndim == 1:
            sample_batch = sample_tensor.view(1, -1)
        elif sample_tensor.ndim == 2:
            sample_batch = sample_tensor[:1]
        else:
            raise ValueError("sample must have shape [seq_len] or [batch, seq_len]")
        model = getattr(self.runtime.process, "model", None)
        trust_region_seq_len = getattr(model, "trust_region_seq_len", None)
        if callable(trust_region_seq_len):
            return int(max(1, int(trust_region_seq_len(sample_batch))))
        mutable_mask = getattr(model, "mutable_mask", None)
        if callable(mutable_mask):
            mask = mutable_mask(sample_batch)
            if isinstance(mask, torch.Tensor) and mask.shape == sample_batch.shape:
                return int(max(1, int(mask[0].sum().item())))
        return int(max(1, sample_batch.shape[1]))

    def candidate_times_for_iteration(
        self,
        iteration: int,
        *,
        base_sample: Optional[torch.Tensor] = None,
    ) -> list[float]:
        candidate = self.candidate

        if candidate.fixed_times is not None:
            return candidate.fixed_times

        if candidate.strategy == "trust_region":
            if self.tr_strategy is None:
                raise RuntimeError("trust_region strategy is not initialised")
            effective_seq_len = self.editable_seq_len_for_sample(base_sample)
            return self.tr_strategy.sample_times(
                num_candidates=int(candidate.num_candidates),
                iteration=int(iteration),
                seq_len=effective_seq_len,
            )

        if candidate.strategy == "random":
            return torch.rand((candidate.num_candidates,), device=self.runtime.device).tolist()

        if candidate.strategy == "uniform":
            return torch.linspace(0.0, 1.0, steps=candidate.num_candidates + 2, device=self.runtime.device)[1:-1].tolist()

        raise RuntimeError(f"Unsupported candidate strategy: {candidate.strategy}")

    def update_trust_region(
        self,
        *,
        best_time: Optional[float],
        best_edits: Optional[float] = None,
        improved: bool,
        mean_edit_error: Optional[float] = None,
    ) -> None:
        if self.candidate.strategy != "trust_region":
            return
        if self.tr_strategy is None:
            raise RuntimeError("trust_region strategy is not initialised")
        self.tr_strategy.update(
            best_time=best_time,
            best_edits=best_edits,
            improved=bool(improved),
            mean_edit_error=mean_edit_error,
        )

    def expand_trust_region_once(self) -> tuple[Optional[float], Optional[float]]:
        if self.candidate.strategy != "trust_region":
            return None, None
        if self.tr_strategy is None:
            raise RuntimeError("trust_region strategy is not initialised")
        expand_once = getattr(self.tr_strategy, "expand_once", None)
        if not callable(expand_once):
            raise RuntimeError("trust_region strategy does not support forced expansion")
        return expand_once()


__all__ = ["CandidateTimePolicy"]
