"""Reward wrapper for Ehrlich benchmark scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from numbers import Integral
from pathlib import Path

import torch
from torch import Tensor

from .ehrlich_instance import EhrlichConfig, EhrlichInstance, make_ehrlich_instance
from .ehrlich_score import ehrlich_score_batch


class EhrlichReward:
    """Callable reward that enforces Ehrlich feasibility and scoring."""

    def __init__(
        self,
        instance_cfg: Mapping[str, object] | None = None,
        *,
        seed: int = 0,
        instance: EhrlichInstance | None = None,
        instance_path: str | Path | None = None,
        quantization_override: int | None = None,
    ) -> None:
        if instance is None and instance_path is not None:
            loaded = torch.load(Path(instance_path).expanduser(), map_location="cpu", weights_only=False)
            if not isinstance(loaded, EhrlichInstance):
                raise TypeError("instance_path did not contain an EhrlichInstance")
            instance = loaded
        if instance is None:
            if instance_cfg is None:
                raise ValueError("EhrlichReward requires instance, instance_path, or instance_cfg")
            cfg = EhrlichConfig(**{k: v for k, v in instance_cfg.items()})
            instance = make_ehrlich_instance(cfg, seed=seed)
        self.instance = instance
        if quantization_override is not None:
            if isinstance(quantization_override, bool) or not isinstance(quantization_override, Integral):
                raise ValueError("quantization_override must be an integer")
            override = int(quantization_override)
            motif_len = int(self.instance.cfg.motif_len)
            if override < 1 or override > motif_len or motif_len % override != 0:
                raise ValueError(
                    "quantization_override must be between 1 and motif_len and evenly divide motif_len"
                )
            self.quantization_override = override
        else:
            self.quantization_override = None
        if self.quantization_override is None:
            self._score_instance = self.instance
        else:
            score_cfg = replace(self.instance.cfg, quantization=self.quantization_override)
            score_cfg.validate()
            self._score_instance = EhrlichInstance(
                cfg=score_cfg,
                transition_probs=self.instance.transition_probs,
                transition_mask=self.instance.transition_mask,
                motifs=self.instance.motifs,
                spacings=self.instance.spacings,
                spacing_max=self.instance.spacing_max,
                base_sequence=self.instance.base_sequence,
            )

    def __call__(self, x: Tensor) -> float:
        if x.ndim != 1:
            raise ValueError("EhrlichReward expects a 1D sequence.")
        score = self.score_batch(x)
        return float(score.item())

    def score_batch(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            scores = ehrlich_score_batch(x, self._score_instance)
            return scores[0]
        if x.ndim != 2:
            raise ValueError("EhrlichReward expects a [L] or [B, L] tensor.")
        return ehrlich_score_batch(x, self._score_instance)

    def state_dict(self) -> dict[str, object]:
        return {
            "instance_cfg": asdict(self.instance.cfg),
            "transition_probs": self.instance.transition_probs,
            "transition_mask": self.instance.transition_mask,
            "motifs": self.instance.motifs,
            "spacings": self.instance.spacings,
            "spacing_max": self.instance.spacing_max,
            "base_sequence": self.instance.base_sequence,
            "quantization_override": self.quantization_override,
        }


__all__ = ["EhrlichReward"]
