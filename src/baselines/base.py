"""Common baseline interfaces and result containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class BaselineResult:
    samples: list[Any]
    scores: np.ndarray
    best_samples: list[Any]
    best_scores: np.ndarray
    oracle_calls: int
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    generated_candidates: int = 0

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=float).reshape(-1)
        self.best_scores = np.asarray(self.best_scores, dtype=float).reshape(-1)
        if self.oracle_calls < 0:
            raise ValueError("oracle_calls must be non-negative")
        if self.generated_candidates < 0:
            raise ValueError("generated_candidates must be non-negative")


class Baseline(ABC):
    """Base class for budgeted optimisation baselines."""

    name: str = "baseline"

    @abstractmethod
    def run(
        self,
        model: Any,
        oracle: Any,
        budget: int,
        batch_size: int,
        seed: int,
        **kwargs: Any,
    ) -> BaselineResult:
        """Run the baseline under a fixed oracle-call budget."""
