"""Shared typing primitives for the GenTuRBO core modules.

This module centralises the small set of aliases and protocols that show up
across datasets, schedules, losses, and processes so the rest of the package
can reuse a single source of truth and keep linting/mypy happy.
"""

from __future__ import annotations

from typing import Callable, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union

import torch
from torch import Tensor

# ---- Generic tensor containers -------------------------------------------------

Batch = Mapping[str, Tensor]
MutableBatch = MutableMapping[str, Tensor]
TensorMap = Mapping[str, Tensor]
MaybeTensor = Optional[Tensor]

# ---- Randomness helpers --------------------------------------------------------

GeneratorLike = Union[int, torch.Generator]

# ---- Transforms ----------------------------------------------------------------

Transform = Callable[[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor]]]
TransformsLike = Optional[Union[Transform, Sequence[Transform]]]

# ---- Schedules -----------------------------------------------------------------

class Schedule(Protocol):
    """Callable schedule that maps normalised time in [0, 1] to a scalar tensor."""

    def __call__(self, timesteps: Tensor) -> Tensor:
        ...


ScheduleFn = Callable[[Tensor], Tensor]

__all__ = [
    "Batch",
    "GeneratorLike",
    "MaybeTensor",
    "MutableBatch",
    "Schedule",
    "ScheduleFn",
    "TensorMap",
    "Transform",
    "TransformsLike",
]
