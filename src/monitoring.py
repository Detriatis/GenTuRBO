"""
Monitoring utilities for ancestor-tree search experiments.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TensorBoardMonitor:
    """
    Simple TensorBoard logger for search diagnostics.
    """

    def __init__(self, log_dir: str | Path, flush_every: int = 100) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.flush_every = int(flush_every)
        self.last_flush_step = 0

    def log_iteration(self, iteration: int, metrics: Dict[str, Optional[float]]) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                self.writer.add_scalar(key, numeric_value, iteration)

        if iteration - self.last_flush_step >= self.flush_every:
            self.writer.flush()
            self.last_flush_step = iteration

    def log_histogram(self, name: str, values: Sequence[float], iteration: int) -> None:
        if not values:
            return
        tensor_values = np.asarray(
            [float(v) for v in values if math.isfinite(v)],
            dtype=np.float32,
        )
        if tensor_values.size:
            self.writer.add_histogram(name, tensor_values, iteration)

        if iteration - self.last_flush_step >= self.flush_every:
            self.writer.flush()
            self.last_flush_step = iteration

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
