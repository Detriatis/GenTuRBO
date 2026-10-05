"""Small wall-clock timing helpers for search diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, Iterator

import torch


@dataclass
class TimingRecorder:
    enabled: bool = False
    sync_cuda: bool = False
    device: Any = None
    seconds: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)

    def _sync(self) -> None:
        if not self.enabled or not self.sync_cuda:
            return
        try:
            device = torch.device(self.device) if self.device is not None else torch.device("cpu")
        except (TypeError, RuntimeError):
            device = torch.device("cpu")
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        torch.cuda.synchronize(device)

    def start(self) -> float:
        if not self.enabled:
            return 0.0
        self._sync()
        return time.perf_counter()

    def stop(self, name: str, start_time: float) -> float:
        if not self.enabled:
            return 0.0
        self._sync()
        elapsed = time.perf_counter() - float(start_time)
        self.seconds[name] = float(self.seconds.get(name, 0.0) + elapsed)
        return elapsed

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start_time = self.start()
        try:
            yield
        finally:
            self.stop(name, start_time)

    def add_seconds(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        self.seconds[name] = float(self.seconds.get(name, 0.0) + float(value))

    def add_seconds_mapping(self, mapping: dict[str, float], *, prefix: str = "") -> None:
        if not self.enabled:
            return
        for name, value in mapping.items():
            key = f"{prefix}{name}" if prefix else str(name)
            self.add_seconds(key, float(value))

    def add_count(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        self.counts[name] = float(self.counts.get(name, 0.0) + float(value))

    def add_counts_mapping(self, mapping: dict[str, float], *, prefix: str = "") -> None:
        if not self.enabled:
            return
        for name, value in mapping.items():
            key = f"{prefix}{name}" if prefix else str(name)
            self.add_count(key, float(value))


__all__ = ["TimingRecorder"]
