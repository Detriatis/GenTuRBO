"""Batched, finite-epoch Boxoban data compatible with the existing trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from core.registry import register
from sokoban.boards import HEIGHT, SEQ_LEN, TILES, WIDTH


@dataclass
class BoxobanConfig:
    root: str = "runs/data/boxoban"
    split: str = "train"
    batch_size: int = 128
    seed: int = 0
    shuffle: bool = True
    augment: bool = True
    expected_count: int | None = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid or test")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.split != "train" and self.augment:
            raise ValueError("held-out splits require augment=false")


class BoxobanIterable(IterableDataset):
    def __init__(self, cfg: BoxobanConfig):
        super().__init__()
        self.cfg = cfg
        path = Path(cfg.root).expanduser() / f"{cfg.split}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Prepare data with: python -m sokoban.prepare --download")
        self.boards = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.boards.dtype != np.uint8 or self.boards.ndim != 2 or self.boards.shape[1] != SEQ_LEN:
            raise ValueError(f"{path}: expected uint8 [N, {SEQ_LEN}]")
        self.num_levels = len(self.boards)
        if self.num_levels == 0 or int(self.boards.max()) >= len(TILES):
            raise ValueError(f"{path}: empty dataset or unknown tile IDs")
        if cfg.expected_count is not None and self.num_levels != cfg.expected_count:
            raise ValueError(f"{path}: expected {cfg.expected_count} levels, found {self.num_levels}")
        self.epoch = 0
        grid = np.arange(SEQ_LEN).reshape(HEIGHT, WIDTH)
        self.symmetries = np.stack(
            [np.rot90(np.fliplr(grid) if flip else grid, k).flatten() for flip in (False, True) for k in range(4)]
        )

    def __len__(self) -> int:
        return math.ceil(self.num_levels / self.cfg.batch_size)

    def __iter__(self):
        info = get_worker_info()
        worker_id, workers = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, self.epoch]))
        self.epoch += 1
        indices = rng.permutation(self.num_levels) if self.cfg.shuffle else np.arange(self.num_levels)
        # All workers use the same permutation and disjoint slices of it.
        indices = indices[worker_id::workers]
        augmentation_rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, self.epoch, worker_id]))
        for start in range(0, len(indices), self.cfg.batch_size):
            board = self.boards[indices[start : start + self.cfg.batch_size]].astype(np.int64)
            if self.cfg.augment:
                choices = augmentation_rng.integers(0, 8, size=len(board))
                board = np.take_along_axis(board, self.symmetries[choices], axis=1)
            yield {"x": torch.from_numpy(np.ascontiguousarray(board))}


@register("dataset", "boxoban")
def build_dataset(**cfg: object) -> BoxobanIterable:
    return BoxobanIterable(BoxobanConfig(**cfg))
