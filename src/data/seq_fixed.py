"""Fixed-sequence dataset for overfitting sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from core.registry import register
from data.base import TransformsLike, to_compose


@dataclass
class SeqFixedConfig:
    sequences: Optional[Sequence[Sequence[int]] | Tensor] = None
    sequences_path: Optional[str] = None
    batch_size: int = 256
    seed: int = 0
    transforms: TransformsLike = None
    emit_index: bool = False


def _make_generator(base_seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(base_seed))
    return g


def _to_sequence_tensor(sequences: Sequence[Sequence[int]] | Tensor) -> Tensor:
    if isinstance(sequences, Tensor):
        seqs = sequences.detach().clone().to(torch.long)
    else:
        seqs = torch.tensor([list(seq) for seq in sequences], dtype=torch.long)
    if seqs.ndim != 2:
        raise ValueError("sequences must be a 2D tensor or a sequence of sequences")
    if seqs.shape[0] == 0 or seqs.shape[1] == 0:
        raise ValueError("sequences must contain at least one non-empty sequence")
    if (seqs < 0).any():
        raise ValueError("sequences must contain non-negative token ids")
    return seqs


def _load_sequences(path: Path) -> Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Sequences path not found: {path}")
    if path.suffix in {".pt", ".pth"}:
        blob = torch.load(path, map_location="cpu")
        if isinstance(blob, dict):
            if "sequences" in blob:
                return _to_sequence_tensor(blob["sequences"])
            if "samples" in blob:
                return _to_sequence_tensor(blob["samples"])
        if torch.is_tensor(blob):
            return _to_sequence_tensor(blob)
        raise TypeError("Unsupported .pt/.pth contents for sequences")
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("sequences", data.get("samples"))
        return _to_sequence_tensor(data)
    raise ValueError("sequences_path must be a .pt/.pth or .json file")


class SeqFixedIterable(IterableDataset):
    def __init__(self, cfg: SeqFixedConfig):
        super().__init__()
        self.cfg = cfg
        self.transforms = to_compose(cfg.transforms)
        if cfg.sequences is None:
            if cfg.sequences_path is None:
                raise ValueError("SeqFixedConfig requires sequences or sequences_path")
            self._sequences = _load_sequences(Path(cfg.sequences_path).expanduser())
        else:
            self._sequences = _to_sequence_tensor(cfg.sequences)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        seed = (self.cfg.seed + 0x9E3779B97F4A7C15 + worker_id) % (2**63 - 1)
        g = _make_generator(seed)
        num_sequences = int(self._sequences.shape[0])

        while True:
            idx = torch.randint(0, num_sequences, (self.cfg.batch_size,), generator=g)
            x = self._sequences[idx]
            c: Optional[Tensor] = idx if self.cfg.emit_index else None
            x, c = self.transforms(x, c)
            out = {"x": x.long()}
            if c is not None:
                out["c"] = c.long()
            yield out


def make_dataset(**cfg: object) -> SeqFixedIterable:  # type: ignore[override]
    return SeqFixedIterable(SeqFixedConfig(**cfg))


@register("dataset", "seq_fixed")
def build_dataset(**cfg: object) -> SeqFixedIterable:  # type: ignore
    return make_dataset(**cfg)


__all__ = ["SeqFixedConfig", "SeqFixedIterable", "make_dataset"]
