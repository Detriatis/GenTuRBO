"""Iterable dataset for sampling Ehrlich Markov sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple, Union, Mapping

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from core.registry import register
from data.base import Compose, Transform, TransformsLike, make_dataloader, to_compose
from ehrlich.ehrlich_generator import sample_markov_sequences
from ehrlich.ehrlich_instance import EhrlichConfig, EhrlichInstance, make_ehrlich_instance



@dataclass
class EhrlichMarkovConfig:
    instance_cfg: Mapping[str, object]
    instance_seed: int = 0
    batch_size: int = 256
    seed: int = 0
    transforms: TransformsLike = None
    emit_instance: bool = False


def _make_generator(base_seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(base_seed))
    return g


class EhrlichMarkovIterable(IterableDataset):
    def __init__(self, cfg: EhrlichMarkovConfig):
        super().__init__()
        self.cfg = cfg
        self.transforms = to_compose(cfg.transforms)
        inst_cfg = EhrlichConfig(**{k: v for k, v in cfg.instance_cfg.items()})
        self.instance: EhrlichInstance = make_ehrlich_instance(inst_cfg, seed=cfg.instance_seed)

    def __iter__(self) -> Iterator[dict[str, Tensor | EhrlichInstance]]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        seed = (self.cfg.seed + 0x9E3779B97F4A7C15 + worker_id) % (2**63 - 1)
        g = _make_generator(seed)
        while True:
            x = sample_markov_sequences(
                self.instance.transition_probs,
                self.instance.cfg.seq_len,
                batch_size=self.cfg.batch_size,
                generator=g,
            )
            x, _ = self.transforms(x, None)
            out: dict[str, Tensor | EhrlichInstance] = {"x": x.long()}
            if self.cfg.emit_instance:
                out["instance"] = self.instance
            yield out


def make_dataset(**cfg: object) -> EhrlichMarkovIterable:  # type: ignore[override]
    return EhrlichMarkovIterable(EhrlichMarkovConfig(**cfg))


@register("dataset", "ehrlich_markov")
def build_dataset(**cfg: object) -> EhrlichMarkovIterable:  # type: ignore
    return make_dataset(**cfg)


__all__ = ["EhrlichMarkovConfig", "EhrlichMarkovIterable", "make_dataset", "make_dataloader"]
