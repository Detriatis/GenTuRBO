"""Shared dataset helpers to reduce duplication."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

Transform = Callable[[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor]]]
TransformsLike = Optional[Union[Transform, Sequence[Transform]]]


class Compose:
    def __init__(self, transforms: Optional[Sequence[Transform]] = None):
        self.transforms = list(transforms) if transforms is not None else []

    def __call__(self, x: Tensor, c: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        for t in self.transforms:
            x, c = t(x, c)
        return x, c

    def append(self, t: Transform) -> None:
        self.transforms.append(t)


def to_compose(tr: TransformsLike) -> Compose:
    if tr is None:
        return Compose([])
    if isinstance(tr, Sequence) and not isinstance(tr, (str, bytes)):
        return Compose(list(tr))
    return Compose([tr])  # type: ignore[list-item]


def make_dataloader(
    ds: IterableDataset,
    *,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    kwargs = {"batch_size": None, "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **kwargs)


__all__ = ["Compose", "Transform", "TransformsLike", "make_dataloader", "to_compose"]
