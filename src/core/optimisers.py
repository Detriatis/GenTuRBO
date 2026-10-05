"""Optimiser builders and registry-backed helpers.

This module mirrors the lightweight factory pattern used throughout the core
package so experiments can request optimisers by name (e.g.
``make_optimiser("adamw", model.parameters(), lr=3e-4)``).  It performs a small
amount of validation on the provided parameter collections before delegating to
the underlying ``torch.optim`` implementations and exposes utilities to inspect
the available entries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Callable, Dict

import torch.optim as optim
from torch import nn
from torch.optim import Optimizer

from core.registry import available as _registry_available
from core.registry import build as _registry_build
from core.registry import register

ParamGroup = Mapping[str, object]
OptimiserParams = Iterable[nn.Parameter] | Sequence[ParamGroup]

__all__ = ["OptimiserParams", "list_optimisers", "make_optimiser"]


# Public helpers ----------------------------------------------------------------


def make_optimiser(kind: str, parameters: OptimiserParams, /, **cfg: object) -> Optimizer:
    """Instantiate a registered optimiser."""
    optimiser = _registry_build("optimiser", kind, parameters=parameters, **cfg)
    if not isinstance(optimiser, Optimizer):
        raise TypeError(
            f"Optimiser '{kind}' did not return torch.optim.Optimizer (got {type(optimiser).__name__})"
        )
    return optimiser


def list_optimisers() -> list[str]:
    """Return the registered optimiser names."""
    return _registry_available("optimiser")


# Internal utilities ------------------------------------------------------------


def _materialize_parameters(parameters: OptimiserParams) -> list[nn.Parameter] | list[dict[str, object]]:
    """Convert parameter iterables or param-group specs into concrete lists."""
    if isinstance(parameters, Sequence):
        if len(parameters) == 0:
            raise ValueError("Optimiser received an empty parameter sequence")
        first = parameters[0]
        if isinstance(first, Mapping):
            groups: list[dict[str, object]] = []
            for idx, group in enumerate(parameters):
                if "params" not in group:
                    raise KeyError(f"Parameter group {idx} is missing the 'params' entry")
                raw_params = group["params"]
                if not isinstance(raw_params, Iterable):
                    raise TypeError(f"Parameter group {idx} 'params' must be iterable")
                param_list = list(raw_params)
                if not param_list:
                    raise ValueError(f"Parameter group {idx} does not contain any parameters")
                group_dict = dict(group)
                group_dict["params"] = param_list
                groups.append(group_dict)
            return groups
    param_list = list(parameters)
    if not param_list:
        raise ValueError("Optimiser requires at least one parameter tensor")
    return param_list


def _register_torch_optimiser(
    name: str,
    ctor: Callable[..., Optimizer],
    *,
    defaults: dict[str, object] | None = None,
) -> None:
    """Register a thin wrapper around a torch optimiser constructor."""

    @register("optimiser", name)
    def _builder(parameters: OptimiserParams, **cfg: object) -> Optimizer:
        params = _materialize_parameters(parameters)
        options: Dict[str, object] = {}
        if defaults:
            options.update(defaults)
        options.update(cfg)
        try:
            return ctor(params, **options)
        except TypeError as exc:  # pragma: no cover - exercised indirectly
            raise TypeError(
                f"Failed to instantiate optimiser '{name}' with cfg={options}"
            ) from exc


# Torch optimisers --------------------------------------------------------------


_register_torch_optimiser("sgd", optim.SGD)
_register_torch_optimiser("sgd_momentum", optim.SGD, defaults={"momentum": 0.9})
_register_torch_optimiser("sgd_nesterov", optim.SGD, defaults={"momentum": 0.9, "nesterov": True})
_register_torch_optimiser("adam", optim.Adam)
_register_torch_optimiser("adamw", optim.AdamW)
_register_torch_optimiser("adamax", optim.Adamax)
_register_torch_optimiser("radam", optim.RAdam)
_register_torch_optimiser("adagrad", optim.Adagrad)
_register_torch_optimiser("adadelta", optim.Adadelta)
_register_torch_optimiser("rmsprop", optim.RMSprop)
_register_torch_optimiser("asgd", optim.ASGD)
_register_torch_optimiser("lbfgs", optim.LBFGS)

if hasattr(optim, "Lion"):
    _register_torch_optimiser("lion", optim.Lion)
