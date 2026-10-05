"""Dataset registrations for discrete sequence experiments."""

import importlib
from typing import Any

__all__ = [
    "seq_markov",
    "seq_fixed",
    "ehrlich_markov",
    "boxoban",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
