"""Load readable JSON experiment configurations with explicit base overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Resolve one optional ``base_config`` relative to each declaring file.

    Overrides are shallow: a child replaces an entire value, including nested
    objects and lists. Missing keys inherit; explicit null values remain null.
    Ordinary paths inside the configuration retain their original interpretation.
    Each call returns fresh objects, so runtime overrides cannot mutate a base.
    """
    return _load_config(Path(path).expanduser().resolve(), ())


def _load_config(path: Path, ancestors: tuple[Path, ...]) -> dict[str, Any]:
    if path in ancestors:
        chain = " -> ".join(str(item) for item in (*ancestors, path))
        raise ValueError(f"base_config cycle: {chain}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"configuration must contain a JSON object: {path}")
    if "base_config" not in values:
        return values
    parent = values.pop("base_config")
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f"base_config must be a non-empty path string: {path}")
    base = _load_config((path.parent / parent).resolve(), (*ancestors, path))
    base.update(values)
    return base
