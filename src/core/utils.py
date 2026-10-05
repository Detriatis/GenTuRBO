"""General-purpose helpers shared across GenTuRBO modules."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np
import torch
from torch import Tensor, nn

from core.types import Batch

__all__ = [
    "seed_everything",
    "resolve_device",
    "batch_to_device",
    "count_parameters",
    "maybe_load_checkpoint",
    "read_checkpoint_metadata",
    "load_checkpoint_metadata",
    "write_checkpoint_metadata",
    "Checkpoint",
    "CheckpointMetadataError",
]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(device: Optional[str | torch.device]) -> torch.device:
    """Resolve a user-provided device spec into a torch.device."""
    if device is None:
        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()
        return torch.device("cuda" if cuda else "mps" if mps else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor in ``batch`` onto ``device``."""
    moved: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Return the number of parameters in ``model``."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


@dataclass
class Checkpoint:
    """Simple checkpoint container."""

    path: Path
    model_state: Mapping[str, Tensor]
    metadata: MutableMapping[str, Any]


class CheckpointMetadataError(RuntimeError):
    """Raised when checkpoint metadata exists but cannot be read safely."""


def maybe_load_checkpoint(path: Optional[str | os.PathLike[str]], model: nn.Module) -> Optional[Checkpoint]:
    """Load ``model`` state dict from ``path`` if it exists."""
    if path is None:
        return None
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint '{ckpt_path}' does not exist")
    metadata = load_checkpoint_metadata(ckpt_path) or {}
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and ("model_state" in state or "metadata" in state):
        metadata = metadata or state.get("metadata", {})
        state = state.get("model_state", state)
    model.load_state_dict(state)
    return Checkpoint(path=ckpt_path, model_state=state, metadata=metadata)


def _metadata_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _json_default(value: object) -> object:
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "__dict__"):
        return asdict(value)  # type: ignore[arg-type]
    return value if isinstance(value, (str, int, float, bool, list, dict, type(None))) else str(value)


def write_checkpoint_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    meta_path = _metadata_sidecar_path(path)
    serialisable = json.loads(json.dumps(metadata, default=_json_default))
    meta_path.write_text(json.dumps(serialisable, indent=2))


def read_checkpoint_metadata(path: Path) -> Optional[dict[str, Any]]:
    meta_path = _metadata_sidecar_path(path)
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointMetadataError(f"cannot read checkpoint metadata sidecar {meta_path}") from exc
    if isinstance(data, dict):
        return data
    raise CheckpointMetadataError(f"checkpoint metadata sidecar must contain an object: {meta_path}")


def load_checkpoint_metadata(path: Path) -> Optional[dict[str, Any]]:
    data = read_checkpoint_metadata(path)
    if data is not None:
        return data
    try:
        blob = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise CheckpointMetadataError(f"cannot load checkpoint metadata from {path}") from exc
    if not isinstance(blob, Mapping):
        raise CheckpointMetadataError(f"checkpoint payload must be a mapping: {path}")
    meta = blob.get("metadata")
    return meta if isinstance(meta, dict) else None
