"""Frozen-candidate writers used by the local kernel-study workflow."""
from __future__ import annotations
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import torch
from ehrlich.ehrlich_reward import EhrlichReward
from genturbo.io import write_jsonl
from .results import KernelStudyResultsError

def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".jsonl.gz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_jsonl(temporary, iter(rows))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _score_frozen_candidates(
    rows: list[dict[str, Any]],
    *,
    condition: Mapping[str, Any],
    repo_root: Path,
    batch_size: int = 512,
) -> None:
    reward = EhrlichReward(
        instance_path=repo_root / str(condition["instance_path"]),
        quantization_override=int(condition["quantization"]),
    )
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        sequences = torch.as_tensor([row["sequence"] for row in batch_rows], dtype=torch.long)
        scores = reward.score_batch(sequences).detach().cpu().tolist()
        for row, value in zip(batch_rows, scores, strict=True):
            if not math.isfinite(float(value)):
                raise KernelStudyResultsError(
                    f"eligible candidate {row['candidate_id']} became invalid during task-specific rescoring"
                )
            row["score"] = float(value)
