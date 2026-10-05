"""Exact final protein protocol overrides applied to the base recipes."""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
from .runner import ProteinStudyCell

HARD5_PROTOCOL = "hard5_online_full_history_v2"

MARGINAL_EI_PROTOCOL = "hard5_marginal_ei_full_history_v1"

_FULL_HISTORY_GP_ARMS = frozenset({"rt_gp", "tr_gp", "adalead", "bo_evo"})

_LEGACY_GP_ARMS = frozenset({"rt_gp", "tr_gp"})

_LAMBO_PRETRAINING_KEYS = frozenset(
    {
        "assets",
        "checkpoint",
        "checkpoint_validation_manifest",
        "checkpoint_validation_samples",
        "corpus",
        "expected_transferred_tensors",
        "pretraining_assets",
        "pretraining_attestation",
        "pretraining_manifest",
        "pretraining_scores",
    }
)

_OUTPUT_KEYS = (
    "initial_batch_out",
    "samples_out",
    "proposal_diagnostics_out",
    "proposal_traces_out",
    "run_manifest_out",
)

class ProteinRecipeError(RuntimeError):
    """Raised when a frozen protein recipe is incomplete."""

def _without_lambo_pretraining_assets(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: nested
        for key, nested in value.items()
        if key not in _LAMBO_PRETRAINING_KEYS
        and not key.startswith("checkpoint_")
        and not key.startswith("pretraining_")
    }

def _v2_resolved_config(cell: ProteinStudyCell, *, root: Path) -> dict[str, Any]:
    resolved: dict[str, Any] = copy.deepcopy(cell.resolved_config)
    for key in _OUTPUT_KEYS:
        resolved[key] = Path(str(resolved[key])).resolve().relative_to(root).as_posix()

    metadata = dict(resolved.get("metadata", {}))
    metadata["pilot_protocol"] = HARD5_PROTOCOL
    if cell.arm in _FULL_HISTORY_GP_ARMS:
        resolved.update(
            {
                "al_buffer_size": 2_000,
                "al_buffer_mode": "recent",
                "al_gp_hyperparameter_refit_interval": 8,
            }
        )
        metadata.update(
            {
                "pilot_training_history": "complete_paid_history",
                "pilot_gp_hyperparameter_refit_interval": 8,
            }
        )
    if cell.arm in _LEGACY_GP_ARMS:
        metadata = {key: value for key, value in metadata.items() if not key.startswith("compatibility_")}
    if cell.arm == "lambo2":
        resolved = _without_lambo_pretraining_assets(resolved)
        nested = resolved.get("lambo2")
        if not isinstance(nested, dict):
            raise ProteinRecipeError("hard5 v2 LaMBO-2 cell has no protocol mapping")
        nested = _without_lambo_pretraining_assets(nested)
        nested.update(
            {
                "initialization": "online_archive_random",
                "training_history": "complete_paid_history",
            }
        )
        resolved["lambo2"] = nested
        metadata.update(
            {
                "pilot_initialization": "online_archive_random",
                "pilot_training_history": "complete_paid_history",
            }
        )
    resolved["metadata"] = metadata
    return resolved

def _marginal_ei_resolved_config(
    cell: ProteinStudyCell,
    *,
    root: Path,
) -> dict[str, Any]:
    resolved: dict[str, Any] = copy.deepcopy(cell.resolved_config)
    for key in _OUTPUT_KEYS:
        resolved[key] = Path(str(resolved[key])).resolve().relative_to(root).as_posix()

    resolved.update(
        {
            "al_buffer_size": 2_000,
            "al_buffer_mode": "recent",
            "al_gp_hyperparameter_refit_interval": 8,
            "al_acquisition": "expected_improvement",
            "al_sequential_selection": False,
            "al_local_penalization": False,
        }
    )
    for key in tuple(resolved):
        if key.startswith("al_qei_"):
            resolved.pop(key)

    metadata = {
        key: value
        for key, value in dict(resolved.get("metadata", {})).items()
        if not key.startswith("compatibility_") and "qei" not in key.lower()
    }
    metadata.update(
        {
            "pilot_protocol": MARGINAL_EI_PROTOCOL,
            "pilot_training_history": "complete_paid_history",
            "pilot_gp_hyperparameter_refit_interval": 8,
        }
    )
    resolved["metadata"] = metadata
    return resolved
