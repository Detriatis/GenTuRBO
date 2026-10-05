"""Strict contract for the one-config ``ehrlich_controls_v1`` study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genturbo.kernel_study.config import StudyConfigs, load_study
from genturbo.serialization import canonical_json_bytes

STUDY = "ehrlich_controls_v1"
SCHEMA_VERSION = 1
CONFIG_FILENAME = "controls.json"
CONDITION = "l256_q4_k8"
ARMS = (
    "unconditional",
    "no_tr_no_gp",
    "tr_no_gp",
    "no_tr_gp",
    "tr_gp",
    "oracle_upper_bound",
)

_BASE_DATASETS_CONFIG = "configs/experiments/ehrlich_kernel_v1/datasets.json"
# Keep the experiment declaration in one place.  The digest pins the complete
# schema and protocol without mirroring the JSON as another 130-line Python
# dictionary that can drift independently.
_EXPECTED_CONFIG_SHA256 = "6404899da9f824323794c4d39ca6005f754dbbad30de64afce190f7db686101c"


class ControlStudyConfigError(ValueError):
    """Raised when the paper-control protocol drifts from its pinned contract."""


@dataclass(frozen=True, slots=True)
class ControlStudyConfig:
    path: Path
    raw: dict[str, Any]
    kernel_study: StudyConfigs

    @property
    def condition(self) -> dict[str, Any]:
        for condition in self.kernel_study.datasets["conditions"]:
            if condition["name"] == CONDITION:
                return condition
        raise ControlStudyConfigError(f"base dataset config is missing {CONDITION}")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlStudyConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControlStudyConfigError(f"{path} must contain a JSON object")
    return value


def _validate(raw: dict[str, Any], *, path: Path) -> None:
    if (raw.get("schema_version"), raw.get("study"), raw.get("workflow")) != (
        SCHEMA_VERSION,
        STUDY,
        "controls",
    ):
        raise ControlStudyConfigError(f"{path} does not identify {STUDY}/controls schema 1")
    if raw.get("base_datasets_config") != _BASE_DATASETS_CONFIG or raw.get("condition") != CONDITION:
        raise ControlStudyConfigError("controls must reuse the pinned ehrlich_kernel_v1 L256 condition")
    if raw.get("run_dir") != f"runs/{STUDY}/controls" or raw.get("results_dir") != f"results/{STUDY}/controls":
        raise ControlStudyConfigError("controls output paths must mirror the one active config")
    observed = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    if observed != _EXPECTED_CONFIG_SHA256:
        raise ControlStudyConfigError(
            f"{path} differs from the versioned {STUDY} protocol; "
            "change the sole experiment config and its contract together"
        )


def load_control_study(path: Path | str, *, repo_root: Path | str) -> ControlStudyConfig:
    """Load the sole controls config and its pinned L256 generator/task declaration."""

    root = Path(repo_root).expanduser().resolve()
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    if config_path.name != CONFIG_FILENAME:
        raise ControlStudyConfigError(f"expected {CONFIG_FILENAME}, got {config_path.name}")
    configs = sorted(item.name for item in config_path.parent.glob("*.json"))
    if configs != [CONFIG_FILENAME]:
        raise ControlStudyConfigError(f"{config_path.parent} must contain exactly {CONFIG_FILENAME}")
    raw = _read_object(config_path)
    _validate(raw, path=config_path)
    base_path = (root / str(raw["base_datasets_config"])).resolve()
    expected = root / _BASE_DATASETS_CONFIG
    if base_path != expected.resolve() or not base_path.is_file():
        raise ControlStudyConfigError("base_datasets_config does not resolve to the active kernel-study dataset config")
    try:
        kernel_study = load_study(base_path.parent, repo_root=root)
    except ValueError as exc:
        raise ControlStudyConfigError(str(exc)) from exc
    study = ControlStudyConfig(config_path, raw, kernel_study)
    _ = study.condition
    return study


__all__ = [
    "ARMS",
    "CONDITION",
    "CONFIG_FILENAME",
    "STUDY",
    "ControlStudyConfig",
    "ControlStudyConfigError",
    "load_control_study",
]
