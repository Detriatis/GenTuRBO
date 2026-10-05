"""Typed configuration and legacy flat-config adaptation for search runs.

The experiment archive contains flat JSON objects because they are convenient to
generate and inspect.  This module keeps that wire format while separating
operational settings from descriptive metadata and validating misspelled keys
before a run creates any output files.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

TraceLevel = Literal["none", "summary", "full"]

# These fields are intentionally descriptive: changing them must never change a
# search trajectory.  They are the complete set used by the checked-in
# interpolant configs (including the initial-batch swap diagnostics).
KNOWN_METADATA_KEYS = frozenset(
    {
        "runner",
        "plot_label",
        "diagnostic",
        "swap_pair",
        "swap_role",
        "source_initial_seed",
        "dynamic_seed",
        "diagnostic_acquired_prefix_size",
        "experiment_status",
        "compatibility_mode",
        "compatibility_commit",
        "compatibility_projection_sha256",
        "compatibility_status",
        "pilot_protocol",
        "pilot_training_history",
        "pilot_gp_hyperparameter_refit_interval",
    }
)


@dataclass(frozen=True, slots=True)
class OutputPolicy:
    """Controls diagnostic output volume without changing search behaviour."""

    trace_level: TraceLevel = "summary"
    trace_record_limit: int | None = None
    compress_jsonl: bool = True

    def __post_init__(self) -> None:
        if self.trace_level not in {"none", "summary", "full"}:
            raise ValueError("trace_level must be one of: none, summary, full")
        if self.trace_record_limit is not None and self.trace_record_limit <= 0:
            raise ValueError("trace_record_limit must be > 0 when provided")
        if self.trace_level == "full" and self.trace_record_limit is None:
            raise ValueError("trace_level='full' requires an explicit trace_record_limit")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> OutputPolicy:
        return cls(
            trace_level=str(values.get("trace_level") or "summary").strip().lower(),  # type: ignore[arg-type]
            trace_record_limit=(
                int(values["trace_record_limit"])
                if values.get("trace_record_limit") is not None
                else None
            ),
            compress_jsonl=bool(values.get("compress_jsonl", True)),
        )


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Validated search configuration with a compatibility flat representation.

    ``values`` contains only operational parser destinations.  ``metadata`` is
    retained in manifests and on the returned namespace, but is never consumed
    by the search algorithm.
    """

    values: Mapping[str, Any] = field(default_factory=dict)
    benchmark_task: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)

    def __post_init__(self) -> None:
        values = dict(self.values)
        if self.benchmark_task is not None:
            values["benchmark_task"] = str(self.benchmark_task)
        values["trace_level"] = self.output_policy.trace_level
        values["trace_record_limit"] = self.output_policy.trace_record_limit
        values["compress_jsonl"] = self.output_policy.compress_jsonl
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_legacy_flat(
        cls,
        raw: Mapping[str, Any],
        *,
        operational_keys: Iterable[str],
    ) -> SearchConfig:
        """Validate and adapt one historical flat JSON search config.

        Unknown fields are reported together in sorted order so generated sweep
        typos can be fixed in one pass.
        """

        if not isinstance(raw, Mapping):
            raise TypeError("search config must contain a JSON object")
        allowed = set(str(key) for key in operational_keys)
        allowed.update({"benchmark_task", "trace_level", "trace_record_limit", "compress_jsonl"})

        nested_metadata = raw.get("metadata", {})
        if nested_metadata is None:
            nested_metadata = {}
        if not isinstance(nested_metadata, Mapping):
            raise TypeError("search config metadata must be a JSON object")

        unknown_metadata = sorted(set(str(key) for key in nested_metadata) - KNOWN_METADATA_KEYS)
        unknown_top_level = sorted(
            str(key)
            for key in raw
            if key not in allowed and key not in KNOWN_METADATA_KEYS and key != "metadata"
        )
        unknown = sorted(set(unknown_metadata + unknown_top_level))
        if unknown:
            raise ValueError(f"Unknown search config keys: {', '.join(unknown)}")

        metadata = dict(nested_metadata)
        for key in KNOWN_METADATA_KEYS:
            if key in raw:
                metadata[key] = raw[key]
        runner = metadata.get("runner")
        if runner is not None and str(runner).strip().lower() != "interpolant":
            raise ValueError(f"search config runner must be 'interpolant', got {runner!r}")

        values = {
            str(key): value
            for key, value in raw.items()
            if key in allowed and key not in {"benchmark_task", "trace_level", "trace_record_limit", "compress_jsonl"}
        }
        benchmark_task = raw.get("benchmark_task")
        if benchmark_task is not None:
            values["benchmark_task"] = str(benchmark_task)

        policy_values = dict(raw)
        output_policy = OutputPolicy.from_mapping(policy_values)
        values["trace_level"] = output_policy.trace_level
        values["trace_record_limit"] = output_policy.trace_record_limit
        values["compress_jsonl"] = output_policy.compress_jsonl
        return cls(
            values=values,
            benchmark_task=(str(benchmark_task) if benchmark_task is not None else None),
            metadata=metadata,
            output_policy=output_policy,
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        operational_keys: Iterable[str],
    ) -> SearchConfig:
        """Named constructor alias for callers that do not use legacy wording."""

        return cls.from_legacy_flat(raw, operational_keys=operational_keys)

    def to_legacy_flat(self) -> dict[str, Any]:
        """Return a flat JSON-compatible object for existing sweep tooling."""

        flat = dict(self.values)
        if self.benchmark_task is not None:
            flat["benchmark_task"] = self.benchmark_task
        flat.update(self.metadata)
        return flat

    def apply_to_namespace(self, args: argparse.Namespace) -> argparse.Namespace:
        """Apply config values below explicit CLI values and above defaults."""

        provided = set(getattr(args, "_provided", ()))
        defaults = dict(getattr(args, "_defaults", {}))
        for key, value in self.values.items():
            if key in provided:
                continue
            if hasattr(args, key) or key in defaults or key == "benchmark_task":
                setattr(args, key, value)

        metadata = dict(self.metadata)
        args.metadata = metadata
        # Retain historical attribute access for analysis scripts while keeping
        # these fields out of operational config handling.
        for key, value in metadata.items():
            setattr(args, key, value)
        args._search_config = self
        args._output_policy = self.output_policy
        return args

    def resolved_flat(self, args: argparse.Namespace) -> dict[str, Any]:
        """Return the manifest representation after precedence resolution."""

        defaults = dict(getattr(args, "_defaults", {}))
        keys = set(defaults)
        keys.update(self.values)
        keys.add("benchmark_task")
        resolved = {
            key: getattr(args, key)
            for key in sorted(keys)
            if hasattr(args, key)
        }
        resolved["metadata"] = dict(self.metadata)
        return resolved


def adapt_legacy_namespace_config(
    args: argparse.Namespace,
    raw: Mapping[str, Any],
) -> tuple[argparse.Namespace, SearchConfig]:
    """Compatibility adapter used by the legacy ``parse_args``/engine façade."""

    defaults = dict(getattr(args, "_defaults", {}))
    operational_keys = set(defaults) or {
        key for key in vars(args) if not key.startswith("_")
    }
    config = SearchConfig.from_legacy_flat(raw, operational_keys=operational_keys)
    return config.apply_to_namespace(args), config


__all__ = [
    "KNOWN_METADATA_KEYS",
    "OutputPolicy",
    "SearchConfig",
    "TraceLevel",
    "adapt_legacy_namespace_config",
]
