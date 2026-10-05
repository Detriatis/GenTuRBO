"""Shared deterministic, atomic serialization helpers."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and reject non-standard numbers."""

    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except ValueError as exc:
            raise ValueError("CSV container values must be finite") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("CSV values must be finite")
    return value


def csv_bytes(
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    preferred_fieldnames: Sequence[str] | None = None,
) -> bytes:
    """Serialize a rectangular table with stable columns and newlines.

    Explicit ``fieldnames`` define the complete schema. Alternatively,
    ``preferred_fieldnames`` are emitted first and any remaining fields are
    appended alphabetically. Container-valued cells use compact canonical JSON
    so their representation is portable and deterministic.
    """

    if fieldnames is not None and preferred_fieldnames is not None:
        raise ValueError("fieldnames and preferred_fieldnames are mutually exclusive")
    if not rows and fieldnames is None and preferred_fieldnames is None:
        return b""
    if fieldnames is not None:
        columns = list(fieldnames)
    elif preferred_fieldnames is not None:
        columns = list(preferred_fieldnames)
        columns.extend(sorted({key for row in rows for key in row} - set(columns)))
    else:
        columns = list(rows[0].keys())
    if len(columns) != len(set(columns)):
        raise ValueError("CSV field names must be unique")
    unknown = sorted({key for row in rows for key in row} - set(columns))
    if unknown:
        raise ValueError(f"CSV rows contain undeclared fields: {unknown}")

    normalized = [{field: _csv_cell(row.get(field)) for field in columns} for row in rows]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    return buffer.getvalue().encode("utf-8")


def atomic_write(path: Path | str, data: bytes) -> None:
    """Atomically replace one file after fully writing its temporary sibling."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path | str, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def atomic_save_figure(
    figure: Any,
    path: Path | str,
    *,
    format: str | None = None,
    **savefig_options: Any,
) -> None:
    """Atomically save a Matplotlib-like figure without duplicating temp-file logic."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = str(format or destination.suffix.lstrip("."))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=f".{suffix}",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format=suffix, **savefig_options)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "atomic_save_figure",
    "atomic_write",
    "atomic_write_json",
    "canonical_json_bytes",
    "csv_bytes",
]
