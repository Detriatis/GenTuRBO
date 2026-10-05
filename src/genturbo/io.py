"""Small, deterministic helpers for plain and gzip-compressed JSON Lines."""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast


@contextmanager
def open_text(path: Path | str, mode: str = "rt") -> Iterator[TextIO]:
    """Open text normally, or through deterministic gzip for ``.gz`` paths.

    Gzip writers omit the source filename and use a zero timestamp, so the same
    bytes always produce the same compressed file. Existing plain files remain
    fully supported.
    """

    target = Path(path)
    if "b" in mode or "t" not in mode:
        raise ValueError("open_text requires an explicit text mode such as 'rt' or 'wt'")
    if target.suffix != ".gz":
        with target.open(mode, encoding="utf-8", newline="") as handle:
            yield cast(TextIO, handle)
        return

    binary_mode = mode.replace("t", "b")
    if mode.startswith("r"):
        with gzip.open(target, mode, encoding="utf-8", newline="") as handle:
            yield cast(TextIO, handle)
        return

    if not mode.startswith(("w", "x")):
        raise ValueError("deterministic gzip output supports only 'wt' and 'xt'")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open(binary_mode) as raw_handle:
        raw = cast(BinaryIO, raw_handle)
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                yield handle


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield object records from plain ``.jsonl`` or ``.jsonl.gz`` input."""

    source = Path(path)
    with open_text(source, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {source} at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object in {source} at line {line_number}")
            yield record


def write_jsonl(path: Path | str, records: Iterator[Mapping[str, Any]]) -> None:
    """Write records with stable key ordering and deterministic compression."""

    with open_text(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


__all__ = ["iter_jsonl", "open_text", "write_jsonl"]
