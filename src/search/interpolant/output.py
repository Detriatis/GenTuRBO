"""JSONL output primitives shared by the search engine and compatibility API."""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast


class _OwnedGzipFile(gzip.GzipFile):
    """Gzip stream that closes the raw file object it owns."""

    def close(self) -> None:
        raw = cast(Any, self.fileobj)
        try:
            super().close()
        finally:
            if raw is not None and callable(getattr(raw, "close", None)):
                raw.close()


class LimitedJsonlWriter:
    """A text-writer façade that stops after a fixed number of JSONL records.

    Search writers emit one complete JSON object per ``write`` call.  Keeping the
    limit at this boundary preserves record ordering and avoids allocating a
    second in-memory trace buffer.
    """

    def __init__(self, stream: TextIO, record_limit: int) -> None:
        if record_limit <= 0:
            raise ValueError("record_limit must be > 0")
        self._stream = stream
        self.record_limit = int(record_limit)
        self.records_written = 0
        self.truncated = False

    @property
    def name(self) -> str:
        return str(getattr(self._stream, "name", "<jsonl>"))

    @property
    def closed(self) -> bool:
        return bool(self._stream.closed)

    def write(self, value: str) -> int:
        record_count = value.count("\n")
        if record_count == 0 and value:
            record_count = 1
        if self.records_written + record_count > self.record_limit:
            remaining = self.record_limit - self.records_written
            if remaining <= 0:
                self.truncated = True
                return len(value)
            pieces = value.splitlines(keepends=True)
            accepted = "".join(pieces[:remaining])
            self._stream.write(accepted)
            self.records_written += remaining
            self.truncated = True
            return len(value)
        result = self._stream.write(value)
        self.records_written += record_count
        return result

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> LimitedJsonlWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def open_jsonl(
    path: str | Path,
    mode: str = "rt",
    *,
    encoding: str = "utf-8",
    record_limit: int | None = None,
) -> TextIO | LimitedJsonlWriter:
    """Open plain or gzip JSONL based on the final path suffix."""

    jsonl_path = Path(path).expanduser()
    text_mode = mode if "t" in mode else f"{mode}t"
    writable = any(flag in text_mode for flag in ("w", "a", "x"))
    if writable:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        jsonl_path = resolve_jsonl_path(jsonl_path)
    stream: TextIO
    if jsonl_path.suffix == ".gz":
        if writable:
            binary_mode = text_mode.replace("t", "b")
            raw_handle = jsonl_path.open(binary_mode)
            raw = cast(BinaryIO, raw_handle)
            compressed = _OwnedGzipFile(
                filename="",
                mode=binary_mode,
                fileobj=raw,
                mtime=0,
            )
            stream = cast(TextIO, io.TextIOWrapper(compressed, encoding=encoding, newline=""))
        else:
            stream = cast(TextIO, gzip.open(jsonl_path, text_mode, encoding=encoding))
    else:
        stream = cast(TextIO, jsonl_path.open(text_mode, encoding=encoding))
    if record_limit is not None:
        if not any(flag in text_mode for flag in ("w", "a", "x")):
            stream.close()
            raise ValueError("record_limit is only valid for writable JSONL streams")
        return LimitedJsonlWriter(stream, record_limit)
    return stream


def resolve_jsonl_path(path: str | Path) -> Path:
    """Resolve a requested JSONL path with plain/gzip compatibility fallback."""

    value = Path(path).expanduser()
    if value.exists():
        return value
    if value.suffix == ".gz":
        plain = value.with_suffix("")
        return plain if plain.exists() else value
    compressed = compressed_jsonl_path(value)
    return compressed if compressed.exists() else value


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read historical plain JSONL and new ``.jsonl.gz`` files alike."""

    with open_jsonl(path, "rt") as opened:
        stream = cast(TextIO, opened)
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record {line_number} in {path} is not an object")
            yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Materialize JSONL records; useful for small archives and tests."""

    return list(iter_jsonl(path))


def compressed_jsonl_path(path: str | Path) -> Path:
    """Return the gzip form of a JSONL path without double-appending suffixes."""

    value = Path(path)
    if value.suffix == ".gz":
        return value
    return Path(f"{value}.gz")


__all__ = [
    "LimitedJsonlWriter",
    "compressed_jsonl_path",
    "iter_jsonl",
    "open_jsonl",
    "read_jsonl",
    "resolve_jsonl_path",
]
