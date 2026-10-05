"""Reproducibility manifests for interpolant search runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genturbo.serialization import atomic_write_json

IMPLEMENTATION_RECORD_ENV = "GENTURBO_IMPLEMENTATION_RECORD"
_SCOPED_SUFFIXES = frozenset({".json", ".py", ".txt", ".toml", ".yaml", ".yml"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017 - LaMBO uses Python 3.10


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scoped_path_sha256(path: str | Path) -> str:
    """Hash one source file or a deterministic tree of source/config files."""

    target = Path(path)
    if target.is_symlink():
        raise ValueError(f"implementation scope cannot contain a symlink: {target}")
    if target.is_file():
        return sha256_file(target)
    if not target.is_dir():
        raise ValueError(f"implementation scope path does not exist: {target}")
    files = sorted(
        candidate
        for candidate in target.rglob("*")
        if candidate.is_file()
        and not candidate.is_symlink()
        and candidate.suffix in _SCOPED_SUFFIXES
        and "__pycache__" not in candidate.parts
    )
    if not files:
        raise ValueError(f"implementation scope directory is empty: {target}")
    digest = hashlib.sha256()
    for candidate in files:
        digest.update(candidate.relative_to(target).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def implementation_record_identity(
    repo_root: str | Path,
    record_path: str | Path,
) -> tuple[str, str]:
    """Validate a versioned experiment scope and return its path and digest."""

    root = Path(repo_root).resolve()
    raw_path = Path(record_path).expanduser()
    path = (raw_path if raw_path.is_absolute() else root / raw_path).resolve()
    try:
        display_path = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"implementation record escapes repository: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"implementation record is missing or non-regular: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read implementation record {path}: {exc}") from exc
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError(f"unsupported implementation record: {path}")
    raw_scope = record.get("paths")
    if not isinstance(raw_scope, dict) or not raw_scope:
        raise ValueError(f"implementation record has no scoped paths: {path}")
    for raw_name, expected in raw_scope.items():
        if not isinstance(raw_name, str) or not raw_name or Path(raw_name).is_absolute():
            raise ValueError(f"invalid implementation scope path: {raw_name!r}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"invalid implementation scope digest for {raw_name!r}")
        candidate = (root / raw_name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"implementation scope escapes repository: {raw_name}") from exc
        actual = scoped_path_sha256(candidate)
        if actual != expected:
            raise ValueError(
                f"implementation scope mismatch for {raw_name}: expected {expected}, got {actual}"
            )
    return display_path, sha256_file(path)


def _git_output(repo_root: Path, args: Sequence[str]) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def repository_identity(repo_root: str | Path = ".") -> tuple[str | None, str | None]:
    """Return ``(HEAD, dirty-worktree SHA-256)`` without mutating Git.

    The identity includes the tracked binary diff plus names and content hashes
    for non-ignored, untracked files outside ``runs/``.  Raw artifacts are
    captured separately as input/output hashes and must not make this operation
    scan tens of gigabytes.
    """

    root = Path(repo_root).resolve()
    head_raw = _git_output(root, ["rev-parse", "HEAD"])
    diff_raw = _git_output(root, ["diff", "--binary", "HEAD", "--", "."])
    head = head_raw.decode().strip() if head_raw else None
    if diff_raw is None:
        return head, None
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(diff_raw)
    has_changes = bool(diff_raw)
    untracked_raw = _git_output(root, ["ls-files", "--others", "--exclude-standard", "-z"])
    if untracked_raw is not None:
        for raw_name in sorted(name for name in untracked_raw.split(b"\0") if name):
            relative = raw_name.decode("utf-8", errors="surrogateescape")
            if relative == "runs" or relative.startswith("runs/"):
                continue
            has_changes = True
            path = root / relative
            digest.update(b"untracked\0")
            digest.update(raw_name)
            digest.update(b"\0")
            if path.is_file() and not path.is_symlink():
                digest.update(sha256_file(path).encode("ascii"))
            else:
                digest.update(b"non-regular")
    dirty_hash = digest.hexdigest() if has_changes else None
    return head, dirty_hash


@dataclass(slots=True)
class RunManifest:
    """Stable wire model for one search execution."""

    resolved_config: dict[str, Any]
    command: list[str]
    seed: int
    git_sha: str | None
    dirty_diff_hash: str | None
    dependency_profile: str
    input_hashes: dict[str, str]
    implementation_record_path: str | None = None
    implementation_record_sha256: str | None = None
    experiment_status: str = "canonical"
    output_hashes: dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=_utc_now)
    completed_at: str | None = None
    archive_uri: str | None = None
    execution_context: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    trace_records_written: int = 0
    trace_truncated: bool = False
    error: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "status": self.status,
            "experiment_status": self.experiment_status,
            "resolved_config": self.resolved_config,
            "command": self.command,
            "seed": int(self.seed),
            "git_sha": self.git_sha,
            "dirty_diff_hash": self.dirty_diff_hash,
            "dependency_profile": self.dependency_profile,
            "input_hashes": self.input_hashes,
            "implementation_record_path": self.implementation_record_path,
            "implementation_record_sha256": self.implementation_record_sha256,
            "output_hashes": self.output_hashes,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "archive_uri": self.archive_uri,
            "execution_context": self.execution_context,
            "trace_records_written": int(self.trace_records_written),
            "trace_truncated": bool(self.trace_truncated),
            "error": self.error,
        }

    def write(self, path: str | Path) -> None:
        manifest_path = Path(path).expanduser()
        atomic_write_json(manifest_path, self.to_dict())


class RunManifestRecorder:
    """Context manager that finalizes output hashes on every exit path."""

    def __init__(
        self,
        manifest: RunManifest,
        path: str | Path,
        *,
        output_paths: Iterable[str | Path] = (),
        trace_writer: Any = None,
        only_changed_outputs: bool = False,
    ) -> None:
        self.manifest = manifest
        self.path = Path(path).expanduser()
        self.output_paths = [Path(item).expanduser() for item in output_paths]
        self.trace_writer = trace_writer
        self.only_changed_outputs = bool(only_changed_outputs)
        self._output_baselines: dict[Path, tuple[int, int, int, int] | None] = {}

    @staticmethod
    def _signature(path: Path) -> tuple[int, int, int, int] | None:
        try:
            metadata = path.stat()
        except OSError:
            return None
        if not path.is_file() or path.is_symlink():
            return None
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )

    def register_output(self, path: str | Path) -> None:
        output = Path(path).expanduser()
        if output not in self.output_paths:
            self.output_paths.append(output)
        self._output_baselines.setdefault(output, self._signature(output))

    def register_outputs(self, paths: Iterable[str | Path]) -> None:
        for path in paths:
            self.register_output(path)

    def __enter__(self) -> RunManifestRecorder:
        self.register_outputs(self.output_paths)
        self.manifest.write(self.path)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.manifest.completed_at = _utc_now()
        self.manifest.status = "failed" if exc is not None else "completed"
        self.manifest.error = (
            f"{type(exc).__name__}: {exc}" if exc is not None else None
        )
        writer = self.trace_writer
        if writer is not None:
            self.manifest.trace_records_written = int(getattr(writer, "records_written", 0))
            self.manifest.trace_truncated = bool(getattr(writer, "truncated", False))
        hashes: dict[str, str] = {}
        for output_path in self.output_paths:
            try:
                current_signature = self._signature(output_path)
                baseline = self._output_baselines.get(output_path)
                if (
                    current_signature is not None
                    and (not self.only_changed_outputs or current_signature != baseline)
                    and output_path.resolve() != self.path.resolve()
                ):
                    hashes[str(output_path)] = sha256_file(output_path)
            except OSError:
                continue
        self.manifest.output_hashes = hashes
        self.manifest.write(self.path)


def build_run_manifest(
    *,
    resolved_config: Mapping[str, Any],
    command: Sequence[str],
    seed: int,
    input_paths: Mapping[str, str | Path | None],
    repo_root: str | Path = ".",
) -> RunManifest:
    root = Path(repo_root).resolve()
    implementation_path = None
    implementation_sha256 = None
    if raw_record := os.environ.get(IMPLEMENTATION_RECORD_ENV):
        implementation_path, implementation_sha256 = implementation_record_identity(
            root,
            raw_record,
        )
    git_sha, dirty_diff_hash = repository_identity(root)
    raw_metadata = resolved_config.get("metadata")
    experiment_status = (
        raw_metadata.get("experiment_status", "canonical")
        if isinstance(raw_metadata, Mapping)
        else "canonical"
    )
    if experiment_status not in {"canonical", "smoke"}:
        raise ValueError(
            "metadata.experiment_status must be either 'canonical' or 'smoke'"
        )
    input_hashes: dict[str, str] = {}
    for name, raw_path in input_paths.items():
        if raw_path is None:
            continue
        path = Path(raw_path).expanduser()
        if path.is_file():
            input_hashes[str(name)] = sha256_file(path)
    execution_context: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if devices := os.environ.get("CUDA_VISIBLE_DEVICES"):
        execution_context["cuda_visible_devices"] = devices
    return RunManifest(
        resolved_config=dict(resolved_config),
        command=list(command),
        seed=int(seed),
        git_sha=git_sha,
        dirty_diff_hash=dirty_diff_hash,
        dependency_profile=os.environ.get("GENTURBO_DEPENDENCY_PROFILE", "default"),
        input_hashes=input_hashes,
        implementation_record_path=implementation_path,
        implementation_record_sha256=implementation_sha256,
        experiment_status=str(experiment_status),
        archive_uri=(
            os.environ.get("GENTURBO_ARCHIVE_URI")
            or os.environ.get("GENTURBO_ARTIFACT_ROOT")
        ),
        execution_context=execution_context,
    )


__all__ = [
    "RunManifest",
    "RunManifestRecorder",
    "IMPLEMENTATION_RECORD_ENV",
    "build_run_manifest",
    "implementation_record_identity",
    "repository_identity",
    "scoped_path_sha256",
    "sha256_file",
]
