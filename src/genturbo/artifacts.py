"""Stable checksums used by experiment input and output validation."""
from __future__ import annotations
import hashlib
import stat
from pathlib import Path

class ArtifactError(RuntimeError):
    """Raised when an artifact operation cannot be completed safely."""


def sha256_file(path: Path, *, require_regular: bool = True) -> str:
    """Hash *path* and reject type changes or writes during the read."""

    path = Path(path)
    before = path.lstat()
    if require_regular and not stat.S_ISREG(before.st_mode):
        raise ArtifactError(f"not a regular file: {path}")
    if stat.S_ISLNK(before.st_mode):
        raise ArtifactError(f"symbolic links are not allowed: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or not stat.S_ISREG(after.st_mode):
        raise ArtifactError(f"file changed while it was being hashed: {path}")
    return digest.hexdigest()
