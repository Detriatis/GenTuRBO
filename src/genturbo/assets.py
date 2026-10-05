"""Verify and import the separately distributed, checksum-pinned experiment inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

MANIFEST_NAME = "GENTURBO_ARTIFACTS.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(manifest: Path | None = None) -> dict[str, Any]:
    path = manifest or Path(__file__).with_name("artifact_manifest.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or not isinstance(result.get("files"), list):
        raise ValueError("Unsupported artifact manifest")
    seen = set()
    for entry in result["files"]:
        name = entry["path"]
        parts = PurePosixPath(name)
        if not name or parts.is_absolute() or ".." in parts.parts or "\\" in name or name in seen:
            raise ValueError(f"Invalid or duplicate artifact path: {name!r}")
        seen.add(name)
    return result


def verify_assets(root: Path, *, groups: Sequence[str] | None = None,
                  manifest: Path | None = None) -> dict[str, Any]:
    """Check all requested files; return missing/mismatched names without changing them."""
    declaration = load_manifest(manifest)
    entries = [entry for entry in declaration["files"] if not groups or entry["group"] in groups]
    if groups and set(groups) - {entry["group"] for entry in declaration["files"]}:
        raise ValueError("Unknown artifact group")
    base = Path(root).resolve()
    missing, mismatched = [], []
    for entry in entries:
        path = base / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
        elif not path.resolve().is_relative_to(base) or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            mismatched.append(entry["path"])
    return {"ok": not missing and not mismatched, "checked": len(entries),
            "missing": missing, "mismatched": mismatched}


def import_bundle(bundle: Path, root: Path, *, manifest: Path | None = None) -> dict[str, Any]:
    """Validate every archive member before importing; never replace a different file."""
    declaration = load_manifest(manifest)
    expected = {entry["path"]: entry for entry in declaration["files"]}
    base = Path(root).resolve()
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".genturbo-import-", dir=base.parent) as temp:
        staging = Path(temp)
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(set(names)) != len(names) or set(names) != set(expected) | {MANIFEST_NAME}:
                raise ValueError("Bundle members do not match the pinned artifact manifest")
            for member in members:
                if not member.isfile():
                    raise ValueError(f"Only ordinary files are accepted: {member.name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read bundle member: {member.name}")
                if member.name == MANIFEST_NAME:
                    if member.size > 20_000_000 or json.load(stream) != declaration:
                        raise ValueError("Bundle manifest differs from this release")
                    continue
                entry = expected[member.name]
                if member.size != entry["bytes"]:
                    raise ValueError(f"Artifact size mismatch: {member.name}")
                target = staging / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as destination:
                    shutil.copyfileobj(stream, destination)
                if sha256_file(target) != entry["sha256"]:
                    raise ValueError(f"Artifact checksum mismatch: {member.name}")
        # Complete preflight before any existing destination is touched.
        for name, entry in expected.items():
            target = base / name
            if not target.resolve().is_relative_to(base):
                raise ValueError(f"Artifact destination escapes root: {name}")
            if target.exists() and (not target.is_file() or sha256_file(target) != entry["sha256"]):
                raise FileExistsError(f"Refusing to replace different artifact: {name}")
        imported = 0
        for name in expected:
            target = base / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staging / name), target)
                imported += 1
    return {"ok": True, "checked": len(expected), "imported": imported}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    imported = commands.add_parser("import", help="Verify and unpack the companion bundle")
    imported.add_argument("bundle", type=Path)
    imported.add_argument("--root", type=Path, default=Path("artifacts"))
    verify = commands.add_parser("verify", help="Check installed inputs against this release")
    verify.add_argument("--root", type=Path, default=Path("artifacts"))
    verify.add_argument("--group", action="append", dest="groups")
    args = parser.parse_args(argv)
    result = import_bundle(args.bundle, args.root) if args.command == "import" else verify_assets(args.root, groups=args.groups)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
