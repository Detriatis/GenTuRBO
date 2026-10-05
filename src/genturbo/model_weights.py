"""Download and verify the exact public protein weights used by the experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.request import urlopen


def model_manifest() -> list[dict[str, Any]]:
    return json.loads(Path(__file__).with_name("model_weights.json").read_text())["files"]


def verify_weight(path: Path, specification: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.name}; run setup with --protein --prefetch-models")
    if path.stat().st_size != specification["bytes"]:
        raise ValueError(f"Model weight size mismatch: {path}")
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    parts = []
    # The official ESM S3 object uses 8 MiB multipart-upload pieces. Its ETag
    # is MD5(concatenated binary part MD5s), followed by the part count.
    part_size = specification.get("multipart_chunk_bytes", 8 * 1024 * 1024)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(part_size), b""):
            sha.update(block)
            md5.update(block)
            parts.append(hashlib.md5(block).digest())
    expected = specification["checksum"]
    algorithm = expected["algorithm"]
    if algorithm == "sha256":
        actual = sha.hexdigest()
    elif algorithm == "md5":
        actual = md5.hexdigest()
    elif algorithm == "s3-multipart-md5":
        actual = hashlib.md5(b"".join(parts)).hexdigest() + f"-{len(parts)}"
    else:
        raise ValueError(f"Unknown model checksum algorithm: {algorithm}")
    if actual != expected["value"]:
        raise ValueError(f"Model weight checksum mismatch: {path}")
    if specification.get("sha256") and sha.hexdigest() != specification["sha256"]:
        raise ValueError(f"Model weight SHA256 checksum mismatch: {path}")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha.hexdigest(),
            "verified_checksum": expected}


def prepare_weights(*, download: bool = False, cache: Path | None = None,
                    names: Sequence[str] | None = None) -> dict[str, Any]:
    if cache is None:
        import torch
        cache = Path(torch.hub.get_dir()) / "checkpoints"
    selected = [item for item in model_manifest() if not names or item["name"] in names]
    if names and set(names) - {item["name"] for item in selected}:
        raise ValueError("Unknown protein model weight name")
    receipts = []
    for item in selected:
        target = cache / item["filename"]
        if not target.exists() and download:
            cache.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".download", dir=cache, delete=False) as stream:
                    temporary = Path(stream.name)
                    with urlopen(item["url"], timeout=120) as response:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            stream.write(block)
                verify_weight(temporary, item)
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        receipts.append(verify_weight(target, item))
    result = {"ok": True, "files": receipts}
    if download:
        (cache / "genturbo-protein-weights.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache", type=Path, help="Torch hub/checkpoints directory")
    parser.add_argument("--name", action="append", choices=[item["name"] for item in model_manifest()])
    args = parser.parse_args(argv)
    print(json.dumps(prepare_weights(download=args.download, cache=args.cache, names=args.name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
