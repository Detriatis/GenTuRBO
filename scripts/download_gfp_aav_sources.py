#!/usr/bin/env python3
"""Download GFP/AAV GGS source files into the benchmark filesystem."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adapters.protein.gfp_aav_benchmark import (  # noqa: E402
    MANIFEST_ENTRIES,
    sha256_file,
    write_manifest_yaml,
    write_task_configs_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmarks/gfp_aav"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    args.root.mkdir(parents=True, exist_ok=True)
    write_manifest_yaml(args.root / "manifest.yaml")
    write_task_configs_json(args.root / "task_configs.json")
    for entry in MANIFEST_ENTRIES.values():
        if entry.upstream_url is None:
            continue
        out_path = args.root / entry.local_path
        if out_path.exists() and not args.overwrite:
            if entry.checksum is not None:
                actual = sha256_file(out_path)
                if actual != entry.checksum:
                    raise RuntimeError(
                        f"checksum mismatch for existing {out_path}: "
                        f"expected {entry.checksum}, got {actual}; rerun with --overwrite"
                    )
            print(f"exists: {out_path}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {entry.upstream_url} -> {out_path}")
        temporary_path = out_path.with_name(f".{out_path.name}.download")
        temporary_path.unlink(missing_ok=True)
        try:
            urlretrieve(entry.upstream_url, temporary_path)
            if entry.checksum is not None:
                actual = sha256_file(temporary_path)
                if actual != entry.checksum:
                    raise RuntimeError(
                        f"checksum mismatch for {entry.upstream_url}: "
                        f"expected {entry.checksum}, got {actual}"
                    )
            temporary_path.replace(out_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
