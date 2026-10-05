#!/usr/bin/env python3
"""Prepare GFP/AAV benchmark task subsets from full GGS DMS CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adapters.protein.gfp_aav_benchmark import (  # noqa: E402
    prepare_gfp_aav_benchmark,
    write_manifest_yaml,
    write_task_configs_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmarks/gfp_aav"))
    parser.add_argument("--task", action="append", default=None, help="Task name to prepare; may be repeated.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing stats/task files.")
    parser.add_argument(
        "--allow-size-mismatch",
        action="store_true",
        help="Write task files even if generated size differs from expected benchmark size.",
    )
    args = parser.parse_args(argv)

    args.root.mkdir(parents=True, exist_ok=True)
    write_manifest_yaml(args.root / "manifest.yaml")
    write_task_configs_json(args.root / "task_configs.json")
    summaries = prepare_gfp_aav_benchmark(
        root=args.root,
        task_names=args.task,
        overwrite=bool(args.overwrite),
        allow_size_mismatch=bool(args.allow_size_mismatch),
    )
    print("task,protein,difficulty,generated_train_size,expected_train_size,output_file")
    for row in summaries:
        print(
            f"{row['task_name']},{row['protein']},{row['difficulty']},"
            f"{row['generated_train_size']},{row['expected_train_size']},{row['output_file']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
