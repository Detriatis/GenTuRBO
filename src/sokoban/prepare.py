"""Prepare pinned, unfiltered Boxoban splits without solver/difficulty labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from sokoban.boards import MASK_ID, SEQ_LEN, TILES, parse_levels

REVISION = "78011d8c86ad423bfb2e1005012393b4c3c24409"
ARCHIVE_URL = f"https://codeload.github.com/google-deepmind/boxoban-levels/zip/{REVISION}"
ARCHIVE_SHA256 = "e2ca9eb749442c62c1f3b598df43b03b1849b299914ec9f6f6d6bf8a8ed693ed"
SPLIT_COUNTS = {"train": 900_000, "valid": 100_000, "test": 1_000}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_archive(destination: Path) -> Path:
    """Explicit download only; dataset construction never accesses the network."""
    if destination.exists():
        if sha256(destination) != ARCHIVE_SHA256:
            raise ValueError(f"Archive checksum mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response, temporary_path.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256(temporary_path) != ARCHIVE_SHA256:
            raise ValueError("Downloaded archive checksum mismatch; no data were prepared")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def prepare_archive(
    archive: Path,
    output: Path,
    *,
    splits: tuple[str, ...] = ("train", "valid"),
    expected_counts: dict[str, int] | None = None,
    expected_sha256: str = ARCHIVE_SHA256,
    revision: str = REVISION,
) -> dict:
    """Validate all records, then atomically publish compact arrays and provenance.

    No malformed board is silently dropped, and no split is merged or ranked.
    Source files are read from the ZIP, never extracted to arbitrary paths.
    Alternate counts/checksums are intended for small offline test fixtures.
    """
    counts = SPLIT_COUNTS if expected_counts is None else expected_counts
    if not splits or len(set(splits)) != len(splits) or any(split not in SPLIT_COUNTS for split in splits):
        raise ValueError("select distinct train, valid and/or test splits")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    actual_sha256 = sha256(archive)
    if actual_sha256 != expected_sha256:
        raise ValueError("Archive checksum mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source_repository": "https://github.com/google-deepmind/boxoban-levels",
        "source_revision": revision,
        "archive_sha256": actual_sha256,
        "subset": "unfiltered",
        "tiles": list(TILES),
        "mask_id": MASK_ID,
        "shape": [10, 10],
        "geometry": "closed wall border; disconnected floor regions allowed",
        "filtering": "none; structural errors abort preparation; no solver or rewards used",
        "ordering": "lexicographic source filename, then original level order within file",
        "splits": {},
    }
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".boxoban-prepare-") as temporary:
        staging = Path(temporary) / "prepared"
        staging.mkdir()
        with zipfile.ZipFile(archive) as source:
            prefix = f"boxoban-levels-{revision}/"
            for filename in ("LICENSE", "LICENSE.txt", "README.md"):
                (staging / f"SOURCE_{filename}").write_bytes(source.read(prefix + filename))
            for split in splits:
                members = sorted(
                    name for name in source.namelist()
                    if name.startswith(f"{prefix}unfiltered/{split}/") and name.endswith(".txt")
                )
                if not members:
                    raise ValueError(f"No source files for unfiltered/{split}")
                count = counts[split]
                array_path = staging / f"{split}.npy"
                array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.uint8, shape=(count, SEQ_LEN))
                position = 0
                records = []
                for index, member in enumerate(members):
                    raw = source.read(member)
                    boards = [board for _, board in parse_levels(raw.decode("ascii"), source=member)]
                    if not boards or position + len(boards) > count:
                        raise ValueError(f"Unexpected level count in {member}")
                    array[position : position + len(boards)] = np.stack(boards)
                    records.append({
                        "path": member.removeprefix(prefix),
                        "start": position,
                        "count": len(boards),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    })
                    position += len(boards)
                    if (index + 1) % 100 == 0:
                        print(f"{split}: validated {position:,}/{count:,} levels", flush=True)
                if position != count:
                    raise ValueError(f"{split}: expected {count} levels, found {position}")
                array.flush()
                del array
                manifest["splits"][split] = {
                    "count": count, "sha256": sha256(array_path), "sources": records,
                }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--download", action="store_true", help="download the pinned public archive")
    source.add_argument("--archive", type=Path, help="use an already downloaded pinned ZIP")
    parser.add_argument("--output", type=Path, default=Path("runs/data/boxoban"))
    parser.add_argument("--include-test", action="store_true", help="also prepare the untouched test split")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a new output directory")
    archive = args.archive
    if args.download:
        archive = download_archive(args.output.parent / "boxoban-source" / f"{REVISION}.zip")
    manifest = prepare_archive(
        archive, args.output, splits=("train", "valid", "test") if args.include_test else ("train", "valid"),
    )
    print(json.dumps({split: value["count"] for split, value in manifest["splits"].items()}, indent=2))
    print(f"Prepared {args.output}; data and provenance stay outside version control under runs/.")


if __name__ == "__main__":
    main()
