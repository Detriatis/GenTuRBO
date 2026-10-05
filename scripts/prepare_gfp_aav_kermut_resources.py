#!/usr/bin/env python3
"""Pack official Kermut arrays into reusable GFP and AAV resources.

The official Kermut archive stores full-construct ProteinMPNN probabilities and
C-alpha coordinates by ProteinGym DMS id.  The GGS benchmark uses a shorter
sequence for each task, so this script applies the two audited mappings and
stores the omitted fixed sequence as ESM embedding context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PROTEINGYM_REVISION = "144fe22b07dfaeec2b366f2346203a9838a55b4c"
PROTEINGYM_REFERENCE_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/"
    f"{PROTEINGYM_REVISION}/reference_files/DMS_substitutions.csv"
)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_name: str
    protein: str
    dms_id: str
    full_length: int
    start: int
    stop: int
    region_mutated: str
    target_sha256: str
    wild_type_sha256: str
    output_name: str

    @property
    def task_length(self) -> int:
        return self.stop - self.start


# Python slices are zero-based and stop-exclusive. ProteinGym's region_mutated
# values are one-based and inclusive; GFP's GGS sequence additionally omits the
# full construct's initiator methionine.
TASK_SPECS: dict[str, TaskSpec] = {
    "gfp": TaskSpec(
        task_name="gfp",
        protein="GFP",
        dms_id="GFP_AEQVI_Sarkisyan_2016",
        full_length=238,
        start=1,
        stop=238,
        region_mutated="3-237",
        target_sha256="9be57dbaa8e6d8b324cd1a84684901808c38b3b84f940c08e00a0323f4215e9f",
        wild_type_sha256="dd295bd4496d5e13da909a2c59492f00f923f7957a774feeaafaf8c5c235828c",
        output_name="gfp_kermut.npz",
    ),
    "aav": TaskSpec(
        task_name="aav",
        protein="AAV",
        dms_id="CAPSD_AAV2S_Sinai_2021",
        full_length=735,
        start=560,
        stop=588,
        region_mutated="561-588",
        target_sha256="3544f4fb2e91766d29fe30811a367b3c8c62676eda35e8e111c3f586d75fa82d",
        wild_type_sha256="7f7ff0b9c7e47da3ee29517e3b95c6ab4e884e8549f54c8959ab37b8934745d4",
        output_name="aav_kermut.npz",
    ),
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def resolve_kermut_data_dir(path: Path) -> Path:
    """Accept either the extracted archive root or its ``data`` directory."""
    path = path.expanduser().resolve()
    candidates = (path, path / "data")
    for candidate in candidates:
        if (candidate / "conditional_probs/ProteinMPNN").is_dir() and (candidate / "structures/coords").is_dir():
            return candidate
    expected = [str(candidate / "conditional_probs/ProteinMPNN") for candidate in candidates]
    raise FileNotFoundError(f"Could not find the extracted Kermut data layout under {path}; expected one of {expected}")


def read_reference_rows(path: Path, dms_ids: set[str]) -> dict[str, Mapping[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"ProteinGym reference file not found at {path}. Download it from {PROTEINGYM_REFERENCE_URL}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"DMS_id", "target_seq", "seq_len", "region_mutated"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"ProteinGym reference file is missing columns {sorted(missing)}")
        rows: dict[str, Mapping[str, str]] = {}
        for row in reader:
            dms_id = row["DMS_id"]
            if dms_id not in dms_ids:
                continue
            if dms_id in rows:
                raise ValueError(f"ProteinGym reference file contains duplicate DMS id {dms_id}")
            rows[dms_id] = row
    missing_ids = dms_ids.difference(rows)
    if missing_ids:
        raise ValueError(f"ProteinGym reference file is missing DMS ids {sorted(missing_ids)}")
    return rows


def _load_numeric_array(path: Path, expected_shape: tuple[int, int], *, probabilities: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Official Kermut array not found: {path}")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Kermut array must be numeric: {path}")
    if array.shape != expected_shape:
        raise ValueError(f"Kermut array {path} has shape {array.shape}; expected {expected_shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Kermut array contains non-finite values: {path}")
    if probabilities and (np.any(array <= 0) or np.any(array > 1)):
        raise ValueError(f"Kermut probability array must contain values in (0, 1]: {path}")
    return array


def _validate_reference(spec: TaskSpec, row: Mapping[str, str]) -> tuple[str, str, str]:
    target = row["target_seq"]
    if target != target.strip().upper() or set(target).difference(ALPHABET):
        raise ValueError(f"{spec.dms_id} target_seq is not an uppercase canonical protein sequence")
    if int(row["seq_len"]) != spec.full_length or len(target) != spec.full_length:
        raise ValueError(f"{spec.dms_id} target length does not match audited length {spec.full_length}")
    if row["region_mutated"] != spec.region_mutated:
        raise ValueError(f"{spec.dms_id} region_mutated is {row['region_mutated']!r}; expected {spec.region_mutated!r}")
    if _sha256_text(target) != spec.target_sha256:
        raise ValueError(f"{spec.dms_id} target sequence does not match the audited ProteinGym reference")

    prefix = target[: spec.start]
    wild_type = target[spec.start : spec.stop]
    suffix = target[spec.stop :]
    if _sha256_text(wild_type) != spec.wild_type_sha256:
        raise ValueError(f"{spec.dms_id} task slice does not match the audited GGS wild type")
    if prefix + wild_type + suffix != target:
        raise AssertionError("Internal slice reconstruction failed")
    return prefix, wild_type, suffix


def _validate_benchmark_sequences(path: Path, wild_type: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"GGS benchmark source not found: {path}")
    saw_wild_type = False
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sequence" not in (reader.fieldnames or ()):
            raise ValueError(f"GGS benchmark source is missing the sequence column: {path}")
        for row in reader:
            row_count += 1
            sequence = row["sequence"]
            if len(sequence) != len(wild_type):
                raise ValueError(f"GGS benchmark source contains a non-{len(wild_type)}-residue sequence: {path}")
            if set(sequence).difference(ALPHABET):
                raise ValueError(f"GGS benchmark source contains a non-canonical sequence: {path}")
            saw_wild_type = saw_wild_type or sequence == wild_type
    if row_count == 0:
        raise ValueError(f"GGS benchmark source is empty: {path}")
    if not saw_wild_type:
        raise ValueError(f"Audited wild type is absent from GGS benchmark source: {path}")


def build_resource(
    *,
    spec: TaskSpec,
    reference_row: Mapping[str, str],
    kermut_data_dir: Path,
    benchmark_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    prefix, wild_type, suffix = _validate_reference(spec, reference_row)
    benchmark_path = benchmark_root / "raw" / spec.protein / "ground_truth.csv"
    _validate_benchmark_sequences(benchmark_path, wild_type)

    probability_path = kermut_data_dir / "conditional_probs/ProteinMPNN" / f"{spec.dms_id}.npy"
    coords_path = kermut_data_dir / "structures/coords" / f"{spec.dms_id}.npy"
    probabilities = _load_numeric_array(
        probability_path,
        (spec.full_length, 20),
        probabilities=True,
    )
    coords = _load_numeric_array(coords_path, (spec.full_length, 3), probabilities=False)

    task_probabilities = np.ascontiguousarray(probabilities[spec.start : spec.stop])
    task_coords = np.ascontiguousarray(coords[spec.start : spec.stop])
    if task_probabilities.shape != (spec.task_length, 20) or task_coords.shape != (spec.task_length, 3):
        raise AssertionError("Internal resource slice produced an unexpected shape")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / spec.output_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing resource: {output_path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{spec.output_name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                wild_type=np.asarray(wild_type),
                alphabet=np.asarray(ALPHABET),
                conditional_probs=task_probabilities,
                coords=task_coords,
                embedding_prefix=np.asarray(prefix),
                embedding_suffix=np.asarray(suffix),
            )
            handle.flush()
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(
        f"{spec.task_name}: {probability_path}[{spec.start}:{spec.stop}] "
        f"{probabilities.shape}->{task_probabilities.shape}; {coords_path}[{spec.start}:{spec.stop}] "
        f"{coords.shape}->{task_coords.shape}; ESM flanks {len(prefix)}+{spec.task_length}+{len(suffix)} "
        f"-> {output_path}"
    )
    return output_path


def prepare_resources(
    *,
    task_names: Sequence[str],
    kermut_root: Path,
    reference_file: Path | None,
    benchmark_root: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    specs = [TASK_SPECS[name] for name in task_names]
    data_dir = resolve_kermut_data_dir(kermut_root)
    resolved_reference = reference_file or data_dir / "DMS_substitutions.csv"
    rows = read_reference_rows(resolved_reference, {spec.dms_id for spec in specs})
    return [
        build_resource(
            spec=spec,
            reference_row=rows[spec.dms_id],
            kermut_data_dir=data_dir,
            benchmark_root=benchmark_root,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        for spec in specs
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kermut-root",
        type=Path,
        required=True,
        help="Extracted kermut_data.zip root, or its data directory.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=None,
        help="Official ProteinGym DMS_substitutions.csv; defaults to <Kermut data>/DMS_substitutions.csv.",
    )
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmarks/gfp_aav"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        action="append",
        choices=tuple(TASK_SPECS),
        dest="tasks",
        help="Protein resource to build; repeat to select both. Defaults to GFP and AAV.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    prepare_resources(
        task_names=args.tasks or tuple(TASK_SPECS),
        kermut_root=args.kermut_root,
        reference_file=args.reference_file,
        benchmark_root=args.benchmark_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
