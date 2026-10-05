"""GFP/AAV benchmark data layout and deterministic task preparation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_GFP_AAV_ROOT = Path("benchmarks/gfp_aav")


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    logical_name: str
    protein: str
    upstream_url: str | None
    local_path: str
    expected_format: str
    required_columns: tuple[str, ...] = ()
    artifact_type: str = "raw"
    generated_from: str | None = None
    generation: str | None = None
    checksum: str | None = None


@dataclass(frozen=True, slots=True)
class GFPAVTaskConfig:
    task_name: str
    protein: str
    difficulty: str
    raw_dataset: str
    output_file: str
    metadata_file: str
    seq_len: int
    filter_percentile: tuple[float, float]
    min_mutant_dist: int
    top_quantile: float
    expected_train_size: int | None
    expected_median_normalized_fitness: float
    expected_train_fitness_range: tuple[float, float]


GGS_REVISION = "033c543c44aa76655e4868bb71e5713c962401ea"
_GGS_RAW_ROOT = f"https://raw.githubusercontent.com/kirjner/GGS/{GGS_REVISION}"

RAW_URLS = {
    "gfp_ground_truth": f"{_GGS_RAW_ROOT}/data/GFP/ground_truth.csv",
    "aav_ground_truth": f"{_GGS_RAW_ROOT}/data/AAV/ground_truth.csv",
    "gfp_oracle_checkpoint": f"{_GGS_RAW_ROOT}/ckpt/GFP/mutations_0/percentile_0.0_1.0/cnn_oracle.ckpt",
    "gfp_oracle_config": f"{_GGS_RAW_ROOT}/ckpt/GFP/mutations_0/percentile_0.0_1.0/config.yaml",
    "aav_oracle_checkpoint": f"{_GGS_RAW_ROOT}/ckpt/AAV/mutations_0/percentile_0.0_1.0/cnn_oracle.ckpt",
    "aav_oracle_config": f"{_GGS_RAW_ROOT}/ckpt/AAV/mutations_0/percentile_0.0_1.0/config.yaml",
}


MANIFEST_ENTRIES: dict[str, ManifestEntry] = {
    "gfp_ground_truth": ManifestEntry(
        logical_name="gfp_ground_truth",
        protein="GFP",
        upstream_url=RAW_URLS["gfp_ground_truth"],
        local_path="raw/GFP/ground_truth.csv",
        expected_format="csv",
        required_columns=("sequence", "score"),
        artifact_type="raw",
        checksum="a29bb2ffe16591ff31dcddd4bec03171f0acf42a2b7b5b65b2eae3a1efec162f",
    ),
    "aav_ground_truth": ManifestEntry(
        logical_name="aav_ground_truth",
        protein="AAV",
        upstream_url=RAW_URLS["aav_ground_truth"],
        local_path="raw/AAV/ground_truth.csv",
        expected_format="csv",
        required_columns=("sequence", "score"),
        artifact_type="raw",
        checksum="cca65e1c1c0ed54b66a77a34ce531f9f907341a3596315568aba785e34a8c9ca",
    ),
    "gfp_oracle_checkpoint": ManifestEntry(
        logical_name="gfp_oracle_checkpoint",
        protein="GFP",
        upstream_url=RAW_URLS["gfp_oracle_checkpoint"],
        local_path="oracle/GFP/cnn_oracle.ckpt",
        expected_format="pytorch/lightning checkpoint",
        artifact_type="oracle",
        checksum="988bd99bcdd8c1ec8bee5174c968dcd5db3adc33ce33a904ed186554cb0d273a",
    ),
    "gfp_oracle_config": ManifestEntry(
        logical_name="gfp_oracle_config",
        protein="GFP",
        upstream_url=RAW_URLS["gfp_oracle_config"],
        local_path="oracle/GFP/config.yaml",
        expected_format="yaml",
        artifact_type="oracle",
        checksum="f6c2d6e112c5a5111e7ceeb517aa94d2fcc06dbf3a1001cda48d4829b3dbd63a",
    ),
    "aav_oracle_checkpoint": ManifestEntry(
        logical_name="aav_oracle_checkpoint",
        protein="AAV",
        upstream_url=RAW_URLS["aav_oracle_checkpoint"],
        local_path="oracle/AAV/cnn_oracle.ckpt",
        expected_format="pytorch/lightning checkpoint",
        artifact_type="oracle",
        checksum="5db64072b22c22b24d1b7356faf950e09d28ff66744ec5168715d05143301564",
    ),
    "aav_oracle_config": ManifestEntry(
        logical_name="aav_oracle_config",
        protein="AAV",
        upstream_url=RAW_URLS["aav_oracle_config"],
        local_path="oracle/AAV/config.yaml",
        expected_format="yaml",
        artifact_type="oracle",
        checksum="ddd45ea9ba7cca90d923cbc657f641d1c3486fb8eda798065776a0f1fba18da6",
    ),
    "gfp_medium_base_seqs": ManifestEntry(
        logical_name="gfp_medium_base_seqs",
        protein="GFP",
        upstream_url=None,
        local_path="tasks/GFP/medium/base_seqs.csv",
        expected_format="csv",
        required_columns=("sequence", "score"),
        artifact_type="generated",
        generated_from="gfp_ground_truth",
        generation="prepare_gfp_aav_benchmark:gfp_medium",
    ),
    "gfp_hard_base_seqs": ManifestEntry(
        logical_name="gfp_hard_base_seqs",
        protein="GFP",
        upstream_url=None,
        local_path="tasks/GFP/hard/base_seqs.csv",
        expected_format="csv",
        required_columns=("sequence", "score"),
        artifact_type="generated",
        generated_from="gfp_ground_truth",
        generation="prepare_gfp_aav_benchmark:gfp_hard",
    ),
    "aav_medium_base_seqs": ManifestEntry(
        logical_name="aav_medium_base_seqs",
        protein="AAV",
        upstream_url=None,
        local_path="tasks/AAV/medium/base_seqs.csv",
        expected_format="csv",
        required_columns=("sequence", "score"),
        artifact_type="generated",
        generated_from="aav_ground_truth",
        generation="prepare_gfp_aav_benchmark:aav_medium",
    ),
    "aav_hard_base_seqs": ManifestEntry(
        logical_name="aav_hard_base_seqs",
        protein="AAV",
        upstream_url=None,
        local_path="tasks/AAV/hard/base_seqs.csv",
        expected_format="csv",
        required_columns=("sequence", "score"),
        artifact_type="generated",
        generated_from="aav_ground_truth",
        generation="prepare_gfp_aav_benchmark:aav_hard",
    ),
    "gfp_fitness_stats": ManifestEntry(
        logical_name="gfp_fitness_stats",
        protein="GFP",
        upstream_url=None,
        local_path="stats/GFP/fitness_stats.json",
        expected_format="json",
        artifact_type="derived",
        generated_from="gfp_ground_truth",
        generation="prepare_gfp_aav_benchmark:fitness_stats",
    ),
    "aav_fitness_stats": ManifestEntry(
        logical_name="aav_fitness_stats",
        protein="AAV",
        upstream_url=None,
        local_path="stats/AAV/fitness_stats.json",
        expected_format="json",
        artifact_type="derived",
        generated_from="aav_ground_truth",
        generation="prepare_gfp_aav_benchmark:fitness_stats",
    ),
}


TASK_CONFIGS: dict[str, GFPAVTaskConfig] = {
    "gfp_medium": GFPAVTaskConfig(
        task_name="gfp_medium",
        protein="GFP",
        difficulty="medium",
        raw_dataset="gfp_ground_truth",
        output_file="tasks/GFP/medium/base_seqs.csv",
        metadata_file="tasks/GFP/medium/metadata.json",
        seq_len=237,
        filter_percentile=(0.2, 0.4),
        min_mutant_dist=6,
        top_quantile=0.99,
        expected_train_size=2828,
        expected_median_normalized_fitness=0.09,
        expected_train_fitness_range=(0.01, 0.62),
    ),
    "gfp_hard": GFPAVTaskConfig(
        task_name="gfp_hard",
        protein="GFP",
        difficulty="hard",
        raw_dataset="gfp_ground_truth",
        output_file="tasks/GFP/hard/base_seqs.csv",
        metadata_file="tasks/GFP/hard/metadata.json",
        seq_len=237,
        filter_percentile=(0.0, 0.3),
        min_mutant_dist=7,
        top_quantile=0.99,
        expected_train_size=2426,
        expected_median_normalized_fitness=0.01,
        expected_train_fitness_range=(0.0, 0.1),
    ),
    "aav_medium": GFPAVTaskConfig(
        task_name="aav_medium",
        protein="AAV",
        difficulty="medium",
        raw_dataset="aav_ground_truth",
        output_file="tasks/AAV/medium/base_seqs.csv",
        metadata_file="tasks/AAV/medium/metadata.json",
        seq_len=28,
        filter_percentile=(0.2, 0.4),
        min_mutant_dist=6,
        top_quantile=0.99,
        expected_train_size=2139,
        expected_median_normalized_fitness=0.32,
        expected_train_fitness_range=(0.29, 0.38),
    ),
    "aav_hard": GFPAVTaskConfig(
        task_name="aav_hard",
        protein="AAV",
        difficulty="hard",
        raw_dataset="aav_ground_truth",
        output_file="tasks/AAV/hard/base_seqs.csv",
        metadata_file="tasks/AAV/hard/metadata.json",
        seq_len=28,
        filter_percentile=(0.0, 0.3),
        min_mutant_dist=7,
        top_quantile=0.99,
        expected_train_size=3448,
        expected_median_normalized_fitness=0.27,
        expected_train_fitness_range=(0.0, 0.33),
    ),
}


def get_gfp_aav_root(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else DEFAULT_GFP_AAV_ROOT


def get_manifest_entries() -> dict[str, ManifestEntry]:
    return dict(MANIFEST_ENTRIES)


def get_task_config(task_name: str) -> GFPAVTaskConfig:
    key = str(task_name).strip().lower()
    if key not in TASK_CONFIGS:
        raise KeyError(f"Unknown GFP/AAV task '{task_name}'")
    return TASK_CONFIGS[key]


def get_task_dataset_path(task_name: str, *, root: str | Path | None = None) -> Path:
    return get_gfp_aav_root(root) / get_task_config(task_name).output_file


def get_task_metadata_path(task_name: str, *, root: str | Path | None = None) -> Path:
    return get_gfp_aav_root(root) / get_task_config(task_name).metadata_file


def get_oracle_checkpoint_path(protein: str, *, root: str | Path | None = None) -> Path:
    return get_gfp_aav_root(root) / f"oracle/{str(protein).upper()}/cnn_oracle.ckpt"


def get_oracle_config_path(protein: str, *, root: str | Path | None = None) -> Path:
    return get_gfp_aav_root(root) / f"oracle/{str(protein).upper()}/config.yaml"


def get_fitness_stats_path(protein: str, *, root: str | Path | None = None) -> Path:
    return get_gfp_aav_root(root) / f"stats/{str(protein).upper()}/fitness_stats.json"


def load_fitness_stats(protein: str, *, root: str | Path | None = None) -> dict[str, Any]:
    return json.loads(get_fitness_stats_path(protein, root=root).read_text())


def load_task_dataframe(task_name: str, *, root: str | Path | None = None) -> list[dict[str, str]]:
    return read_csv_records(get_task_dataset_path(task_name, root=root))


def read_csv_records(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        return [dict(row) for row in reader]


def write_csv_records(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def validate_raw_records(
    rows: Sequence[Mapping[str, str]],
    *,
    path: str | Path,
    seq_len: int,
    required_columns: Sequence[str] = ("sequence", "score"),
) -> None:
    if not rows:
        raise ValueError(f"{path} contains no rows")
    columns = set(rows[0].keys())
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    bad_lengths = sorted({len(str(row["sequence"])) for row in rows if len(str(row["sequence"])) != int(seq_len)})
    if bad_lengths:
        raise ValueError(f"{path} contains sequence lengths {bad_lengths}; expected {int(seq_len)}")
    for idx, row in enumerate(rows, start=2):
        try:
            float(row["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{idx} has a non-numeric score") from exc


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    q_value = float(q)
    if q_value < 0.0 or q_value > 1.0:
        raise ValueError("quantile q must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q_value
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - float(lower)
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sequence_distance(a: str, b: str, *, stop_at: int | None = None) -> int:
    if len(a) != len(b):
        raise ValueError("GFP/AAV sequence_distance expects fixed-length sequences")
    distance = 0
    cutoff = None if stop_at is None else int(stop_at)
    for left, right in zip(a, b):
        if left == right:
            continue
        distance += 1
        if cutoff is not None and distance >= cutoff:
            return distance
    return distance


def _is_far_from_all_top_sequences(sequence: str, top_sequences: Sequence[str], min_mutant_dist: int) -> bool:
    threshold = int(min_mutant_dist)
    for top_sequence in top_sequences:
        if sequence_distance(sequence, top_sequence, stop_at=threshold) < threshold:
            return False
    return True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def generate_fitness_stats(
    source_csv: str | Path,
    *,
    protein: str,
    seq_len: int,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = read_csv_records(source_csv)
    validate_raw_records(rows, path=source_csv, seq_len=seq_len)
    scores = [float(row["score"]) for row in rows]
    stats = {
        "protein": str(protein).upper(),
        "source_dataset": str(source_csv),
        "num_rows": len(rows),
        "sequence_length": int(seq_len),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": sum(scores) / float(len(scores)),
        "score_median": quantile(scores, 0.5),
        "score_quantiles": {
            str(q): quantile(scores, q)
            for q in (0.0, 0.2, 0.3, 0.4, 0.99, 1.0)
        },
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def normalize_score(score: float, fitness_stats: Mapping[str, Any]) -> float:
    score_min = float(fitness_stats["score_min"])
    score_max = float(fitness_stats["score_max"])
    if score_max <= score_min:
        raise ValueError("fitness_stats score_max must be greater than score_min")
    return (float(score) - score_min) / (score_max - score_min)


def generate_task_subset(
    source_csv: str | Path,
    task_config: GFPAVTaskConfig,
    *,
    output_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    allow_size_mismatch: bool = False,
    overwrite: bool = False,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    out_path = Path(output_path) if output_path is not None else None
    meta_path = Path(metadata_path) if metadata_path is not None else None
    for path in (out_path, meta_path):
        if path is not None and path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; rerun with overwrite=True to regenerate")

    rows = read_csv_records(source_csv)
    validate_raw_records(rows, path=source_csv, seq_len=task_config.seq_len)
    fieldnames = list(rows[0].keys())
    scores = [float(row["score"]) for row in rows]
    lower_score = quantile(scores, task_config.filter_percentile[0])
    upper_score = quantile(scores, task_config.filter_percentile[1])
    top_score = quantile(scores, task_config.top_quantile)
    top_sequences = [str(row["sequence"]) for row in rows if float(row["score"]) >= top_score]
    candidates = [
        row
        for row in rows
        if lower_score <= float(row["score"]) <= upper_score
    ]
    kept = [
        row
        for row in candidates
        if _is_far_from_all_top_sequences(str(row["sequence"]), top_sequences, task_config.min_mutant_dist)
    ]

    generated_size = len(kept)
    if (
        task_config.expected_train_size is not None
        and generated_size != int(task_config.expected_train_size)
        and not allow_size_mismatch
    ):
        raise ValueError(
            f"{task_config.task_name} generated {generated_size} rows, expected "
            f"{task_config.expected_train_size}. Use allow_size_mismatch=True to write anyway."
        )
    kept_scores = [float(row["score"]) for row in kept]
    metadata = {
        "task_name": task_config.task_name,
        "protein": task_config.protein,
        "difficulty": task_config.difficulty,
        "source_dataset": str(source_csv),
        "source_dataset_sha256": sha256_file(source_csv),
        "seq_len": task_config.seq_len,
        "filter_percentile": list(task_config.filter_percentile),
        "top_quantile": task_config.top_quantile,
        "min_mutant_dist": task_config.min_mutant_dist,
        "lower_score_threshold": lower_score,
        "upper_score_threshold": upper_score,
        "top_score_threshold": top_score,
        "num_top_sequences": len(top_sequences),
        "generated_train_size": generated_size,
        "expected_train_size": task_config.expected_train_size,
        "median_score": quantile(kept_scores, 0.5) if kept_scores else None,
        "min_score": min(kept_scores) if kept_scores else None,
        "max_score": max(kept_scores) if kept_scores else None,
        "expected_median_normalized_fitness": task_config.expected_median_normalized_fitness,
        "expected_train_fitness_range": list(task_config.expected_train_fitness_range),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_version": _git_commit(),
    }
    if out_path is not None:
        write_csv_records(out_path, kept, fieldnames)
    if meta_path is not None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return kept, metadata


def _raw_dataset_path(root: Path, logical_name: str) -> Path:
    entry = MANIFEST_ENTRIES[logical_name]
    return root / entry.local_path


def prepare_gfp_aav_benchmark(
    *,
    root: str | Path | None = None,
    allow_size_mismatch: bool = False,
    overwrite: bool = False,
    task_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    bench_root = get_gfp_aav_root(root)
    selected_tasks = list(task_names) if task_names is not None else list(TASK_CONFIGS)
    summaries: list[dict[str, Any]] = []
    for protein in ("GFP", "AAV"):
        raw_key = f"{protein.lower()}_ground_truth"
        source_path = _raw_dataset_path(bench_root, raw_key)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing raw dataset: {source_path}")
        seq_len = 237 if protein == "GFP" else 28
        stats_path = get_fitness_stats_path(protein, root=bench_root)
        if stats_path.exists() and not overwrite:
            raise FileExistsError(f"{stats_path} exists; rerun with overwrite=True to regenerate")
        generate_fitness_stats(source_path, protein=protein, seq_len=seq_len, output_path=stats_path)

    for task_name in selected_tasks:
        cfg = get_task_config(task_name)
        source_path = _raw_dataset_path(bench_root, cfg.raw_dataset)
        kept, metadata = generate_task_subset(
            source_path,
            cfg,
            output_path=bench_root / cfg.output_file,
            metadata_path=bench_root / cfg.metadata_file,
            allow_size_mismatch=allow_size_mismatch,
            overwrite=overwrite,
        )
        summaries.append(
            {
                "task_name": cfg.task_name,
                "protein": cfg.protein,
                "difficulty": cfg.difficulty,
                "generated_train_size": len(kept),
                "expected_train_size": cfg.expected_train_size,
                "output_file": cfg.output_file,
                "metadata_file": cfg.metadata_file,
                "lower_score_threshold": metadata["lower_score_threshold"],
                "upper_score_threshold": metadata["upper_score_threshold"],
                "top_score_threshold": metadata["top_score_threshold"],
            }
        )
    return summaries


def write_manifest_yaml(path: str | Path) -> None:
    rows = [asdict(entry) for entry in MANIFEST_ENTRIES.values()]
    lines = ["entries:"]
    for row in rows:
        lines.append(f"  - logical_name: {row['logical_name']}")
        for key in (
            "protein",
            "upstream_url",
            "local_path",
            "expected_format",
            "artifact_type",
            "generated_from",
            "generation",
            "checksum",
        ):
            value = row[key]
            if value is None:
                lines.append(f"    {key}: null")
            else:
                lines.append(f"    {key}: {json.dumps(value)}")
        columns = list(row["required_columns"])
        lines.append(f"    required_columns: {json.dumps(columns)}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_task_configs_json(path: str | Path) -> None:
    payload = {
        name: {
            **asdict(config),
            "filter_percentile": list(config.filter_percentile),
            "expected_train_fitness_range": list(config.expected_train_fitness_range),
        }
        for name, config in TASK_CONFIGS.items()
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_GFP_AAV_ROOT",
    "GGS_REVISION",
    "GFPAVTaskConfig",
    "MANIFEST_ENTRIES",
    "ManifestEntry",
    "RAW_URLS",
    "TASK_CONFIGS",
    "generate_fitness_stats",
    "generate_task_subset",
    "get_fitness_stats_path",
    "get_gfp_aav_root",
    "get_manifest_entries",
    "get_oracle_checkpoint_path",
    "get_oracle_config_path",
    "get_task_config",
    "get_task_dataset_path",
    "get_task_metadata_path",
    "load_fitness_stats",
    "load_task_dataframe",
    "normalize_score",
    "prepare_gfp_aav_benchmark",
    "quantile",
    "read_csv_records",
    "sequence_distance",
    "sha256_file",
    "validate_raw_records",
    "write_manifest_yaml",
    "write_task_configs_json",
]
