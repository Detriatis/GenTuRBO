"""Deterministic result primitives for the ``ehrlich_kernel_v1`` study.

This module deliberately contains no experiment orchestration.  It turns full
proposal traces into small, publication-facing tables and validates that a
cohort is complete before any result writer is allowed to publish it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from genturbo.io import iter_jsonl
from genturbo.numeric import pearson_correlation
from genturbo.serialization import atomic_write, csv_bytes


class KernelStudyResultsError(RuntimeError):
    """Raised when kernel-study inputs cannot produce trustworthy results."""


@dataclass(frozen=True)
class TraceSource:
    """A proposal-trace stream and the immutable provenance attached to it."""

    path: Path
    condition: str
    seed: int


@dataclass(frozen=True)
class ClassifiedProposals:
    """Normalised trace rows split into nested decision populations."""

    all_records: tuple[dict[str, Any], ...]
    eligible_records: tuple[dict[str, Any], ...]
    selected_records: tuple[dict[str, Any], ...]

    def records_for(self, population: Literal["all", "eligible", "selected"]) -> tuple[dict[str, Any], ...]:
        """Return the requested nested proposal population."""

        if population == "all":
            return self.all_records
        if population == "eligible":
            return self.eligible_records
        if population == "selected":
            return self.selected_records
        raise KernelStudyResultsError(f"unknown proposal population: {population}")


ScoreCallback = Callable[[tuple[int, ...], Mapping[str, Any]], float | None]


def _token_sequence(value: Any, *, field: str, source: Path, row_index: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(token, bool) or not isinstance(token, int) for token in value)
    ):
        raise KernelStudyResultsError(
            f"{source}: row {row_index} has invalid {field}; expected a non-empty integer list"
        )
    return tuple(int(token) for token in value)


def absolute_hamming(left: Sequence[int], right: Sequence[int]) -> int:
    """Return absolute token Hamming distance for fixed-length sequences."""

    if len(left) != len(right):
        raise KernelStudyResultsError(
            f"absolute Hamming distance requires equal lengths, received {len(left)} and {len(right)}"
        )
    return sum(a != b for a, b in zip(left, right, strict=True))


def _finite_number(value: Any, *, field: str, source: Path, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KernelStudyResultsError(f"{source}: row {row_index} has non-numeric {field}")
    number = float(value)
    if not math.isfinite(number):
        raise KernelStudyResultsError(f"{source}: row {row_index} has non-finite {field}")
    return number


def _normalise_trace_record(
    raw: Mapping[str, Any],
    *,
    source: TraceSource,
    row_index: int,
    scorer: ScoreCallback | None,
) -> dict[str, Any]:
    ancestor = _token_sequence(raw.get("ancestor_seq"), field="ancestor_seq", source=source.path, row_index=row_index)
    child_field = "to_seq" if "to_seq" in raw else "seq"
    child = _token_sequence(raw.get(child_field), field=child_field, source=source.path, row_index=row_index)
    expected = _finite_number(
        raw.get("expected_edits"), field="expected_edits", source=source.path, row_index=row_index
    )
    if expected < 0.0:
        raise KernelStudyResultsError(f"{source.path}: row {row_index} has negative expected_edits")
    realised = absolute_hamming(ancestor, child)

    acquisition_candidate = raw.get("acquisition_candidate") is True
    invalid = raw.get("invalid_generated_proposal") is True
    eligible = acquisition_candidate and not invalid
    selected_flag = raw.get("acquisition_selected", raw.get("oracle_evaluated", False)) is True
    if selected_flag and not eligible:
        raise KernelStudyResultsError(
            f"{source.path}: row {row_index} is selected but is not a valid acquisition candidate"
        )

    # Retain the analysis projection, not three large raw sequence copies. The
    # immutable full diagnostics remain in the source trace.
    record = {
        "condition": source.condition,
        "source_seed": source.seed,
        "trace_path": source.path.as_posix(),
        "trace_row": row_index,
        "batch_id": raw.get("batch_id"),
        "sequence": list(child),
        "expected_edits": expected,
        "realised_edits": realised,
        "signed_edit_error": realised - expected,
        "absolute_edit_error": abs(realised - expected),
        "acquisition_candidate": acquisition_candidate,
        "acquisition_eligible": eligible,
        "acquisition_selected": selected_flag,
        "invalid_generated_proposal": invalid,
        "duplicate_exact_proposal": raw.get("duplicate_exact_proposal") is True,
        "duplicate_generated_leaf": raw.get("duplicate_generated_leaf") is True,
        "cached_generated_proposal": raw.get("cached_generated_proposal") is True,
    }
    if scorer is not None:
        score = scorer(child, record)
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise KernelStudyResultsError(
                    f"{source.path}: scoring callback returned a non-finite score at row {row_index}"
                )
            record["rescored_value"] = float(score)
        else:
            record["rescored_value"] = None
    return record


def read_proposal_traces(
    sources: Iterable[TraceSource],
    *,
    scorer: ScoreCallback | None = None,
) -> ClassifiedProposals:
    """Read and classify plain or gzip-compressed full proposal traces.

    Classification is nested: ``selected`` is a subset of ``eligible``, which
    is a subset of ``all``.  Acquisition eligibility is defined by the runner's
    candidate flag plus generator validity; duplicate/cache decisions therefore
    remain owned by the proposal runner rather than being reimplemented here.
    """

    ordered_sources = sorted(
        sources,
        key=lambda item: (item.condition, item.seed, item.path.as_posix()),
    )
    if not ordered_sources:
        raise KernelStudyResultsError("at least one proposal trace is required")

    all_records: list[dict[str, Any]] = []
    eligible_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    seen_source_keys: set[tuple[str, int]] = set()
    for source in ordered_sources:
        if not isinstance(source.condition, str) or not source.condition.strip():
            raise KernelStudyResultsError("trace condition must be a non-empty string")
        if isinstance(source.seed, bool) or not isinstance(source.seed, int) or source.seed < 0:
            raise KernelStudyResultsError(f"trace seed must be a non-negative integer: {source.seed!r}")
        key = (source.condition, source.seed)
        if key in seen_source_keys:
            raise KernelStudyResultsError(
                f"multiple trace streams were supplied for {source.condition}/seed{source.seed}"
            )
        seen_source_keys.add(key)
        if not source.path.is_file():
            raise KernelStudyResultsError(f"proposal trace does not exist: {source.path}")
        source_count = 0
        for row_index, raw in enumerate(iter_jsonl(source.path)):
            record = _normalise_trace_record(
                raw,
                source=source,
                row_index=row_index,
                scorer=scorer,
            )
            all_records.append(record)
            if record["acquisition_eligible"]:
                eligible_records.append(record)
            if record["acquisition_selected"]:
                selected_records.append(record)
            source_count += 1
        if source_count == 0:
            raise KernelStudyResultsError(f"proposal trace is empty: {source.path}")

    return ClassifiedProposals(
        all_records=tuple(all_records),
        eligible_records=tuple(eligible_records),
        selected_records=tuple(selected_records),
    )


def locality_summary_rows(classified: ClassifiedProposals) -> list[dict[str, Any]]:
    """Summarise expected-versus-realised absolute edits for each population.

    Rows are grouped by condition and population.  The per-proposal normalised
    records remain available from :func:`read_proposal_traces` for scatter or
    calibration plots.
    """

    summaries: list[dict[str, Any]] = []
    for population in ("all", "eligible", "selected"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in classified.records_for(population):
            grouped[str(record["condition"])].append(record)
        for condition in sorted(grouped):
            records = grouped[condition]
            desired = [float(record["expected_edits"]) for record in records]
            realised = [float(record["realised_edits"]) for record in records]
            errors = [actual - target for target, actual in zip(desired, realised, strict=True)]
            summaries.append(
                {
                    "condition": condition,
                    "population": population,
                    "proposal_count": len(records),
                    "source_seed_count": len({int(record["source_seed"]) for record in records}),
                    "expected_edits_mean": fmean(desired),
                    "realised_edits_mean": fmean(realised),
                    "signed_edit_error_mean": fmean(errors),
                    "absolute_edit_error_mean": fmean(abs(error) for error in errors),
                    "edit_rmse": math.sqrt(fmean(error * error for error in errors)),
                    "expected_realised_pearson": pearson_correlation(desired, realised),
                }
            )
    return summaries


def locality_calibration_rows(classified: ClassifiedProposals) -> list[dict[str, Any]]:
    """Return deterministic expected-edit calibration points for publication plots."""

    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for population in ("all", "eligible", "selected"):
        for record in classified.records_for(population):
            key = (str(record["condition"]), population, float(record["expected_edits"]))
            grouped[key].append(record)
    rows: list[dict[str, Any]] = []
    for condition, population, expected in sorted(grouped):
        records = grouped[(condition, population, expected)]
        realised = [float(record["realised_edits"]) for record in records]
        rows.append(
            {
                "condition": condition,
                "population": population,
                "expected_edits": expected,
                "proposal_count": len(records),
                "realised_edits_mean": fmean(realised),
                "realised_edits_min": min(realised),
                "realised_edits_max": max(realised),
                "absolute_edit_error_mean": fmean(abs(value - expected) for value in realised),
            }
        )
    return rows


def _candidate_id(condition: str, sequence: Sequence[int]) -> str:
    payload = json.dumps(
        {"condition": condition, "sequence": list(sequence)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def freeze_eligible_candidates(
    classified: ClassifiedProposals,
    *,
    scorer: ScoreCallback | None = None,
) -> list[dict[str, Any]]:
    """Freeze every valid acquisition-eligible sequence with seed provenance.

    Sequences are unique within condition.  All source-seed memberships are
    retained so leave-one-source-seed-out splits can exclude cross-seed leaks.
    A scorer, when supplied, is called once per unique candidate.
    """

    occurrences: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    for record in classified.eligible_records:
        sequence = tuple(int(token) for token in record["sequence"])
        occurrences[(str(record["condition"]), sequence)].append(record)

    frozen: list[dict[str, Any]] = []
    for (condition, sequence), records in sorted(occurrences.items()):
        ordered = sorted(records, key=lambda row: (int(row["source_seed"]), int(row["trace_row"])))
        source_seeds = sorted({int(row["source_seed"]) for row in ordered})
        candidate: dict[str, Any] = {
            "candidate_id": _candidate_id(condition, sequence),
            "condition": condition,
            "sequence": list(sequence),
            "sequence_length": len(sequence),
            "source_seed_memberships": source_seeds,
            "occurrence_count": len(ordered),
            "selected_occurrence_count": sum(bool(row["acquisition_selected"]) for row in ordered),
        }
        if scorer is not None:
            score = scorer(sequence, ordered[0])
            if score is not None:
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                    raise KernelStudyResultsError(
                        f"scoring callback returned a non-finite score for candidate {candidate['candidate_id']}"
                    )
                candidate["score"] = float(score)
            else:
                candidate["score"] = None
        frozen.append(candidate)
    return frozen


def selected_batch_diversity_rows(classified: ClassifiedProposals) -> list[dict[str, Any]]:
    """Compute pairwise absolute-Hamming diversity within selected batches."""

    grouped: dict[tuple[str, int, int], list[tuple[int, ...]]] = defaultdict(list)
    for record in classified.selected_records:
        batch_id = record.get("batch_id")
        if isinstance(batch_id, bool) or not isinstance(batch_id, int):
            raise KernelStudyResultsError(f"{record['trace_path']}: row {record['trace_row']} has invalid batch_id")
        grouped[(str(record["condition"]), int(record["source_seed"]), batch_id)].append(
            tuple(int(token) for token in record["sequence"])
        )

    rows: list[dict[str, Any]] = []
    for (condition, seed, batch_id), sequences in sorted(grouped.items()):
        distances = [
            absolute_hamming(sequences[left], sequences[right])
            for left in range(len(sequences))
            for right in range(left + 1, len(sequences))
        ]
        rows.append(
            {
                "condition": condition,
                "source_seed": seed,
                "batch_id": batch_id,
                "selected_count": len(sequences),
                "unique_sequence_count": len(set(sequences)),
                "pair_count": len(distances),
                "pairwise_hamming_mean": fmean(distances) if distances else None,
                "pairwise_hamming_min": min(distances) if distances else None,
                "pairwise_hamming_max": max(distances) if distances else None,
            }
        )
    return rows


def require_complete_manifests(
    manifests: Mapping[str, Path | str],
    *,
    expected_cell_ids: Iterable[str],
    expected_status: Literal["completed", "smoke"] = "completed",
) -> dict[str, dict[str, Any]]:
    """Load a cohort only when its exact expected cell set has finished."""

    expected = tuple(sorted(str(cell_id) for cell_id in expected_cell_ids))
    if not expected or len(expected) != len(set(expected)):
        raise KernelStudyResultsError("expected cell IDs must be non-empty and unique")
    actual = tuple(sorted(str(cell_id) for cell_id in manifests))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise KernelStudyResultsError(
            f"incomplete cohort: missing={missing or 'none'}, unexpected={unexpected or 'none'}"
        )

    loaded: dict[str, dict[str, Any]] = {}
    incomplete: list[str] = []
    for cell_id in expected:
        path = Path(manifests[cell_id])
        if not path.is_file():
            incomplete.append(f"{cell_id}: missing manifest")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KernelStudyResultsError(f"cannot read manifest for {cell_id}: {path}") from exc
        if not isinstance(payload, dict):
            raise KernelStudyResultsError(f"manifest for {cell_id} is not a JSON object: {path}")
        status = payload.get("status")
        if status != expected_status:
            incomplete.append(f"{cell_id}: status={status!r}")
            continue
        recorded_id = payload.get("cell_id")
        if recorded_id is not None and recorded_id != cell_id:
            raise KernelStudyResultsError(f"manifest cell ID mismatch for {cell_id}: recorded {recorded_id!r}")
        loaded[cell_id] = payload
    if incomplete:
        raise KernelStudyResultsError("incomplete cohort: " + "; ".join(incomplete))
    return loaded


def write_csv_atomic(
    path: Path | str,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Atomically write a compact CSV with deterministic columns and rows."""

    fields = list(fieldnames) if fieldnames is not None else sorted({key for row in rows for key in row})
    if len(fields) != len(set(fields)):
        raise KernelStudyResultsError("CSV field names must be unique")
    unknown = sorted({key for row in rows for key in row} - set(fields))
    if unknown:
        raise KernelStudyResultsError(f"CSV rows contain undeclared fields: {unknown}")
    destination = Path(path)
    try:
        payload = csv_bytes(rows, fieldnames=fields) if fields else b""
        atomic_write(destination, payload)
    except (TypeError, ValueError) as exc:
        raise KernelStudyResultsError("result CSV is not finite and serialisable") from exc
    return destination


__all__ = [
    "ClassifiedProposals",
    "KernelStudyResultsError",
    "ScoreCallback",
    "TraceSource",
    "absolute_hamming",
    "freeze_eligible_candidates",
    "locality_calibration_rows",
    "locality_summary_rows",
    "read_proposal_traces",
    "require_complete_manifests",
    "selected_batch_diversity_rows",
    "write_csv_atomic",
]
