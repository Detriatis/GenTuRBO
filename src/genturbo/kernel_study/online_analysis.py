"""Read-only, manifest-aware summaries for Ehrlich kernel sweeps.

The analyzer treats run manifests as provenance claims rather than filenames:
the selected manifest is the unique one whose recorded output digests match
every current JSONL output in its run directory.  Study-specific reporting
reads proposal traces once for the metrics it publishes; this provenance layer
does not redundantly rescore the full proposal pool.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from genturbo.artifacts import sha256_file
from genturbo.io import iter_jsonl

SCHEMA_VERSION = 1
_SEED_DIRECTORY = re.compile(r"^seed(?P<seed>\d+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_JSONL_SUFFIXES = (".jsonl", ".jsonl.gz")
_STREAM_DEFAULTS = {
    "initial": "initial_batch.jsonl",
    "samples": "samples.jsonl",
    "diagnostics": "proposal_diagnostics.jsonl",
    "traces": "proposal_traces.jsonl",
}
_STREAM_CONFIG_KEYS = {
    "initial": "initial_batch_out",
    "samples": "samples_out",
    "diagnostics": "proposal_diagnostics_out",
    "traces": "proposal_traces_out",
}
_INPUT_CONFIG_KEYS = {
    "checkpoint": "checkpoint",
    "runtime_config": "config",
    "reward_config": "reward_config",
    "transition_mask": "transition_mask_path",
    "initial_batch": "initial_batch_path",
    "target_sequence": "target_sequence",
    "diagnostic_full_dataset_best_csv": "diagnostic_full_dataset_best_csv",
    "pretraining_manifest": "pretraining_manifest",
    "lambo_requirements": "lambo_requirements",
}


class EhrlichSweepError(RuntimeError):
    """Raised when a run cannot be summarized without weakening provenance."""


@dataclass(slots=True)
class _ManifestCandidate:
    path: Path
    data: dict[str, Any] | None
    outputs: dict[str, Path]
    mismatch_reasons: list[str]

    @property
    def matches(self) -> bool:
        return self.data is not None and not self.mismatch_reasons


@dataclass(slots=True)
class _SelectedManifest:
    path: Path
    data: dict[str, Any]
    outputs: dict[str, Path]
    stale: list[_ManifestCandidate]


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EhrlichSweepError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EhrlichSweepError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EhrlichSweepError(f"{label} must contain a JSON object: {path}")
    return value


def _is_jsonl(path: Path) -> bool:
    return path.name.endswith(_JSONL_SUFFIXES)


def _current_jsonl_outputs(run_dir: Path) -> set[str]:
    current: set[str] = set()
    for path in run_dir.iterdir():
        if not _is_jsonl(path):
            continue
        if path.is_symlink() or not path.is_file():
            raise EhrlichSweepError(f"run output is not a regular file: {path}")
        current.add(path.name)
    return current


def _manifest_candidate(
    path: Path,
    *,
    run_dir: Path,
    current_jsonl: set[str],
    digest_cache: dict[Path, str],
) -> _ManifestCandidate:
    try:
        data = _load_json_object(path, label="run manifest")
    except EhrlichSweepError as exc:
        return _ManifestCandidate(path=path, data=None, outputs={}, mismatch_reasons=[str(exc)])

    reasons: list[str] = []
    raw_hashes = data.get("output_hashes")
    if not isinstance(raw_hashes, Mapping) or not raw_hashes:
        return _ManifestCandidate(
            path=path,
            data=data,
            outputs={},
            mismatch_reasons=["output_hashes is missing or empty"],
        )

    outputs: dict[str, Path] = {}
    for raw_name, raw_digest in raw_hashes.items():
        if not isinstance(raw_name, str) or not raw_name:
            reasons.append("output_hashes contains a non-string path")
            continue
        if not isinstance(raw_digest, str) or _SHA256.fullmatch(raw_digest) is None:
            reasons.append(f"invalid SHA-256 for {raw_name!r}")
            continue
        basename = Path(raw_name).name
        if not basename or basename in outputs:
            reasons.append(f"duplicate or empty output basename {basename!r}")
            continue
        current = run_dir / basename
        outputs[basename] = current
        if current.is_symlink() or not current.is_file():
            reasons.append(f"recorded output is missing or non-regular: {basename}")
            continue
        digest = digest_cache.setdefault(current, sha256_file(current))
        if digest != raw_digest:
            reasons.append(f"checksum mismatch: {basename}")

    unclaimed = sorted(current_jsonl - set(outputs))
    if unclaimed:
        reasons.append(f"current JSONL outputs are not claimed: {', '.join(unclaimed)}")
    return _ManifestCandidate(path=path, data=data, outputs=outputs, mismatch_reasons=reasons)


def select_manifest(run_dir: Path | str) -> _SelectedManifest:
    """Select the unique manifest that describes the run directory's current outputs."""

    directory = Path(run_dir).expanduser().resolve()
    manifests = sorted(directory.glob("run_manifest*.json"))
    if not manifests:
        raise EhrlichSweepError(f"no run manifests found in {directory}")
    current_jsonl = _current_jsonl_outputs(directory)
    if not current_jsonl:
        raise EhrlichSweepError(f"no JSONL outputs found in {directory}")

    digest_cache: dict[Path, str] = {}
    candidates = [
        _manifest_candidate(
            path,
            run_dir=directory,
            current_jsonl=current_jsonl,
            digest_cache=digest_cache,
        )
        for path in manifests
    ]
    matches = [candidate for candidate in candidates if candidate.matches]
    if len(matches) != 1:
        details = "; ".join(
            f"{candidate.path.name}: " + ("matches" if candidate.matches else ", ".join(candidate.mismatch_reasons))
            for candidate in candidates
        )
        if not matches:
            raise EhrlichSweepError(f"no manifest matches current outputs in {directory}: {details}")
        names = ", ".join(candidate.path.name for candidate in matches)
        raise EhrlichSweepError(f"ambiguous matching manifests in {directory}: {names}")

    selected = matches[0]
    assert selected.data is not None
    return _SelectedManifest(
        path=selected.path,
        data=selected.data,
        outputs=selected.outputs,
        stale=[candidate for candidate in candidates if candidate is not selected],
    )


def discover_run_directories(run_root: Path | str) -> list[Path]:
    """Discover strict ``<method>/seed<N>`` run directories beneath a sweep root."""

    root = Path(run_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise EhrlichSweepError(f"run root is not a regular directory: {root}")
    runs: list[Path] = []
    for method_dir in sorted(root.iterdir()):
        if method_dir.is_symlink() or not method_dir.is_dir():
            continue
        for seed_dir in sorted(method_dir.iterdir()):
            if seed_dir.is_symlink() or not seed_dir.is_dir():
                continue
            if _SEED_DIRECTORY.fullmatch(seed_dir.name) is None:
                continue
            if any(seed_dir.glob("run_manifest*.json")):
                runs.append(seed_dir)
    if not runs:
        raise EhrlichSweepError(f"no <method>/seed<N> run directories found below {root}")
    return runs


def _infer_repo_root(run_root: Path, explicit: Path | str | None) -> Path | None:
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if root.is_symlink() or not root.is_dir():
            raise EhrlichSweepError(f"repository root is not a regular directory: {root}")
        return root
    for candidate in (run_root, *run_root.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _reward_spec(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    raw = resolved_config.get("oracle") or resolved_config.get("reward")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EhrlichSweepError("resolved reward is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise EhrlichSweepError("resolved config must contain an Ehrlich reward mapping")
    if raw.get("name") != "ehrlich":
        raise EhrlichSweepError(f"resolved reward must be 'ehrlich', got {raw.get('name')!r}")
    return raw


def _command_search_config(command: Any) -> Any:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return None
    try:
        index = command.index("--search-config")
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _recorded_input_paths(manifest: Mapping[str, Any]) -> dict[str, Any]:
    resolved = manifest.get("resolved_config")
    if not isinstance(resolved, Mapping):
        return {}
    paths = {name: resolved.get(config_key) for name, config_key in _INPUT_CONFIG_KEYS.items()}
    paths["search_config"] = resolved.get("search_config") or _command_search_config(manifest.get("command"))
    reward = _reward_spec(resolved)
    paths["reward_instance"] = reward.get("instance_path")
    return paths


def _resolve_input(
    raw_path: Any,
    *,
    expected_digest: str,
    repo_root: Path | None,
    run_root: Path,
) -> Path:
    if not isinstance(raw_path, (str, os.PathLike)) or not str(raw_path).strip():
        raise EhrlichSweepError("manifest records an input hash without its resolved path")
    path = Path(raw_path).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if repo_root is not None:
            candidates.append(repo_root / path)
        candidates.append(Path.cwd() / path)
        candidates.append(run_root / path)

    seen: set[Path] = set()
    existing: list[Path] = []
    for candidate in candidates:
        canonical = candidate.resolve(strict=False)
        if canonical in seen:
            continue
        seen.add(canonical)
        if candidate.is_symlink():
            continue
        if candidate.is_file():
            existing.append(candidate)
    if not existing:
        raise EhrlichSweepError(f"recorded input is missing: {path}")
    matching = [candidate for candidate in existing if sha256_file(candidate) == expected_digest]
    if not matching:
        raise EhrlichSweepError(f"recorded input checksum mismatch: {path}")
    return matching[0].resolve()


def _validate_manifest(
    selected: _SelectedManifest,
    *,
    run_dir: Path,
    run_root: Path,
    repo_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = selected.data
    if manifest.get("schema_version") != 1:
        raise EhrlichSweepError(f"unsupported manifest schema in {selected.path}")
    if manifest.get("status") != "completed" or manifest.get("error") is not None:
        raise EhrlichSweepError(f"selected manifest is not a successful completion: {selected.path}")
    started = manifest.get("started_at")
    completed = manifest.get("completed_at")
    if not isinstance(started, str) or not isinstance(completed, str):
        raise EhrlichSweepError(f"manifest timestamps are incomplete: {selected.path}")
    try:
        started_at = datetime.fromisoformat(started.replace("Z", "+00:00"))
        completed_at = datetime.fromisoformat(completed.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EhrlichSweepError(f"manifest timestamps are invalid: {selected.path}") from exc
    if completed_at < started_at:
        raise EhrlichSweepError(f"manifest completion precedes its start: {selected.path}")

    seed_match = _SEED_DIRECTORY.fullmatch(run_dir.name)
    assert seed_match is not None
    expected_seed = int(seed_match.group("seed"))
    method = run_dir.parent.name
    resolved = manifest.get("resolved_config")
    if not isinstance(resolved, dict):
        raise EhrlichSweepError(f"resolved_config is missing from {selected.path}")
    if manifest.get("seed") != expected_seed or resolved.get("seed") != expected_seed:
        raise EhrlichSweepError(f"seed mapping mismatch in {run_dir}")
    metadata = resolved.get("metadata")
    plot_label = metadata.get("plot_label") if isinstance(metadata, Mapping) else None
    if not isinstance(plot_label, str) or not plot_label:
        raise EhrlichSweepError(f"metadata.plot_label is missing from {selected.path}")
    if plot_label != method:
        raise EhrlichSweepError(
            f"method mapping mismatch in {run_dir}: directory={method!r}, metadata.plot_label={plot_label!r}"
        )
    _reward_spec(resolved)

    command = manifest.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        raise EhrlichSweepError(f"manifest command is invalid: {selected.path}")
    git_sha = manifest.get("git_sha")
    if git_sha is not None and (not isinstance(git_sha, str) or _GIT_SHA.fullmatch(git_sha) is None):
        raise EhrlichSweepError(f"manifest git_sha is invalid: {selected.path}")
    dirty_hash = manifest.get("dirty_diff_hash")
    if dirty_hash is not None and (not isinstance(dirty_hash, str) or _SHA256.fullmatch(dirty_hash) is None):
        raise EhrlichSweepError(f"manifest dirty_diff_hash is invalid: {selected.path}")
    implementation_path = manifest.get("implementation_record_path")
    implementation_hash = manifest.get("implementation_record_sha256")
    if (implementation_path is None) != (implementation_hash is None):
        raise EhrlichSweepError(f"manifest implementation record is incomplete: {selected.path}")
    if implementation_path is not None and (
        not isinstance(implementation_path, str)
        or not implementation_path
        or Path(implementation_path).is_absolute()
        or not isinstance(implementation_hash, str)
        or _SHA256.fullmatch(implementation_hash) is None
    ):
        raise EhrlichSweepError(f"manifest implementation record is invalid: {selected.path}")
    if not isinstance(manifest.get("dependency_profile"), str) or not manifest["dependency_profile"]:
        raise EhrlichSweepError(f"manifest dependency_profile is invalid: {selected.path}")

    raw_input_hashes = manifest.get("input_hashes")
    if not isinstance(raw_input_hashes, Mapping) or "reward_instance" not in raw_input_hashes:
        raise EhrlichSweepError(f"manifest must hash the Ehrlich reward instance: {selected.path}")
    path_specs = _recorded_input_paths(manifest)
    verified_inputs: dict[str, Path] = {}
    for name, digest in raw_input_hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise EhrlichSweepError(f"manifest input hash is invalid: {name!r}")
        if name not in path_specs:
            raise EhrlichSweepError(f"cannot validate unknown manifest input {name!r}")
        verified_inputs[name] = _resolve_input(
            path_specs[name],
            expected_digest=digest,
            repo_root=repo_root,
            run_root=run_root,
        )
    return resolved, verified_inputs


def _stream_candidates(
    label: str,
    *,
    selected: _SelectedManifest,
    resolved: Mapping[str, Any],
) -> list[Path]:
    names = {_STREAM_DEFAULTS[label], f"{_STREAM_DEFAULTS[label]}.gz"}
    configured = resolved.get(_STREAM_CONFIG_KEYS[label])
    if isinstance(configured, str) and configured:
        names.add(Path(configured).name)
    return [path for name, path in selected.outputs.items() if name in names]


def _resolve_streams(
    selected: _SelectedManifest,
    resolved: Mapping[str, Any],
) -> dict[str, Path | None]:
    streams: dict[str, Path | None] = {}
    for label in _STREAM_DEFAULTS:
        candidates = _stream_candidates(label, selected=selected, resolved=resolved)
        if len(candidates) > 1:
            raise EhrlichSweepError(
                f"both plain and gzip (or multiple custom) {label} streams are present in {selected.path.parent}"
            )
        streams[label] = candidates[0] if candidates else None

    if streams["samples"] is None:
        raise EhrlichSweepError(f"selected manifest has no samples output: {selected.path}")
    initial_expected = _optional_nonnegative_int(
        resolved.get("initial_oracle_evaluations"),
        label="initial_oracle_evaluations",
    )
    if initial_expected and streams["initial"] is None:
        initial_input = resolved.get("initial_batch_path")
        if not isinstance(initial_input, str) or not initial_input:
            raise EhrlichSweepError(f"selected manifest has neither an initial-batch output nor input: {selected.path}")
    trace_level = str(resolved.get("trace_level") or "summary")
    if trace_level == "full" and streams["traces"] is None:
        raise EhrlichSweepError(f"full trace run has no proposal trace output: {selected.path}")
    if trace_level != "full" and streams["traces"] is not None:
        raise EhrlichSweepError(f"non-full run unexpectedly contains proposal traces: {selected.path}")
    return streams


def _read_rows(path: Path | None) -> list[dict[str, Any]]:
    return [] if path is None else list(iter_jsonl(path))


def _initial_stream_source(
    streams: Mapping[str, Path | None],
    verified_inputs: Mapping[str, Path],
) -> Path | None:
    # Online arms all point at the dataset workflow's shared archive.  Prefer
    # that checksum-verified input over a per-run output copy when both exist.
    return verified_inputs.get("initial_batch") or streams.get("initial")


def _optional_nonnegative_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EhrlichSweepError(f"{label} must be a non-negative integer")
    return int(value)


def _required_nonnegative_int(value: Any, *, label: str) -> int:
    parsed = _optional_nonnegative_int(value, label=label)
    if parsed is None:
        raise EhrlichSweepError(f"{label} is required")
    return parsed


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row_score(row: Mapping[str, Any]) -> float | None:
    for key in ("score_for_analysis", "diagnostic_score", "score"):
        value = _finite_number(row.get(key))
        if value is not None:
            return value
    return None


def _validate_sample_only_copy(
    initial_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Prove the engine's duplicate sample-only streams describe one cohort."""

    if len(initial_rows) != len(sample_rows):
        raise EhrlichSweepError(
            "sample-only initial/samples row-count mismatch: "
            f"initial={len(initial_rows)}, samples={len(sample_rows)}"
        )
    for index, (initial, sample) in enumerate(zip(initial_rows, sample_rows, strict=True)):
        if initial.get("seq") != sample.get("seq"):
            raise EhrlichSweepError(f"sample-only initial/samples sequence mismatch at row {index}")
        initial_score = _row_score(initial)
        sample_score = _row_score(sample)
        if initial_score is None or sample_score is None:
            agrees = initial_score is None and sample_score is None
        else:
            agrees = math.isclose(initial_score, sample_score, rel_tol=1e-12, abs_tol=1e-12)
        if not agrees:
            raise EhrlichSweepError(f"sample-only initial/samples score mismatch at row {index}")


def _trajectory_metrics(
    initial_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the recorded oracle trajectory with infeasible Ehrlich scores floored at zero."""

    initial_scores = [_row_score(row) or 0.0 for row in initial_rows]
    selected_scores = [_row_score(row) or 0.0 for row in sample_rows]
    initial_incumbent = max(initial_scores, default=0.0)
    incumbent = initial_incumbent
    improvement_count = 0
    for score in selected_scores:
        if score > incumbent:
            improvement_count += 1
            incumbent = score

    running_best: list[float] = []
    best = 0.0
    for score in [*initial_scores, *selected_scores]:
        best = max(best, score)
        running_best.append(best)
    auc = sum(running_best)
    post_initial = running_best[len(initial_scores) :]
    return {
        "selected_score_mean": (sum(selected_scores) / len(selected_scores) if selected_scores else None),
        "selected_best_score": max(selected_scores) if selected_scores else None,
        "initial_incumbent_score": initial_incumbent,
        "final_best_score": best if running_best else None,
        "improvement_over_initial": max(0.0, incumbent - initial_incumbent),
        "sequential_improvement_count": improvement_count,
        "sequential_improvement_rate": (improvement_count / len(selected_scores) if selected_scores else None),
        "best_so_far_auc": auc if running_best else None,
        "best_so_far_mean": auc / len(running_best) if running_best else None,
        "normalized_post_initial_auc": (sum(post_initial) / len(post_initial) if post_initial else None),
        "best_so_far_trajectory": running_best,
        "trajectory_oracle_calls": len(running_best),
    }


def _validate_budget(
    *,
    resolved: Mapping[str, Any],
    initial_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sample_only = bool(resolved.get("sample_only"))
    initial_evaluated = sum(row.get("oracle_evaluated") is True for row in initial_rows)
    if initial_rows and initial_evaluated != len(initial_rows):
        raise EhrlichSweepError("initial_batch contains rows not marked oracle_evaluated")
    configured_initial = (
        None
        if sample_only
        else _optional_nonnegative_int(
            resolved.get("initial_oracle_evaluations"),
            label="initial_oracle_evaluations",
        )
    )
    if configured_initial is not None and initial_evaluated != configured_initial:
        raise EhrlichSweepError(
            f"initial oracle count mismatch: config={configured_initial}, output={initial_evaluated}"
        )

    sample_evaluated = sum(row.get("oracle_evaluated") is True for row in sample_rows)
    if bool(resolved.get("active_learning")) and sample_evaluated != len(sample_rows):
        raise EhrlichSweepError("active-learning samples contain non-oracle rows")

    running_budget_total = initial_evaluated
    running_physical_total = initial_evaluated
    running_privileged_total = 0
    accepted_evaluated = 0
    diagnostic_scored = 0
    previous_batch = -1
    used_legacy_scored_count = False
    diagnostic_budget_debits: list[tuple[int, int]] = []
    for index, row in enumerate(diagnostic_rows):
        batch_id = _required_nonnegative_int(row.get("batch_id"), label=f"diagnostics[{index}].batch_id")
        if batch_id < previous_batch:
            raise EhrlichSweepError("diagnostic batch ids must be nondecreasing")
        previous_batch = batch_id
        evaluated = _required_nonnegative_int(
            row.get("oracle_evaluated_count"),
            label=f"diagnostics[{index}].oracle_evaluated_count",
        )
        raw_scored = row.get("oracle_scored_count")
        if raw_scored is None:
            scored = evaluated
            used_legacy_scored_count = True
        else:
            scored = _required_nonnegative_int(
                raw_scored,
                label=f"diagnostics[{index}].oracle_scored_count",
            )
        if scored < evaluated:
            raise EhrlichSweepError("oracle_scored_count cannot be smaller than oracle_evaluated_count")
        raw_debit = row.get("oracle_budget_debit_count")
        if raw_debit is None:
            debit = scored
        else:
            debit = _required_nonnegative_int(
                raw_debit,
                label=f"diagnostics[{index}].oracle_budget_debit_count",
            )
        privileged = _required_nonnegative_int(
            row.get("oracle_privileged_scored_count", 0),
            label=f"diagnostics[{index}].oracle_privileged_scored_count",
        )
        if scored != debit + privileged:
            raise EhrlichSweepError(
                "oracle accounting mismatch: "
                f"scored={scored}, budget_debit={debit}, privileged={privileged}"
            )
        if debit < evaluated:
            raise EhrlichSweepError("oracle_budget_debit_count cannot be smaller than oracle_evaluated_count")
        diagnostic_budget_debits.append((batch_id, debit))
        running_budget_total += debit
        running_physical_total += scored
        running_privileged_total += privileged
        reported_total = _required_nonnegative_int(
            row.get("oracle_total_after_iteration"),
            label=f"diagnostics[{index}].oracle_total_after_iteration",
        )
        if reported_total != running_budget_total:
            raise EhrlichSweepError(
                f"oracle total mismatch in batch {batch_id}: expected {running_budget_total}, got {reported_total}"
            )
        cumulative_fields = (
            ("oracle_budget_total_after_iteration", running_budget_total),
            ("oracle_physical_total_after_iteration", running_physical_total),
            ("oracle_privileged_total_after_iteration", running_privileged_total),
        )
        for field, expected_total in cumulative_fields:
            raw_cumulative = row.get(field)
            if raw_cumulative is None:
                continue
            cumulative = _required_nonnegative_int(
                raw_cumulative,
                label=f"diagnostics[{index}].{field}",
            )
            if cumulative != expected_total:
                raise EhrlichSweepError(
                    f"{field} mismatch in batch {batch_id}: expected {expected_total}, got {cumulative}"
                )
        if row.get("proposal_batch_discarded") is not True:
            accepted_evaluated += evaluated
        diagnostic_scored += scored

    if diagnostic_rows and accepted_evaluated != sample_evaluated:
        raise EhrlichSweepError(
            f"selected sample count {sample_evaluated} does not match accepted diagnostics {accepted_evaluated}"
        )
    if not diagnostic_rows:
        running_budget_total += sample_evaluated
        running_physical_total += sample_evaluated
        accepted_evaluated = sample_evaluated
        diagnostic_scored = sample_evaluated

    maximum = _optional_nonnegative_int(
        resolved.get("max_oracle_evaluations"),
        label="max_oracle_evaluations",
    )
    if maximum is not None and running_budget_total > maximum:
        raise EhrlichSweepError(
            f"oracle budget exceeded: total={running_budget_total}, maximum={maximum}"
        )
    iterations = _optional_nonnegative_int(resolved.get("num_iterations"), label="num_iterations")
    overflow_batches = [batch_id for batch_id, _debit in diagnostic_budget_debits if batch_id >= (iterations or 0)]
    if iterations is not None and overflow_batches:
        batch_size = _optional_nonnegative_int(resolved.get("al_oracle_batch_size"), label="al_oracle_batch_size")
        nominal_total = initial_evaluated + sum(
            debit for batch_id, debit in diagnostic_budget_debits if batch_id < iterations
        )
        exact_budget_recovery = (
            maximum is not None
            and batch_size is not None
            and initial_evaluated + iterations * batch_size == maximum
            and nominal_total < maximum
            and running_budget_total == maximum
            and max(overflow_batches) < iterations * 2
        )
        if not exact_budget_recovery:
            raise EhrlichSweepError("diagnostic batch id exceeds the resolved iteration count")
    return {
        "initial_oracle_evaluations": initial_evaluated,
        "selected_oracle_evaluations": accepted_evaluated,
        "diagnostic_oracle_evaluations": diagnostic_scored - accepted_evaluated,
        # Keep the historical field as an alias for the charged/archive
        # budget.  New reporting also exposes physical and privileged work so
        # oracle-best upper bounds cannot be mistaken for ordinary oracle cost.
        "oracle_evaluations_total": running_budget_total,
        "oracle_budgeted_evaluations_total": running_budget_total,
        "oracle_physical_evaluations_total": running_physical_total,
        "oracle_privileged_evaluations_total": running_privileged_total,
        "max_oracle_evaluations": maximum,
        "oracle_budget_exhausted": maximum is not None and running_budget_total == maximum,
        "oracle_scored_count_source": "legacy_evaluated_count" if used_legacy_scored_count else "oracle_scored_count",
    }


def _validate_cohort(runs: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "git_sha",
        "dirty_diff_hash",
        "dependency_profile",
        "checkpoint_sha256",
        "runtime_config_sha256",
        "reward_instance_sha256",
    )
    first = runs[0]
    for field in fields:
        if first.get(field) is None and field != "dirty_diff_hash":
            raise EhrlichSweepError(f"cohort provenance is missing {field}")
        values = {run.get(field) for run in runs}
        if len(values) != 1:
            raise EhrlichSweepError(f"mixed cohort provenance for {field}")


def _validate_recorded_oracle_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    instance_path: Path,
    reward_spec: Mapping[str, Any],
    reward_cache: dict[tuple[Path, int | None], Any],
) -> None:
    """Re-score only archived selections, never the unselected proposal pool."""

    if not rows:
        return
    try:
        import torch

        from ehrlich.ehrlich_reward import EhrlichReward
    except ImportError as exc:  # pragma: no cover - exercised only without the CPU/search profile
        raise EhrlichSweepError("PyTorch and the Ehrlich package are required to verify oracle scores") from exc

    quantization_raw = reward_spec.get("quantization_override")
    quantization = int(quantization_raw) if quantization_raw is not None else None
    cache_key = (instance_path, quantization)
    reward = reward_cache.get(cache_key)
    if reward is None:
        reward = EhrlichReward(
            instance_path=instance_path,
            quantization_override=quantization,
        )
        reward_cache[cache_key] = reward

    sequences: list[list[int]] = []
    for index, row in enumerate(rows):
        sequence = row.get("seq")
        if (
            not isinstance(sequence, list)
            or not sequence
            or any(isinstance(token, bool) or not isinstance(token, int) for token in sequence)
        ):
            raise EhrlichSweepError(f"selected oracle row {index} has an invalid sequence")
        sequences.append(sequence)
    with torch.no_grad():
        rescored = reward.score_batch(torch.as_tensor(sequences, dtype=torch.long))
    offline_scores = [float(value) for value in rescored.detach().cpu().view(-1).tolist()]
    if len(offline_scores) != len(rows):
        raise EhrlichSweepError("Ehrlich oracle returned the wrong number of selected scores")
    for index, (row, offline_score) in enumerate(zip(rows, offline_scores, strict=True)):
        recorded_score = _finite_number(row.get("score"))
        agrees = (
            recorded_score is not None
            and math.isclose(recorded_score, offline_score, rel_tol=1e-9, abs_tol=1e-9)
            if math.isfinite(offline_score)
            else recorded_score is None
        )
        if not agrees:
            raise EhrlichSweepError(f"recorded oracle score disagrees with Ehrlich rescoring at selected row {index}")


def _analyze_run(
    run_dir: Path,
    *,
    run_root: Path,
    repo_root: Path | None,
    reward_cache: dict[tuple[Path, int | None], Any],
) -> dict[str, Any]:
    selected = select_manifest(run_dir)
    resolved, verified_inputs = _validate_manifest(
        selected,
        run_dir=run_dir,
        run_root=run_root,
        repo_root=repo_root,
    )
    streams = _resolve_streams(selected, resolved)
    initial_source = _initial_stream_source(streams, verified_inputs)
    initial_rows = _read_rows(initial_source)
    sample_rows = _read_rows(streams["samples"])
    diagnostic_rows = _read_rows(streams["diagnostics"])
    # ``sample_only`` serializes the same unconditional cohort both as the
    # engine's initial state and as the public samples stream.  The latter is
    # the canonical 2,000-call trajectory; counting both would duplicate every
    # oracle call.  Both files remain checksum-verified by the manifest.
    if bool(resolved.get("sample_only")):
        _validate_sample_only_copy(initial_rows, sample_rows)
        trajectory_initial_rows: list[dict[str, Any]] = []
    else:
        trajectory_initial_rows = initial_rows
    budget = _validate_budget(
        resolved=resolved,
        initial_rows=trajectory_initial_rows,
        sample_rows=sample_rows,
        diagnostic_rows=diagnostic_rows,
    )
    _validate_recorded_oracle_scores(
        sample_rows,
        instance_path=verified_inputs["reward_instance"],
        reward_spec=_reward_spec(resolved),
        reward_cache=reward_cache,
    )

    method = run_dir.parent.name
    seed_match = _SEED_DIRECTORY.fullmatch(run_dir.name)
    assert seed_match is not None
    seed = int(seed_match.group("seed"))
    trace_path = streams["traces"]
    trace_rows = _required_nonnegative_int(
        selected.data.get("trace_records_written", 0),
        label="manifest trace_records_written",
    )
    if trace_path is None and trace_rows:
        raise EhrlichSweepError(f"manifest records proposal traces but no trace output exists in {run_dir}")
    if trace_path is not None:
        observed_trace_rows = sum(1 for _row in iter_jsonl(trace_path))
        if observed_trace_rows != trace_rows:
            raise EhrlichSweepError(
                f"trace row count mismatch in {run_dir}: "
                f"manifest={trace_rows}, output={observed_trace_rows}"
            )

    trajectory = _trajectory_metrics(trajectory_initial_rows, sample_rows)
    trajectory_complete = trajectory["trajectory_oracle_calls"] == budget["oracle_evaluations_total"]
    if not trajectory_complete:
        trajectory["best_so_far_auc"] = None
        trajectory["best_so_far_mean"] = None
        trajectory["normalized_post_initial_auc"] = None
        trajectory["best_so_far_trajectory"] = []
    stale_details = [
        {
            "manifest": candidate.path.name,
            "reasons": candidate.mismatch_reasons,
        }
        for candidate in selected.stale
    ]
    input_hashes = selected.data["input_hashes"]
    run_summary = {
        "method": method,
        "seed": seed,
        "feature_space": resolved.get("al_feature_space"),
        "acquisition": resolved.get("al_acquisition"),
        "run_directory": run_dir.relative_to(run_root).as_posix(),
        "manifest": selected.path.name,
        "stale_manifest_count": len(selected.stale),
        "stale_manifests": stale_details,
        "git_sha": selected.data.get("git_sha"),
        "dirty_diff_hash": selected.data.get("dirty_diff_hash"),
        "dependency_profile": selected.data.get("dependency_profile"),
        "implementation_record_path": selected.data.get("implementation_record_path"),
        "implementation_record_sha256": selected.data.get("implementation_record_sha256"),
        "started_at": selected.data.get("started_at"),
        "completed_at": selected.data.get("completed_at"),
        "input_hash_count": len(verified_inputs),
        "checkpoint_sha256": input_hashes.get("checkpoint"),
        "runtime_config_sha256": input_hashes.get("runtime_config"),
        "reward_instance_sha256": input_hashes.get("reward_instance"),
        "search_config_sha256": input_hashes.get("search_config"),
        "initial_batch_sha256": input_hashes.get("initial_batch"),
        "output_hash_count": len(selected.outputs),
        "initial_rows": len(trajectory_initial_rows),
        "serialized_initial_rows": len(initial_rows),
        "sample_rows": len(sample_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "trace_rows": trace_rows,
        "has_full_traces": trace_path is not None,
        "trace_truncated": bool(selected.data.get("trace_truncated", False)),
        "trajectory_covers_all_oracle_calls": trajectory_complete,
        **trajectory,
        **budget,
    }
    return run_summary


def analyze_sweep(
    run_root: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate and summarize Ehrlich ``<arm>/seedN`` runs without mutating them.

    The study-specific four-arm matrix and exact budget are checked by the
    results processor, which has the four canonical configs in hand.  This
    layer validates only claims that can be proven from each raw run.
    """

    root = Path(run_root).expanduser().resolve()
    repository = _infer_repo_root(root, repo_root)
    run_directories = discover_run_directories(root)
    runs: list[dict[str, Any]] = []
    reward_cache: dict[tuple[Path, int | None], Any] = {}
    for run_dir in run_directories:
        run_summary = _analyze_run(
            run_dir,
            run_root=root,
            repo_root=repository,
            reward_cache=reward_cache,
        )
        runs.append(run_summary)
    runs.sort(key=lambda row: (str(row["method"]), int(row["seed"])))
    _validate_cohort(runs)
    try:
        displayed_root = root.relative_to(repository).as_posix() if repository is not None else str(root)
    except ValueError:
        displayed_root = str(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "genturbo.kernel_study.online_analysis",
        "study": "ehrlich_kernel_v1",
        "run_root": displayed_root,
        "run_count": len(runs),
        "stale_manifest_count": sum(int(run["stale_manifest_count"]) for run in runs),
        "runs": runs,
    }


__all__ = [
    "EhrlichSweepError",
    "analyze_sweep",
    "discover_run_directories",
    "select_manifest",
]
