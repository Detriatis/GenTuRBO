"""Reward-blind pretraining and generic online execution for LaMBO-2.

Non-Ehrlich resolved cells provide ``lambo2.reward_adapter`` as
``{module, class, kwargs}``; the constructed class must expose ``score_batch``.
Sequence length, alphabet, prepaid archive size, checkpoint, pretraining
attestation, and optional score-free corpus inputs are resolved per cell.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from genturbo.artifacts import sha256_file
from genturbo.io import iter_jsonl, open_text, write_jsonl
from genturbo.serialization import atomic_write_json, canonical_json_bytes

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_ONLINE_ARCHIVE_RANDOM = "online_archive_random"
_PRETRAINED_GENERATOR = "pretrained_generator_checkpoint"
_COMPLETE_PAID_HISTORY = "complete_paid_history"
_EXACT_DATASET_TARGET = "genturbo.sequence_baseline_study.lambo2._exact_in_memory_numpy_dataset"


class LaMBO2ProtocolError(RuntimeError):
    """Raised when the isolated LaMBO-2 protocol is unavailable or has drifted."""


def _set_lambo_dependency_profile() -> str:
    """Set the local fallback without replacing an explicit runtime profile."""

    return os.environ.setdefault("GENTURBO_DEPENDENCY_PROFILE", "lambo2-py310")


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaMBO2ProtocolError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaMBO2ProtocolError(f"{path} must contain a JSON object")
    return value


def _lambo_config(study_config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _load_json(study_config)
    lambo = raw.get("lambo2")
    if not isinstance(lambo, dict) or not isinstance(lambo.get("pretraining"), dict):
        raise LaMBO2ProtocolError("study config has no LaMBO-2 pretraining declaration")
    return raw, lambo


def pretraining_plan(study_config: Path | str, *, repo_root: Path | str) -> dict[str, Any]:
    """Return the immutable reward-free pretraining contract without importing Cortex."""

    root = Path(repo_root).resolve()
    config_path = _resolve(root, study_config)
    _raw, lambo = _lambo_config(config_path)
    spec = dict(lambo["pretraining"])
    source = _resolve(root, str(spec["markov_training_config"]))
    reference_checkpoint = _resolve(root, str(spec["reference_checkpoint"]))
    checkpoint = _resolve(root, str(spec["checkpoint"]))
    manifest = _resolve(root, str(spec["manifest"]))
    batch_size = int(spec["batch_size"])
    sampled_corpus_size = int(spec["sampled_corpus_size"])
    train_fraction = float(spec["adapter_train_fraction"])
    training_corpus_size = int(spec["training_corpus_size"])
    epochs = int(spec["epochs"])
    sequence_length = int(spec["sequence_length"])
    if int(sampled_corpus_size * train_fraction) != training_corpus_size:
        raise LaMBO2ProtocolError("LaMBO-2 adapter training split no longer matches its declared corpus")
    if training_corpus_size % batch_size:
        raise LaMBO2ProtocolError("LaMBO-2 training corpus must divide exactly into batches")
    steps = training_corpus_size // batch_size * epochs
    token_exposures = steps * batch_size * sequence_length
    expected = int(spec["matched_token_exposures"])
    reference = (
        int(spec["reference_updates"])
        * int(spec["reference_batch_size"])
        * int(spec["reference_sequence_length"])
    )
    if token_exposures != expected or token_exposures != reference:
        raise LaMBO2ProtocolError(f"pretraining token exposures drifted: {token_exposures} != {expected}")
    if not source.is_file() or not reference_checkpoint.is_file():
        raise LaMBO2ProtocolError("LaMBO-2 pretraining source config or reference CTMC checkpoint is missing")
    status = "pending"
    if checkpoint.exists() or manifest.exists():
        status = "invalid"
        if checkpoint.is_file() and manifest.is_file():
            record = _load_json(manifest)
            expected_spec = hashlib.sha256(canonical_json_bytes(spec)).hexdigest()
            if (
                record.get("checkpoint_sha256") == sha256_file(checkpoint)
                and record.get("pretraining_spec_sha256") == expected_spec
                and record.get("oracle_labels_seen") == 0
                and record.get("task_motifs_seen") is False
            ):
                status = "completed"
    return {
        "status": status,
        "study_config": str(config_path),
        "markov_training_config": str(source),
        "reference_checkpoint": str(reference_checkpoint),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "sampled_corpus_size": sampled_corpus_size,
        "adapter_train_fraction": train_fraction,
        "training_corpus_size": training_corpus_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "gradient_steps": steps,
        "sequence_length": sequence_length,
        "matched_token_exposures": token_exposures,
        "oracle_labels_seen": 0,
        "task_motifs_seen": False,
        "device": str(spec["device"]),
        "estimated_hours": list(spec["estimated_hours"]),
    }


def _patch_mps_corruption() -> None:
    """Use float32 uniforms only where Cortex's float64 mask sampling is unsupported."""

    import torch
    from cortex.corruption import MaskCorruptionProcess

    def corrupt(
        self: Any,
        x_start: Any,
        corrupt_frac: float,
        mask_val: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        del self, args, kwargs
        dtype = torch.float32 if x_start.device.type == "mps" else torch.float64
        is_corrupted = torch.rand_like(x_start, dtype=dtype) < corrupt_frac
        return torch.where(is_corrupted, torch.full_like(x_start, mask_val), x_start), is_corrupted

    MaskCorruptionProcess._corrupt = corrupt


def _markov_distribution(training_config: Path) -> Any:
    """Recreate only the generator's training distribution; no task instance is read."""

    from ehrlich.ehrlich_instance import EhrlichConfig, make_ehrlich_instance

    raw = _load_json(training_config)
    dataset = raw.get("dataset_cfg")
    if raw.get("dataset") != "ehrlich_markov" or not isinstance(dataset, dict):
        raise LaMBO2ProtocolError("LaMBO-2 requires the pinned reward-free ehrlich_markov training config")
    instance_cfg = dataset.get("instance_cfg")
    if not isinstance(instance_cfg, dict):
        raise LaMBO2ProtocolError("generator training config has no instance_cfg")
    instance = make_ehrlich_instance(EhrlichConfig(**instance_cfg), seed=int(dataset["instance_seed"]))
    return instance.transition_probs


def _unlabeled_corpus(
    transition_probs: Any,
    *,
    count: int,
    sequence_length: int,
    seed: int,
    chunk_size: int = 2048,
) -> Any:
    """Sample deterministic protein strings from the Markov law, without labels."""

    import numpy as np
    import torch

    from ehrlich.ehrlich_generator import sample_markov_sequences

    generator = torch.Generator().manual_seed(int(seed))
    sequences: list[str] = []
    while len(sequences) < count:
        batch = sample_markov_sequences(
            transition_probs,
            sequence_length,
            min(chunk_size, count - len(sequences)),
            generator=generator,
        )
        sequences.extend(" ".join(AMINO_ACIDS[token] for token in row) for row in batch.tolist())
    return np.asarray(sequences, dtype=object)


def pretrain_generator(study_config: Path | str, *, repo_root: Path | str) -> dict[str, Any]:
    """Train the LaMBO denoising head on unlabeled Ehrlich samples exactly once."""

    import hydra
    import lightning as lightning
    from omegaconf import OmegaConf
    from poli_baselines.solvers.bayesian_optimization.lambo2.solver import DEFAULT_CONFIG_DIR

    from search.interpolant.manifest import repository_identity

    _patch_mps_corruption()
    root = Path(repo_root).resolve()
    config_path = _resolve(root, study_config)
    _raw, lambo = _lambo_config(config_path)
    spec = dict(lambo["pretraining"])
    plan = pretraining_plan(config_path, repo_root=root)
    checkpoint = Path(plan["checkpoint"])
    manifest_path = Path(plan["manifest"])
    if checkpoint.exists() or manifest_path.exists():
        if not checkpoint.is_file() or not manifest_path.is_file():
            raise LaMBO2ProtocolError("partial LaMBO-2 pretraining output exists; refusing to overwrite it")
        manifest = _load_json(manifest_path)
        expected_spec = hashlib.sha256(canonical_json_bytes(spec)).hexdigest()
        if (
            manifest.get("checkpoint_sha256") != sha256_file(checkpoint)
            or manifest.get("pretraining_spec_sha256") != expected_spec
            or manifest.get("oracle_labels_seen") != 0
            or manifest.get("task_motifs_seen") is not False
        ):
            raise LaMBO2ProtocolError("existing LaMBO-2 checkpoint does not match its manifest")
        return {**plan, "outcome": "skipped_completed", "checkpoint_sha256": sha256_file(checkpoint)}

    training_config = Path(plan["markov_training_config"])
    corpus = _unlabeled_corpus(
        _markov_distribution(training_config),
        count=int(plan["sampled_corpus_size"]),
        sequence_length=int(plan["sequence_length"]),
        seed=int(spec["seed"]),
    )
    overrides = [
        "tasks=[protein_seq]",
        "branches=[protein_generation]",
        f"accelerator={spec['device']}",
        "trainer.precision=32",
        f"batch_size={int(spec['batch_size'])}",
        f"max_epochs={int(spec['epochs'])}",
        f"max_sequence_length={int(spec['sequence_length'])}",
    ]
    with hydra.initialize_config_dir(config_dir=str(DEFAULT_CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="generic_training", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    lightning.seed_everything(int(spec["seed"]), workers=True)
    model = hydra.utils.instantiate(cfg.tree)
    model.build_tree(cfg, skip_task_setup=True)
    task = model.task_dict["protein_seq"]
    for stage in ("test", "fit"):
        task.data_module.setup(stage=stage, dataset_kwargs={"data": {"tokenized_seq": corpus}})
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        max_epochs=int(spec["epochs"]),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=model.get_dataloader(split="train"))
    if int(trainer.global_step) != int(plan["gradient_steps"]):
        raise LaMBO2ProtocolError(
            f"pretraining stopped at {trainer.global_step} steps; expected {plan['gradient_steps']}"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    trainer.save_checkpoint(temporary)
    os.replace(temporary, checkpoint)
    git_sha, dirty_hash = repository_identity(root)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "method": "lambo2_generator_only_pretraining",
        "oracle_labels_seen": 0,
        "task_motifs_seen": False,
        "markov_training_config": str(spec["markov_training_config"]),
        "markov_training_config_sha256": sha256_file(training_config),
        "reference_checkpoint_sha256": sha256_file(Path(plan["reference_checkpoint"])),
        "study_config_sha256": sha256_file(config_path),
        "pretraining_spec_sha256": hashlib.sha256(canonical_json_bytes(spec)).hexdigest(),
        "checkpoint": str(spec["checkpoint"]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_sha": git_sha,
        "dirty_diff_hash": dirty_hash,
        "adapter_commit": lambo["adapter_commit"],
        "poli_commit": lambo["poli_commit"],
        "cortex_version": lambo["cortex_version"],
        **{
            key: plan[key]
            for key in (
                "sampled_corpus_size",
                "adapter_train_fraction",
                "training_corpus_size",
                "batch_size",
                "epochs",
                "gradient_steps",
                "sequence_length",
                "matched_token_exposures",
            )
        },
    }
    atomic_write_json(manifest_path, manifest)
    return {**plan, "outcome": "completed", "checkpoint_sha256": manifest["checkpoint_sha256"]}


def _alphabet_map(alphabet: str) -> dict[str, int]:
    if len(alphabet) < 2 or len(set(alphabet)) != len(alphabet):
        raise LaMBO2ProtocolError("LaMBO-2 alphabet must contain at least two unique tokens")
    return {token: index for index, token in enumerate(alphabet)}


def _strings_to_ids(
    values: Any,
    *,
    alphabet: str = AMINO_ACIDS,
    sequence_length: int = 256,
) -> list[list[int]]:
    import numpy as np

    array = np.asarray(values)
    rows = array.reshape(len(array), -1)
    strings = [
        (str(row[0]) if len(row) == 1 else "".join(str(token) for token in row)).replace(" ", "")
        for row in rows
    ]
    try:
        token_of = _alphabet_map(alphabet)
        encoded = [[token_of[token] for token in sequence] for sequence in strings]
    except KeyError as exc:
        raise LaMBO2ProtocolError(f"LaMBO-2 emitted a non-protein token: {exc.args[0]!r}") from exc
    if any(len(sequence) != sequence_length for sequence in encoded):
        raise LaMBO2ProtocolError(
            f"LaMBO-2 emitted a sequence with length other than {sequence_length}"
        )
    return encoded


def _initial_archive(
    path: Path,
    *,
    expected_count: int = 64,
    sequence_length: int = 256,
    alphabet: str = AMINO_ACIDS,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    import numpy as np

    rows = list(iter_jsonl(path))
    if len(rows) != expected_count:
        raise LaMBO2ProtocolError(
            f"LaMBO-2 requires exactly {expected_count} initial rows, got {len(rows)}"
        )
    sequences = [row.get("seq") for row in rows]
    if any(not isinstance(sequence, list) or len(sequence) != sequence_length for sequence in sequences):
        raise LaMBO2ProtocolError("LaMBO-2 initial archive contains an invalid sequence")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 or token >= len(alphabet)
        for sequence in sequences
        for token in sequence
    ):
        raise LaMBO2ProtocolError("LaMBO-2 initial archive contains a token outside its alphabet")
    x0 = np.asarray([[alphabet[int(token)] for token in sequence] for sequence in sequences])
    scores: list[list[float]] = []
    for row in rows:
        if row.get("score_is_finite", True) is not True:
            scores.append([-float("inf")])
            continue
        score = float(row["score"])
        scores.append([score if math.isfinite(score) else -float("inf")])
    y0 = np.asarray(scores, dtype=float)
    return x0, y0, rows


def _transfer_generator_checkpoint(model_path: Path, pretrained_path: Path) -> int:
    import torch

    online = torch.load(model_path, map_location="cpu", weights_only=False)
    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    online_state = online.get("state_dict")
    pretrained_state = pretrained.get("state_dict")
    if not isinstance(online_state, dict) or not isinstance(pretrained_state, dict):
        raise LaMBO2ProtocolError("LaMBO-2 checkpoint has no state_dict")
    transferable = {
        key: value
        for key, value in pretrained_state.items()
        if key in online_state and tuple(value.shape) == tuple(online_state[key].shape)
    }
    if not transferable or set(transferable) != set(pretrained_state):
        raise LaMBO2ProtocolError("generator-only checkpoint does not match the online LaMBO-2 model")
    online_state.update(transferable)
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    torch.save(online, temporary)
    os.replace(temporary, model_path)
    return len(transferable)


def _fit_initial_objective_model(
    solver: Any,
    pretrained_path: Path,
    *,
    expected_transferred_tensors: int,
    epochs: int,
) -> None:
    """Transfer the reward-free generator, then persist the adapter's initial scored fit."""

    transferred = _transfer_generator_checkpoint(solver.model_path, pretrained_path)
    if transferred != expected_transferred_tensors:
        raise LaMBO2ProtocolError(
            f"transferred {transferred} generator tensors, expected {expected_transferred_tensors}"
        )
    if epochs < 1:
        raise LaMBO2ProtocolError("LaMBO-2 objective heads require an initial scored fit")
    solver._train_model_with_history(
        load_checkpoint_from=solver.model_path,
        save_checkpoint_to=solver.model_path,
        max_epochs=epochs,
    )


def _nearest_ancestor(
    child: Sequence[int],
    candidates: Any,
    *,
    alphabet: str = AMINO_ACIDS,
    sequence_length: int = 256,
) -> tuple[list[int], int]:
    encoded = _strings_to_ids(candidates, alphabet=alphabet, sequence_length=sequence_length)
    distances = [sum(left != right for left, right in zip(parent, child, strict=True)) for parent in encoded]
    index = min(range(len(encoded)), key=distances.__getitem__)
    return encoded[index], distances[index]


def _json_line(stream: Any, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
    stream.write("\n")


def _sample_row(
    sequence: list[int],
    score: float,
    *,
    batch_id: int,
    seq_idx: int,
    alphabet: str = AMINO_ACIDS,
) -> dict[str, Any]:
    finite = math.isfinite(score)
    sequence_string = "".join(alphabet[token] for token in sequence)
    return {
        "batch_id": batch_id,
        "seq_idx": seq_idx,
        "seq": sequence,
        "seq_length": len(sequence),
        "score": score if finite else None,
        "score_is_finite": finite,
        "score_for_analysis": score if finite else None,
        "decoded_sequence": sequence_string,
        "sequence_string": sequence_string,
        "oracle_evaluated": True,
        "oracle_rejected": not finite,
        "smiles": sequence_string,
    }


def _trace_row(
    sample: Mapping[str, Any],
    *,
    ancestor: list[int],
    edits: int,
    duplicate: bool,
) -> dict[str, Any]:
    return {
        **sample,
        "acquisition_candidate": True,
        "acquisition_selected": True,
        "duplicate_exact_proposal": duplicate,
        "duplicate_generated_leaf": False,
        "cached_generated_proposal": False,
        "invalid_generated_proposal": False,
        "expected_edits": float(edits),
        "proposal_logical_edits": edits,
        "ancestor_to_child_edits": edits,
        "ancestor_inference": "nearest_lambo_candidate_point",
        "ancestor_seq": ancestor,
        "from_seq": ancestor,
        "to_seq": sample["seq"],
    }


def _diagnostic_row(
    batch_id: int,
    samples: Sequence[Mapping[str, Any]],
    *,
    total: int,
    protocol: Mapping[str, Any],
    bound_input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    unique = {tuple(int(token) for token in row["seq"]) for row in samples}
    rejected = sum(row["oracle_rejected"] is True for row in samples)
    row = {
        "batch_id": batch_id,
        "generated_proposals": len(samples),
        "deduped_proposals": len(unique),
        "duplicate_exact_proposals": len(samples) - len(unique),
        "duplicate_generated_leaves": 0,
        "cached_generated_proposals": 0,
        "invalid_generated_proposals": 0,
        "acquisition_batch_size": len(samples),
        "oracle_evaluated_count": len(samples),
        "oracle_rejected_count": rejected,
        "oracle_budget_debit_count": len(samples),
        "oracle_scored_count": len(samples),
        "oracle_privileged_scored_count": 0,
        "oracle_budget_total_after_iteration": total,
        "oracle_physical_total_after_iteration": total,
        "oracle_privileged_total_after_iteration": 0,
        "oracle_total_after_iteration": total,
        "proposal_batch_discarded": False,
        "proposal_operator": "guided_discrete_diffusion",
        "lambo_direct_design_count": len(samples),
        "lambo_objective_refit_epochs": int(protocol["retraining_epochs"]),
        "lambo_declared_objective_refit_epochs": int(
            protocol.get("objective_refit_epochs_per_round", protocol["retraining_epochs"])
        ),
        "lambo_refit_schedule": "complete_observed_history_before_guided_batch",
        "lambo_bound_input_hashes": dict(sorted(bound_input_hashes.items())),
    }
    if protocol.get("initialization") == _ONLINE_ARCHIVE_RANDOM:
        row.update(
            {
                "lambo_initialization": _ONLINE_ARCHIVE_RANDOM,
                "lambo_training_history": _COMPLETE_PAID_HISTORY,
                "lambo_initial_training_epochs": int(protocol["initial_training_epochs"]),
                "lambo_training_rows_before_refit": total - len(samples),
                "lambo_exact_dataset_rows": True,
            }
        )
    return row


def _online_reward_adapter(
    resolved: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Any, dict[str, Path]]:
    """Instantiate the declared task reward, retaining the Ehrlich v2 fallback.

    New task adapters declare ``lambo2.reward_adapter`` with JSON fields
    ``module``, ``class``, and ``kwargs``. Keyword values are passed unchanged;
    relative file paths therefore resolve from the repository-root worker cwd.
    Existing Ehrlich cells omit this declaration and continue to construct the
    historical :class:`EhrlichReward` from the top-level reward mapping.
    """

    lambo = resolved.get("lambo2")
    adapter = lambo.get("reward_adapter") if isinstance(lambo, Mapping) else None
    if adapter is None:
        from ehrlich.ehrlich_reward import EhrlichReward

        reward_spec = resolved["reward"]
        reward_path = _resolve(repo_root, reward_spec["instance_path"])
        return (
            EhrlichReward(
                instance_path=reward_path,
                quantization_override=int(reward_spec["quantization_override"]),
            ),
            {"reward_instance": reward_path},
        )
    if not isinstance(adapter, Mapping):
        raise LaMBO2ProtocolError("lambo2.reward_adapter must be a mapping")
    module_name = adapter.get("module")
    class_name = adapter.get("class")
    kwargs = adapter.get("kwargs")
    if (
        not isinstance(module_name, str)
        or not module_name
        or not isinstance(class_name, str)
        or not class_name
        or not isinstance(kwargs, Mapping)
    ):
        raise LaMBO2ProtocolError(
            "lambo2.reward_adapter requires non-empty module/class strings and kwargs mapping"
        )
    try:
        adapter_class = getattr(importlib.import_module(module_name), class_name)
        reward = adapter_class(**dict(kwargs))
    except (AttributeError, ImportError, TypeError) as exc:
        raise LaMBO2ProtocolError(
            f"cannot construct LaMBO-2 reward adapter {module_name}.{class_name}: {exc}"
        ) from exc
    if not callable(getattr(reward, "score_batch", None)):
        raise LaMBO2ProtocolError("LaMBO-2 reward adapter must expose score_batch")
    input_paths: dict[str, Path] = {}
    for key, value in kwargs.items():
        if isinstance(value, str):
            candidate = _resolve(repo_root, value)
            if candidate.is_file():
                input_paths[f"reward_adapter_{key}"] = candidate
    return reward, input_paths


def _declared_corpus_inputs(
    lambo: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Path]:
    """Resolve optional score-free corpus files declared by a generic cell."""

    corpus = lambo.get("corpus")
    if corpus is None:
        return {}
    if not isinstance(corpus, Mapping) or not corpus:
        raise LaMBO2ProtocolError("lambo2.corpus must be a non-empty mapping")
    inputs: dict[str, Path] = {}
    for key, value in corpus.items():
        if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
            raise LaMBO2ProtocolError("lambo2.corpus keys and paths must be non-empty strings")
        path = _resolve(repo_root, value)
        if not path.is_file():
            raise LaMBO2ProtocolError(f"declared LaMBO-2 corpus input is missing: {path}")
        inputs[f"lambo_corpus_{key}"] = path
    return inputs


def _exact_in_memory_numpy_dataset(
    data: Mapping[str, Any],
    train: bool = True,
    random_seed: int = 0xDEADBEEF,
) -> Any:
    """Construct Cortex's in-memory dataset without its implicit 80/20 split.

    LaMBO already supplies an explicit, complete history to each task data
    module.  Cortex 0.0.4's :class:`NumpyDataset` independently keeps only 80%
    of that mapping for ``train=True``.  Reusing the upstream class and then
    restoring the supplied frame preserves its item/collator behaviour while
    making the task module's declared ``lengths: [1.0, 0.0]`` authoritative.
    """

    import pandas as pd
    from cortex.data.dataset import NumpyDataset

    columns = list(data)
    if not columns:
        raise ValueError("LaMBO-2 exact in-memory dataset requires at least one column")
    row_count = len(data[columns[0]])
    if any(len(data[column]) != row_count for column in columns):
        raise ValueError("all LaMBO-2 in-memory dataset columns must have equal length")
    dataset = NumpyDataset(data=dict(data), train=train, random_seed=random_seed)
    dataset._data = pd.DataFrame({column: data[column] for column in columns})
    dataset.columns = columns
    return dataset


def _initialization_inputs(
    resolved: Mapping[str, Any],
    lambo: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[str, Path | None, dict[str, Path]]:
    """Resolve only the inputs used by the selected initialization protocol."""

    declared_mode = lambo.get("initialization")
    mode = _PRETRAINED_GENERATOR if declared_mode is None else str(declared_mode)
    if mode == _ONLINE_ARCHIVE_RANDOM:
        if lambo.get("training_history") != _COMPLETE_PAID_HISTORY:
            raise LaMBO2ProtocolError("online_archive_random requires training_history=complete_paid_history")
        if int(lambo.get("initial_training_epochs", 0)) < 1:
            raise LaMBO2ProtocolError("online_archive_random requires at least one initial training epoch")
        return mode, None, {}
    if mode != _PRETRAINED_GENERATOR:
        raise LaMBO2ProtocolError(f"unsupported LaMBO-2 initialization: {mode!r}")

    checkpoint_value = resolved.get("checkpoint", lambo.get("checkpoint"))
    manifest_value = resolved.get("pretraining_manifest", lambo.get("pretraining_manifest"))
    if not isinstance(checkpoint_value, str) or not isinstance(manifest_value, str):
        raise LaMBO2ProtocolError("LaMBO-2 checkpoint and pretraining_manifest must be resolved paths")
    pretrained_path = _resolve(repo_root, checkpoint_value)
    pretraining_manifest = _resolve(repo_root, manifest_value)
    pretraining_record = _load_json(pretraining_manifest)
    expected_attestation = lambo.get(
        "pretraining_attestation",
        {"oracle_labels_seen": 0, "task_motifs_seen": False},
    )
    if (
        not isinstance(expected_attestation, Mapping)
        or not expected_attestation
        or any(not isinstance(key, str) or not key for key in expected_attestation)
        or pretraining_record.get("checkpoint_sha256") != sha256_file(pretrained_path)
        or any(pretraining_record.get(key) != value for key, value in expected_attestation.items())
    ):
        raise LaMBO2ProtocolError(
            "online cell's generator checkpoint is not the declared label-free pretraining output"
        )
    # Protein cells declare these diagnostics; historical Ehrlich cells do not.
    validation_keys = ("checkpoint_validation_samples", "checkpoint_validation_manifest")
    validation_inputs: dict[str, Path] = {}
    if any(key in resolved for key in validation_keys):
        for key in validation_keys:
            value = resolved.get(key)
            if not isinstance(value, str) or not value:
                raise LaMBO2ProtocolError(
                    "checkpoint validation samples and manifest must be declared together as paths"
                )
            path = _resolve(repo_root, value)
            if not path.is_file():
                raise LaMBO2ProtocolError(f"declared LaMBO-2 checkpoint validation input is missing: {path}")
            validation_inputs[key] = path
    corpus_inputs = _declared_corpus_inputs(lambo, repo_root=repo_root)
    return (
        mode,
        pretrained_path,
        {
            "checkpoint": pretrained_path,
            "pretraining_manifest": pretraining_manifest,
            **validation_inputs,
            **corpus_inputs,
        },
    )


def _solver_overrides(
    resolved: Mapping[str, Any],
    lambo: Mapping[str, Any],
    *,
    initialization: str,
    sequence_length: int,
    cell_dir: Path,
) -> list[str]:
    """Build the pinned upstream overrides for one initialization mode."""

    overrides = [
        (
            f"max_epochs={int(lambo['initial_training_epochs'])}"
            if initialization == _ONLINE_ARCHIVE_RANDOM
            else "max_epochs=0"
        ),
        f"batch_size={int(lambo['training_batch_size'])}",
        f"num_samples={int(resolved['al_oracle_batch_size'])}",
        f"max_sequence_length={sequence_length}",
        f"accelerator={lambo['device']}",
        "trainer.precision=32",
        f"num_steps={int(lambo['num_steps'])}",
        f"num_mutations_per_step={int(lambo['mutations_per_step'])}",
        f"optim.max_guidance_updates={int(lambo['max_guidance_updates'])}",
        f"data_dir={cell_dir / 'model'}",
        "+trainer.logger=false",
        "+trainer.enable_checkpointing=false",
        "+trainer.enable_progress_bar=false",
        "+trainer.enable_model_summary=false",
    ]
    if initialization == _ONLINE_ARCHIVE_RANDOM:
        overrides.extend(
            [
                f"tasks.protein_property.generic_task.data_module.dataset_config._target_={_EXACT_DATASET_TARGET}",
                f"tasks.protein_generation.protein_seq.data_module.dataset_config._target_={_EXACT_DATASET_TARGET}",
            ]
        )
    return overrides


def run_online(search_config: Path | str, *, repo_root: Path | str) -> None:
    """Run one resolved matched-online LaMBO-2 cell and emit standard study streams."""

    import numpy as np
    import torch
    from poli.core.abstract_black_box import AbstractBlackBox
    from poli.core.black_box_information import BlackBoxInformation
    from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2

    from search.interpolant.manifest import RunManifestRecorder, build_run_manifest

    _patch_mps_corruption()
    root = Path(repo_root).resolve()
    config_path = _resolve(root, search_config)
    resolved = _load_json(config_path)
    if resolved.get("runner") != "lambo2" or resolved.get("metadata", {}).get("plot_label") != "lambo2":
        raise LaMBO2ProtocolError("resolved config does not identify a LaMBO-2 cell")
    lambo = resolved["lambo2"]
    sequence_length = int(resolved["sequence_length"])
    initial_count = int(resolved["initial_oracle_evaluations"])
    alphabet = str(lambo.get("alphabet", AMINO_ACIDS))
    _alphabet_map(alphabet)
    task_name = str(lambo.get("task_name", "ehrlich_l256_q4_k8"))
    initial_path = _resolve(root, resolved["initial_batch_path"])
    reward, reward_inputs = _online_reward_adapter(resolved, repo_root=root)
    initialization, pretrained_path, initialization_inputs = _initialization_inputs(
        resolved,
        lambo,
        repo_root=root,
    )

    class TaskBlackBox(AbstractBlackBox):
        def __init__(self) -> None:
            super().__init__(evaluation_budget=int(resolved["max_oracle_evaluations"]) - initial_count)
            self.reward = reward

        def get_black_box_info(self) -> Any:
            return BlackBoxInformation(task_name, sequence_length, True, True, True, list(alphabet))

        def _black_box(self, x: Any, context: Any = None) -> Any:
            del context
            scores = self.reward.score_batch(
                torch.as_tensor(
                    _strings_to_ids(x, alphabet=alphabet, sequence_length=sequence_length),
                    dtype=torch.long,
                )
            )
            values = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
            return np.asarray(values, dtype=float).reshape(-1, 1)

    x0, y0, initial_rows = _initial_archive(
        initial_path,
        expected_count=initial_count,
        sequence_length=sequence_length,
        alphabet=alphabet,
    )
    cell_dir = config_path.parent
    overrides = _solver_overrides(
        resolved,
        lambo,
        initialization=initialization,
        sequence_length=sequence_length,
        cell_dir=cell_dir,
    )
    solver = LaMBO2(
        TaskBlackBox(),
        x0=x0,
        y0=y0,
        overrides=overrides,
        seed=int(resolved["seed"]),
        max_epochs_for_retraining=int(lambo["retraining_epochs"]),
    )
    if initialization == _PRETRAINED_GENERATOR:
        if pretrained_path is None:  # pragma: no cover - guarded by _initialization_inputs
            raise LaMBO2ProtocolError("pretrained LaMBO-2 initialization has no checkpoint")
        _fit_initial_objective_model(
            solver,
            pretrained_path,
            expected_transferred_tensors=int(lambo["expected_transferred_tensors"]),
            epochs=int(lambo["initial_training_epochs"]),
        )
        solver.cfg.max_epochs = int(lambo["retraining_epochs"])

    captured_candidates: list[Any] = []
    original_candidates = solver.get_candidate_points

    def capture_candidates() -> Any:
        candidates = original_candidates()
        captured_candidates.append(candidates.copy())
        return candidates

    solver.get_candidate_points = capture_candidates
    outputs = {
        key: _resolve(root, resolved[key])
        for key in ("initial_batch_out", "samples_out", "proposal_diagnostics_out", "proposal_traces_out")
    }
    manifest_path = _resolve(root, resolved["run_manifest_out"])
    _set_lambo_dependency_profile()
    manifest = build_run_manifest(
        resolved_config=resolved,
        command=[
            sys.executable,
            "-m",
            "genturbo.sequence_baseline_study.lambo2",
            "online",
            "--search-config",
            str(config_path),
        ],
        seed=int(resolved["seed"]),
        input_paths={
            "runtime_config": _resolve(root, resolved["config"]),
            "initial_batch": initial_path,
            "search_config": config_path,
            "lambo_requirements": _resolve(root, resolved["lambo_requirements"]),
            **initialization_inputs,
            **reward_inputs,
        },
        repo_root=root,
    )
    required_binding_names = {
        "lambo_requirements",
        *initialization_inputs,
    }
    missing_bindings = required_binding_names - manifest.input_hashes.keys()
    if missing_bindings:
        raise LaMBO2ProtocolError(
            "LaMBO-2 run manifest could not bind required inputs: "
            + ", ".join(sorted(missing_bindings))
        )
    bound_input_hashes = {
        name: manifest.input_hashes[name] for name in sorted(required_binding_names)
    }
    selected_target = int(resolved["max_oracle_evaluations"]) - initial_count
    batches, remainder = divmod(selected_target, int(resolved["al_oracle_batch_size"]))
    if remainder or batches != int(resolved["num_iterations"]):
        raise LaMBO2ProtocolError("LaMBO-2 selected budget is not an exact batch matrix")
    archive = {tuple(int(token) for token in row["seq"]) for row in initial_rows}
    with RunManifestRecorder(manifest, manifest_path, output_paths=outputs.values()) as recorder:
        write_jsonl(outputs["initial_batch_out"], iter(initial_rows))
        with (
            open_text(outputs["samples_out"], "wt") as samples_stream,
            open_text(outputs["proposal_diagnostics_out"], "wt") as diagnostics_stream,
            open_text(outputs["proposal_traces_out"], "wt") as traces_stream,
        ):
            for batch_id in range(batches):
                captured_candidates.clear()
                designs, scores = solver.step()
                expected_batch = int(resolved["al_oracle_batch_size"])
                if len(designs) != expected_batch or len(captured_candidates) != 1:
                    raise LaMBO2ProtocolError(
                        f"LaMBO-2 did not return exactly one {expected_batch}-design batch"
                    )
                sequences = _strings_to_ids(
                    designs,
                    alphabet=alphabet,
                    sequence_length=sequence_length,
                )
                numeric_scores = np.asarray(scores, dtype=float).reshape(-1).tolist()
                batch_rows: list[dict[str, Any]] = []
                for offset, (sequence, score) in enumerate(zip(sequences, numeric_scores, strict=True)):
                    sample = _sample_row(
                        sequence,
                        score,
                        batch_id=batch_id,
                        seq_idx=initial_count + batch_id * int(resolved["al_oracle_batch_size"]) + offset,
                        alphabet=alphabet,
                    )
                    ancestor, edits = _nearest_ancestor(
                        sequence,
                        captured_candidates[0],
                        alphabet=alphabet,
                        sequence_length=sequence_length,
                    )
                    key = tuple(sequence)
                    trace = _trace_row(sample, ancestor=ancestor, edits=edits, duplicate=key in archive)
                    archive.add(key)
                    batch_rows.append(sample)
                    _json_line(samples_stream, sample)
                    _json_line(traces_stream, trace)
                total = initial_count + (batch_id + 1) * int(resolved["al_oracle_batch_size"])
                _json_line(
                    diagnostics_stream,
                    _diagnostic_row(
                        batch_id,
                        batch_rows,
                        total=total,
                        protocol=lambo,
                        bound_input_hashes=bound_input_hashes,
                    ),
                )
        recorder.manifest.trace_records_written = selected_target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    pretrain = commands.add_parser("pretrain")
    pretrain.add_argument("--study-config", type=Path, required=True)
    online = commands.add_parser("online")
    online.add_argument("--search-config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "pretrain":
        print(json.dumps(pretrain_generator(args.study_config, repo_root=args.repo_root), indent=2, sort_keys=True))
    else:
        run_online(args.search_config, repo_root=args.repo_root)
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())


__all__ = [
    "AMINO_ACIDS",
    "LaMBO2ProtocolError",
    "pretrain_generator",
    "pretraining_plan",
    "run_online",
]
