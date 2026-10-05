"""Rollout and reward evaluation for interpolant proposals."""

from __future__ import annotations

import hashlib
import inspect
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch
import torch.nn.functional as F

from eval.seq_eval import logical_edit_diagnostics, score_batch
from search.interpolant.policies import CandidateRecord, Proposal, ProposalBatch
from search.interpolant.timing import TimingRecorder


@dataclass
class ProposalRetentionByTime:
    t_value: float
    count: int
    mean_retained_frac: float
    mean_retained_frac_corrupted: Optional[float]
    mean_retained_frac_uncorrupted: Optional[float]
    mean_retained_frac_logical: Optional[float]


@dataclass
class ProposalFilterByTime:
    t_value: float
    generated_proposals: int
    deduped_proposals: int
    duplicate_exact_proposals: int
    duplicate_generated_leaves: int
    cached_generated_proposals: int
    invalid_generated_proposals: int
    acquisition_batch_size: int


@dataclass
class ProposalChangeStats:
    generated_proposals: int
    deduped_proposals: int
    duplicate_exact_proposals: int
    duplicate_generated_leaves: int
    cached_generated_proposals: int
    invalid_generated_proposals: int
    acquisition_batch_size: int
    inferred_corrupt_masks: int
    model_changes_total: int
    model_changes_on_corrupted: int
    model_changes_on_uncorrupted: int
    proposal_logical_edits_total: int
    model_logical_edits_total: int
    dropped_proposal_logical_edits: int
    introduced_logical_edits: int
    changed_proposals_only_corrupted: int
    changed_proposals_with_uncorrupted: int
    mean_retained_frac: Optional[float]
    mean_retained_frac_corrupted: Optional[float]
    mean_retained_frac_uncorrupted: Optional[float]
    mean_retained_frac_logical: Optional[float]
    retention_by_time: list[ProposalRetentionByTime]
    filter_by_time: list[ProposalFilterByTime]


@dataclass
class ProposalTrace:
    base_index: int
    t_value: float
    expected_edits: float
    proposal_corrupted_tokens: int
    model_changes_total: int
    model_changes_on_corrupted: int
    model_changes_on_uncorrupted: int
    ancestor_to_child_edits: int
    ancestor_seq: torch.Tensor
    from_seq: torch.Tensor
    to_seq: torch.Tensor
    base_score: float
    score: float
    proposal_logical_edits: int = 0
    model_logical_edits: int = 0
    retained_proposal_logical_edits: int = 0
    dropped_proposal_logical_edits: int = 0
    introduced_logical_edits: int = 0
    retained_proposal_logical_frac: Optional[float] = None
    oracle_evaluated: bool = True
    smiles: Optional[str] = None
    surrogate_mean: Optional[float] = None
    surrogate_std: Optional[float] = None
    expected_improvement: Optional[float] = None
    acquisition_value: Optional[float] = None
    local_kernel_log_prob: Optional[float] = None
    local_kernel_log_prob_sequence: Optional[float] = None
    region_id: Optional[int] = None
    acquisition_candidate: bool = False
    acquisition_selected: bool = False
    duplicate_exact_proposal: bool = False
    duplicate_generated_leaf: bool = False
    cached_generated_proposal: bool = False
    invalid_generated_proposal: bool = False
    oracle_rejected: bool = False
    diagnostic_score: Optional[float] = None
    diagnostic_score_evaluated: bool = False
    full_batch_rank_by_score: Optional[int] = None
    diagnostic_injected_global_best: bool = False
    diagnostic_injected_source: Optional[str] = None
    diagnostic_injected_full_dataset_score: Optional[float] = None
    parent_seq_hash: Optional[str] = None
    parent_score: Optional[float] = None
    parent_archive_index: Optional[int] = None
    parent_archive_probability: Optional[float] = None
    parent_selection_log_prob: Optional[float] = None
    archive_size: Optional[int] = None
    archive_snapshot_id: Optional[str] = None
    hamming_conditional_log_prob: Optional[float] = None
    proposal_log_prob: Optional[float] = None
    proposal_probability_method: Optional[str] = None


@dataclass(frozen=True)
class _ProposalTraceMetrics:
    proposal_corrupted_tokens: int
    model_changes_total: int
    model_changes_on_corrupted: int
    model_changes_on_uncorrupted: int
    ancestor_to_child_edits: int
    proposal_logical_edits: int
    model_logical_edits: int
    retained_proposal_logical_edits: int
    dropped_proposal_logical_edits: int
    introduced_logical_edits: int
    retained_proposal_logical_frac: Optional[float]


def _trace_sequence_hash(sequence: torch.Tensor) -> str:
    sequence_cpu = sequence.detach().to("cpu", dtype=torch.long).contiguous().view(-1)
    return hashlib.sha1(sequence_cpu.numpy().tobytes()).hexdigest()


def _trace_tensors_cpu(tensors: list[Optional[torch.Tensor]]) -> list[Optional[torch.Tensor]]:
    """Copy trace-only tensors in batches, without a GPU sync for every row."""
    result: list[Optional[torch.Tensor]] = [None] * len(tensors)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, tensor in enumerate(tensors):
        if tensor is None:
            continue
        if tensor.device.type == "cpu":
            result[idx] = tensor.detach()
        else:
            groups[(tensor.device, tensor.dtype, tuple(tensor.shape))].append(idx)
    for indices in groups.values():
        batch = torch.stack([tensors[idx].detach() for idx in indices]).cpu()
        for row, idx in enumerate(indices):
            result[idx] = batch[row]
    return result


@dataclass
class DiagnosticInjectedCandidate:
    sequence: torch.Tensor
    sequence_string: str
    source: str
    full_dataset_score: Optional[float] = None


@dataclass
class ProposalEvaluation:
    candidate_leaves: list[torch.Tensor]
    best_candidate_state: torch.Tensor
    best_candidate_reward: float
    best_candidate_t: Optional[float]
    best_candidate_edits: Optional[float]
    tr_best_t: Optional[float]
    tr_best_edits: Optional[float]
    evaluated_proposals: list[Proposal]
    evaluated_rewards: torch.Tensor
    candidate_rewards: torch.Tensor
    oracle_evaluated_count: int
    oracle_budget_debit_count: int
    oracle_scored_count: int
    oracle_privileged_scored_count: int
    oracle_rejected_count: int
    active_learning_threshold_exhausted: bool
    active_learning_max_expected_improvement: Optional[float]
    active_learning_num_above_threshold: int
    proposal_collapse_detected: bool
    proposal_collapse_acquisition_fraction: Optional[float]
    proposal_collapse_min_acquisition_fraction: Optional[float]
    proposal_collapse_min_acquisition_count: Optional[int]
    proposal_collapse_target_count: Optional[int]
    full_batch_scored_count: int
    full_batch_top_score: Optional[float]
    full_batch_top_smiles: Optional[str]
    full_batch_top_t_value: Optional[float]
    full_batch_top_selected: Optional[bool]
    full_batch_best_selected_score: Optional[float]
    diagnostic_injected_global_best_present: bool
    diagnostic_injected_global_best_acquisition_candidate: bool
    diagnostic_injected_global_best_selected: bool
    diagnostic_injected_global_best_rank_by_acquisition: Optional[int]
    diagnostic_injected_global_best_surrogate_mean: Optional[float]
    diagnostic_injected_global_best_surrogate_std: Optional[float]
    diagnostic_injected_global_best_expected_improvement: Optional[float]
    diagnostic_injected_global_best_acquisition_value: Optional[float]
    diagnostic_injected_global_best_sequence: Optional[str]
    diagnostic_injected_global_best_full_dataset_score: Optional[float]
    change_stats: ProposalChangeStats
    proposal_traces_all: list[ProposalTrace]
    proposal_traces: list[ProposalTrace]
    timing_seconds: dict[str, float]
    timing_counts: dict[str, float]
    proposal_generation_diagnostics: dict[str, Any]
    candidate_records: list[CandidateRecord]


def _score_oracle_batch(
    batch: torch.Tensor,
    reward_fn: Any,
    *,
    context: str,
) -> torch.Tensor:
    """Call a stateful oracle exactly once and attach failure context."""

    expected = int(batch.shape[0]) if batch.ndim > 1 else 1
    try:
        scores = score_batch(batch, reward_fn)
    except Exception as exc:
        raise RuntimeError(f"oracle scoring failed during {context} for {expected} candidate(s)") from exc
    if scores.ndim == 0:
        scores = scores.view(1)
    if int(scores.numel()) != expected:
        raise RuntimeError(
            f"oracle scoring during {context} returned {int(scores.numel())} score(s) for {expected} candidate(s)"
        )
    return scores.view(-1)


def _observe_oracle_batch(
    selector: Any,
    batch: torch.Tensor,
    scores: torch.Tensor,
    *,
    context: str,
) -> None:
    try:
        selector.observe_oracle_batch(batch, scores)
    except Exception as exc:
        raise RuntimeError(
            f"oracle observation failed during {context} for {int(batch.shape[0])} candidate(s)"
        ) from exc


def _sequence_distance(
    adapter: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    force_hamming: bool = False,
) -> int:
    if force_hamming:
        return int((a.view(-1) != b.view(-1)).sum().item())
    model = getattr(getattr(adapter, "process", None), "model", None)
    custom_distance = getattr(model, "sequence_edit_distance", None)
    if callable(custom_distance):
        return int(custom_distance(a, b))
    return int((a.view(-1) != b.view(-1)).sum().item())


def _logical_edit_summary(
    adapter: Any,
    ancestor: torch.Tensor,
    proposal: torch.Tensor,
    leaf: torch.Tensor,
) -> dict[str, Optional[float]]:
    model = getattr(getattr(adapter, "process", None), "model", None)
    ignore_token_id = getattr(model, "pad_token_id", None)
    if ignore_token_id is not None:
        ignore_token_id = int(ignore_token_id)
    return logical_edit_diagnostics(
        ancestor,
        proposal,
        leaf,
        ignore_token_id=ignore_token_id,
    )


def _resolved_corrupt_mask(
    proposal: Proposal,
    base_state: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    corrupt_mask = proposal.corrupt_mask
    if corrupt_mask is None:
        return proposal.xt[0] != base_state[0], True
    corrupt_mask_1d = corrupt_mask.to(
        device=proposal.xt.device,
        dtype=torch.bool,
    ).view(-1)
    if corrupt_mask_1d.numel() != int(proposal.xt.shape[1]):
        return proposal.xt[0] != base_state[0], True
    return corrupt_mask_1d, False


def _proposal_trace_metrics(
    *,
    adapter: Any,
    base_state: torch.Tensor,
    proposal: Proposal,
    leaf: torch.Tensor,
    corrupt_mask_1d: torch.Tensor,
    model_changed_1d: torch.Tensor,
    force_hamming: bool,
    ancestor_metrics: Optional[_ProposalTraceMetrics] = None,
    distance_inputs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> _ProposalTraceMetrics:
    changed_total = int(model_changed_1d.sum().item())
    changed_on_corrupted = int((model_changed_1d & corrupt_mask_1d).sum().item())
    changed_on_uncorrupted = int((model_changed_1d & (~corrupt_mask_1d)).sum().item())

    if ancestor_metrics is None:
        ancestor_to_child_edits = _sequence_distance(
            adapter,
            leaf[0] if distance_inputs is None else distance_inputs[0],
            base_state[0] if distance_inputs is None else distance_inputs[1],
            force_hamming=force_hamming,
        )
        logical = _logical_edit_summary(
            adapter,
            base_state[0],
            proposal.xt[0],
            leaf[0],
        )
        proposal_logical_edits = int(logical["proposal_edits"] or 0.0)
        model_logical_edits = int(logical["model_edits"] or 0.0)
        retained_proposal_logical_edits = int(logical["retained_proposal_edits"] or 0.0)
        dropped_proposal_logical_edits = int(logical["dropped_proposal_edits"] or 0.0)
        introduced_logical_edits = int(logical["introduced_leaf_edits"] or 0.0)
        retained_proposal_logical_frac = logical["retained_proposal_frac"]
    else:
        ancestor_to_child_edits = ancestor_metrics.ancestor_to_child_edits
        proposal_logical_edits = ancestor_metrics.proposal_logical_edits
        model_logical_edits = ancestor_metrics.model_logical_edits
        retained_proposal_logical_edits = ancestor_metrics.retained_proposal_logical_edits
        dropped_proposal_logical_edits = ancestor_metrics.dropped_proposal_logical_edits
        introduced_logical_edits = ancestor_metrics.introduced_logical_edits
        retained_proposal_logical_frac = ancestor_metrics.retained_proposal_logical_frac

    return _ProposalTraceMetrics(
        proposal_corrupted_tokens=int(corrupt_mask_1d.sum().item()),
        model_changes_total=changed_total,
        model_changes_on_corrupted=changed_on_corrupted,
        model_changes_on_uncorrupted=changed_on_uncorrupted,
        ancestor_to_child_edits=ancestor_to_child_edits,
        proposal_logical_edits=proposal_logical_edits,
        model_logical_edits=model_logical_edits,
        retained_proposal_logical_edits=retained_proposal_logical_edits,
        dropped_proposal_logical_edits=dropped_proposal_logical_edits,
        introduced_logical_edits=introduced_logical_edits,
        retained_proposal_logical_frac=(
            float(retained_proposal_logical_frac) if retained_proposal_logical_frac is not None else None
        ),
    )


def _estimate_predictive_kernel_log_probs(
    proposal_batch: ProposalBatch,
    *,
    unique_props: list[Proposal],
    unique_leaf_rows: list[Optional[torch.Tensor]],
    adapter: Any,
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    """Estimate log K_t(y | x) for each unique proposal leaf.

    For each parent/time group, this computes the Monte Carlo estimate
    logmeanexp_m log p_theta(y | x_t^(m), t), using the initially mixed states
    as the MC samples and the denoised leaves as candidate endpoints.
    """

    n_unique = len(unique_props)
    logk_per_token: list[Optional[float]] = [None for _ in range(n_unique)]
    logk_sequence: list[Optional[float]] = [None for _ in range(n_unique)]
    if n_unique == 0:
        return logk_per_token, logk_sequence

    process = getattr(adapter, "process", None)
    model = getattr(process, "model", None)
    if not callable(model):
        return logk_per_token, logk_sequence
    cfg = getattr(process, "cfg", None)
    vocab_size = int(getattr(cfg, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        return logk_per_token, logk_sequence
    device = getattr(adapter, "device", unique_props[0].xt.device)
    temperature = float(getattr(adapter, "temperature", 1.0))
    pad_token_id = getattr(model, "pad_token_id", None)
    if pad_token_id is not None:
        pad_token_id = int(pad_token_id)

    start_groups: dict[tuple[int, float], list[Proposal]] = defaultdict(list)
    for proposal in proposal_batch.proposals:
        group_key = (int(proposal.base_index), round(float(proposal.t_value), 8))
        start_groups[group_key].append(proposal)

    candidate_groups: dict[tuple[int, float], list[int]] = defaultdict(list)
    for unique_idx, proposal in enumerate(unique_props):
        if unique_leaf_rows[unique_idx] is None:
            continue
        group_key = (int(proposal.base_index), round(float(proposal.t_value), 8))
        candidate_groups[group_key].append(unique_idx)

    model_was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            for group_key, candidate_indices in candidate_groups.items():
                starts = start_groups.get(group_key)
                if not starts:
                    continue
                xt_batch = torch.cat([proposal.xt for proposal in starts], dim=0).to(device=device)
                if xt_batch.ndim != 2:
                    continue
                t_value = float(starts[0].t_value)
                t_batch = torch.full((int(xt_batch.shape[0]),), t_value, device=device)
                logits = model(xt_batch, timesteps=t_batch)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                    continue
                if int(logits.shape[0]) != int(xt_batch.shape[0]) or int(logits.shape[-1]) < vocab_size:
                    continue
                log_probs = F.log_softmax(logits[..., :vocab_size] / max(temperature, 1e-6), dim=-1)

                leaves = torch.cat(
                    [
                        unique_leaf_rows[unique_idx]
                        for unique_idx in candidate_indices
                        if unique_leaf_rows[unique_idx] is not None
                    ],
                    dim=0,
                ).to(device=device, dtype=torch.long)
                if leaves.ndim != 2 or int(leaves.shape[1]) != int(log_probs.shape[1]):
                    continue
                valid_token_mask = (leaves >= 0) & (leaves < vocab_size)
                if pad_token_id is not None:
                    valid_token_mask = valid_token_mask & (leaves != pad_token_id)
                safe_leaves = leaves.clamp(min=0, max=max(0, vocab_size - 1))
                expanded_targets = safe_leaves.unsqueeze(0).expand(int(log_probs.shape[0]), -1, -1)
                token_log_probs = torch.gather(
                    log_probs.unsqueeze(1).expand(-1, int(leaves.shape[0]), -1, -1),
                    dim=-1,
                    index=expanded_targets.unsqueeze(-1),
                ).squeeze(-1)
                token_log_probs = token_log_probs.masked_fill(~valid_token_mask.unsqueeze(0), 0.0)
                lengths = valid_token_mask.sum(dim=-1).clamp_min(1).to(dtype=token_log_probs.dtype)
                sequence_log_probs = token_log_probs.sum(dim=-1)
                group_logk_sequence = torch.logsumexp(sequence_log_probs, dim=0) - math.log(float(log_probs.shape[0]))
                group_logk_per_token = group_logk_sequence / lengths
                for local_idx, unique_idx in enumerate(candidate_indices):
                    logk_sequence[unique_idx] = float(group_logk_sequence[local_idx].detach().cpu().item())
                    logk_per_token[unique_idx] = float(group_logk_per_token[local_idx].detach().cpu().item())
    finally:
        if model_was_training:
            model.train()

    return logk_per_token, logk_sequence


def evaluate_proposals(
    proposal_batch: ProposalBatch,
    *,
    adapter: Any,
    reward_fn: Any,
    track_trust_region: bool,
    rollout_use_corrupt_mask: bool = False,
    rollout_denoise_steps: Optional[int] = None,
    legacy_hamming_tr_error: bool = False,
    active_learning_selector: Any = None,
    active_learning_budget_override: Optional[int] = None,
    score_full_proposal_batch: bool = False,
    diagnostic_injected_candidate: Optional[DiagnosticInjectedCandidate] = None,
    diagnostic_injection_affects_collapse: bool = False,
    proposal_collapse_min_acq_fraction: Optional[float] = None,
    proposal_collapse_min_acq_count: Optional[int] = None,
    proposal_collapse_target_count: Optional[int] = None,
    proposal_collapse_block_selection: bool = True,
    profile_timings: bool = False,
    profile_sync_cuda: bool = False,
) -> ProposalEvaluation:
    base_samples = proposal_batch.base_samples
    base_rewards = proposal_batch.base_rewards
    timings = TimingRecorder(
        enabled=bool(profile_timings),
        sync_cuda=bool(profile_sync_cuda),
        device=getattr(adapter, "device", base_samples.device),
    )
    total_start = timings.start()
    generated_proposal_count = int(len(proposal_batch.proposals))
    oracle_scored_count = 0
    oracle_rejected_count = 0
    selector_acquisition = str(getattr(active_learning_selector, "_acquisition", "") or "")
    oracle_selection_mode = selector_acquisition == "oracle_best"
    selector_config = getattr(active_learning_selector, "config", None)
    ehrlich_black_box_feasibility = bool(getattr(selector_config, "ehrlich_black_box_feasibility", False))
    oracle_best_budget_mode = (
        str(getattr(selector_config, "oracle_best_budget_mode", "physical") or "physical").strip().lower()
    )
    if oracle_best_budget_mode not in {"physical", "selected_archive"}:
        raise ValueError("active_learning oracle_best_budget_mode must be one of: physical, selected_archive")
    if oracle_best_budget_mode == "selected_archive" and not oracle_selection_mode:
        raise ValueError("active_learning oracle_best_budget_mode='selected_archive' requires oracle_best acquisition")
    timings.add_count("generated_proposals", generated_proposal_count)

    best_candidate_state = base_samples[:1]
    if base_rewards.ndim > 0:
        best_candidate_reward = float(base_rewards[0].item())
    else:
        best_candidate_reward = float(base_rewards.item())
    best_candidate_t: Optional[float] = None
    best_candidate_edits: Optional[float] = None
    tr_best_t: Optional[float] = None
    tr_best_edits: Optional[float] = None
    tr_best_reward = float("-inf")
    base_index_to_row = {int(base_idx): row for row, base_idx in enumerate(proposal_batch.base_indices)}

    inferred_corrupt_masks = 0
    model_changes_total = 0
    model_changes_on_corrupted = 0
    model_changes_on_uncorrupted = 0
    changed_proposals_only_corrupted = 0
    changed_proposals_with_uncorrupted = 0
    retained_sum = 0.0
    retained_count = 0
    retained_sum_corrupted = 0.0
    retained_count_corrupted = 0
    retained_sum_uncorrupted = 0.0
    retained_count_uncorrupted = 0
    logical_retained_sum = 0.0
    logical_retained_count = 0
    proposal_logical_edits_total = 0
    model_logical_edits_total = 0
    dropped_proposal_logical_edits = 0
    introduced_logical_edits = 0
    retention_by_time_acc: dict[float, dict[str, float]] = {}
    # Traces for deduplicated proposals only (used for internal diagnostics/updates).
    proposal_traces: list[ProposalTrace] = []
    # Traces aligned to the original proposal list (includes duplicates).
    proposal_traces_all: list[ProposalTrace] = []

    needs_rollout = any(not proposal.direct_leaf for proposal in proposal_batch.proposals)
    supports_update_mask = False
    supports_denoise_steps = False
    supports_t_is_time = False
    if needs_rollout:
        with timings.measure("signature_check_seconds"):
            try:
                rollout_sig = inspect.signature(adapter.rollout_from_batch)
                supports_update_mask = "update_mask" in rollout_sig.parameters
                supports_denoise_steps = "denoise_steps" in rollout_sig.parameters
                supports_t_is_time = "t_is_time" in rollout_sig.parameters
            except (TypeError, ValueError):
                pass
    if needs_rollout and rollout_use_corrupt_mask and not supports_update_mask:
        raise RuntimeError("masked rollout requested, but adapter.rollout_from_batch does not expose update_mask")
    enforce_masked_rollout = bool(needs_rollout and rollout_use_corrupt_mask)
    if needs_rollout and rollout_denoise_steps is not None and not supports_denoise_steps:
        raise RuntimeError("adapter.rollout_from_batch does not support denoise_steps")
    if enforce_masked_rollout:
        for proposal_idx, proposal in enumerate(proposal_batch.proposals):
            if proposal.direct_leaf:
                continue
            seq_len = int(proposal.xt.shape[1])
            mask = proposal.corrupt_mask
            if mask is None:
                raise ValueError(f"masked rollout proposal {proposal_idx} is missing corrupt_mask")
            if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
                raise TypeError(f"masked rollout proposal {proposal_idx} corrupt_mask must be a boolean tensor")
            if tuple(mask.shape) != (seq_len,):
                raise ValueError(
                    f"masked rollout proposal {proposal_idx} corrupt_mask has shape "
                    f"{tuple(mask.shape)}; expected ({seq_len},)"
                )

    # Deduplicate identical proposals before rollout.
    with timings.measure("dedupe_exact_proposals_seconds"):
        unique_props: list[Proposal] = []
        key_to_unique_idx: dict[tuple[bool, float, bytes, Optional[bytes]], int] = {}
        original_to_unique: list[int] = []
        original_is_duplicate_exact: list[bool] = []
        for proposal in proposal_batch.proposals:
            xt_cpu = proposal.xt.detach().to("cpu").contiguous()
            mask_key: Optional[bytes] = None
            if enforce_masked_rollout and not proposal.direct_leaf:
                assert proposal.corrupt_mask is not None
                mask_cpu = proposal.corrupt_mask.detach().to("cpu").contiguous()
                mask_key = mask_cpu.numpy().tobytes()
            t_key = round(float(proposal.t_value), 8)
            key = (bool(proposal.direct_leaf), t_key, xt_cpu.numpy().tobytes(), mask_key)
            if key in key_to_unique_idx:
                original_to_unique.append(key_to_unique_idx[key])
                original_is_duplicate_exact.append(True)
                continue
            key_to_unique_idx[key] = len(unique_props)
            original_to_unique.append(len(unique_props))
            original_is_duplicate_exact.append(False)
            unique_props.append(proposal)
    duplicate_exact_proposal_count = max(0, generated_proposal_count - len(unique_props))
    candidate_records = [CandidateRecord(proposal=proposal) for proposal in unique_props]
    timings.add_count("deduped_proposals", len(unique_props))
    timings.add_count("duplicate_exact_proposals", duplicate_exact_proposal_count)

    filter_by_time_acc: dict[float, dict[str, float]] = {}

    def _filter_time_key(t_value: float) -> float:
        return round(float(t_value), 6)

    def _filter_acc_for_t(t_value: float) -> dict[str, float]:
        return filter_by_time_acc.setdefault(
            _filter_time_key(t_value),
            {
                "t_sum": 0.0,
                "t_count": 0.0,
                "generated": 0.0,
                "deduped": 0.0,
                "duplicate_exact": 0.0,
                "duplicate_leaf": 0.0,
                "cached": 0.0,
                "invalid": 0.0,
                "acquisition": 0.0,
            },
        )

    for original_idx, proposal in enumerate(proposal_batch.proposals):
        acc = _filter_acc_for_t(float(proposal.t_value))
        acc["generated"] += 1.0
        acc["t_sum"] += float(proposal.t_value)
        acc["t_count"] += 1.0
        if original_idx < len(original_is_duplicate_exact) and original_is_duplicate_exact[original_idx]:
            acc["duplicate_exact"] += 1.0
    for proposal in unique_props:
        acc = _filter_acc_for_t(float(proposal.t_value))
        acc["deduped"] += 1.0

    candidate_leaves: list[torch.Tensor] = []
    unique_leaf_rows: list[Optional[torch.Tensor]] = []
    unique_rewards = torch.empty((0,), dtype=torch.float, device=base_samples.device)
    active_learning_threshold_exhausted = False
    active_learning_max_expected_improvement: Optional[float] = None
    active_learning_num_above_threshold = 0
    proposal_collapse_detected = False
    proposal_collapse_acquisition_fraction: Optional[float] = None
    proposal_collapse_min_acquisition_fraction: Optional[float] = None
    proposal_collapse_min_acquisition_count: Optional[int] = None
    proposal_collapse_effective_target_count: Optional[int] = (
        max(1, int(proposal_collapse_target_count)) if proposal_collapse_target_count is not None else None
    )
    diagnostic_injected_unique_idx: Optional[int] = None
    diagnostic_injected_global_best_present = False
    diagnostic_injected_global_best_acquisition_candidate = False
    diagnostic_injected_global_best_selected = False
    diagnostic_injected_global_best_rank_by_acquisition: Optional[int] = None
    diagnostic_injected_global_best_surrogate_mean: Optional[float] = None
    diagnostic_injected_global_best_surrogate_std: Optional[float] = None
    diagnostic_injected_global_best_expected_improvement: Optional[float] = None
    diagnostic_injected_global_best_acquisition_value: Optional[float] = None
    if unique_props:
        grouped: dict[float, list[int]] = defaultdict(list)
        for idx, proposal in enumerate(unique_props):
            if not proposal.direct_leaf:
                grouped[round(float(proposal.t_value), 8)].append(idx)

        unique_leaf_rows = [None for _ in unique_props]
        for idx, proposal in enumerate(unique_props):
            if proposal.direct_leaf:
                leaf = proposal.xt.clone()
                unique_leaf_rows[idx] = leaf
                candidate_records[idx].leaf = leaf

        # Roll out all proposals sharing the same t value in a single batched call.
        for t_key in sorted(grouped.keys()):
            idxs = grouped[t_key]
            t_value = float(unique_props[idxs[0]].t_value)
            rollout_t: float = t_value
            rollout_time_kw: dict[str, bool] = {}
            if supports_t_is_time:
                rollout_time_kw["t_is_time"] = True
            else:
                total_steps = float(getattr(adapter, "T", 1))
                rollout_t = t_value * total_steps
            xt_batch = torch.cat([unique_props[i].xt for i in idxs], dim=0)
            if enforce_masked_rollout:
                mask_rows: list[torch.Tensor] = []
                for unique_idx in idxs:
                    mask = unique_props[unique_idx].corrupt_mask
                    assert mask is not None
                    mask_rows.append(mask.to(device=xt_batch.device))
                update_mask_batch = torch.stack(mask_rows, dim=0)
                start = timings.start()
                leaves_batch = adapter.rollout_from_batch(
                    xt_batch,
                    rollout_t,
                    update_mask=update_mask_batch,
                    denoise_steps=rollout_denoise_steps,
                    **rollout_time_kw,
                )[-1]
                timings.stop("rollout_seconds", start)
            else:
                start = timings.start()
                leaves_batch = adapter.rollout_from_batch(
                    xt_batch,
                    rollout_t,
                    denoise_steps=rollout_denoise_steps,
                    **rollout_time_kw,
                )[-1]
                timings.stop("rollout_seconds", start)
            timings.add_count("rollout_batches", 1)
            timings.add_count("rollout_unique_proposals", len(idxs))

            for row_idx, unique_idx in enumerate(idxs):
                leaf = leaves_batch[row_idx : row_idx + 1]
                unique_leaf_rows[unique_idx] = leaf
                candidate_records[unique_idx].leaf = leaf

        unique_rewards = torch.full((len(unique_props),), float("nan"), dtype=torch.float, device=base_samples.device)
        unique_oracle_evaluated = [False for _ in unique_props]
        unique_selected = [False for _ in unique_props]
        unique_smiles: list[Optional[str]] = [None for _ in unique_props]
        unique_surrogate_mean: list[Optional[float]] = [None for _ in unique_props]
        unique_surrogate_std: list[Optional[float]] = [None for _ in unique_props]
        unique_expected_improvement: list[Optional[float]] = [None for _ in unique_props]
        unique_acquisition_value: list[Optional[float]] = [None for _ in unique_props]
        unique_local_kernel_log_prob: list[Optional[float]] = [None for _ in unique_props]
        unique_local_kernel_log_prob_sequence: list[Optional[float]] = [None for _ in unique_props]
        unique_diagnostic_score: list[Optional[float]] = [None for _ in unique_props]
        unique_diagnostic_score_evaluated = [False for _ in unique_props]
        unique_full_batch_rank_by_score: list[Optional[int]] = [None for _ in unique_props]
        unique_is_injected_global_best = [False for _ in unique_props]
        unique_oracle_rejected = [False for _ in unique_props]
        available_unique_indices = [idx for idx, leaf in enumerate(unique_leaf_rows) if leaf is not None]
        unique_leaf_keys: list[Optional[bytes]] = [None for _ in unique_props]
        for idx in available_unique_indices:
            leaf = unique_leaf_rows[idx]
            if leaf is None:
                continue
            unique_leaf_keys[idx] = leaf.detach().to("cpu").contiguous().view(-1).numpy().tobytes()
        unique_is_cached = [False for _ in unique_props]
        if active_learning_selector is not None and available_unique_indices:
            all_leaf_batch = torch.cat([unique_leaf_rows[idx] for idx in available_unique_indices], dim=0)
            start = timings.start()
            try:
                cached_scores_all = list(active_learning_selector.cached_scores(all_leaf_batch))
            except Exception as exc:
                raise RuntimeError(
                    f"active-learning score-cache lookup failed for {len(available_unique_indices)} candidate(s)"
                ) from exc
            timings.stop("cache_lookup_seconds", start)
            if len(cached_scores_all) != len(available_unique_indices):
                raise RuntimeError(
                    "active-learning score-cache returned "
                    f"{len(cached_scores_all)} rows for {len(available_unique_indices)} candidate(s)"
                )
            for row_idx, unique_idx in enumerate(available_unique_indices):
                cached_score = cached_scores_all[row_idx]
                unique_is_cached[unique_idx] = cached_score is not None
                candidate_records[unique_idx].cached = cached_score is not None

        decoder_context = "reward"
        try:
            decode_batch = getattr(reward_fn, "decode_batch", None)
        except Exception as exc:
            raise RuntimeError("reward decoder capability lookup failed") from exc
        if not callable(decode_batch) and active_learning_selector is not None:
            decoder_context = "active-learning selector candidate"
            try:
                decode_batch = getattr(active_learning_selector, "decode_candidate_batch", None)
            except Exception as exc:
                raise RuntimeError("active-learning selector candidate decoder capability lookup failed") from exc
        if callable(decode_batch) and available_unique_indices:
            start = timings.start()
            try:
                all_leaf_batch = torch.cat([unique_leaf_rows[idx] for idx in available_unique_indices], dim=0)
                decoded_smiles = list(decode_batch(all_leaf_batch))
            except Exception as exc:
                raise RuntimeError(
                    f"{decoder_context} decoder failed for {len(available_unique_indices)} candidate(s)"
                ) from exc
            timings.stop("decode_batch_seconds", start)
            if len(decoded_smiles) != len(available_unique_indices):
                raise RuntimeError(
                    f"{decoder_context} decoder returned {len(decoded_smiles)} rows for "
                    f"{len(available_unique_indices)} candidate(s)"
                )
            for row_idx, unique_idx in enumerate(available_unique_indices):
                unique_smiles[unique_idx] = decoded_smiles[row_idx]
                candidate_records[unique_idx].decoded = decoded_smiles[row_idx]

        score_full_pool_for_ranking = bool(score_full_proposal_batch) or bool(oracle_selection_mode)
        if selector_acquisition in {"weighted_expected_improvement", "weighted_batched_expected_improvement"}:
            with timings.measure("predictive_kernel_seconds"):
                (
                    unique_local_kernel_log_prob,
                    unique_local_kernel_log_prob_sequence,
                ) = _estimate_predictive_kernel_log_probs(
                    proposal_batch,
                    unique_props=unique_props,
                    unique_leaf_rows=unique_leaf_rows,
                    adapter=adapter,
                )

        with timings.measure("dedupe_generated_leaves_seconds"):
            duplicate_generated_leaf_count = 0
            deduped_leaf_unique_indices: list[int] = []
            original_is_duplicate_generated_leaf = [False for _ in proposal_batch.proposals]
            unique_is_duplicate_generated_leaf = [False for _ in unique_props]
            seen_leaf_keys: set[bytes] = set()
            for original_idx, unique_idx in enumerate(original_to_unique):
                leaf_key = unique_leaf_keys[unique_idx] if 0 <= unique_idx < len(unique_leaf_keys) else None
                if leaf_key is not None:
                    if leaf_key in seen_leaf_keys:
                        duplicate_generated_leaf_count += 1
                        if 0 <= original_idx < len(original_is_duplicate_generated_leaf):
                            original_is_duplicate_generated_leaf[original_idx] = True
                        if 0 <= unique_idx < len(unique_is_duplicate_generated_leaf):
                            unique_is_duplicate_generated_leaf[unique_idx] = True
                            candidate_records[unique_idx].duplicate_leaf = True
                        if 0 <= unique_idx < len(unique_props):
                            _filter_acc_for_t(float(unique_props[unique_idx].t_value))["duplicate_leaf"] += 1.0
                        continue
                    else:
                        seen_leaf_keys.add(leaf_key)
                        deduped_leaf_unique_indices.append(unique_idx)

        # Staged counts make the acquisition filter accounting non-overlapping:
        # total - duplicate leaves - invalid leaves - cached leaves == acquisition_batch_size.
        cached_generated_proposal_count = 0
        invalid_generated_proposal_count = 0
        acquisition_unique_indices: list[int] = []
        unique_is_invalid = [False for _ in unique_props]
        unique_is_acquisition_candidate = [False for _ in unique_props]
        for unique_idx in deduped_leaf_unique_indices:
            leaf_key = unique_leaf_keys[unique_idx] if 0 <= unique_idx < len(unique_leaf_keys) else None
            is_invalid = (
                callable(decode_batch)
                and 0 <= unique_idx < len(unique_smiles)
                and leaf_key is not None
                and unique_smiles[unique_idx] is None
            )
            if is_invalid:
                invalid_generated_proposal_count += 1
                unique_is_invalid[unique_idx] = True
                candidate_records[unique_idx].invalid = True
                _filter_acc_for_t(float(unique_props[unique_idx].t_value))["invalid"] += 1.0
                continue
            if 0 <= unique_idx < len(unique_is_cached) and unique_is_cached[unique_idx]:
                cached_generated_proposal_count += 1
                _filter_acc_for_t(float(unique_props[unique_idx].t_value))["cached"] += 1.0
                continue
            acquisition_unique_indices.append(unique_idx)
            unique_is_acquisition_candidate[unique_idx] = True
            candidate_records[unique_idx].acquisition_candidate = True
            _filter_acc_for_t(float(unique_props[unique_idx].t_value))["acquisition"] += 1.0
        acquisition_batch_size = len(acquisition_unique_indices)
        if active_learning_selector is not None:
            effective_min_acq_count: Optional[int] = None
            if proposal_collapse_min_acq_count is not None:
                effective_min_acq_count = max(0, int(proposal_collapse_min_acq_count))
            if proposal_collapse_min_acq_fraction is not None:
                proposal_collapse_min_acquisition_fraction = float(proposal_collapse_min_acq_fraction)
                collapse_denominator = int(
                    proposal_collapse_effective_target_count
                    if proposal_collapse_effective_target_count is not None
                    else max(1, generated_proposal_count)
                )
                fraction_min_count = int(
                    math.ceil(proposal_collapse_min_acquisition_fraction * float(collapse_denominator))
                )
                effective_min_acq_count = max(int(effective_min_acq_count or 0), fraction_min_count)
            if effective_min_acq_count is not None:
                proposal_collapse_min_acquisition_count = int(effective_min_acq_count)
                collapse_denominator = int(
                    proposal_collapse_effective_target_count
                    if proposal_collapse_effective_target_count is not None
                    else max(1, generated_proposal_count)
                )
                proposal_collapse_acquisition_fraction = float(acquisition_batch_size) / float(collapse_denominator)
                proposal_collapse_detected = int(acquisition_batch_size) < int(effective_min_acq_count)

        if diagnostic_injected_candidate is not None and active_learning_selector is not None:
            injected_leaf = diagnostic_injected_candidate.sequence.to(
                device=base_samples.device,
                dtype=torch.long,
            ).view(1, -1)
            if int(injected_leaf.shape[1]) != int(base_samples.shape[1]):
                raise ValueError(
                    "diagnostic_injected_candidate sequence length "
                    f"{int(injected_leaf.shape[1])} does not match base sample length "
                    f"{int(base_samples.shape[1])}"
                )
            injected_key = injected_leaf.detach().to("cpu").contiguous().view(-1).numpy().tobytes()
            existing_idx: Optional[int] = None
            for unique_idx, leaf_key in enumerate(unique_leaf_keys):
                if leaf_key == injected_key:
                    existing_idx = int(unique_idx)
                    break

            if existing_idx is None:
                if base_rewards.ndim > 0 and int(base_rewards.numel()) > 0:
                    base_row = int(torch.argmax(base_rewards.detach()).item())
                else:
                    base_row = 0
                base_row = max(0, min(base_row, int(base_samples.shape[0]) - 1))
                base_index = int(proposal_batch.base_indices[base_row])
                base_state = base_samples[base_row : base_row + 1].to(device=base_samples.device)
                corrupt_mask = (injected_leaf[0] != base_state[0]).detach().to(dtype=torch.bool)
                injected_proposal = Proposal(
                    base_index=base_index,
                    t_value=-1.0,
                    edits_value=0.0,
                    xt=injected_leaf.clone(),
                    corrupt_mask=corrupt_mask,
                )
                existing_idx = len(unique_props)
                unique_props.append(injected_proposal)
                candidate_records.append(
                    CandidateRecord(
                        proposal=injected_proposal,
                        leaf=injected_leaf,
                        decoded=diagnostic_injected_candidate.sequence_string,
                    )
                )
                unique_leaf_rows.append(injected_leaf)
                unique_rewards = torch.cat(
                    [
                        unique_rewards,
                        torch.full((1,), float("nan"), dtype=torch.float, device=base_samples.device),
                    ],
                    dim=0,
                )
                unique_oracle_evaluated.append(False)
                unique_selected.append(False)
                unique_smiles.append(diagnostic_injected_candidate.sequence_string)
                unique_surrogate_mean.append(None)
                unique_surrogate_std.append(None)
                unique_expected_improvement.append(None)
                unique_acquisition_value.append(None)
                unique_local_kernel_log_prob.append(None)
                unique_local_kernel_log_prob_sequence.append(None)
                unique_diagnostic_score.append(None)
                unique_diagnostic_score_evaluated.append(False)
                unique_full_batch_rank_by_score.append(None)
                unique_is_injected_global_best.append(True)
                unique_oracle_rejected.append(False)
                available_unique_indices.append(existing_idx)
                unique_leaf_keys.append(injected_key)
                unique_is_cached.append(False)
                unique_is_invalid.append(False)
                unique_is_acquisition_candidate.append(False)
                unique_is_duplicate_generated_leaf.append(False)
            else:
                unique_is_injected_global_best[existing_idx] = True
                unique_smiles[existing_idx] = (
                    unique_smiles[existing_idx] or diagnostic_injected_candidate.sequence_string
                )
                candidate_records[existing_idx].decoded = unique_smiles[existing_idx]

            diagnostic_injected_unique_idx = int(existing_idx)
            diagnostic_injected_global_best_present = True
            if diagnostic_injected_unique_idx not in available_unique_indices:
                available_unique_indices.append(diagnostic_injected_unique_idx)

            try:
                cached_values = list(active_learning_selector.cached_scores(injected_leaf))
            except Exception as exc:
                raise RuntimeError(
                    "active-learning score-cache lookup failed for diagnostic injected candidate"
                ) from exc
            if len(cached_values) != 1:
                raise RuntimeError(
                    f"active-learning score-cache returned {len(cached_values)} rows for one diagnostic candidate"
                )
            cached_score = cached_values[0]
            if cached_score is not None:
                unique_is_cached[diagnostic_injected_unique_idx] = True
                candidate_records[diagnostic_injected_unique_idx].cached = True

            injects_into_acquisition = not bool(unique_is_cached[diagnostic_injected_unique_idx])
            injects_into_acquisition = injects_into_acquisition and not bool(
                unique_is_invalid[diagnostic_injected_unique_idx]
            )
            if injects_into_acquisition:
                if diagnostic_injected_unique_idx not in acquisition_unique_indices:
                    acquisition_unique_indices.append(diagnostic_injected_unique_idx)
                unique_is_acquisition_candidate[diagnostic_injected_unique_idx] = True
                candidate_records[diagnostic_injected_unique_idx].acquisition_candidate = True
                diagnostic_injected_global_best_acquisition_candidate = True
                if bool(diagnostic_injection_affects_collapse):
                    acquisition_batch_size += 1
                    if proposal_collapse_min_acquisition_count is not None:
                        proposal_collapse_acquisition_fraction = float(acquisition_batch_size) / float(
                            proposal_collapse_effective_target_count
                            if proposal_collapse_effective_target_count is not None
                            else max(1, generated_proposal_count)
                        )
                        proposal_collapse_detected = int(acquisition_batch_size) < int(
                            proposal_collapse_min_acquisition_count
                        )

        selected_unique_indices: list[int] = []
        if available_unique_indices:
            if active_learning_selector is None:
                selected_unique_indices = list(available_unique_indices)
                if active_learning_budget_override is not None:
                    oracle_budget = max(
                        0,
                        min(
                            int(active_learning_budget_override),
                            len(selected_unique_indices),
                        ),
                    )
                    selected_unique_indices = selected_unique_indices[:oracle_budget]
            else:
                available_unique_indices = list(acquisition_unique_indices)
                block_collapsed_selection = bool(proposal_collapse_detected) and bool(proposal_collapse_block_selection)
                if block_collapsed_selection:
                    pass
                elif not available_unique_indices:
                    active_learning_threshold_exhausted = True
                if active_learning_budget_override is None:
                    select_budget = int(active_learning_selector.config.oracle_batch_size)
                else:
                    # The engine passes the run's *remaining* oracle budget.
                    # It is a hard cap, not permission to enlarge the
                    # selector's configured per-iteration batch.
                    select_budget = min(
                        int(active_learning_selector.config.oracle_batch_size),
                        int(active_learning_budget_override),
                    )
                oracle_budget = max(0, min(select_budget, len(available_unique_indices)))
                if block_collapsed_selection or not available_unique_indices:
                    pass
                elif oracle_selection_mode:
                    if oracle_budget > 0:
                        oracle_pool_indices = list(available_unique_indices)
                        if active_learning_budget_override is not None and oracle_best_budget_mode == "physical":
                            oracle_pool_indices = oracle_pool_indices[: max(0, int(active_learning_budget_override))]
                        if not oracle_pool_indices:
                            active_learning_threshold_exhausted = True
                            oracle_pool_indices = []
                        available_leaf_batch = torch.cat(
                            [unique_leaf_rows[idx] for idx in oracle_pool_indices],
                            dim=0,
                        )
                        start = timings.start()
                        oracle_rewards_all = _score_oracle_batch(
                            available_leaf_batch,
                            reward_fn,
                            context="oracle-best pool ranking",
                        )
                        timings.stop("oracle_upper_bound_score_seconds", start)
                        oracle_scored_count += int(available_leaf_batch.shape[0])
                        if oracle_rewards_all.ndim == 0:
                            oracle_rewards_all = oracle_rewards_all.view(1)
                        finite_items: list[tuple[float, int, int]] = []
                        for local_idx, unique_idx in enumerate(oracle_pool_indices):
                            if local_idx >= int(oracle_rewards_all.numel()):
                                continue
                            score_value = float(oracle_rewards_all[local_idx].item())
                            unique_diagnostic_score[unique_idx] = score_value
                            unique_diagnostic_score_evaluated[unique_idx] = True
                            unique_acquisition_value[unique_idx] = score_value
                            if math.isfinite(score_value):
                                finite_items.append((score_value, int(local_idx), int(unique_idx)))
                        finite_items.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                        selected_items = finite_items[:oracle_budget]
                        active_learning_num_above_threshold = len(finite_items)
                        timings.add_count("oracle_upper_bound_pool_scored_count", len(finite_items))
                        if finite_items:
                            active_learning_max_expected_improvement = float(finite_items[0][0])
                        for _, _, unique_idx in selected_items:
                            unique_selected[unique_idx] = True
                            selected_unique_indices.append(unique_idx)

                        if selected_items:
                            selected_leaf_batch = torch.cat(
                                [unique_leaf_rows[unique_idx] for _, _, unique_idx in selected_items],
                                dim=0,
                            )
                            selected_rewards = torch.as_tensor(
                                [score_value for score_value, _, _ in selected_items],
                                dtype=torch.float,
                                device=base_samples.device,
                            )
                            start = timings.start()
                            _observe_oracle_batch(
                                active_learning_selector,
                                selected_leaf_batch,
                                selected_rewards,
                                context="oracle-best acquisition",
                            )
                            timings.stop("acquisition_observe_seconds", start)
                            for row_idx, (_, _, unique_idx) in enumerate(selected_items):
                                unique_rewards[unique_idx] = selected_rewards[row_idx]
                                unique_oracle_evaluated[unique_idx] = True
                                unique_diagnostic_score[unique_idx] = float(selected_rewards[row_idx].item())
                                unique_diagnostic_score_evaluated[unique_idx] = True
                elif (
                    bool(getattr(active_learning_selector.config, "sequential_selection", False)) and oracle_budget > 0
                ):
                    remaining_unique_indices = list(available_unique_indices)
                    while remaining_unique_indices and len(selected_unique_indices) < oracle_budget:
                        remaining_leaf_batch = torch.cat(
                            [unique_leaf_rows[idx] for idx in remaining_unique_indices],
                            dim=0,
                        )
                        start = timings.start()
                        selection = active_learning_selector.select(
                            remaining_leaf_batch,
                            budget=min(1, oracle_budget - len(selected_unique_indices)),
                            local_kernel_log_probs=[
                                unique_local_kernel_log_prob[idx] for idx in remaining_unique_indices
                            ],
                        )
                        timings.stop("acquisition_select_seconds", start)
                        if selection.max_expected_improvement is not None:
                            if active_learning_max_expected_improvement is None or float(
                                selection.max_expected_improvement
                            ) > float(active_learning_max_expected_improvement):
                                active_learning_max_expected_improvement = float(selection.max_expected_improvement)
                        active_learning_num_above_threshold = max(
                            int(active_learning_num_above_threshold),
                            int(selection.num_candidates_above_threshold),
                        )
                        for local_idx, unique_idx in enumerate(remaining_unique_indices):
                            unique_smiles[unique_idx] = selection.smiles[local_idx]
                            unique_surrogate_mean[unique_idx] = selection.surrogate_mean[local_idx]
                            unique_surrogate_std[unique_idx] = selection.surrogate_std[local_idx]
                            unique_expected_improvement[unique_idx] = selection.expected_improvement[local_idx]
                            unique_acquisition_value[unique_idx] = selection.acquisition_value[local_idx]
                            unique_local_kernel_log_prob[unique_idx] = selection.local_kernel_log_prob[local_idx]
                        if not selection.selected_indices:
                            if not selected_unique_indices:
                                active_learning_threshold_exhausted = bool(selection.threshold_exhausted)
                            break
                        local_idx = int(selection.selected_indices[0])
                        if local_idx < 0 or local_idx >= len(remaining_unique_indices):
                            break
                        unique_idx = remaining_unique_indices.pop(local_idx)
                        unique_selected[unique_idx] = True
                        selected_leaf_batch = unique_leaf_rows[unique_idx]
                        start = timings.start()
                        selected_rewards = _score_oracle_batch(
                            selected_leaf_batch,
                            reward_fn,
                            context="sequential acquisition",
                        )
                        timings.stop("oracle_score_seconds", start)
                        oracle_scored_count += int(selected_leaf_batch.shape[0])
                        if selected_rewards.ndim == 0:
                            selected_rewards = selected_rewards.view(1)
                        start = timings.start()
                        _observe_oracle_batch(
                            active_learning_selector,
                            selected_leaf_batch,
                            selected_rewards,
                            context="sequential acquisition",
                        )
                        timings.stop("acquisition_observe_seconds", start)
                        unique_rewards[unique_idx] = selected_rewards[0]
                        unique_oracle_evaluated[unique_idx] = True
                        unique_diagnostic_score[unique_idx] = float(selected_rewards[0].item())
                        selected_unique_indices.append(unique_idx)
                else:
                    available_leaf_batch = torch.cat([unique_leaf_rows[idx] for idx in available_unique_indices], dim=0)
                    start = timings.start()
                    if proposal_batch.preselected:
                        # The proposal method already chose this paid batch by
                        # posterior mean. Do not make a second acquisition pass
                        # or consume additional random draws.
                        from search.interpolant.active_learning import ActiveLearningSelection
                        n = len(available_unique_indices)
                        selection = ActiveLearningSelection(
                            selected_indices=list(range(min(n, oracle_budget))),
                            smiles=[unique_smiles[i] for i in available_unique_indices],
                            surrogate_mean=[None] * n, surrogate_std=[None] * n,
                            expected_improvement=[None] * n, acquisition_value=[None] * n,
                            local_kernel_log_prob=[None] * n,
                        )
                    else:
                        selection = active_learning_selector.select(
                            available_leaf_batch,
                            budget=oracle_budget,
                            local_kernel_log_probs=[unique_local_kernel_log_prob[idx] for idx in available_unique_indices],
                        )
                    timings.stop("acquisition_select_seconds", start)
                    active_learning_threshold_exhausted = bool(selection.threshold_exhausted)
                    active_learning_max_expected_improvement = selection.max_expected_improvement
                    active_learning_num_above_threshold = int(selection.num_candidates_above_threshold)
                    for local_idx, unique_idx in enumerate(available_unique_indices):
                        unique_smiles[unique_idx] = selection.smiles[local_idx]
                        unique_surrogate_mean[unique_idx] = selection.surrogate_mean[local_idx]
                        unique_surrogate_std[unique_idx] = selection.surrogate_std[local_idx]
                        unique_expected_improvement[unique_idx] = selection.expected_improvement[local_idx]
                        unique_acquisition_value[unique_idx] = selection.acquisition_value[local_idx]
                        unique_local_kernel_log_prob[unique_idx] = selection.local_kernel_log_prob[local_idx]
                    seen_selected: set[int] = set()
                    for local_idx in selection.selected_indices:
                        if local_idx < 0 or local_idx >= len(available_unique_indices):
                            continue
                        unique_idx = available_unique_indices[local_idx]
                        if unique_idx in seen_selected:
                            continue
                        seen_selected.add(unique_idx)
                        unique_selected[unique_idx] = True
                        selected_unique_indices.append(unique_idx)

                    if selected_unique_indices:
                        selected_leaf_batch = torch.cat(
                            [unique_leaf_rows[idx] for idx in selected_unique_indices], dim=0
                        )
                        start = timings.start()
                        oracle_rewards = _score_oracle_batch(
                            selected_leaf_batch,
                            reward_fn,
                            context="batched acquisition",
                        )
                        timings.stop("oracle_score_seconds", start)
                        oracle_scored_count += int(selected_leaf_batch.shape[0])
                        if oracle_rewards.ndim == 0:
                            oracle_rewards = oracle_rewards.view(1)
                        start = timings.start()
                        _observe_oracle_batch(
                            active_learning_selector,
                            selected_leaf_batch,
                            oracle_rewards,
                            context="batched acquisition",
                        )
                        timings.stop("acquisition_observe_seconds", start)
                        for row_idx, unique_idx in enumerate(selected_unique_indices):
                            unique_rewards[unique_idx] = oracle_rewards[row_idx]
                            unique_oracle_evaluated[unique_idx] = True
                            unique_diagnostic_score[unique_idx] = float(oracle_rewards[row_idx].item())
            if active_learning_selector is None and selected_unique_indices:
                selected_leaf_batch = torch.cat([unique_leaf_rows[idx] for idx in selected_unique_indices], dim=0)
                start = timings.start()
                selected_rewards = _score_oracle_batch(
                    selected_leaf_batch,
                    reward_fn,
                    context="proposal evaluation",
                )
                timings.stop("oracle_score_seconds", start)
                oracle_scored_count += int(selected_leaf_batch.shape[0])
                if selected_rewards.ndim == 0:
                    selected_rewards = selected_rewards.view(1)
                for row_idx, unique_idx in enumerate(selected_unique_indices):
                    unique_selected[unique_idx] = True
                    unique_rewards[unique_idx] = selected_rewards[row_idx]
                    unique_oracle_evaluated[unique_idx] = True
                    unique_diagnostic_score[unique_idx] = float(selected_rewards[row_idx].item())

        if diagnostic_injected_unique_idx is not None:
            idx = int(diagnostic_injected_unique_idx)
            if 0 <= idx < len(unique_selected):
                diagnostic_injected_global_best_selected = bool(unique_selected[idx])
                diagnostic_injected_global_best_acquisition_candidate = bool(unique_is_acquisition_candidate[idx])
                diagnostic_injected_global_best_surrogate_mean = unique_surrogate_mean[idx]
                diagnostic_injected_global_best_surrogate_std = unique_surrogate_std[idx]
                diagnostic_injected_global_best_expected_improvement = unique_expected_improvement[idx]
                diagnostic_injected_global_best_acquisition_value = unique_acquisition_value[idx]

                ranked_acq_items: list[tuple[float, int]] = []
                for candidate_idx in acquisition_unique_indices:
                    if candidate_idx < 0 or candidate_idx >= len(unique_acquisition_value):
                        continue
                    value = unique_acquisition_value[candidate_idx]
                    if value is None or not math.isfinite(float(value)):
                        continue
                    ranked_acq_items.append((float(value), int(candidate_idx)))
                ranked_acq_items.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                for rank_idx, (_, candidate_idx) in enumerate(ranked_acq_items, start=1):
                    if candidate_idx == idx:
                        diagnostic_injected_global_best_rank_by_acquisition = int(rank_idx)
                        break

        if (
            score_full_pool_for_ranking
            and active_learning_selector is not None
            and (not proposal_collapse_detected or not proposal_collapse_block_selection)
            and acquisition_unique_indices
        ):
            diagnostic_unique_indices = [
                idx
                for idx in acquisition_unique_indices
                if (
                    not unique_oracle_evaluated[idx]
                    and unique_leaf_rows[idx] is not None
                    and unique_diagnostic_score[idx] is None
                )
            ]
            if active_learning_budget_override is not None:
                diagnostic_budget = max(
                    0,
                    int(active_learning_budget_override) - int(oracle_scored_count),
                )
                diagnostic_unique_indices = diagnostic_unique_indices[:diagnostic_budget]
            if diagnostic_unique_indices:
                diagnostic_leaf_batch = torch.cat(
                    [unique_leaf_rows[idx] for idx in diagnostic_unique_indices],
                    dim=0,
                )
                start = timings.start()
                diagnostic_rewards = _score_oracle_batch(
                    diagnostic_leaf_batch,
                    reward_fn,
                    context="full-proposal diagnostics",
                )
                timings.stop("full_batch_diagnostic_score_seconds", start)
                oracle_scored_count += int(diagnostic_leaf_batch.shape[0])
                if diagnostic_rewards.ndim == 0:
                    diagnostic_rewards = diagnostic_rewards.view(1)
                for row_idx, unique_idx in enumerate(diagnostic_unique_indices):
                    score_value = float(diagnostic_rewards[row_idx].item())
                    unique_diagnostic_score[unique_idx] = score_value
                    unique_diagnostic_score_evaluated[unique_idx] = True

        # Black-box Ehrlich baselines may only learn transition validity from a
        # paid oracle outcome. Keep those candidates in the acquisition pool,
        # then mark the raw non-finite result after scoring without replacing it.
        if ehrlich_black_box_feasibility:
            for unique_idx in range(len(unique_props)):
                score_value: float | None = None
                if unique_oracle_evaluated[unique_idx]:
                    score_value = float(unique_rewards[unique_idx].item())
                elif unique_diagnostic_score_evaluated[unique_idx]:
                    diagnostic_value = unique_diagnostic_score[unique_idx]
                    if diagnostic_value is not None:
                        score_value = float(diagnostic_value)
                if score_value is None or not (math.isinf(score_value) and score_value < 0.0):
                    continue
                if not unique_oracle_rejected[unique_idx]:
                    oracle_rejected_count += 1
                unique_oracle_rejected[unique_idx] = True

        full_batch_top_score: Optional[float] = None
        full_batch_top_smiles: Optional[str] = None
        full_batch_top_t_value: Optional[float] = None
        full_batch_top_selected: Optional[bool] = None
        full_batch_best_selected_score: Optional[float] = None
        full_batch_scored_count = 0
        if (
            score_full_pool_for_ranking
            and active_learning_selector is not None
            and (not proposal_collapse_detected or not proposal_collapse_block_selection)
        ):
            full_batch_scored_items: list[tuple[float, int]] = []
            for unique_idx in acquisition_unique_indices:
                score_value = unique_diagnostic_score[unique_idx]
                if score_value is None and unique_oracle_evaluated[unique_idx]:
                    score_value = float(unique_rewards[unique_idx].item())
                    unique_diagnostic_score[unique_idx] = score_value
                if score_value is None or not math.isfinite(float(score_value)):
                    continue
                full_batch_scored_items.append((float(score_value), int(unique_idx)))
            full_batch_scored_items.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            for rank_idx, (_, unique_idx) in enumerate(full_batch_scored_items, start=1):
                unique_full_batch_rank_by_score[unique_idx] = int(rank_idx)
            full_batch_scored_count = len(full_batch_scored_items)
            if full_batch_scored_items:
                full_batch_top_score, full_batch_top_idx = full_batch_scored_items[0]
                full_batch_top_smiles = unique_smiles[full_batch_top_idx]
                full_batch_top_t_value = float(unique_props[full_batch_top_idx].t_value)
                full_batch_top_selected = bool(unique_selected[full_batch_top_idx])
                selected_scored = [
                    float(score) for score, unique_idx in full_batch_scored_items if unique_selected[unique_idx]
                ]
                if selected_scored:
                    full_batch_best_selected_score = max(selected_scored)

        candidate_reward_values: list[float] = []
        unique_trace_templates: dict[int, ProposalTrace] = {}
        trace_metrics_by_context: dict[tuple[int, int], tuple[torch.Tensor, _ProposalTraceMetrics]] = {}
        trace_start = timings.start()
        # These copies are for diagnostics only. Selection, rollout, oracle and
        # archive updates continue to use the original tensors and devices.
        trace_base_samples = base_samples.detach().cpu()
        trace_base_rewards = base_rewards.detach().cpu()
        trace_rewards = unique_rewards.detach().cpu()
        trace_leaves = _trace_tensors_cpu(unique_leaf_rows)
        trace_sources = {id(p): p for p in [*unique_props, *proposal_batch.proposals]}
        trace_xt = _trace_tensors_cpu([p.xt for p in trace_sources.values()])
        trace_masks = _trace_tensors_cpu([p.corrupt_mask for p in trace_sources.values()])
        trace_proposals = {
            key: replace(p, xt=xt, corrupt_mask=mask)
            for (key, p), xt, mask in zip(trace_sources.items(), trace_xt, trace_masks)
        }
        model = getattr(getattr(adapter, "process", None), "model", None)
        # Custom distance callbacks retain their original input device contract.
        custom_trace_distance = callable(getattr(model, "sequence_edit_distance", None)) and not legacy_hamming_tr_error
        for idx, proposal in enumerate(unique_props):
            proposal = trace_proposals[id(proposal)]
            leaf = trace_leaves[idx]
            if leaf is None:
                continue

            base_row = base_index_to_row.get(int(proposal.base_index))
            if base_row is None or base_row < 0 or base_row >= int(base_samples.shape[0]):
                continue
            base_state = trace_base_samples[base_row : base_row + 1]
            if proposal.xt.shape != base_state.shape or leaf.shape != proposal.xt.shape:
                continue
            corrupt_mask_1d, corrupt_mask_inferred = _resolved_corrupt_mask(
                proposal,
                base_state,
            )
            inferred_corrupt_masks += int(corrupt_mask_inferred)
            retained_1d = leaf[0] == proposal.xt[0]
            retained_frac = float(retained_1d.float().mean().item())
            retained_sum += retained_frac
            retained_count += 1
            acc = retention_by_time_acc.setdefault(
                round(float(proposal.t_value), 6),
                {
                    "count": 0.0,
                    "sum": 0.0,
                    "count_corr": 0.0,
                    "sum_corr": 0.0,
                    "count_uncorr": 0.0,
                    "sum_uncorr": 0.0,
                    "count_logical": 0.0,
                    "sum_logical": 0.0,
                    "t_sum": 0.0,
                },
            )
            acc["count"] += 1.0
            acc["sum"] += retained_frac
            acc["t_sum"] += float(proposal.t_value)

            if bool(corrupt_mask_1d.any().item()):
                retained_frac_corr = float(retained_1d[corrupt_mask_1d].float().mean().item())
                retained_sum_corrupted += retained_frac_corr
                retained_count_corrupted += 1
                acc["count_corr"] += 1.0
                acc["sum_corr"] += retained_frac_corr
            if bool((~corrupt_mask_1d).any().item()):
                retained_frac_uncorr = float(retained_1d[~corrupt_mask_1d].float().mean().item())
                retained_sum_uncorrupted += retained_frac_uncorr
                retained_count_uncorrupted += 1
                acc["count_uncorr"] += 1.0
                acc["sum_uncorr"] += retained_frac_uncorr

            model_changed_1d = leaf[0] != proposal.xt[0]
            trace_metrics = _proposal_trace_metrics(
                adapter=adapter,
                base_state=base_state,
                proposal=proposal,
                leaf=leaf,
                corrupt_mask_1d=corrupt_mask_1d,
                model_changed_1d=model_changed_1d,
                force_hamming=bool(legacy_hamming_tr_error),
                distance_inputs=(unique_leaf_rows[idx][0], base_samples[base_row]) if custom_trace_distance else None,
            )
            trace_metrics_by_context[(idx, base_row)] = (
                corrupt_mask_1d.detach().clone(),
                trace_metrics,
            )
            proposal_logical_edits_total += trace_metrics.proposal_logical_edits
            model_logical_edits_total += trace_metrics.model_logical_edits
            dropped_proposal_logical_edits += trace_metrics.dropped_proposal_logical_edits
            introduced_logical_edits += trace_metrics.introduced_logical_edits
            if trace_metrics.retained_proposal_logical_frac is not None:
                logical_retained_sum += trace_metrics.retained_proposal_logical_frac
                logical_retained_count += 1
                acc["count_logical"] += 1.0
                acc["sum_logical"] += trace_metrics.retained_proposal_logical_frac

            selected_for_scoring = bool(unique_selected[idx] or unique_oracle_evaluated[idx])
            selected_for_oracle = bool(unique_oracle_evaluated[idx])
            score_value = float(trace_rewards[idx].item()) if selected_for_scoring else float("nan")
            trace = ProposalTrace(
                base_index=int(proposal.base_index),
                t_value=float(proposal.t_value),
                expected_edits=float(proposal.edits_value),
                proposal_corrupted_tokens=trace_metrics.proposal_corrupted_tokens,
                model_changes_total=trace_metrics.model_changes_total,
                model_changes_on_corrupted=trace_metrics.model_changes_on_corrupted,
                model_changes_on_uncorrupted=trace_metrics.model_changes_on_uncorrupted,
                ancestor_to_child_edits=trace_metrics.ancestor_to_child_edits,
                ancestor_seq=base_state[0].detach().cpu().clone(),
                from_seq=proposal.xt[0].detach().cpu().clone(),
                to_seq=leaf[0].detach().cpu().clone(),
                base_score=float(trace_base_rewards[base_row].item()),
                score=score_value,
                proposal_logical_edits=trace_metrics.proposal_logical_edits,
                model_logical_edits=trace_metrics.model_logical_edits,
                retained_proposal_logical_edits=(trace_metrics.retained_proposal_logical_edits),
                dropped_proposal_logical_edits=(trace_metrics.dropped_proposal_logical_edits),
                introduced_logical_edits=trace_metrics.introduced_logical_edits,
                retained_proposal_logical_frac=(trace_metrics.retained_proposal_logical_frac),
                oracle_evaluated=selected_for_oracle,
                smiles=unique_smiles[idx],
                surrogate_mean=unique_surrogate_mean[idx],
                surrogate_std=unique_surrogate_std[idx],
                expected_improvement=unique_expected_improvement[idx],
                acquisition_value=unique_acquisition_value[idx],
                local_kernel_log_prob=unique_local_kernel_log_prob[idx],
                local_kernel_log_prob_sequence=unique_local_kernel_log_prob_sequence[idx],
                acquisition_candidate=bool(unique_is_acquisition_candidate[idx]),
                acquisition_selected=bool(unique_selected[idx]),
                duplicate_exact_proposal=False,
                duplicate_generated_leaf=False,
                cached_generated_proposal=bool(unique_is_cached[idx]),
                invalid_generated_proposal=bool(unique_is_invalid[idx]),
                oracle_rejected=bool(unique_oracle_rejected[idx]),
                diagnostic_score=unique_diagnostic_score[idx],
                diagnostic_score_evaluated=bool(unique_diagnostic_score_evaluated[idx]),
                full_batch_rank_by_score=unique_full_batch_rank_by_score[idx],
                diagnostic_injected_global_best=bool(unique_is_injected_global_best[idx]),
                diagnostic_injected_source=(
                    diagnostic_injected_candidate.source
                    if bool(unique_is_injected_global_best[idx]) and diagnostic_injected_candidate is not None
                    else None
                ),
                diagnostic_injected_full_dataset_score=(
                    diagnostic_injected_candidate.full_dataset_score
                    if bool(unique_is_injected_global_best[idx]) and diagnostic_injected_candidate is not None
                    else None
                ),
                parent_seq_hash=(
                    proposal.parent_seq_hash or _trace_sequence_hash(base_state[0])
                ),
                parent_score=(
                    float(proposal.parent_score)
                    if proposal.parent_score is not None
                    else float(trace_base_rewards[base_row].item())
                ),
                parent_archive_index=proposal.parent_archive_index,
                parent_archive_probability=proposal.parent_archive_probability,
                parent_selection_log_prob=proposal.parent_selection_log_prob,
                archive_size=proposal.archive_size,
                archive_snapshot_id=proposal.archive_snapshot_id,
                hamming_conditional_log_prob=proposal.hamming_conditional_log_prob,
                proposal_log_prob=proposal.proposal_log_prob,
                proposal_probability_method=proposal.proposal_probability_method,
            )
            unique_trace_templates[idx] = trace

            if selected_for_scoring:
                candidate_leaves.append(unique_leaf_rows[idx])
                candidate_reward_values.append(score_value)
                proposal_traces.append(trace)
                if score_value > best_candidate_reward:
                    best_candidate_reward = score_value
                    best_candidate_state = unique_leaf_rows[idx]
                    best_candidate_t = float(proposal.t_value)
                    best_candidate_edits = float(proposal.edits_value)
                if (
                    track_trust_region
                    and not bool(unique_is_injected_global_best[idx])
                    and score_value > tr_best_reward
                ):
                    tr_best_reward = score_value
                    tr_best_t = float(proposal.t_value)
                    tr_best_edits = float(proposal.edits_value)
            if bool(unique_is_injected_global_best[idx]):
                proposal_traces_all.append(trace)

            if trace_metrics.model_changes_total <= 0:
                continue
            model_changes_total += trace_metrics.model_changes_total
            model_changes_on_corrupted += trace_metrics.model_changes_on_corrupted
            model_changes_on_uncorrupted += trace_metrics.model_changes_on_uncorrupted
            if trace_metrics.model_changes_on_uncorrupted == 0 and trace_metrics.model_changes_on_corrupted > 0:
                changed_proposals_only_corrupted += 1
            if trace_metrics.model_changes_on_uncorrupted > 0:
                changed_proposals_with_uncorrupted += 1
        timings.stop("trace_build_deduped_seconds", trace_start)

    oracle_evaluated_count = int(sum(1 for trace in proposal_traces if trace.oracle_evaluated))
    oracle_privileged_scored_count = (
        max(0, int(oracle_scored_count) - int(oracle_evaluated_count))
        if oracle_best_budget_mode == "selected_archive"
        else 0
    )
    oracle_budget_debit_count = (
        int(oracle_evaluated_count) if oracle_best_budget_mode == "selected_archive" else int(oracle_scored_count)
    )
    candidate_rewards = torch.as_tensor(candidate_reward_values, dtype=torch.float, device=base_samples.device)

    if proposal_batch.proposals and unique_props:
        if active_learning_selector is None and active_learning_budget_override is None:
            evaluated_proposals = list(proposal_batch.proposals)
            evaluated_rewards = torch.empty(
                (len(proposal_batch.proposals),), dtype=torch.float, device=base_samples.device
            )
            for original_idx, unique_idx in enumerate(original_to_unique):
                evaluated_rewards[original_idx] = unique_rewards[unique_idx]
        else:
            selected_original_indices = [
                original_idx
                for original_idx, unique_idx in enumerate(original_to_unique)
                if unique_selected[unique_idx]
                and not original_is_duplicate_exact[original_idx]
                and not original_is_duplicate_generated_leaf[original_idx]
            ]
            evaluated_proposals = [proposal_batch.proposals[idx] for idx in selected_original_indices]
            evaluated_rewards = torch.empty(
                (len(selected_original_indices),), dtype=torch.float, device=base_samples.device
            )
            for row_idx, original_idx in enumerate(selected_original_indices):
                evaluated_rewards[row_idx] = unique_rewards[original_to_unique[original_idx]]
    else:
        evaluated_proposals = []
        evaluated_rewards = torch.empty((0,), dtype=torch.float, device=base_samples.device)

    if proposal_batch.proposals and unique_props:
        trace_all_start = timings.start()
        for original_idx, proposal in enumerate(proposal_batch.proposals):
            proposal = trace_proposals[id(proposal)]
            unique_idx = original_to_unique[original_idx]
            if unique_idx < 0 or unique_idx >= len(unique_leaf_rows):
                continue
            leaf = trace_leaves[unique_idx]
            if leaf is None:
                continue

            base_row = base_index_to_row.get(int(proposal.base_index))
            if base_row is None or base_row < 0 or base_row >= int(base_samples.shape[0]):
                continue
            base_state = trace_base_samples[base_row : base_row + 1]
            if proposal.xt.shape != base_state.shape or leaf.shape != proposal.xt.shape:
                continue

            trace_template = unique_trace_templates.get(unique_idx)
            if trace_template is None:
                continue
            corrupt_mask_1d, _ = _resolved_corrupt_mask(proposal, base_state)
            model_changed_1d = leaf[0] != proposal.xt[0]
            context_key = (unique_idx, base_row)
            cached_context = trace_metrics_by_context.get(context_key)
            if cached_context is not None and torch.equal(
                corrupt_mask_1d,
                cached_context[0],
            ):
                trace_metrics = cached_context[1]
            else:
                trace_metrics = _proposal_trace_metrics(
                    adapter=adapter,
                    base_state=base_state,
                    proposal=proposal,
                    leaf=leaf,
                    corrupt_mask_1d=corrupt_mask_1d,
                    model_changed_1d=model_changed_1d,
                    force_hamming=bool(legacy_hamming_tr_error),
                    ancestor_metrics=(cached_context[1] if cached_context is not None else None),
                    distance_inputs=(unique_leaf_rows[unique_idx][0], base_samples[base_row]) if custom_trace_distance else None,
                )
                if cached_context is None:
                    trace_metrics_by_context[context_key] = (
                        corrupt_mask_1d.detach().clone(),
                        trace_metrics,
                    )
            original_is_unique = not (
                original_is_duplicate_exact[original_idx] or original_is_duplicate_generated_leaf[original_idx]
            )
            original_oracle_evaluated = bool(unique_oracle_evaluated[unique_idx]) and original_is_unique
            proposal_traces_all.append(
                replace(
                    trace_template,
                    base_index=int(proposal.base_index),
                    t_value=float(proposal.t_value),
                    expected_edits=float(proposal.edits_value),
                    proposal_corrupted_tokens=trace_metrics.proposal_corrupted_tokens,
                    model_changes_total=trace_metrics.model_changes_total,
                    model_changes_on_corrupted=(trace_metrics.model_changes_on_corrupted),
                    model_changes_on_uncorrupted=(trace_metrics.model_changes_on_uncorrupted),
                    ancestor_to_child_edits=trace_metrics.ancestor_to_child_edits,
                    ancestor_seq=base_state[0].detach().cpu().clone(),
                    from_seq=proposal.xt[0].detach().cpu().clone(),
                    to_seq=leaf[0].detach().cpu().clone(),
                    base_score=float(trace_base_rewards[base_row].item()),
                    score=(float(trace_rewards[unique_idx].item()) if original_oracle_evaluated else float("nan")),
                    proposal_logical_edits=trace_metrics.proposal_logical_edits,
                    model_logical_edits=trace_metrics.model_logical_edits,
                    retained_proposal_logical_edits=(trace_metrics.retained_proposal_logical_edits),
                    dropped_proposal_logical_edits=(trace_metrics.dropped_proposal_logical_edits),
                    introduced_logical_edits=trace_metrics.introduced_logical_edits,
                    retained_proposal_logical_frac=(trace_metrics.retained_proposal_logical_frac),
                    oracle_evaluated=original_oracle_evaluated,
                    acquisition_candidate=(bool(unique_is_acquisition_candidate[unique_idx]) and original_is_unique),
                    acquisition_selected=(bool(unique_selected[unique_idx]) and original_is_unique),
                    duplicate_exact_proposal=bool(original_is_duplicate_exact[original_idx]),
                    duplicate_generated_leaf=bool(original_is_duplicate_generated_leaf[original_idx]),
                    diagnostic_injected_global_best=False,
                    diagnostic_injected_source=None,
                    diagnostic_injected_full_dataset_score=None,
                    parent_seq_hash=(
                        proposal.parent_seq_hash or _trace_sequence_hash(base_state[0])
                    ),
                    parent_score=(
                        float(proposal.parent_score)
                        if proposal.parent_score is not None
                        else float(trace_base_rewards[base_row].item())
                    ),
                    parent_archive_index=proposal.parent_archive_index,
                    parent_archive_probability=proposal.parent_archive_probability,
                    parent_selection_log_prob=proposal.parent_selection_log_prob,
                    archive_size=proposal.archive_size,
                    archive_snapshot_id=proposal.archive_snapshot_id,
                    hamming_conditional_log_prob=proposal.hamming_conditional_log_prob,
                    proposal_log_prob=proposal.proposal_log_prob,
                    proposal_probability_method=proposal.proposal_probability_method,
                )
            )
        timings.stop("trace_build_all_seconds", trace_all_start)
    else:
        proposal_traces_all = list(proposal_traces)
        full_batch_scored_count = 0
        full_batch_top_score = None
        full_batch_top_smiles = None
        full_batch_top_t_value = None
        full_batch_top_selected = None
        full_batch_best_selected_score = None

    retention_by_time: list[ProposalRetentionByTime] = []
    for t_key in sorted(retention_by_time_acc.keys()):
        acc = retention_by_time_acc[t_key]
        count = int(acc["count"])
        mean_retained_frac = float(acc["sum"] / acc["count"]) if acc["count"] > 0.0 else 0.0
        mean_t_value = float(acc["t_sum"] / acc["count"]) if acc["count"] > 0.0 else 0.0
        mean_retained_frac_corrupted: Optional[float]
        if acc["count_corr"] > 0.0:
            mean_retained_frac_corrupted = float(acc["sum_corr"] / acc["count_corr"])
        else:
            mean_retained_frac_corrupted = None
        mean_retained_frac_uncorrupted: Optional[float]
        if acc["count_uncorr"] > 0.0:
            mean_retained_frac_uncorrupted = float(acc["sum_uncorr"] / acc["count_uncorr"])
        else:
            mean_retained_frac_uncorrupted = None
        mean_retained_frac_logical: Optional[float]
        if acc["count_logical"] > 0.0:
            mean_retained_frac_logical = float(acc["sum_logical"] / acc["count_logical"])
        else:
            mean_retained_frac_logical = None
        retention_by_time.append(
            ProposalRetentionByTime(
                t_value=mean_t_value,
                count=count,
                mean_retained_frac=mean_retained_frac,
                mean_retained_frac_corrupted=mean_retained_frac_corrupted,
                mean_retained_frac_uncorrupted=mean_retained_frac_uncorrupted,
                mean_retained_frac_logical=mean_retained_frac_logical,
            )
        )

    filter_by_time: list[ProposalFilterByTime] = []
    for t_key in sorted(filter_by_time_acc.keys()):
        acc = filter_by_time_acc[t_key]
        count = float(acc["t_count"])
        mean_t_value = float(acc["t_sum"] / count) if count > 0.0 else float(t_key)
        filter_by_time.append(
            ProposalFilterByTime(
                t_value=mean_t_value,
                generated_proposals=int(acc["generated"]),
                deduped_proposals=int(acc["deduped"]),
                duplicate_exact_proposals=int(acc["duplicate_exact"]),
                duplicate_generated_leaves=int(acc["duplicate_leaf"]),
                cached_generated_proposals=int(acc["cached"]),
                invalid_generated_proposals=int(acc["invalid"]),
                acquisition_batch_size=int(acc["acquisition"]),
            )
        )

    timings.add_count("duplicate_generated_leaves", duplicate_generated_leaf_count if unique_props else 0)
    timings.add_count("cached_generated_proposals", cached_generated_proposal_count if unique_props else 0)
    timings.add_count("invalid_generated_proposals", invalid_generated_proposal_count if unique_props else 0)
    timings.add_count("acquisition_batch_size", acquisition_batch_size if unique_props else 0)
    timings.add_count("oracle_evaluated_count", oracle_evaluated_count)
    timings.add_count("oracle_budget_debit_count", oracle_budget_debit_count)
    timings.add_count("oracle_scored_count", oracle_scored_count)
    timings.add_count("oracle_privileged_scored_count", oracle_privileged_scored_count)
    timings.add_count("oracle_rejected_count", oracle_rejected_count)
    timings.add_count("full_batch_scored_count", full_batch_scored_count if active_learning_selector is not None else 0)
    timings.stop("total_seconds", total_start)

    change_stats = ProposalChangeStats(
        generated_proposals=generated_proposal_count,
        deduped_proposals=len(unique_props),
        duplicate_exact_proposals=duplicate_exact_proposal_count,
        duplicate_generated_leaves=duplicate_generated_leaf_count if unique_props else 0,
        cached_generated_proposals=cached_generated_proposal_count if unique_props else 0,
        invalid_generated_proposals=invalid_generated_proposal_count if unique_props else 0,
        acquisition_batch_size=acquisition_batch_size if unique_props else 0,
        inferred_corrupt_masks=inferred_corrupt_masks,
        model_changes_total=model_changes_total,
        model_changes_on_corrupted=model_changes_on_corrupted,
        model_changes_on_uncorrupted=model_changes_on_uncorrupted,
        proposal_logical_edits_total=proposal_logical_edits_total,
        model_logical_edits_total=model_logical_edits_total,
        dropped_proposal_logical_edits=dropped_proposal_logical_edits,
        introduced_logical_edits=introduced_logical_edits,
        changed_proposals_only_corrupted=changed_proposals_only_corrupted,
        changed_proposals_with_uncorrupted=changed_proposals_with_uncorrupted,
        mean_retained_frac=(retained_sum / retained_count) if retained_count > 0 else None,
        mean_retained_frac_corrupted=(
            retained_sum_corrupted / retained_count_corrupted if retained_count_corrupted > 0 else None
        ),
        mean_retained_frac_uncorrupted=(
            retained_sum_uncorrupted / retained_count_uncorrupted if retained_count_uncorrupted > 0 else None
        ),
        mean_retained_frac_logical=(
            logical_retained_sum / logical_retained_count if logical_retained_count > 0 else None
        ),
        retention_by_time=retention_by_time,
        filter_by_time=filter_by_time,
    )

    if unique_props:
        for unique_idx, record in enumerate(candidate_records):
            raw_score = float(unique_rewards[unique_idx].item())
            record.leaf = unique_leaf_rows[unique_idx]
            record.decoded = unique_smiles[unique_idx]
            record.score = raw_score if math.isfinite(raw_score) else None
            record.cached = bool(unique_is_cached[unique_idx])
            record.invalid = bool(unique_is_invalid[unique_idx])
            record.duplicate_leaf = bool(unique_is_duplicate_generated_leaf[unique_idx])
            record.acquisition_candidate = bool(unique_is_acquisition_candidate[unique_idx])
            record.acquisition_selected = bool(unique_selected[unique_idx])
            record.oracle_evaluated = bool(unique_oracle_evaluated[unique_idx])
            record.surrogate_mean = unique_surrogate_mean[unique_idx]
            record.surrogate_std = unique_surrogate_std[unique_idx]
            record.expected_improvement = unique_expected_improvement[unique_idx]
            record.acquisition_value = unique_acquisition_value[unique_idx]

    return ProposalEvaluation(
        candidate_leaves=candidate_leaves,
        best_candidate_state=best_candidate_state,
        best_candidate_reward=best_candidate_reward,
        best_candidate_t=best_candidate_t,
        best_candidate_edits=best_candidate_edits,
        tr_best_t=tr_best_t,
        tr_best_edits=tr_best_edits,
        evaluated_proposals=evaluated_proposals,
        evaluated_rewards=evaluated_rewards,
        candidate_rewards=candidate_rewards,
        oracle_evaluated_count=oracle_evaluated_count,
        oracle_budget_debit_count=oracle_budget_debit_count,
        oracle_scored_count=oracle_scored_count,
        oracle_privileged_scored_count=oracle_privileged_scored_count,
        oracle_rejected_count=oracle_rejected_count,
        active_learning_threshold_exhausted=active_learning_threshold_exhausted
        if active_learning_selector is not None
        else False,
        active_learning_max_expected_improvement=(
            active_learning_max_expected_improvement if active_learning_selector is not None else None
        ),
        active_learning_num_above_threshold=(
            active_learning_num_above_threshold if active_learning_selector is not None else 0
        ),
        proposal_collapse_detected=bool(proposal_collapse_detected),
        proposal_collapse_acquisition_fraction=(
            proposal_collapse_acquisition_fraction if active_learning_selector is not None else None
        ),
        proposal_collapse_min_acquisition_fraction=(
            proposal_collapse_min_acquisition_fraction if active_learning_selector is not None else None
        ),
        proposal_collapse_min_acquisition_count=(
            proposal_collapse_min_acquisition_count if active_learning_selector is not None else None
        ),
        proposal_collapse_target_count=(
            proposal_collapse_effective_target_count if active_learning_selector is not None else None
        ),
        full_batch_scored_count=full_batch_scored_count if active_learning_selector is not None else 0,
        full_batch_top_score=full_batch_top_score if active_learning_selector is not None else None,
        full_batch_top_smiles=full_batch_top_smiles if active_learning_selector is not None else None,
        full_batch_top_t_value=full_batch_top_t_value if active_learning_selector is not None else None,
        full_batch_top_selected=full_batch_top_selected if active_learning_selector is not None else None,
        full_batch_best_selected_score=full_batch_best_selected_score if active_learning_selector is not None else None,
        diagnostic_injected_global_best_present=bool(diagnostic_injected_global_best_present),
        diagnostic_injected_global_best_acquisition_candidate=bool(
            diagnostic_injected_global_best_acquisition_candidate
        ),
        diagnostic_injected_global_best_selected=bool(diagnostic_injected_global_best_selected),
        diagnostic_injected_global_best_rank_by_acquisition=diagnostic_injected_global_best_rank_by_acquisition,
        diagnostic_injected_global_best_surrogate_mean=diagnostic_injected_global_best_surrogate_mean,
        diagnostic_injected_global_best_surrogate_std=diagnostic_injected_global_best_surrogate_std,
        diagnostic_injected_global_best_expected_improvement=(diagnostic_injected_global_best_expected_improvement),
        diagnostic_injected_global_best_acquisition_value=diagnostic_injected_global_best_acquisition_value,
        diagnostic_injected_global_best_sequence=(
            diagnostic_injected_candidate.sequence_string if diagnostic_injected_candidate is not None else None
        ),
        diagnostic_injected_global_best_full_dataset_score=(
            diagnostic_injected_candidate.full_dataset_score if diagnostic_injected_candidate is not None else None
        ),
        change_stats=change_stats,
        proposal_traces_all=proposal_traces_all,
        proposal_traces=proposal_traces,
        timing_seconds=dict(timings.seconds),
        timing_counts=dict(timings.counts),
        proposal_generation_diagnostics=dict(proposal_batch.generation_diagnostics),
        candidate_records=candidate_records,
    )


__all__ = [
    "DiagnosticInjectedCandidate",
    "CandidateRecord",
    "ProposalChangeStats",
    "ProposalEvaluation",
    "ProposalFilterByTime",
    "ProposalRetentionByTime",
    "ProposalTrace",
    "evaluate_proposals",
]
