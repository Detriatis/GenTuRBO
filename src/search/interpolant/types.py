"""Shared dataclasses for interpolant search policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class CandidateConfig:
    strategy: str
    source: str
    mixing_mode: str
    num_candidates: int
    mixes_per_time: int
    fixed_times: Optional[list[float]]


@dataclass
class TrustRegionConfig:
    kind: str
    eps: float
    log_every: int
    domain: str = "edits"
    mean_edits: Optional[float] = None
    mean_time: Optional[float] = None
    edits_delta: float = 0.0
    edits_min: float = 0.0
    edits_max: Optional[float] = None
    edits_init: Optional[float] = None
    time_init: Optional[float] = None
    time_min: float = 0.0
    time_max: float = 1.0
    tau_success: int = 1
    tau_failure: int = 1
    region_scalar: float = 2.0
    region_expand_scalar: Optional[float] = None
    region_contract_scalar: Optional[float] = None
    region_shrink_scalar: Optional[float] = None
    alpha: float = 1.0
    alpha_k: float = 0.0
    alpha_min: float = 0.25
    alpha_max: float = 2.0


@dataclass
class Proposal:
    base_index: int
    t_value: float
    edits_value: float
    xt: torch.Tensor = None
    corrupt_mask: Optional[torch.Tensor] = None
    direct_leaf: bool = False
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


@dataclass
class CandidateRecord:
    """One proposal and its evaluation state.

    This is the typed replacement boundary for evaluator-local parallel arrays.
    Legacy ``ProposalTrace`` fields are populated from these records and retain
    their historical JSONL ordering; additive trace metadata is appended.
    """

    proposal: Proposal
    leaf: torch.Tensor | None = None
    decoded: str | None = None
    score: float | None = None
    cached: bool = False
    invalid: bool = False
    duplicate_exact: bool = False
    duplicate_leaf: bool = False
    acquisition_candidate: bool = False
    acquisition_selected: bool = False
    oracle_evaluated: bool = False
    surrogate_mean: float | None = None
    surrogate_std: float | None = None
    expected_improvement: float | None = None
    acquisition_value: float | None = None


@dataclass
class ProposalBatch:
    base_indices: list[int]
    base_samples: torch.Tensor
    base_rewards: torch.Tensor
    top_reward: float
    proposals: list[Proposal]
    generation_diagnostics: dict[str, Any] = field(default_factory=dict)
    preselected: bool = False


@dataclass
class WeightedArchiveEntry:
    """One unique sequence tracked in the weighted top-k archive."""

    seq: torch.Tensor
    reward: float
    seq_hash: str
    added_iter: int
    last_updated_iter: int


@dataclass
class WeightedArchiveSnapshot:
    """Weights used for archive sampling at a given iteration."""

    iteration: int
    seq_hashes: list[str]
    rewards: torch.Tensor
    weights: torch.Tensor
    temperature: float


@dataclass
class WeightedArchiveDraw:
    """One sampled parent draw from the weighted archive."""

    iteration: int
    archive_index: int
    seq_hash: str
    reward: float
    weight: float


__all__ = [
    "CandidateConfig",
    "CandidateRecord",
    "TrustRegionConfig",
    "Proposal",
    "ProposalBatch",
    "WeightedArchiveEntry",
    "WeightedArchiveSnapshot",
    "WeightedArchiveDraw",
]
