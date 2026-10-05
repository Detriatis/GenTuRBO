"""Backwards-compatible policy exports for interpolant search.

Implementation is split across focused modules:
- candidate_config.py
- proposal_policy.py
- survivor_policy.py
- time_policy.py
- types.py
"""

from search.interpolant.candidate_config import resolve_candidate_config
from search.interpolant.proposal_policy import InterpolantProposalPolicy
from search.interpolant.survivor_policy import BestSurvivorSelectionPolicy
from search.interpolant.types import CandidateConfig, CandidateRecord, Proposal, ProposalBatch, TrustRegionConfig

__all__ = [
    "BestSurvivorSelectionPolicy",
    "CandidateConfig",
    "CandidateRecord",
    "InterpolantProposalPolicy",
    "Proposal",
    "ProposalBatch",
    "TrustRegionConfig",
    "resolve_candidate_config",
]
