"""Interpolant-based search package."""

import argparse

from search.interpolant.args import parse_args
from search.interpolant.config import OutputPolicy, SearchConfig
from search.interpolant.manifest import RunManifest
from search.interpolant.policies import (
    BestSurvivorSelectionPolicy,
    CandidateConfig,
    CandidateRecord,
    InterpolantProposalPolicy,
    Proposal,
    ProposalBatch,
    TrustRegionConfig,
    resolve_candidate_config,
)
from search.interpolant.runtime import SearchRuntime


def run_interpolant_search(args: argparse.Namespace) -> None:
    """Compatibility facade that avoids importing the full engine eagerly."""

    from search.interpolant.engine import run_interpolant_search as _run_interpolant_search

    return _run_interpolant_search(args)

__all__ = [
    "BestSurvivorSelectionPolicy",
    "CandidateConfig",
    "CandidateRecord",
    "InterpolantProposalPolicy",
    "OutputPolicy",
    "Proposal",
    "ProposalBatch",
    "RunManifest",
    "SearchConfig",
    "SearchRuntime",
    "TrustRegionConfig",
    "parse_args",
    "resolve_candidate_config",
    "run_interpolant_search",
]
