"""Focused budget regression tests for the reproducibility release."""
from __future__ import annotations

from types import SimpleNamespace
import torch
import search.interpolant.evaluator as ev
from search.interpolant.active_learning import ActiveLearningSelection
from search.interpolant.types import Proposal, ProposalBatch
import pytest
from eval.seq_eval import sequence_edit_distance
from search.interpolant.proposal_policy import InterpolantProposalPolicy
from search.interpolant.types import CandidateConfig, TrustRegionConfig

class _DummyAdapter:

    def __init__(self, mapping: dict[tuple[int, ...], tuple[int, ...]]) -> None:
        self.mapping = {tuple((int(v) for v in key)): tuple((int(v) for v in value)) for key, value in mapping.items()}
        self.T = 1
        self.process = SimpleNamespace(model=SimpleNamespace())
        self.rollout_batches: list[torch.Tensor] = []

    def rollout_from_batch(self, xt_batch: torch.Tensor, t_value: float, denoise_steps: int | None=None) -> list[torch.Tensor]:
        _ = (t_value, denoise_steps)
        self.rollout_batches.append(xt_batch.detach().cpu().clone())
        leaves = []
        for row in xt_batch.detach().cpu():
            key = tuple((int(v) for v in row.tolist()))
            leaves.append(torch.tensor(self.mapping[key], dtype=xt_batch.dtype, device=xt_batch.device))
        return [torch.stack(leaves, dim=0)]

class _DummySelector:

    def __init__(self, *, oracle_batch_size: int, cached_rows: list[tuple[int, ...]], acquisition: str='', oracle_best_budget_mode: str='physical') -> None:
        self.config = SimpleNamespace(oracle_batch_size=oracle_batch_size, oracle_best_budget_mode=oracle_best_budget_mode, sequential_selection=False)
        self._acquisition = acquisition
        self._cached = {tuple((int(v) for v in row)): float(idx + 1) for idx, row in enumerate(cached_rows)}
        self.select_batches: list[torch.Tensor] = []
        self.observed_batches: list[torch.Tensor] = []

    def cached_scores(self, batch: torch.Tensor) -> list[float | None]:
        out: list[float | None] = []
        for row in batch.detach().cpu():
            out.append(self._cached.get(tuple((int(v) for v in row.tolist()))))
        return out

    def select(self, batch: torch.Tensor, *, budget: int | None=None, local_kernel_log_probs: list[float | None] | None=None) -> ActiveLearningSelection:
        _ = local_kernel_log_probs
        self.select_batches.append(batch.detach().cpu().clone())
        n = int(batch.shape[0])
        k = n if budget is None else min(int(budget), n)
        return ActiveLearningSelection(selected_indices=list(range(k)), smiles=[None] * n, surrogate_mean=[None] * n, surrogate_std=[None] * n, expected_improvement=[None] * n, acquisition_value=[float(idx) for idx in range(n)], local_kernel_log_prob=[None] * n, max_expected_improvement=None, num_candidates_above_threshold=n, threshold_exhausted=False)

    def observe_oracle_batch(self, batch: torch.Tensor, scores: torch.Tensor) -> None:
        _ = scores
        self.observed_batches.append(batch.detach().cpu().clone())

class _DummyReward:

    def __init__(self, decode_map: dict[tuple[int, ...], str | None]) -> None:
        self.decode_map = {tuple((int(v) for v in key)): value for key, value in decode_map.items()}

    def decode_batch(self, batch: torch.Tensor) -> list[str | None]:
        out: list[str | None] = []
        for row in batch.detach().cpu():
            out.append(self.decode_map.get(tuple((int(v) for v in row.tolist()))))
        return out

def _make_proposal_batch(rows: list[tuple[int, ...]]) -> ProposalBatch:
    proposals = [Proposal(base_index=0, t_value=0.5, edits_value=1.0, xt=torch.tensor([row], dtype=torch.long)) for row in rows]
    return ProposalBatch(base_indices=[0], base_samples=torch.tensor([[0, 0, 0]], dtype=torch.long), base_rewards=torch.tensor([0.0], dtype=torch.float), top_reward=0.0, proposals=proposals)

def test_direct_leaf_bypasses_rollout_and_preserves_exact_edits(monkeypatch) -> None:
    monkeypatch.setattr(ev, 'score_batch', lambda batch, reward_fn: batch[:, 0].to(dtype=torch.float))
    adapter = _DummyAdapter({})
    proposal_batch = _make_proposal_batch([(1, 2, 0)])
    proposal_batch.proposals[0].direct_leaf = True
    proposal_batch.proposals[0].edits_value = 2.0
    result = ev.evaluate_proposals(proposal_batch, adapter=adapter, reward_fn=object(), track_trust_region=False)
    assert adapter.rollout_batches == []
    assert result.candidate_records[0].leaf.tolist() == [[1, 2, 0]]
    assert result.proposal_traces_all[0].ancestor_to_child_edits == 2
    assert result.proposal_traces_all[0].from_seq.tolist() == [1, 2, 0]
    assert result.proposal_traces_all[0].to_seq.tolist() == [1, 2, 0]

def test_selected_direct_leaf_duplicate_is_counted_once(monkeypatch) -> None:
    monkeypatch.setattr(ev, 'score_batch', lambda batch, reward_fn: batch[:, 0].to(dtype=torch.float))
    adapter = _DummyAdapter({})
    selector = _DummySelector(oracle_batch_size=1, cached_rows=[])
    proposal_batch = _make_proposal_batch([(1, 2, 0), (1, 2, 0)])
    for proposal in proposal_batch.proposals:
        proposal.direct_leaf = True
    result = ev.evaluate_proposals(proposal_batch, adapter=adapter, reward_fn=object(), track_trust_region=False, active_learning_selector=selector)
    assert len(result.evaluated_proposals) == 1
    assert [trace.acquisition_selected for trace in result.proposal_traces_all] == [True, False]
    assert [trace.oracle_evaluated for trace in result.proposal_traces_all] == [True, False]
    assert result.proposal_traces_all[1].duplicate_exact_proposal is True

def test_active_learning_selects_only_fresh_deduplicated_leaves(monkeypatch) -> None:
    monkeypatch.setattr(ev, 'score_batch', lambda batch, reward_fn: batch[:, 0].to(dtype=torch.float))
    adapter = _DummyAdapter({(1, 0, 0): (8, 0, 0), (2, 0, 0): (8, 0, 0), (3, 0, 0): (9, 0, 0), (4, 0, 0): (7, 0, 0)})
    selector = _DummySelector(oracle_batch_size=4, cached_rows=[(9, 0, 0)])
    proposal_batch = _make_proposal_batch([(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)])
    result = ev.evaluate_proposals(proposal_batch, adapter=adapter, reward_fn=object(), track_trust_region=False, active_learning_selector=selector)
    assert len(selector.select_batches) == 1
    assert selector.select_batches[0].tolist() == [[8, 0, 0], [7, 0, 0]]
    assert len(selector.observed_batches) == 1
    assert selector.observed_batches[0].tolist() == [[8, 0, 0], [7, 0, 0]]
    assert result.oracle_evaluated_count == 2
    assert result.oracle_scored_count == 2
    assert [proposal.xt[0].tolist() for proposal in result.evaluated_proposals] == [[1, 0, 0], [4, 0, 0]]
    assert result.evaluated_rewards.tolist() == [8.0, 7.0]
    assert result.change_stats.generated_proposals == 4
    assert result.change_stats.duplicate_exact_proposals == 0
    assert result.change_stats.duplicate_generated_leaves == 1
    assert result.change_stats.cached_generated_proposals == 1
    assert result.change_stats.invalid_generated_proposals == 0
    assert result.change_stats.acquisition_batch_size == 2

def test_active_learning_acquisition_batch_excludes_duplicate_invalid_and_cached_leaves(monkeypatch) -> None:
    monkeypatch.setattr(ev, 'score_batch', lambda batch, reward_fn: batch[:, 0].to(dtype=torch.float))
    adapter = _DummyAdapter({(1, 0, 0): (8, 0, 0), (2, 0, 0): (8, 0, 0), (3, 0, 0): (9, 0, 0), (4, 0, 0): (7, 0, 0), (5, 0, 0): (6, 0, 0)})
    reward = _DummyReward({(8, 0, 0): 'C', (9, 0, 0): None, (7, 0, 0): 'N', (6, 0, 0): 'O'})
    selector = _DummySelector(oracle_batch_size=5, cached_rows=[(7, 0, 0)])
    proposal_batch = _make_proposal_batch([(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0)])
    result = ev.evaluate_proposals(proposal_batch, adapter=adapter, reward_fn=reward, track_trust_region=False, active_learning_selector=selector)
    assert len(selector.select_batches) == 1
    assert selector.select_batches[0].tolist() == [[8, 0, 0], [6, 0, 0]]
    assert result.change_stats.generated_proposals == 5
    assert result.change_stats.duplicate_generated_leaves == 1
    assert result.change_stats.invalid_generated_proposals == 1
    assert result.change_stats.cached_generated_proposals == 1
    assert result.change_stats.acquisition_batch_size == 2

def test_active_learning_marks_all_cached_pool_as_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(ev, 'score_batch', lambda batch, reward_fn: batch[:, 0].to(dtype=torch.float))
    adapter = _DummyAdapter({(1, 0, 0): (9, 0, 0), (2, 0, 0): (9, 0, 0)})
    selector = _DummySelector(oracle_batch_size=2, cached_rows=[(9, 0, 0)])
    proposal_batch = _make_proposal_batch([(1, 0, 0), (2, 0, 0)])
    result = ev.evaluate_proposals(proposal_batch, adapter=adapter, reward_fn=object(), track_trust_region=False, active_learning_selector=selector)
    assert selector.select_batches == []
    assert selector.observed_batches == []
    assert result.oracle_evaluated_count == 0
    assert result.oracle_scored_count == 0
    assert result.active_learning_threshold_exhausted is True
    assert result.evaluated_proposals == []
    assert result.evaluated_rewards.numel() == 0
    assert result.change_stats.generated_proposals == 2
    assert result.change_stats.duplicate_generated_leaves == 1
    assert result.change_stats.cached_generated_proposals == 1
    assert result.change_stats.invalid_generated_proposals == 0
    assert result.change_stats.acquisition_batch_size == 0

def test_oracle_best_counts_and_caps_every_oracle_scored_row(monkeypatch) -> None:
    scored_batches: list[list[list[int]]] = []

    def _record_score(batch: torch.Tensor, reward_fn: object) -> torch.Tensor:
        _ = reward_fn
        scored_batches.append(batch.detach().cpu().tolist())
        return batch[:, 0].to(dtype=torch.float)
    monkeypatch.setattr(ev, 'score_batch', _record_score)
    adapter = _DummyAdapter({(1, 0, 0): (4, 0, 0), (2, 0, 0): (9, 0, 0), (3, 0, 0): (7, 0, 0)})
    selector = _DummySelector(oracle_batch_size=2, cached_rows=[], acquisition='oracle_best')
    result = ev.evaluate_proposals(_make_proposal_batch([(1, 0, 0), (2, 0, 0), (3, 0, 0)]), adapter=adapter, reward_fn=object(), track_trust_region=False, active_learning_selector=selector, active_learning_budget_override=1)
    assert scored_batches == [[[4, 0, 0]]]
    assert result.oracle_evaluated_count == 1
    assert result.oracle_budget_debit_count == 1
    assert result.oracle_scored_count == 1
    assert result.oracle_privileged_scored_count == 0
    assert result.oracle_budget_debit_count + result.oracle_privileged_scored_count == result.oracle_scored_count
    assert result.full_batch_scored_count == 1

def test_oracle_best_selected_archive_budget_scores_full_pool(monkeypatch) -> None:
    scored_batches: list[list[list[int]]] = []

    def _record_score(batch: torch.Tensor, reward_fn: object) -> torch.Tensor:
        _ = reward_fn
        scored_batches.append(batch.detach().cpu().tolist())
        return batch[:, 0].to(dtype=torch.float)
    monkeypatch.setattr(ev, 'score_batch', _record_score)
    adapter = _DummyAdapter({(1, 0, 0): (4, 0, 0), (2, 0, 0): (9, 0, 0), (3, 0, 0): (7, 0, 0)})
    selector = _DummySelector(oracle_batch_size=2, cached_rows=[], acquisition='oracle_best', oracle_best_budget_mode='selected_archive')
    result = ev.evaluate_proposals(_make_proposal_batch([(1, 0, 0), (2, 0, 0), (3, 0, 0)]), adapter=adapter, reward_fn=object(), track_trust_region=False, active_learning_selector=selector, active_learning_budget_override=1)
    assert scored_batches == [[[4, 0, 0], [9, 0, 0], [7, 0, 0]]]
    assert result.oracle_evaluated_count == 1
    assert result.oracle_budget_debit_count == 1
    assert result.oracle_scored_count == 3
    assert result.oracle_privileged_scored_count == 2
    assert result.oracle_budget_debit_count + result.oracle_privileged_scored_count == result.oracle_scored_count
    assert result.evaluated_rewards.tolist() == [9.0]
    assert result.full_batch_scored_count == 3

def test_full_pool_diagnostics_share_the_hard_oracle_budget(monkeypatch) -> None:
    scored_batch_sizes: list[int] = []

    def _record_score(batch: torch.Tensor, reward_fn: object) -> torch.Tensor:
        _ = reward_fn
        scored_batch_sizes.append(int(batch.shape[0]))
        return batch[:, 0].to(dtype=torch.float)
    monkeypatch.setattr(ev, 'score_batch', _record_score)
    adapter = _DummyAdapter({(1, 0, 0): (4, 0, 0), (2, 0, 0): (9, 0, 0), (3, 0, 0): (7, 0, 0)})
    selector = _DummySelector(oracle_batch_size=1, cached_rows=[])
    result = ev.evaluate_proposals(_make_proposal_batch([(1, 0, 0), (2, 0, 0), (3, 0, 0)]), adapter=adapter, reward_fn=object(), track_trust_region=False, active_learning_selector=selector, active_learning_budget_override=2, score_full_proposal_batch=True)
    assert scored_batch_sizes == [1, 1]
    assert result.oracle_evaluated_count == 1
    assert result.oracle_scored_count == 2
    assert result.full_batch_scored_count == 2

class _PadAwareModel:

    def mutable_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x != 0

    def sequence_edit_distance(self, a: torch.Tensor, b: torch.Tensor) -> int:
        return int(sequence_edit_distance(a, b, ignore_token_id=0))

def test_trust_region_times_use_mutable_tokens_for_padded_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = SimpleNamespace(device=torch.device('cpu'), seq_len=4, process=SimpleNamespace(model=_PadAwareModel()), transition_mask=None)
    candidate = CandidateConfig(strategy='trust_region', source='topk', mixing_mode='random', num_candidates=1, mixes_per_time=1, fixed_times=None)
    trust_region = TrustRegionConfig(kind='AdditiveTrustRegion', mean_edits=1.0, eps=0.0, log_every=0, edits_min=0.0, edits_max=4.0, edits_init=1.0)
    args = SimpleNamespace(candidate_k=2, topk_use_archive=False, topk_archive_size=None, topk_weight_n=None, topk_archive_d_min=None, topk_unique=None)
    policy = InterpolantProposalPolicy(args=args, runtime=runtime, candidate=candidate, trust_region=trust_region)
    monkeypatch.setattr(policy, '_sample_interpolant', lambda base_sample, t_tensor, elite_probs: base_sample.clone())
    batch = torch.tensor([[5, 6, 0, 0], [5, 6, 7, 8]], dtype=torch.long)
    rewards = torch.tensor([2.0, 1.0], dtype=torch.float)
    proposal_batch = policy.propose(iteration=0, batch=batch, rewards=rewards, elite_probs=None)
    assert len(proposal_batch.proposals) == 2
    first, second = proposal_batch.proposals
    assert first.base_index == 0
    assert first.edits_value == pytest.approx(1.0)
    assert first.t_value == pytest.approx(0.5)
    assert second.base_index == 1
    assert second.edits_value == pytest.approx(1.0)
    assert second.t_value == pytest.approx(0.75)

def test_scalar_edit_domain_trust_region_uses_separate_expansion_and_contraction() -> None:
    trust_region = TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0, domain='edits', mean_edits=4.0, edits_min=1.0, edits_max=16.0, edits_init=4.0, tau_success=1, tau_failure=1, region_scalar=2.0, region_expand_scalar=3.0, region_shrink_scalar=0.25)
    strategy = InterpolantProposalPolicy(args=SimpleNamespace(candidate_k=1, topk_use_archive=False, topk_archive_size=None, topk_weight_n=None, topk_archive_d_min=None, topk_unique=None), runtime=SimpleNamespace(device=torch.device('cpu'), seq_len=16, process=SimpleNamespace(model=_PadAwareModel()), transition_mask=None), candidate=CandidateConfig(strategy='trust_region', source='best', mixing_mode='random', num_candidates=1, mixes_per_time=1, fixed_times=None), trust_region=trust_region).time_policy.tr_strategy
    assert strategy is not None
    strategy.update(best_time=0.25, best_edits=4.0, improved=True)
    assert strategy.config.mean_edits == pytest.approx(12.0)
    strategy.update(best_time=0.75, best_edits=12.0, improved=False)
    assert strategy.config.mean_edits == pytest.approx(3.0)
