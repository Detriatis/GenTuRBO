"""Focused proposals regression tests for the reproducibility release."""
from __future__ import annotations

import math
from types import SimpleNamespace
import pytest
import torch
from search.interpolant.proposal_policy import InterpolantProposalPolicy
from search.interpolant.types import CandidateConfig, TrustRegionConfig

def _args(operator: str) -> SimpleNamespace:
    return SimpleNamespace(proposal_operator=operator, baseline_population_size=2, baseline_generations=4, baseline_fitness_threshold=0.25, baseline_mutation_edits=1, baseline_parent_temperature=10.0, baseline_ucb_beta=0.2, baseline_children_per_selection=8, al_oracle_batch_size=8, candidate_k=1, topk_use_archive=False, topk_archive_size=None, topk_weight_n=None, topk_archive_d_min=None, topk_unique=None)

def _official_adalead_args(*, surrogate_queries: int=128, paid_batch_size: int=8, eval_batch_size: int=20) -> SimpleNamespace:
    args = _args('adalead_official')
    args.al_oracle_batch_size = paid_batch_size
    args.baseline_adalead_surrogate_queries = surrogate_queries
    args.baseline_adalead_paid_batch_size = paid_batch_size
    args.baseline_adalead_mu = 1.0
    args.baseline_adalead_eval_batch_size = eval_batch_size
    args.baseline_adalead_rho = 0
    args.baseline_adalead_recombination_rate = 0.0
    args.baseline_adalead_alphabet = 'ABCD'
    args.baseline_adalead_rng_backend = 'torch_global_seeded'
    args.baseline_fitness_threshold = 0.05
    return args

def _runtime(*, transition_mask: torch.Tensor | None=None) -> SimpleNamespace:
    return SimpleNamespace(device=torch.device('cpu'), seq_len=6, vocab_size=4, process=SimpleNamespace(model=SimpleNamespace()), transition_mask=transition_mask)

def test_hamming_operator_makes_exact_edits_without_hidden_feasibility_mask() -> None:
    transition = torch.eye(4, dtype=torch.bool)
    policy = InterpolantProposalPolicy(args=_args('hamming'), runtime=_runtime(transition_mask=transition), candidate=CandidateConfig('trust_region', 'best', 'random', 1, 8, None), trust_region=TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0, domain='edits', mean_edits=3.0, edits_min=1.0, edits_max=6.0, edits_init=3.0))
    parent = torch.zeros((1, 6), dtype=torch.long)
    proposals = policy.propose(iteration=0, batch=parent, rewards=torch.tensor([1.0]), elite_probs=None).proposals
    assert len(proposals) == 8
    for proposal in proposals:
        child = proposal.xt[0]
        assert proposal.direct_leaf is True
        assert int((child != parent[0]).sum().item()) == 3
        assert proposal.edits_value == 3.0
        assert not bool(transition[child[:-1], child[1:]].all().item())

def test_score_weighted_archive_hamming_proposals_record_exact_probabilities() -> None:
    args = _args('hamming')
    args.topk_use_archive = True
    args.topk_archive_size = 8
    args.topk_weight_n = 2.0
    policy = InterpolantProposalPolicy(args=args, runtime=_runtime(), candidate=CandidateConfig('random', 'topk', 'random', 1, 2, None), trust_region=TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0))
    batch = torch.tensor([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long)
    rewards = torch.tensor([0.0, 1.0], dtype=torch.float)
    torch.manual_seed(4)
    proposals = policy.propose(iteration=3, batch=batch, rewards=rewards, elite_probs=None).proposals
    assert len(proposals) == 2
    assert policy.last_archive_snapshot is not None
    assert len(policy.last_archive_draws) == 2
    expected_hamming_log_prob = -math.log(6 * 3)
    snapshot_ids = {proposal.archive_snapshot_id for proposal in proposals}
    assert len(snapshot_ids) == 1
    assert next(iter(snapshot_ids)).startswith('archive-3-')
    for proposal, draw in zip(proposals, policy.last_archive_draws, strict=True):
        assert proposal.parent_seq_hash == draw.seq_hash
        assert proposal.parent_score == pytest.approx(draw.reward)
        assert proposal.parent_archive_index == draw.archive_index
        assert proposal.parent_archive_probability == pytest.approx(draw.weight)
        assert proposal.parent_selection_log_prob == pytest.approx(math.log(draw.weight))
        assert proposal.archive_size == 2
        assert proposal.hamming_conditional_log_prob == pytest.approx(expected_hamming_log_prob)
        assert proposal.proposal_log_prob == pytest.approx(math.log(draw.weight) + expected_hamming_log_prob)
        assert proposal.proposal_probability_method == 'archive_softmax_x_exact_hamming'

def test_generator_exact_corruptions_conditions_interpolant_draw() -> None:
    args = _args('generator')
    args.generator_exact_corruptions = 1
    runtime = _runtime()
    draws = iter([torch.zeros((1, 6), dtype=torch.long), torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.long), torch.tensor([[0, 0, 2, 0, 0, 0]], dtype=torch.long)])
    draw_count = 0

    def sample_interpolant(**_kwargs) -> torch.Tensor:
        nonlocal draw_count
        draw_count += 1
        return next(draws)
    runtime.process.model.sample_interpolant = sample_interpolant
    policy = InterpolantProposalPolicy(args=args, runtime=runtime, candidate=CandidateConfig('uniform', 'best', 'random', 1, 1, [0.5]), trust_region=TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0))
    parent = torch.zeros((1, 6), dtype=torch.long)
    [proposal] = policy.propose(iteration=0, batch=parent, rewards=torch.tensor([1.0]), elite_probs=None).proposals
    assert draw_count == 3
    assert proposal.direct_leaf is False
    assert proposal.edits_value == 1.0
    assert proposal.xt.tolist() == [[0, 0, 2, 0, 0, 0]]
    assert proposal.corrupt_mask is not None
    assert proposal.corrupt_mask.tolist() == [False, False, True, False, False, False]

class _MomentSelector:

    def __init__(self, *, std: float=0.0) -> None:
        self.std = std
        self.fit_calls = 0
        self.batch_sizes: list[int] = []

    def fit_surrogate_moments_predictor(self):
        self.fit_calls += 1

        def predict(batch: torch.Tensor):
            self.batch_sizes.append(len(batch))
            means = batch.float().sum(dim=1)
            return (means, torch.full_like(means, self.std))
        return predict

class _OfficialAdaLeadSelector:

    def __init__(self, score_by_code: dict[int, float] | None=None) -> None:
        self.score_by_code = score_by_code
        self.fit_calls = 0
        self.prediction_batches: list[torch.Tensor] = []

    @staticmethod
    def code(batch: torch.Tensor) -> torch.Tensor:
        powers = 4 ** torch.arange(batch.shape[1], dtype=torch.long)
        return (batch.detach().cpu().long() * powers).sum(dim=1)

    def fit_surrogate_predictor(self):
        self.fit_calls += 1

        def predict(batch: torch.Tensor):
            self.prediction_batches.append(batch.detach().cpu().clone())
            codes = self.code(batch)
            if self.score_by_code is None:
                means = codes.double()
            else:
                means = torch.as_tensor([self.score_by_code.get(int(code), 0.0) for code in codes], dtype=torch.double)
            return (means, -means)
        return predict

def _base4_sequence(value: int, *, length: int=6) -> torch.Tensor:
    digits: list[int] = []
    for _ in range(length):
        digits.append(value % 4)
        value //= 4
    return torch.tensor([digits], dtype=torch.long)

def test_official_adalead_mutation_is_independent_mu_over_l_with_full_alphabet(monkeypatch) -> None:
    runtime = _runtime(transition_mask=torch.eye(4, dtype=torch.bool))
    runtime.process.model.mutable_mask = lambda sequence: torch.zeros_like(sequence, dtype=torch.bool)
    policy = InterpolantProposalPolicy(args=_official_adalead_args(), runtime=runtime, candidate=CandidateConfig('random', 'best', 'random', 1, 8, None), trust_region=TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0), active_learning_selector=_OfficialAdaLeadSelector())
    draws = torch.tensor([0.1, 0.9, 0.1, 0.9, 0.9, 0.9])
    replacement_sizes: list[tuple[int, ...]] = []
    monkeypatch.setattr(torch, 'rand', lambda length, *, device: draws[:length].to(device))

    def replacements(low, high, size, *, device):
        assert (low, high) == (0, 4)
        replacement_sizes.append(size)
        return torch.tensor([0, 3], device=device)
    monkeypatch.setattr(torch, 'randint', replacements)
    parent = torch.zeros((1, 6), dtype=torch.long)
    child = policy._mutate_adalead_official(parent)
    assert replacement_sizes == [(2,)]
    assert torch.equal(child, torch.tensor([[0, 0, 3, 0, 0, 0]]))

def test_official_adalead_uses_exact_128_queries_and_returns_top_eight_means(monkeypatch) -> None:
    selector = _OfficialAdaLeadSelector()
    policy = InterpolantProposalPolicy(args=_official_adalead_args(), runtime=_runtime(), candidate=CandidateConfig('random', 'best', 'random', 1, 8, None), trust_region=TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0), active_learning_selector=selector)
    next_code = 1

    def unique_mutant(_parent: torch.Tensor) -> torch.Tensor:
        nonlocal next_code
        child = _base4_sequence(next_code)
        next_code += 1
        return child
    monkeypatch.setattr(policy, '_mutate_adalead_official', unique_mutant)
    proposal_batch = policy.propose(iteration=0, batch=torch.zeros((1, 6), dtype=torch.long), rewards=torch.tensor([1.0]), elite_probs=None)
    assert selector.fit_calls == 1
    assert sum((len(batch) for batch in selector.prediction_batches)) == 128
    assert [len(batch) for batch in selector.prediction_batches] == [8] * 16
    assert len(proposal_batch.proposals) == 8
    assert proposal_batch.generation_diagnostics == {'operator': 'adalead_official', 'surrogate_query_count': 128, 'surrogate_root_query_count': 24, 'surrogate_mutant_query_count': 104, 'paid_target_count': 8, 'unique_generated_count': 104}
    assert all((proposal.direct_leaf for proposal in proposal_batch.proposals))
    assert [int(code) for code in selector.code(torch.cat([proposal.xt for proposal in proposal_batch.proposals], dim=0))] == list(range(104, 96, -1))

def test_official_adalead_preserves_threshold_resize_and_upstream_node_order(monkeypatch) -> None:
    selector = _OfficialAdaLeadSelector(score_by_code={0: 100.0, 1: 100.0, 2: 100.0})
    policy = InterpolantProposalPolicy(args=_official_adalead_args(surrogate_queries=32, paid_batch_size=4, eval_batch_size=20), runtime=_runtime(), candidate=CandidateConfig('random', 'best', 'random', 1, 4, None), trust_region=TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0), active_learning_selector=selector)
    mutation_parent_codes: list[int] = []
    next_code = 10

    def record_parent(parent: torch.Tensor) -> torch.Tensor:
        nonlocal next_code
        mutation_parent_codes.append(int(selector.code(parent)[0]))
        child = _base4_sequence(next_code)
        next_code += 1
        return child
    monkeypatch.setattr(policy, '_mutate_adalead_official', record_parent)
    archive = torch.cat([_base4_sequence(0), _base4_sequence(0), _base4_sequence(1), _base4_sequence(2)], dim=0)
    policy.propose(iteration=0, batch=archive, rewards=torch.tensor([100.0, 100.0, 90.0, 96.0]), elite_probs=None)
    assert selector.code(selector.prediction_batches[0]).tolist() == [0, 0, 2, 0]
    assert mutation_parent_codes[:4] == [0, 0, 0, 2]
    assert [int(selector.code(row)[0]) for row, _ in policy._adalead_measured_rows] == [0, 0, 1, 2]
    later = torch.cat([_base4_sequence(0), _base4_sequence(3)], dim=0)
    policy._remember_baseline_archive(later, torch.tensor([100.0, 101.0]))
    policy._remember_baseline_archive(later, torch.tensor([100.0, 101.0]))
    assert [int(selector.code(row)[0]) for row, _ in policy._adalead_measured_rows] == [0, 0, 1, 2, 3]

def test_bo_evo_screens_m_candidates_and_returns_k_direct_leaves(monkeypatch) -> None:
    torch.manual_seed(0)
    selector = _MomentSelector()
    policy = InterpolantProposalPolicy(args=_args('bo_evo'), runtime=_runtime(), candidate=CandidateConfig('random', 'best', 'random', 1, 64, None), trust_region=TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0), active_learning_selector=selector)
    assert not hasattr(policy, '_crossover')
    archive = torch.tensor([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=torch.long)
    proposal_batch = policy.propose(iteration=0, batch=archive, rewards=torch.tensor([0.0, 1.0]), elite_probs=None)
    assert len(proposal_batch.proposals) == 8
    assert all((proposal.direct_leaf for proposal in proposal_batch.proposals))
    assert selector.fit_calls == 1
    assert selector.batch_sizes == [8] * 8
    assert proposal_batch.generation_diagnostics == {'operator': 'bo_evo', 'surrogate_query_count': 64, 'paid_target_count': 8, 'children_per_selection': 8, 'ucb_beta': 0.2, 'parent_temperature': 10.0, 'crossover_rate': 0.0, 'mutation_operator': 'independent_1_over_l_conditioned_nonempty'}

def test_bo_evo_selects_each_leaf_by_mean_plus_beta_std(monkeypatch) -> None:
    args = _args('bo_evo')
    args.al_oracle_batch_size = 1
    args.baseline_children_per_selection = 2
    selector = _MomentSelector()
    policy = InterpolantProposalPolicy(args=args, runtime=_runtime(), candidate=CandidateConfig('random', 'best', 'random', 1, 2, None), trust_region=TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0), active_learning_selector=selector)
    children = [torch.tensor([[1, 0, 0, 0, 0, 0]]), torch.tensor([[0, 1, 0, 0, 0, 0]])]
    monkeypatch.setattr(policy, '_mutate_bo_evo', lambda _parent: children.pop(0))

    def moments_predictor():

        def predict(batch: torch.Tensor):
            if len(batch) == 1:
                return (torch.tensor([0.0]), torch.tensor([0.0]))
            return (torch.tensor([1.0, 0.9]), torch.tensor([0.0, 1.0]))
        return predict
    monkeypatch.setattr(selector, 'fit_surrogate_moments_predictor', moments_predictor)
    initial = torch.zeros((1, 6), dtype=torch.long)
    proposal = policy.propose(iteration=0, batch=initial, rewards=torch.tensor([0.0]), elite_probs=None).proposals[0]
    assert torch.equal(proposal.xt, torch.tensor([[0, 1, 0, 0, 0, 0]]))

def test_bo_evo_parent_pool_becomes_exact_previous_paid_batch() -> None:
    torch.manual_seed(0)
    policy = InterpolantProposalPolicy(args=_args('bo_evo'), runtime=_runtime(), candidate=CandidateConfig('random', 'best', 'random', 1, 64, None), trust_region=TrustRegionConfig(kind='ScalarTrustRegion', eps=0.0, log_every=0), active_learning_selector=_MomentSelector())
    initial = torch.tensor([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=torch.long)
    generated = policy.propose(iteration=0, batch=initial, rewards=torch.tensor([0.0, 1.0]), elite_probs=None).proposals
    policy.update_from_evaluations(proposals=generated[:2], proposal_rewards=torch.tensor([float('-inf'), 8.0]), base_rewards=torch.tensor([0.0, 1.0]))
    assert policy._bo_evo_parents is not None
    assert [score for _, score in policy._bo_evo_parents] == [0.0, 8.0]
    assert [policy._sequence_key(row) for row, _ in policy._bo_evo_parents] == [policy._sequence_key(proposal.xt) for proposal in generated[:2]]
