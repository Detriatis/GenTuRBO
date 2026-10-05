"""Small regression gate for the scientific differences in this release."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from search.interpolant.proposal_policy import InterpolantProposalPolicy
from search.interpolant.types import CandidateConfig, Proposal, ProposalBatch, TrustRegionConfig
from sokoban.adalead_official import OfficialAdaLead
from sokoban.optimise import PilotConfig, load_config, make_controller, proposal_parents

ROOT = Path(__file__).resolve().parents[1]


def policy(predictor, vocab=20, *, ehrlich=True):
    args = SimpleNamespace(
        proposal_operator='adalead_official', baseline_fitness_threshold=.05,
        al_ehrlich_black_box_feasibility=ehrlich, al_oracle_batch_size=8,
        baseline_adalead_alphabet='ABCDEFGHIJKLMNOPQRST'[:vocab],
        baseline_adalead_rng_backend='torch_global_seeded',
        candidate_k=1, topk_use_archive=False, topk_archive_size=None,
        topk_weight_n=None, topk_archive_d_min=None, topk_unique=None,
    )
    runtime = SimpleNamespace(device=torch.device('cpu'), vocab_size=vocab, seq_len=256,
                              process=SimpleNamespace(model=SimpleNamespace()))
    return InterpolantProposalPolicy(
        args=args, runtime=runtime,
        candidate=CandidateConfig('random', 'best', 'random', 1, 8, None),
        trust_region=TrustRegionConfig('ScalarTrustRegion', 0, 0),
        active_learning_selector=SimpleNamespace(fit_surrogate_predictor=lambda: predictor))


@pytest.mark.parametrize('seed', [0, 7, 19])
def test_ehrlich_adalead_matches_independent_port_queries_proposals_and_rng(seed):
    torch.manual_seed(100)
    initial = torch.randint(7, (64, 256)); initial[1] = initial[0]
    scores = torch.linspace(0, 1, 64); scores[0] = scores[1] = 1
    calls = []
    def predict(batch):
        calls.append(batch.clone())
        return batch.double().sum(1).numpy(), None
    reference = OfficialAdaLead(SimpleNamespace(
        adalead_mu=1., oracle_batch=8, adalead_surrogate_queries=128,
        adalead_eval_batch_size=20, adalead_fitness_threshold=.05))
    reference.observe(initial, scores)
    torch.manual_seed(seed)
    expected = reference.propose(predict, count=8)
    expected_calls = torch.cat(calls); calls.clear()
    expected_rng = torch.get_rng_state()
    candidate = policy(predict, vocab=7)
    torch.manual_seed(seed)
    actual = candidate.propose(iteration=1, batch=initial, rewards=scores, elite_probs=None)
    assert torch.equal(torch.cat([p.xt for p in actual.proposals]), expected)
    assert torch.equal(torch.cat(calls), expected_calls)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual.preselected
    assert actual.generation_diagnostics['surrogate_query_count'] == 128
    assert actual.generation_diagnostics['selected_predicted_means'] == sorted(
        actual.generation_diagnostics['selected_predicted_means'], reverse=True)


def test_paid_infeasible_ehrlich_rows_are_zero_targets_and_remain_unavailable():
    candidate = policy(None)
    initial = torch.zeros((2, 256), dtype=torch.long)
    candidate._remember_baseline_archive(initial, torch.full((2,), -torch.inf))
    child = torch.ones((1, 256), dtype=torch.long)
    candidate.update_from_evaluations(proposals=[Proposal(0, .9, 1, child, direct_leaf=True)],
                                      proposal_rewards=torch.tensor([-torch.inf]), base_rewards=torch.tensor([0.]))
    candidate._remember_baseline_archive(initial, torch.full((2,), -torch.inf))
    assert [score for _, score in candidate._adalead_measured_rows] == [0, 0, 0]
    assert len(candidate._adalead_paid_keys) == 2
    # Preserve the separate, existing protein protocol.
    protein = policy(None, ehrlich=False)
    protein._remember_baseline_archive(initial, torch.full((2,), -torch.inf))
    assert not protein._adalead_measured_rows


def test_preselected_batch_preserves_order_rng_and_paid_invalid_budget(monkeypatch):
    import search.interpolant.evaluator as ev
    class Selector:
        config = SimpleNamespace(oracle_batch_size=8, oracle_best_budget_mode='physical', sequential_selection=False)
        _acquisition = ''
        def cached_scores(self, batch): return [None] * len(batch)
        def select(self, *args, **kwargs): pytest.fail('second acquisition pass')
        def observe_oracle_batch(self, batch, scores): pass
    class Adapter:
        T = 1
        process = SimpleNamespace(model=SimpleNamespace())
        def rollout_from_batch(self, *args, **kwargs): pytest.fail('generator invoked')
    proposals = ProposalBatch(
        base_indices=[0], base_samples=torch.zeros((1, 3), dtype=torch.long),
        base_rewards=torch.tensor([0.]), top_reward=0., preselected=True,
        generation_diagnostics=dict(operator="adalead_official", selection="top_surrogate_mean", surrogate_query_count=128),
        proposals=[Proposal(0, .5, 1., torch.tensor([[i, 0, 0]]), direct_leaf=True)
                   for i in range(8, 0, -1)])
    monkeypatch.setattr(ev, 'score_batch', lambda batch, reward: torch.full((len(batch),), -torch.inf))
    rng = torch.get_rng_state()
    result = ev.evaluate_proposals(proposals, adapter=Adapter(), reward_fn=object(),
                                   track_trust_region=False, active_learning_selector=Selector())
    assert result.oracle_evaluated_count == result.oracle_budget_debit_count == 8
    assert result.oracle_scored_count == 8 and result.oracle_privileged_scored_count == 0
    assert [p.xt[0, 0].item() for p in result.evaluated_proposals] == list(range(8, 0, -1))
    assert torch.equal(rng, torch.get_rng_state())
    from search.interpolant.engine import _write_proposal_diagnostics_jsonl
    output = io.StringIO()
    _write_proposal_diagnostics_jsonl(output, iteration=1, eval_result=result,
        oracle_budget_total=8, oracle_physical_total=8, oracle_privileged_total=0)
    assert json.loads(output.getvalue())["proposal_generation"] == proposals.generation_diagnostics


def test_sokoban_recipe_settings_and_adaptive_parent_policy():
    configs = {name: PilotConfig(**load_config(ROOT/f'configs/search/sokoban/{name}.json'))
               for name in ('generator', 'genturbo', 'grid_ga', 'adalead_official', 'hamming')}
    assert all(c.budget == 2000 and list(c.seeds) == list(range(20)) for c in configs.values())
    cfg = configs['genturbo']
    assert cfg.oracle_batch == 8 and cfg.gp_buffer == 2000 and not cfg.local_penalization
    assert configs['grid_ga'].ga_editable_region == 'all'
    assert configs['adalead_official'].adalead_surrogate_queries == 128
    assert configs['hamming'].parent_policy == cfg.parent_policy == 'best'
    archive = {str(i): dict(key=str(i), score=s, query=i, canonical_board=[i]*100)
               for i, s in enumerate([2, 8, 4, 8])}
    _, ids, boards = proposal_parents(cfg, archive, torch.Generator().manual_seed(0))
    assert set(ids) == {0} and torch.all(boards == 3)
    ctrl = make_controller(cfg)
    ctrl.update(best_time=None, improved=False)
    assert ctrl.config.mean_edits == 15
    ctrl.update(best_time=None, improved=True)
    assert ctrl.config.mean_edits == 22.5


def test_sokoban_exact_solver_charges_symmetry_cache_and_invalid_rows():
    from sokoban.boards import encode_board
    from sokoban.search import SokobanSwitchReward
    from sokoban.planning import solution_switches
    board = encode_board([
        '##########', '##########', '####. #@##', '###.  $ ##', '##  $#  ##',
        '## $ #####', '### $.####', '#  .  ####', '#        #', '##########'])
    with SokobanSwitchReward(workers=1) as oracle:
        batch = torch.tensor(np.stack([board, np.rot90(board.reshape(10, 10)).flatten(), np.zeros(100)]), dtype=torch.long)
        assert oracle.score_batch(batch).tolist() == [4, 4, -1]
        assert len(oracle.records) == 3 and oracle.records[1]['cache_hit']
        assert oracle.physical_solves == 2
        record = oracle.records[0]
        assert record['objective_status'] == 'exact'
        assert solution_switches(record['canonical_board'], record['solution'])['switches'] == 4
