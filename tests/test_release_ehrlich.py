"""Focused ehrlich regression tests for the reproducibility release."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
import pytest
import torch
from ehrlich.ehrlich_instance import EhrlichConfig, EhrlichInstance, make_ehrlich_instance
from ehrlich.ehrlich_reward import EhrlichReward
from ehrlich.ehrlich_score import ehrlich_score_batch, is_feasible
from search.interpolant.args import parse_args
from search.interpolant.engine import run_interpolant_search
from search.interpolant.output import read_jsonl
from search.interpolant.proposal_policy import InterpolantProposalPolicy
from search.interpolant.types import Proposal, ProposalBatch
_TRACE_FIELD_ORDER = ['batch_id', 'seq_idx', 'seq', 'seq_length', 'score', 'score_is_finite', 'score_for_analysis', 'decoded_sequence', 'sequence_string', 'oracle_evaluated', 'acquisition_candidate', 'acquisition_selected', 'duplicate_exact_proposal', 'duplicate_generated_leaf', 'cached_generated_proposal', 'invalid_generated_proposal', 'oracle_rejected', 't_value', 'expected_edits', 'smiles', 'surrogate_mean', 'surrogate_std', 'expected_improvement', 'acquisition_value', 'local_kernel_log_prob', 'local_kernel_log_prob_sequence', 'diagnostic_score', 'diagnostic_score_evaluated', 'full_batch_rank_by_score', 'diagnostic_injected_global_best', 'diagnostic_injected_source', 'diagnostic_injected_full_dataset_score', 'region_id', 'proposal_corrupted_tokens', 'model_changes_total', 'model_changes_on_corrupted', 'model_changes_on_uncorrupted', 'proposal_logical_edits', 'model_logical_edits', 'retained_proposal_logical_edits', 'dropped_proposal_logical_edits', 'introduced_logical_edits', 'retained_proposal_logical_frac', 'ancestor_to_child_edits', 'ancestor_seq', 'from_seq', 'to_seq', 'parent_seq_hash', 'parent_score', 'parent_archive_index', 'parent_archive_probability', 'parent_selection_log_prob', 'archive_size', 'archive_snapshot_id', 'hamming_conditional_log_prob', 'proposal_log_prob', 'proposal_probability_method']

def _write_test_assets(root: Path) -> tuple[Path, Path, EhrlichInstance]:
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / 'uniform.pt'
    instance_path = root / 'instance.pt'
    runtime_config = {'process': 'flow_mask_ctmc', 'process_cfg': {'vocab_size': 2, 'seq_len': 6, 'mask_id': None, 'interpolant': 'uniform', 'sample_time_schedule': 'linear'}, 'model': 'uniform_categorical_denoiser', 'model_cfg': {'vocab_size': 2, 'seq_len': 6, 'mask_id': None, 'device': 'cpu'}, 'device': 'cpu'}
    torch.save({'metadata': {'config': runtime_config, 'data_shape': [6]}}, checkpoint_path)
    instance = make_ehrlich_instance(EhrlichConfig(vocab_size=2, seq_len=6, motif_count=1, motif_len=2, quantization=1, bandwidth=1, temperature=0.5), seed=7)
    torch.save(instance, instance_path)
    return (checkpoint_path, instance_path, instance)

def _run_search(root: Path, *, feature_space: str='ehrlich_onehot', initial_oracle_evaluations: int=3, max_oracle_evaluations: int=4, num_iterations: int=2, config_overrides: dict[str, Any] | None=None, explicit_run_manifest: bool=True) -> dict[str, Any]:
    checkpoint_path, instance_path, instance = _write_test_assets(root)
    output_paths = {'instance': instance_path, 'initial': root / 'initial.jsonl.gz', 'samples': root / 'samples.jsonl.gz', 'traces': root / 'traces.jsonl.gz', 'diagnostics': root / 'diagnostics.jsonl.gz', 'manifest': root / ('manifest.json' if explicit_run_manifest else 'run_manifest.json')}
    search_config = {'runner': 'interpolant', 'checkpoint': str(checkpoint_path), 'device': 'cpu', 'reward': {'name': 'ehrlich', 'instance_path': str(instance_path)}, 'batch_size': 3, 'initial_oracle_evaluations': initial_oracle_evaluations, 'num_iterations': num_iterations, 'flow_steps': 2, 'num_times': 2, 'mixes_per_time': 3, 'candidate_strategy': 'random', 'candidate_source': 'best', 'active_learning': True, 'al_feature_space': feature_space, 'al_oracle_batch_size': 2, 'al_buffer_size': 8, 'al_acquisition': 'expected_improvement', 'al_gp_noise_variance': 0.001, 'max_oracle_evaluations': max_oracle_evaluations, 'seed': 123, 'trace_level': 'full', 'trace_record_limit': 100, 'initial_batch_out': str(output_paths['initial']), 'samples_out': str(output_paths['samples']), 'proposal_traces_out': str(output_paths['traces']), 'proposal_diagnostics_out': str(output_paths['diagnostics'])}
    if explicit_run_manifest:
        search_config['run_manifest_out'] = str(output_paths['manifest'])
    if config_overrides:
        search_config.update(config_overrides)
    config_path = root / 'search.json'
    config_path.write_text(json.dumps(search_config, indent=2) + '\n')
    args = parse_args(['--search-config', str(config_path)])
    run_interpolant_search(args)
    return {'instance': instance, 'config_path': config_path, 'paths': output_paths, 'initial': read_jsonl(output_paths['initial']), 'samples': read_jsonl(output_paths['samples']), 'traces': read_jsonl(output_paths['traces']), 'diagnostics': read_jsonl(output_paths['diagnostics']), 'manifest': json.loads(output_paths['manifest'].read_text())}

def _assert_oracle_record_matches_ehrlich(record: dict[str, Any], instance: EhrlichInstance) -> None:
    sequence = torch.as_tensor(record['seq'], dtype=torch.long)
    feasible = bool(is_feasible(sequence, instance.transition_mask))
    assert record['score_is_finite'] is feasible
    expected_score = float(ehrlich_score_batch(sequence, instance).item())
    if feasible:
        assert record['score'] == pytest.approx(expected_score)
    else:
        assert record['score'] is None
        assert math.isinf(expected_score) and expected_score < 0.0

def test_ehrlich_real_loop_is_deterministic_and_honours_hard_oracle_budget(tmp_path: Path) -> None:
    first = _run_search(tmp_path / 'first')
    second = _run_search(tmp_path / 'second')
    diffusion = _run_search(tmp_path / 'diffusion', feature_space='ehrlich_diffusion')
    initial = first['initial']
    samples = first['samples']
    traces = first['traces']
    diagnostics = first['diagnostics']
    manifest = first['manifest']
    instance = first['instance']
    assert len(initial) == 3
    assert [record['batch_id'] for record in initial] == [-1, -1, -1]
    assert [record['seq_idx'] for record in initial] == [0, 1, 2]
    assert all((record['oracle_evaluated'] for record in initial))
    assert len(diagnostics) == 1
    assert diagnostics[0]['batch_id'] == 0
    assert diagnostics[0]['oracle_evaluated_count'] == 1
    assert diagnostics[0]['oracle_budget_debit_count'] == 1
    assert diagnostics[0]['oracle_scored_count'] == 1
    assert diagnostics[0]['oracle_privileged_scored_count'] == 0
    assert diagnostics[0]['oracle_budget_total_after_iteration'] == 4
    assert diagnostics[0]['oracle_physical_total_after_iteration'] == 4
    assert diagnostics[0]['oracle_privileged_total_after_iteration'] == 0
    assert diagnostics[0]['oracle_total_after_iteration'] == 4
    assert len(samples) == 1
    assert samples[0]['batch_id'] == 0
    assert samples[0]['oracle_evaluated'] is True
    assert samples[0]['acquisition_selected'] is True
    assert len(traces) == 6
    assert [record['batch_id'] for record in traces] == [0] * 6
    assert [record['seq_idx'] for record in traces] == list(range(6))
    assert list(traces[0]) == _TRACE_FIELD_ORDER
    stats = diagnostics[0]
    assert stats['generated_proposals'] - stats['duplicate_generated_leaves'] - stats['invalid_generated_proposals'] - stats['cached_generated_proposals'] == stats['acquisition_batch_size']
    for record in initial + samples:
        _assert_oracle_record_matches_ehrlich(record, instance)
    for record in traces:
        sequence = torch.as_tensor(record['seq'], dtype=torch.long)
        feasible = bool(is_feasible(sequence, instance.transition_mask))
        assert record['invalid_generated_proposal'] is (not feasible)
        if record['oracle_evaluated']:
            _assert_oracle_record_matches_ehrlich(record, instance)
    assert manifest['status'] == 'completed'
    assert manifest['seed'] == 123
    assert manifest['resolved_config']['max_oracle_evaluations'] == 4
    assert manifest['resolved_config']['al_oracle_best_budget_mode'] == 'physical'
    assert set(manifest['input_hashes']) >= {'checkpoint', 'reward_instance', 'search_config'}
    expected_instance_hash = hashlib.sha256(first['paths']['instance'].read_bytes()).hexdigest()
    assert manifest['input_hashes']['reward_instance'] == expected_instance_hash
    assert manifest['trace_records_written'] == 6
    assert manifest['trace_truncated'] is False
    assert set(manifest['output_hashes']) >= {str(first['paths'][name]) for name in ('initial', 'samples', 'traces', 'diagnostics')}
    for name in ('initial', 'samples', 'traces', 'diagnostics'):
        assert first['paths'][name].read_bytes() == second['paths'][name].read_bytes()
    assert diffusion['diagnostics'][0]['oracle_total_after_iteration'] == 4
    assert diffusion['manifest']['resolved_config']['al_feature_space'] == 'ehrlich_diffusion'
    assert any((record['surrogate_mean'] is not None for record in diffusion['traces']))

def test_remaining_oracle_budget_caps_but_does_not_enlarge_al_batch(tmp_path: Path) -> None:
    run = _run_search(tmp_path / 'bounded_batches', initial_oracle_evaluations=2, max_oracle_evaluations=6, num_iterations=2)
    assert [row['oracle_evaluated_count'] for row in run['diagnostics']] == [2, 2]
    assert [row['oracle_total_after_iteration'] for row in run['diagnostics']] == [4, 6]
    assert len(run['samples']) == 4

def test_exact_budget_schedule_replaces_a_fully_collapsed_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_propose = InterpolantProposalPolicy.propose

    def collapse_first_batch(self: InterpolantProposalPolicy, *, iteration: int, batch: torch.Tensor, rewards: torch.Tensor, elite_probs: torch.Tensor | None, max_proposals: int | None=None) -> ProposalBatch:
        if iteration:
            return original_propose(self, iteration=iteration, batch=batch, rewards=rewards, elite_probs=elite_probs, max_proposals=max_proposals)
        proposals = [Proposal(base_index=index, t_value=0.0, edits_value=0.0, xt=batch[index:index + 1].clone(), direct_leaf=True) for index in range(len(batch))]
        return ProposalBatch(base_indices=list(range(len(batch))), base_samples=batch, base_rewards=rewards, top_reward=float(rewards.max().item()), proposals=proposals)
    monkeypatch.setattr(InterpolantProposalPolicy, 'propose', collapse_first_batch)
    run = _run_search(tmp_path / 'collapsed_batch', initial_oracle_evaluations=2, max_oracle_evaluations=6, num_iterations=2)
    assert [row['oracle_evaluated_count'] for row in run['diagnostics']] == [0, 2, 2]
    assert [row['oracle_total_after_iteration'] for row in run['diagnostics']] == [2, 4, 6]
    assert len(run['samples']) == 4

def test_oracle_best_can_budget_selected_archive_while_counting_physical_scores(tmp_path: Path) -> None:
    run = _run_search(tmp_path / 'oracle_selected_archive_budget', initial_oracle_evaluations=3, max_oracle_evaluations=5, num_iterations=2, config_overrides={'al_acquisition': 'oracle_best', 'al_oracle_best_budget_mode': 'selected_archive'})
    assert len(run['diagnostics']) == 1
    row = run['diagnostics'][0]
    assert row['oracle_evaluated_count'] == 2
    assert row['oracle_budget_debit_count'] == 2
    assert row['oracle_scored_count'] > row['oracle_budget_debit_count']
    assert row['oracle_privileged_scored_count'] == row['oracle_scored_count'] - row['oracle_budget_debit_count']
    assert row['oracle_budget_total_after_iteration'] == 5
    assert row['oracle_total_after_iteration'] == 5
    assert row['oracle_physical_total_after_iteration'] == 3 + row['oracle_scored_count']
    assert row['oracle_privileged_total_after_iteration'] == row['oracle_privileged_scored_count']
    assert len(run['samples']) == 2
    assert run['manifest']['resolved_config']['al_oracle_best_budget_mode'] == 'selected_archive'

def test_full_batch_diagnostics_are_debited_before_threshold_resampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scored_rows: list[int] = []
    original_score_batch = EhrlichReward.score_batch

    def _counted_score_batch(self: EhrlichReward, batch: torch.Tensor) -> torch.Tensor:
        scored_rows.append(int(batch.shape[0]) if batch.ndim == 2 else 1)
        return original_score_batch(self, batch)
    monkeypatch.setattr(EhrlichReward, 'score_batch', _counted_score_batch)
    run = _run_search(tmp_path / 'threshold_resample', initial_oracle_evaluations=3, max_oracle_evaluations=4, num_iterations=1, config_overrides={'al_ei_threshold': 1e+30, 'al_max_resample_batches': 2, 'al_score_full_proposal_batch': True})
    assert scored_rows == [3, 1]
    diagnostics = run['diagnostics']
    assert len(diagnostics) == 3
    assert [row['proposal_batch_discarded'] for row in diagnostics] == [True, True, False]
    assert [row['resample_count'] for row in diagnostics] == [1, 2, 2]
    assert [row['oracle_evaluated_count'] for row in diagnostics] == [0, 0, 0]
    assert [row['oracle_scored_count'] for row in diagnostics] == [1, 0, 0]
    assert sum((row['oracle_scored_count'] for row in diagnostics)) == 1
    assert [row['oracle_total_after_iteration'] for row in diagnostics] == [4, 4, 4]
    assert run['samples'] == []
    assert not any((record['diagnostic_score_evaluated'] for record in run['traces']))

def test_second_identical_run_is_rejected_without_touching_first_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / 'isolated_run'
    first = _run_search(run_root, num_iterations=1, explicit_run_manifest=False)
    protected_names = ('initial', 'samples', 'traces', 'diagnostics', 'manifest')
    first_bytes = {name: first['paths'][name].read_bytes() for name in protected_names}
    with pytest.raises(FileExistsError, match='fresh samples_out path'):
        _run_search(run_root, num_iterations=1, explicit_run_manifest=False)
    assert {name: first['paths'][name].read_bytes() for name in protected_names} == first_bytes
    assert not (run_root / 'run_manifest_1.json').exists()
