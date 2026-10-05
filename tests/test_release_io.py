"""Focused io regression tests for the reproducibility release."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from search.interpolant.args import parse_args
from search.interpolant.config import OutputPolicy, SearchConfig
from search.interpolant.engine import _initialize_search_state, _load_initial_batch_jsonl, _validate_output_paths
from search.interpolant.evaluator import evaluate_proposals
from search.interpolant.output import iter_jsonl, open_jsonl
from search.interpolant.survivor_policy import BestSurvivorSelectionPolicy
from search.interpolant.types import Proposal, ProposalBatch
from genturbo.configuration import load_config

def _batch(*, mask: torch.Tensor | None=None) -> ProposalBatch:
    return ProposalBatch(base_indices=[0], base_samples=torch.tensor([[0, 0, 0]], dtype=torch.long), base_rewards=torch.tensor([0.0]), top_reward=0.0, proposals=[Proposal(base_index=0, t_value=0.5, edits_value=1.0, xt=torch.tensor([[1, 0, 0]], dtype=torch.long), corrupt_mask=mask)])

class _NoMaskAdapter:
    T = 1
    device = torch.device('cpu')
    process = SimpleNamespace(model=SimpleNamespace())

    def rollout_from_batch(self, batch: torch.Tensor, t: float, denoise_steps: int | None=None):
        _ = (t, denoise_steps)
        return [batch]

class _Reward:

    def __call__(self, row: torch.Tensor) -> float:
        return float(row[0].item())

def test_cli_value_equal_to_default_still_wins_over_config() -> None:
    args = parse_args(['--elite-prior-alpha', '0.1'])
    config = SearchConfig.from_legacy_flat({'runner': 'interpolant', 'elite_prior_alpha': 0.9}, operational_keys=args._defaults)
    config.apply_to_namespace(args)
    assert args.elite_prior_alpha == 0.1

def test_legacy_config_separates_metadata_and_rejects_sorted_unknowns() -> None:
    args = parse_args([])
    config = SearchConfig.from_legacy_flat({'runner': 'interpolant', 'plot_label': 'control', 'diagnostic': 'initial_swap', 'benchmark_task': 'gfp_hard'}, operational_keys=args._defaults)
    assert config.benchmark_task == 'gfp_hard'
    assert dict(config.metadata) == {'runner': 'interpolant', 'plot_label': 'control', 'diagnostic': 'initial_swap'}
    with pytest.raises(ValueError, match='Unknown search config keys: aaa, zzz'):
        SearchConfig.from_legacy_flat({'zzz': 1, 'aaa': 2}, operational_keys=args._defaults)

def test_gzip_jsonl_bytes_are_deterministic_and_plain_name_falls_back(tmp_path: Path) -> None:
    compressed = tmp_path / 'records.jsonl.gz'
    with open_jsonl(compressed, 'wt') as stream:
        stream.write('{"index":0}\n')
    first = compressed.read_bytes()
    with open_jsonl(compressed, 'wt') as stream:
        stream.write('{"index":0}\n')
    assert compressed.read_bytes() == first
    assert int.from_bytes(first[4:8], 'little') == 0
    assert list(iter_jsonl(tmp_path / 'records.jsonl')) == [{'index': 0}]

def test_initial_archive_reconstructs_explicit_infeasible_scores_without_rescoring(tmp_path: Path) -> None:
    archive = tmp_path / 'initial_batch.jsonl.gz'
    with open_jsonl(archive, 'wt') as stream:
        stream.write(json.dumps({'seq': [0, 1, 2], 'score': 1.25, 'score_is_finite': True}) + '\n')
        stream.write(json.dumps({'seq': [2, 1, 0], 'score': None, 'score_is_finite': False}) + '\n')
    batch, scores = _load_initial_batch_jsonl(archive, device=torch.device('cpu'), seq_len=3)
    assert batch.tolist() == [[0, 1, 2], [2, 1, 0]]
    assert scores is not None
    assert scores.tolist()[0] == pytest.approx(1.25)
    assert scores.tolist()[1] == float('-inf')

def test_loaded_explicit_infeasible_archive_advances_rng_without_oracle_rescore(tmp_path: Path) -> None:
    archive = tmp_path / 'initial_batch.jsonl.gz'
    with open_jsonl(archive, 'wt') as stream:
        stream.write(json.dumps({'seq': [2, 1, 0], 'score': None, 'score_is_finite': False}) + '\n')

    class _Process:

        def __init__(self) -> None:
            self.calls: list[tuple[int, int, torch.device]] = []

        def sample(self, *, num_samples: int, steps: int, device: torch.device) -> torch.Tensor:
            self.calls.append((num_samples, steps, device))
            return torch.zeros((num_samples, 3), dtype=torch.long, device=device)

    def _unexpected_oracle(_batch: torch.Tensor) -> torch.Tensor:
        raise AssertionError('stored explicit infeasible scores must not be rescored')
    process = _Process()
    runtime = SimpleNamespace(device=torch.device('cpu'), seq_len=3, process=process, adapter=SimpleNamespace(T=7), reward_fn=_unexpected_oracle)
    args = SimpleNamespace(initial_batch_path=str(archive), initial_batch_advance_rng=True, initial_batch_rescore_with_reward=False, initial_batch_out=None, samples_out='/dev/null', _output_policy=OutputPolicy(trace_level='none'))
    state = _initialize_search_state(args=args, runtime=runtime, planned_num_samples=1, max_oracle_evaluations=4, pad_token_id=None, active_learning_selector=None, selection_policy=BestSurvivorSelectionPolicy(), profile_timings=False, profile_sync_cuda=False)
    assert process.calls == [(1, 7, torch.device('cpu'))]
    assert state.loaded_rewards is not None
    assert state.rewards.tolist() == [float('-inf')]
    assert state.sample_source == 'loaded_initial_batch'

def test_occupied_output_set_fails_before_any_file_is_truncated(tmp_path: Path) -> None:
    outputs = {'samples': tmp_path / 'samples.jsonl.gz', 'initial': tmp_path / 'initial.jsonl.gz', 'diagnostics': tmp_path / 'diagnostics.jsonl.gz', 'traces': tmp_path / 'traces.jsonl.gz', 'timings': tmp_path / 'timings.jsonl.gz', 'leaves': tmp_path / 'leaves.pt', 'manifest': tmp_path / 'run_manifest.json'}
    expected = {name: f'existing {name}'.encode() for name in outputs}
    for name, path in outputs.items():
        path.write_bytes(expected[name])
    log_dir = tmp_path / 'tensorboard'
    log_dir.mkdir()
    event_file = log_dir / 'events.out.tfevents.existing'
    event_file.write_bytes(b'existing tensorboard event')
    args = SimpleNamespace(_output_policy=OutputPolicy(trace_level='full', trace_record_limit=10), samples_out=str(outputs['samples']), initial_batch_out=str(outputs['initial']), write_proposal_diagnostics=True, proposal_diagnostics_out=str(outputs['diagnostics']), proposal_traces_out=str(outputs['traces']), timings_out=str(outputs['timings']), leaves_out=str(outputs['leaves']), log_dir=str(log_dir))
    with pytest.raises(FileExistsError, match='fresh samples_out path'):
        _validate_output_paths(args, manifest_path=outputs['manifest'], profile_timings=True)
    assert {name: path.read_bytes() for name, path in outputs.items()} == expected
    assert event_file.read_bytes() == b'existing tensorboard event'

def test_cache_failure_is_not_reclassified_as_a_cache_miss() -> None:

    class BrokenCache:
        config = SimpleNamespace(oracle_batch_size=1, sequential_selection=False)
        _acquisition = ''

        def cached_scores(self, batch: torch.Tensor):
            _ = batch
            raise OSError('cache database unavailable')
    with pytest.raises(RuntimeError, match='score-cache lookup failed') as error:
        evaluate_proposals(_batch(), adapter=_NoMaskAdapter(), reward_fn=_Reward(), track_trust_region=False, active_learning_selector=BrokenCache())
    assert isinstance(error.value.__cause__, OSError)

def test_stateful_oracle_failure_is_called_once_and_propagated() -> None:

    class BrokenOracle:

        def __init__(self) -> None:
            self.calls = 0

        def score_batch(self, batch: torch.Tensor):
            _ = batch
            self.calls += 1
            raise TimeoutError('oracle timed out after accepting request')
    reward = BrokenOracle()
    with pytest.raises(RuntimeError, match='oracle scoring failed') as error:
        evaluate_proposals(_batch(), adapter=_NoMaskAdapter(), reward_fn=reward, track_trust_region=False)
    assert reward.calls == 1
    assert isinstance(error.value.__cause__, TimeoutError)

def _write(path: Path, values: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding='utf-8')
    return path

def test_relative_bases_shallow_overrides_nulls_and_fresh_values(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path / 'base.json', {'seed': 7, 'reward': {'name': 'old', 'device': 'cuda'}, 'order': [3, 1, 2], 'noise': 0.1})
    middle = _write(tmp_path / 'nested/middle.json', {'base_config': '../base.json', 'noise': None})
    child = _write(tmp_path / 'child.json', {'base_config': 'nested/middle.json', 'reward': {'name': 'new'}})
    monkeypatch.chdir(tmp_path.parent)
    result = load_config(child)
    assert result == {'seed': 7, 'reward': {'name': 'new'}, 'order': [3, 1, 2], 'noise': None}
    result['order'].append(9)
    result['reward']['name'] = 'mutated'
    assert load_config(child)['order'] == [3, 1, 2]
    assert load_config(child)['reward'] == {'name': 'new'}
    assert load_config(middle)['reward'] == {'name': 'old', 'device': 'cuda'}

@pytest.mark.parametrize('parent', [None, '', '  ', 1, [], {}])
def test_invalid_base_is_rejected(tmp_path: Path, parent: object) -> None:
    config = _write(tmp_path / 'config.json', {'base_config': parent})
    with pytest.raises(ValueError, match='base_config must be a non-empty path string'):
        load_config(config)

def test_non_object_missing_base_and_cycles_fail_before_execution(tmp_path: Path) -> None:
    child = _write(tmp_path / 'child.json', {'base_config': 'base.json'})
    with pytest.raises(FileNotFoundError, match='base.json'):
        load_config(child)
    _write(tmp_path / 'base.json', [])
    with pytest.raises(TypeError, match='configuration must contain a JSON object'):
        load_config(child)
    _write(tmp_path / 'base.json', {'base_config': 'nested/../child.json'})
    with pytest.raises(ValueError, match='base_config cycle.*child.json.*base.json.*child.json'):
        load_config(child)
