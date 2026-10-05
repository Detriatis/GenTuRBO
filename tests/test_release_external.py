"""Focused external regression tests for the reproducibility release."""
from __future__ import annotations

import sys
import types
from pathlib import Path
import torch
from genturbo.io import write_jsonl
from genturbo.sequence_baseline_study import lambo2 as lambo2_protocol
from genturbo.sequence_baseline_study.lambo2 import AMINO_ACIDS, LaMBO2ProtocolError, _declared_corpus_inputs, _diagnostic_row, _exact_in_memory_numpy_dataset, _fit_initial_objective_model, _initial_archive, _initialization_inputs, _online_reward_adapter, _strings_to_ids, _transfer_generator_checkpoint, _unlabeled_corpus

def test_holo_prepaid_archive_runs_one_generation_without_rescoring_starts() -> None:
    from baselines.holo_genetic_algorithm import HoloGeneticAlgorithmBaseline

    initial = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])

    class CountingOracle:
        def __init__(self) -> None:
            self.batches: list[torch.Tensor] = []

        def score_batch(self, samples: torch.Tensor) -> torch.Tensor:
            self.batches.append(samples.clone())
            return samples.sum(dim=1).float()

    baseline = HoloGeneticAlgorithmBaseline(
        population_size=4, mutation_prob=0.5, recombine_prob=0.5,
        survival_quantile=0.5, vocab_size=3,
    )
    oracle = CountingOracle()
    result = baseline.run(
        None, oracle, budget=6, batch_size=4, seed=0,
        initial_population=initial, initial_scores=initial.sum(dim=1).tolist(),
        device='cpu',
    )
    assert len(oracle.batches) == 1 and oracle.batches[0].shape == (4, 4)
    assert result.oracle_calls == result.generated_candidates == 4
    assert result.metadata['initial_oracle_evaluations'] == 2
    assert result.metadata['total_charged_oracle_calls'] == 6
    assert result.metadata['prepaid_initial_scores'] is True
    assert len(result.trace) == 1 and result.trace[0]['charged_oracle_calls'] == 6
    assert result.scores.tolist() == oracle.batches[0].sum(dim=1).float().tolist()
    repeat_oracle = CountingOracle()
    repeated = baseline.run(
        None, repeat_oracle, budget=6, batch_size=4, seed=0,
        initial_population=initial, initial_scores=initial.sum(dim=1).tolist(),
        device='cpu',
    )
    assert torch.equal(oracle.batches[0], repeat_oracle.batches[0])
    assert result.scores.tolist() == repeated.scores.tolist()
    assert initial.tolist() == [[0, 1, 2, 0], [2, 1, 0, 2]]

class _ConfiguredReward:

    def __init__(self, *, offset: float, artifact_path: str) -> None:
        self.offset = float(offset)
        self.artifact_path = artifact_path

    def score_batch(self, values: torch.Tensor) -> torch.Tensor:
        return values.to(dtype=torch.float32).sum(dim=1) + self.offset

def test_unlabeled_corpus_is_deterministic_and_contains_only_protein_tokens() -> None:
    transition = torch.eye(20)
    first = _unlabeled_corpus(transition, count=4, sequence_length=8, seed=3, chunk_size=2)
    second = _unlabeled_corpus(transition, count=4, sequence_length=8, seed=3, chunk_size=2)
    assert first.tolist() == second.tolist()
    assert all((len(sequence.split()) == 8 for sequence in first))
    assert all((set(sequence.split()) <= set(AMINO_ACIDS) for sequence in first))
    assert _strings_to_ids([' '.join((AMINO_ACIDS[0] for _ in range(256)))]) == [[0] * 256]

def test_online_helpers_use_resolved_archive_shape_alphabet_and_reward_adapter(tmp_path: Path) -> None:
    archive = tmp_path / 'initial.jsonl'
    artifact = tmp_path / 'reward.bin'
    artifact.write_bytes(b'task reward\n')
    write_jsonl(archive, iter([{'seq': [0, 1, 1, 0], 'score': 1.0, 'score_is_finite': True}, {'seq': [1, 0, 0, 1], 'score': 2.0, 'score_is_finite': True}]))
    encoded = _strings_to_ids(['A B B A'], alphabet='AB', sequence_length=4)
    x0, y0, rows = _initial_archive(archive, expected_count=2, sequence_length=4, alphabet='AB')
    reward, inputs = _online_reward_adapter({'lambo2': {'reward_adapter': {'module': __name__, 'class': '_ConfiguredReward', 'kwargs': {'offset': 3.0, 'artifact_path': str(artifact)}}}}, repo_root=tmp_path)
    assert encoded == [[0, 1, 1, 0]]
    assert x0.tolist() == [['A', 'B', 'B', 'A'], ['B', 'A', 'A', 'B']]
    assert y0.reshape(-1).tolist() == [1.0, 2.0]
    assert len(rows) == 2
    assert isinstance(reward, _ConfiguredReward)
    assert reward.offset == 3.0
    assert inputs == {'reward_adapter_artifact_path': artifact}
    validation = tmp_path / 'validation.csv'
    validation.write_text('sequence\nABBA\n', encoding='utf-8')
    corpus_inputs = _declared_corpus_inputs({'corpus': {'train': str(artifact), 'validation': str(validation)}}, repo_root=tmp_path)
    assert corpus_inputs == {'lambo_corpus_train': artifact, 'lambo_corpus_validation': validation}

def test_initial_archive_accepts_finite_scores_without_optional_flag(tmp_path: Path) -> None:
    archive = tmp_path / 'initial.jsonl'
    write_jsonl(archive, iter([{'seq': [0, 0], 'score': 1.5}, {'seq': [0, 1], 'score': 2.5, 'score_is_finite': True}, {'seq': [1, 0], 'score': 3.5, 'score_is_finite': False}, {'seq': [1, 1], 'score': float('inf'), 'score_is_finite': True}]))
    _x0, y0, _rows = _initial_archive(archive, expected_count=4, sequence_length=2, alphabet='AB')
    assert y0.reshape(-1).tolist() == [1.5, 2.5, -float('inf'), -float('inf')]

def test_online_archive_random_does_not_resolve_pretraining_inputs(monkeypatch, tmp_path: Path) -> None:

    def unexpected_resolution(*_args: object, **_kwargs: object) -> None:
        raise AssertionError('online initialization must not resolve a pretraining artifact')
    monkeypatch.setattr(lambo2_protocol, '_resolve', unexpected_resolution)
    monkeypatch.setattr(lambo2_protocol, '_load_json', unexpected_resolution)
    monkeypatch.setattr(lambo2_protocol, '_declared_corpus_inputs', unexpected_resolution)
    mode, checkpoint, inputs = _initialization_inputs({'checkpoint': 'missing-generator.ckpt', 'pretraining_manifest': 'missing-pretraining.json'}, {'initialization': 'online_archive_random', 'training_history': 'complete_paid_history', 'initial_training_epochs': 1, 'corpus': {'train': 'missing-corpus.csv'}}, repo_root=tmp_path)
    assert mode == 'online_archive_random'
    assert checkpoint is None
    assert inputs == {}

def test_exact_in_memory_dataset_keeps_all_128_rows(monkeypatch) -> None:

    class DummyFrame:

        def __init__(self, data: dict[str, object]) -> None:
            self.data = data

        def __len__(self) -> int:
            return len(next(iter(self.data.values())))

    class DummyNumpyDataset:

        def __init__(self, data: dict[str, list[object]], train: bool, random_seed: int) -> None:
            del random_seed
            split = int(len(next(iter(data.values()))) * 0.8)
            selection = slice(None, split) if train else slice(split, None)
            self._data = DummyFrame({key: value[selection] for key, value in data.items()})
            self.upstream_row_count = len(self._data)
            self.columns = list(data)
    pandas = types.ModuleType('pandas')
    pandas.DataFrame = DummyFrame
    cortex = types.ModuleType('cortex')
    cortex_data = types.ModuleType('cortex.data')
    cortex_dataset = types.ModuleType('cortex.data.dataset')
    cortex_dataset.NumpyDataset = DummyNumpyDataset
    cortex.data = cortex_data
    cortex_data.dataset = cortex_dataset
    monkeypatch.setitem(sys.modules, 'pandas', pandas)
    monkeypatch.setitem(sys.modules, 'cortex', cortex)
    monkeypatch.setitem(sys.modules, 'cortex.data', cortex_data)
    monkeypatch.setitem(sys.modules, 'cortex.data.dataset', cortex_dataset)
    dataset = _exact_in_memory_numpy_dataset({'tokenized_seq': [f'SEQ{index}' for index in range(128)], 'generic_task': [float(index) for index in range(128)]}, train=True, random_seed=7)
    assert dataset.upstream_row_count == 102
    assert len(dataset._data) == 128
    assert dataset.columns == ['tokenized_seq', 'generic_task']

def test_online_diagnostic_proves_full_history_without_pretraining_hashes() -> None:
    row = _diagnostic_row(0, [{'seq': [0, 1], 'oracle_rejected': False}, {'seq': [1, 0], 'oracle_rejected': False}], total=130, protocol={'initialization': 'online_archive_random', 'training_history': 'complete_paid_history', 'initial_training_epochs': 1, 'retraining_epochs': 1, 'objective_refit_epochs_per_round': 1}, bound_input_hashes={'lambo_requirements': 'requirements-hash'})
    assert row['lambo_initialization'] == 'online_archive_random'
    assert row['lambo_training_history'] == 'complete_paid_history'
    assert row['lambo_initial_training_epochs'] == 1
    assert row['lambo_training_rows_before_refit'] == 128
    assert row['lambo_exact_dataset_rows'] is True
    assert row['lambo_bound_input_hashes'] == {'lambo_requirements': 'requirements-hash'}

def test_generator_checkpoint_transfer_cannot_touch_online_only_parameters(tmp_path: Path) -> None:
    online_path = tmp_path / 'online.ckpt'
    pretrained_path = tmp_path / 'generator.ckpt'
    torch.save({'state_dict': {'root.weight': torch.zeros(2), 'property.weight': torch.ones(2)}}, online_path)
    torch.save({'state_dict': {'root.weight': torch.full((2,), 3.0)}}, pretrained_path)
    assert _transfer_generator_checkpoint(online_path, pretrained_path) == 1
    transferred = torch.load(online_path, map_location='cpu', weights_only=False)['state_dict']
    assert torch.equal(transferred['root.weight'], torch.full((2,), 3.0))
    assert torch.equal(transferred['property.weight'], torch.ones(2))
    torch.save({'state_dict': {'root.weight': torch.ones(2), 'unexpected.weight': torch.ones(1)}}, pretrained_path)
    try:
        _transfer_generator_checkpoint(online_path, pretrained_path)
    except LaMBO2ProtocolError as exc:
        assert 'does not match' in str(exc)
    else:
        raise AssertionError('unexpected pretrained parameters must be rejected')

def test_initial_objective_fit_is_persisted_after_generator_transfer(tmp_path: Path) -> None:
    online_path = tmp_path / 'online.ckpt'
    pretrained_path = tmp_path / 'generator.ckpt'
    torch.save({'state_dict': {'root.weight': torch.zeros(2), 'property.weight': torch.ones(2)}}, online_path)
    torch.save({'state_dict': {'root.weight': torch.full((2,), 3.0)}}, pretrained_path)

    class DummySolver:
        model_path = online_path
        calls: list[dict[str, object]] = []

        def _train_model_with_history(self, **kwargs: object) -> None:
            self.calls.append(kwargs)
    solver = DummySolver()
    _fit_initial_objective_model(solver, pretrained_path, expected_transferred_tensors=1, epochs=1)
    assert solver.calls == [{'load_checkpoint_from': online_path, 'save_checkpoint_to': online_path, 'max_epochs': 1}]
    state = torch.load(online_path, map_location='cpu', weights_only=False)['state_dict']
    assert torch.equal(state['root.weight'], torch.full((2,), 3.0))
