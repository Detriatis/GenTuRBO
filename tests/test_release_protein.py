"""Focused protein regression tests for the reproducibility release."""
from __future__ import annotations

class _ProteinDecoder:
    """Token decoder without a fitness oracle, for selector isolation checks."""
    decode_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    def __init__(self, **kwargs): pass
    def decode_batch(self, batch):
        return ["".join(self.decode_alphabet[int(x)] for x in row) for row in batch]

import pytest
import torch
from adapters.protein.esm import ESM2Featurizer
from adapters.protein.gp import EmbeddingGPRegressor
from adapters.protein.validation import CANONICAL_AA
from core.registry import build as build_from_registry
from flows.ctmc_adapter import CTMCFlowAdapter
from models.evodiff import EvoDiffD3PMDenoiser
from processes.dfm_mask import MaskingCTMCFlowProcess
from search.interpolant.active_learning import ActiveLearningConfig, OneHotTanimotoExpectedImprovementSelector
from pathlib import Path
from torch import nn
from adapters.protein.fitness_benchmarks import GGS_ALPHABET, KirjnerCNNFitnessOracle, OracleNormalizationStats, evaluate_generated_sequences, tokenize_ggs_sequences

class _FakeAlphabet:
    padding_idx = 0

    def get_batch_converter(self):
        aa_to_id = {aa: idx + 3 for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

        def convert(data):
            max_len = max((len(seq) for _, seq in data))
            tokens = torch.zeros((len(data), max_len + 2), dtype=torch.long)
            labels = []
            seqs = []
            for row, (label, seq) in enumerate(data):
                labels.append(label)
                seqs.append(seq)
                tokens[row, 0] = 1
                for col, aa in enumerate(seq, start=1):
                    tokens[row, col] = aa_to_id[aa]
                tokens[row, len(seq) + 1] = 2
            return (labels, seqs, tokens)
        return convert

class _FakeESMModel:

    def eval(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, tokens, *, repr_layers, return_contacts=False):
        _ = return_contacts
        layer = int(repr_layers[0])
        base = tokens.to(torch.float).unsqueeze(-1)
        reps = torch.cat([base, base + 1.0, base + 2.0, base + 3.0], dim=-1)
        return {'representations': {layer: reps}}

def test_esm_featurizer_mean_pools_residues_and_excludes_special_tokens() -> None:
    featurizer = ESM2Featurizer(model_name='esm2_t30_150M_UR50D', model_loader=lambda _name: (_FakeESMModel(), _FakeAlphabet()))
    embeddings = featurizer.featurize(['ACD', 'KK'])
    assert embeddings.shape == (2, 4)
    assert torch.allclose(embeddings[0], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.allclose(embeddings[1], torch.tensor([11.0, 12.0, 13.0, 14.0]))

def test_embedding_gp_sets_and_freezes_rbf_lengthscale() -> None:

    class _RawLengthscale:

        def __init__(self) -> None:
            self.requires_grad = True

        def requires_grad_(self, value: bool) -> None:
            self.requires_grad = bool(value)

    class _RBFKernel:

        def __init__(self) -> None:
            self.lengthscale = torch.ones(1, 1, dtype=torch.double)
            self.raw_lengthscale = _RawLengthscale()

    class _ScaleKernel:

        def __init__(self) -> None:
            self.base_kernel = _RBFKernel()
    covar = _ScaleKernel()
    gp = EmbeddingGPRegressor()
    used = gp._set_kernel_lengthscale(covar, n_features=9)
    assert used == pytest.approx(3.0)
    assert torch.allclose(covar.base_kernel.lengthscale, torch.full((1, 1), 3.0, dtype=torch.double))
    assert covar.base_kernel.raw_lengthscale.requires_grad is False

def test_protein_onehot_selector_propagates_strict_gp_training_policy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Regressor:

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
    monkeypatch.setattr('search.interpolant.active_learning.TanimotoGPRegressor', _Regressor)
    selector = OneHotTanimotoExpectedImprovementSelector(reward_fn=_ProteinDecoder(min_len=1), config=ActiveLearningConfig(enabled=True, onehot_max_len=6, gp_noise_variance=None, gp_outputscale=1.0, gp_learn_outputscale=False, gp_fit_mean_and_noise_only=True))
    selector._make_gp_regressor()
    assert captured['outputscale'] == 1.0
    assert captured['learn_outputscale'] is False
    assert captured['noise_variance'] is None
    assert captured['fit_mean_and_noise_only'] is True

class _OffsetTokenizer:
    all_aas = CANONICAL_AA
    K = 30
    mask_id = 29

    def __init__(self) -> None:
        self._aa_to_id = {aa: idx + 5 for idx, aa in enumerate(self.all_aas)}

    def tokenize(self, seq: str) -> list[int]:
        return [self._aa_to_id[aa] for aa in seq]

class _OffsetEvoDiffModel:

    def eval(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        _ = (sample, timestep)
        logits = torch.full((*sample.shape, 30), -20.0, device=sample.device)
        logits[..., _OffsetTokenizer().tokenize('K')[0]] = 20.0
        return logits

def _fake_evodiff_backend(_model_type: str):
    k = _OffsetTokenizer.K
    timesteps = 4
    q = torch.full((timesteps, k, k), 1.0 / float(k), dtype=torch.float)
    q_bar = torch.full((timesteps, k, k), 1.0 / float(k), dtype=torch.float)
    return (_OffsetEvoDiffModel(), _OffsetTokenizer(), 'd3pm_uniform', timesteps, q_bar, q)

def test_evodiff_denoiser_maps_evodiff_logits_to_ctmc_residue_order() -> None:
    model = EvoDiffD3PMDenoiser(model_type='fake', seq_len=4, device='cpu', backend_loader=_fake_evodiff_backend)
    tokens = torch.zeros((2, 4), dtype=torch.long)
    logits = model(tokens, timesteps=torch.zeros((2,), dtype=torch.float))
    assert logits.shape == (2, 4, len(CANONICAL_AA))
    assert torch.argmax(logits[0, 0]).item() == CANONICAL_AA.index('K')
    assert model._skip_genturbo_checkpoint_load is True

def test_evodiff_denoiser_uses_native_corruption_token_space() -> None:
    model = EvoDiffD3PMDenoiser(model_type='fake', seq_len=4, device='cpu', backend_loader=_fake_evodiff_backend)
    base = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    xt_clean = model.sample_interpolant(base_sample=base, t_tensor=torch.ones((1,), dtype=torch.float))
    assert xt_clean.tolist() == [_OffsetTokenizer().tokenize('ACDE')]
    assert model.corrupt_mask(base, xt_clean).tolist() == [[False, False, False, False]]
    xt_changed = xt_clean.clone()
    xt_changed[0, 2] = _OffsetTokenizer().tokenize('K')[0]
    assert model.corrupt_mask(base, xt_changed).tolist() == [[False, False, True, False]]
    rolled = model.rollout_from_batch_native(xt_clean, t_start=1.0)
    assert rolled[-1].tolist() == base.tolist()

def test_evodiff_denoiser_plugs_into_ctmc_rollout_adapter() -> None:
    model = build_from_registry('model', 'protein_evodiff_d3pm', model_type='fake', seq_len=4, device='cpu', backend_loader=_fake_evodiff_backend)
    process = MaskingCTMCFlowProcess(model=model, vocab_size=len(CANONICAL_AA), seq_len=4, mask_id=len(CANONICAL_AA), interpolant='uniform', device='cpu', stop_when_filled=False)
    adapter = CTMCFlowAdapter(process=process, steps=2, max_time=0.5, device='cpu')
    rollout = adapter.rollout_from(torch.tensor([0, 1, 2, 3], dtype=torch.long), 0)
    assert len(rollout) >= 2
    assert all((tuple(state.shape) == (4,) for state in rollout))
    assert int(rollout[-1].min().item()) >= 0
    assert int(rollout[-1].max().item()) < len(CANONICAL_AA)

class _FixedOracleModel(nn.Module):

    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer('values', torch.as_tensor(values, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values[:int(x.shape[0])]

def _minimal_ggs_state_dict() -> dict[str, torch.Tensor]:
    return {'predictor.encoder.weight': torch.zeros((2, 20, 1)), 'predictor.encoder.bias': torch.zeros(2), 'predictor.embedding.layer.weight': torch.zeros((4, 2)), 'predictor.embedding.layer.bias': torch.zeros(4), 'predictor.decoder.weight': torch.zeros((1, 4)), 'predictor.decoder.bias': torch.zeros(1)}

def test_ggs_tokenizer_uses_official_alphabet_and_integer_tokens() -> None:
    assert GGS_ALPHABET == 'ARNDCQEGHILKMFPSTWYV'
    tokens = tokenize_ggs_sequences(['ARN', 'DCQ'], sequence_length=3)
    assert tokens.dtype == torch.long
    assert tuple(tokens.shape) == (2, 3)
    assert tokens.tolist() == [[0, 1, 2], [3, 4, 5]]
    with pytest.raises(ValueError, match='unknown residues'):
        tokenize_ggs_sequences(['ARB'], sequence_length=3)
    with pytest.raises(ValueError, match='does not match expected'):
        tokenize_ggs_sequences(['ARND'], sequence_length=3)

def test_checkpoint_loading_rejects_missing_or_unexpected_keys(tmp_path: Path) -> None:
    stats = OracleNormalizationStats(protein='GFP', full_dms_size=1, y_min=0.0, y_max=1.0)
    missing_path = tmp_path / 'missing.ckpt'
    missing_state = _minimal_ggs_state_dict()
    del missing_state['predictor.decoder.bias']
    torch.save({'state_dict': missing_state}, missing_path)
    with pytest.raises(RuntimeError, match='Strict GGS CNN checkpoint load failed'):
        KirjnerCNNFitnessOracle(model_path=missing_path, stats=stats, sequence_length=3)
    unexpected_path = tmp_path / 'unexpected.ckpt'
    unexpected_state = _minimal_ggs_state_dict()
    unexpected_state['predictor.extra.weight'] = torch.zeros(1)
    torch.save({'state_dict': unexpected_state}, unexpected_path)
    with pytest.raises(RuntimeError, match='Strict GGS CNN checkpoint load failed'):
        KirjnerCNNFitnessOracle(model_path=unexpected_path, stats=stats, sequence_length=3)

def test_oracle_predict_returns_raw_unclipped_values() -> None:
    oracle = KirjnerCNNFitnessOracle(model=_FixedOracleModel([-5.0, 15.0]), stats=OracleNormalizationStats(protein='GFP', full_dms_size=2, y_min=0.0, y_max=10.0), sequence_length=3)
    assert oracle.predict(['AAA', 'CCC']).tolist() == pytest.approx([-5.0, 15.0])
    assert oracle.normalized_fitness(['AAA', 'CCC']).tolist() == pytest.approx([-0.5, 1.5])

def test_oracle_scorer_uses_full_dms_min_max_for_normalization() -> None:
    oracle = KirjnerCNNFitnessOracle(model=_FixedOracleModel([10.0, 15.0, 20.0]), stats=OracleNormalizationStats(protein='GFP', full_dms_size=56086, y_min=10.0, y_max=20.0), sequence_length=3)
    normalized = oracle.normalized_fitness(['AAA', 'CCC', 'DDD'])
    assert normalized.tolist() == pytest.approx([0.0, 0.5, 1.0])

def test_oracle_metric_reports_median_normalized_fitness_not_mean() -> None:
    oracle = KirjnerCNNFitnessOracle(model=_FixedOracleModel([0.0, 0.0, 10.0]), stats=OracleNormalizationStats(protein='AAV', full_dms_size=44156, y_min=0.0, y_max=10.0), sequence_length=3)
    result = evaluate_generated_sequences(['AAA', 'CCC', 'DDD'], oracle=oracle)
    assert result.normalized_fitness == pytest.approx([0.0, 0.0, 1.0])
    assert result.median_normalized_fitness == pytest.approx(0.0)
