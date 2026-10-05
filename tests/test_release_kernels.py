"""Focused kernels regression tests for the reproducibility release."""
from __future__ import annotations

class _ProteinDecoder:
    """Token decoder without a fitness oracle, for selector isolation checks."""
    decode_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    def __init__(self, **kwargs): pass
    def decode_batch(self, batch):
        return ["".join(self.decode_alphabet[int(x)] for x in row) for row in batch]

import numpy as np
import pytest
import torch
from adapters.protein.kermut import KermutExpectedImprovementSelector, KermutKernel, KermutResources
from adapters.protein.validation import CANONICAL_AA
from search.interpolant.active_learning import ActiveLearningConfig
from surrogate.model import SpacedNGramKernel
from types import SimpleNamespace
from surrogate.features import AdditiveSpacedNGramGPRegressor, SpacedNGramGPRegressor, TanimotoGPRegressor

def _resources(length: int=3) -> KermutResources:
    rng = np.random.default_rng(11)
    return KermutResources(wild_type='A' * length, alphabet=CANONICAL_AA, conditional_probs=rng.dirichlet(np.ones(20), size=length) * 0.97, coords=rng.normal(size=(length, 3)) * 4)

def _features() -> torch.Tensor:
    return torch.tensor([[0, 0, 0, 0.1, 0.2], [1, 0, 0, 0.2, 0.1], [2, 0, 3, 0.8, 0.2], [0, 4, 3, 0.6, 0.1]], dtype=torch.double)

def _reference(resources: KermutResources, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.empty((len(x), len(y)))
    length = len(resources.wild_type)
    probs = resources.conditional_probs
    for row, a in enumerate(x):
        for col, b in enumerate(y):
            structure = 0.0
            for i in np.flatnonzero(a[:length]):
                for j in np.flatnonzero(b[:length]):
                    h = np.sqrt(0.5 * np.square(np.sqrt(probs[i]) - np.sqrt(probs[j])).sum())
                    p = abs(np.log(probs[i, int(a[i])]) - np.log(probs[j, int(b[j])]))
                    d = np.linalg.norm(resources.coords[i] - resources.coords[j])
                    structure += np.exp(-h - p - d)
            sequence = np.exp(-0.5 * np.square(a[length:] - b[length:]).sum())
            result[row, col] = 0.5 * (structure + sequence)
    return result

def test_kernel_matches_paper_mutation_pair_sum_psd_and_gradients() -> None:
    resources = _resources()
    x = _features()
    kernel = KermutKernel(resources).double()
    actual = kernel(x).to_dense()
    np.testing.assert_allclose(actual.detach(), _reference(resources, x.numpy(), x.numpy()), atol=1e-10)
    torch.testing.assert_close(actual, actual.T)
    assert torch.linalg.eigvalsh(actual).min() >= -1e-10
    assert actual[0, 0].item() == pytest.approx(0.5)
    torch.testing.assert_close(kernel(x, diag=True), actual.diagonal())
    actual.sum().backward()
    assert all((param.grad is not None and torch.isfinite(param.grad).all() for param in kernel.parameters()))

def test_kernel_rectangular_broadcast_batches_and_all_wild_type() -> None:
    kernel = KermutKernel(_resources()).double()
    x = _features()
    y = torch.stack((x[1:3], x[2:4])).unsqueeze(0)
    result = kernel(x, y).to_dense()
    assert result.shape == (1, 2, 4, 2)
    for idx in range(2):
        torch.testing.assert_close(result[0, idx], kernel(x, y[0, idx]).to_dense())
    torch.testing.assert_close(kernel(x[:1], x[:1]).to_dense(), torch.tensor([[0.5]], dtype=torch.double))

def test_kernel_respects_reference_residues_and_probability_alphabet_order() -> None:
    resources = KermutResources(**{**vars(_resources()), 'wild_type': 'ACD'})
    reversed_alphabet = resources.alphabet[::-1]
    reordered = KermutResources(**{**vars(resources), 'alphabet': reversed_alphabet, 'conditional_probs': resources.conditional_probs[:, ::-1]})
    sequences = ['ACD', 'CCD', 'ACE', 'CDD']
    embeddings = _features()[:, 3:]

    def packed(alphabet):
        tokens = torch.tensor([[alphabet.index(aa) for aa in seq] for seq in sequences], dtype=torch.double)
        return torch.cat((tokens, embeddings), dim=1)
    standard = KermutKernel(resources).double()(packed(resources.alphabet)).to_dense()
    permuted = KermutKernel(reordered).double()(packed(reversed_alphabet)).to_dense()
    torch.testing.assert_close(standard, permuted)
    assert standard[0, 0].item() == pytest.approx(0.5)

def test_resources_preflight_and_upstream_non_normalized_probabilities(tmp_path) -> None:
    resources = _resources()
    path = tmp_path / 'reference.npz'
    np.savez(path, wild_type=resources.wild_type, alphabet=resources.alphabet, conditional_probs=resources.conditional_probs, coords=resources.coords)
    loaded = KermutResources.load(path, expected_length=3)
    np.testing.assert_array_equal(loaded.conditional_probs, resources.conditional_probs)
    assert np.all(loaded.conditional_probs.sum(1) < 1)
    assert loaded.embedding_prefix == '' and loaded.embedding_suffix == ''
    context_path = tmp_path / 'reference_with_context.npz'
    np.savez(context_path, **{**vars(resources), 'embedding_prefix': np.asarray('M'), 'embedding_suffix': np.asarray('WY')})
    contextual = KermutResources.load(context_path, expected_length=3)
    assert contextual.embedding_prefix == 'M' and contextual.embedding_suffix == 'WY'
    with pytest.raises(ValueError, match='task length'):
        KermutResources.load(path, expected_length=237)
    for field, bad in [('alphabet', CANONICAL_AA[:-1] + 'A'), ('conditional_probs', np.zeros((3, 20))), ('conditional_probs', np.ones((2, 20))), ('coords', np.full((3, 3), np.nan)), ('coords', np.ones((2, 3)))]:
        with pytest.raises(ValueError):
            KermutResources(**{**vars(resources), field: bad})
    for field in ('embedding_prefix', 'embedding_suffix'):
        with pytest.raises(ValueError, match='non-canonical'):
            KermutResources(**{**vars(resources), field: 'B'})

class _FakeESM:

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def featurize(self, sequences: list[str]) -> torch.Tensor:
        self.calls.append(sequences)
        return torch.tensor([[float(seq.count('C')), -0.4] for seq in sequences])

def test_selector_embeds_aav_window_with_both_flanks_and_caches_full_constructs() -> None:
    resources = KermutResources(**{**vars(_resources()), 'embedding_prefix': 'GG', 'embedding_suffix': 'WY'})
    esm = _FakeESM()
    selector = KermutExpectedImprovementSelector(reward_fn=_ProteinDecoder(min_len=1), config=ActiveLearningConfig(), resources=resources, featurizer=esm)
    first, valid = selector._make_fingerprint_features(['AAA', 'CAA', 'AAA'])
    second, _ = selector._make_fingerprint_features(['CAA', 'DAA'])
    assert valid.tolist() == [True, True, True]
    assert esm.calls == [['GGAAAWY', 'GGCAAWY'], ['GGDAAWY']]
    np.testing.assert_array_equal(first[:, :3], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(second[:, :3], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

def test_selector_reuses_cache_and_onehot_similarity_with_actual_gp(monkeypatch) -> None:
    import botorch.fit
    monkeypatch.setattr(botorch.fit, 'fit_gpytorch_mll', lambda mll: mll)
    resources = _resources()
    esm = _FakeESM()
    selector = KermutExpectedImprovementSelector(reward_fn=_ProteinDecoder(min_len=1), config=ActiveLearningConfig(buffer_size=2000, min_train_size=2, acquisition='expected_improvement'), resources=resources, featurizer=esm)
    observed = torch.tensor([[0, 0, 0], [1, 0, 0]])
    selector.observe_oracle_batch(observed, torch.tensor([0.0, 0.5]))
    candidates = torch.tensor([[1, 0, 0], [1, 1, 0], [2, 0, 0]])
    result = selector.select(candidates, budget=2)
    assert set(result.selected_indices) == {1, 2}
    assert all((value is not None and np.isfinite(value) for value in result.expected_improvement))
    assert esm.calls == [['AAA', 'CAA'], ['CCA', 'DAA']]
    selector.select(candidates, budget=2)
    assert len(esm.calls) == 2
    features, _ = selector._make_fingerprint_features(['AAA', 'CAA'])
    assert selector._feature_similarity(*features) == pytest.approx(2 / 4)
    assert selector.cached_scores(observed) == [0.0, 0.5]

def test_spaced_ngram_kernel_is_normalized_psd_and_shift_invariant() -> None:
    pytest.importorskip('gpytorch')
    pytest.importorskip('scipy')
    sequences = torch.tensor([[1, 0, 2, 0, 0], [0, 1, 0, 2, 0], [1, 2, 0, 0, 0], [2, 0, 1, 0, 2]], dtype=torch.double)
    kernel = SpacedNGramKernel(offsets=torch.tensor([[0, 2]]), vocab_size=3)
    gram = kernel(sequences, sequences).to_dense()
    assert torch.allclose(gram, gram.T, atol=1e-10)
    assert torch.allclose(torch.diag(gram), torch.ones(4, dtype=torch.double), atol=1e-10)
    assert float(torch.linalg.eigvalsh(gram).min()) >= -1e-10
    assert gram[0, 1] > gram[0, 2]

def _forced_spaced_ngram_kernel(*, compressed: bool) -> SpacedNGramKernel:
    kernel = SpacedNGramKernel(offsets=torch.tensor([[0, 1], [0, 2]]), vocab_size=3)
    kernel._MAX_DIRECT_SPARSE_FEATURES = 0 if compressed else kernel.num_features
    return kernel

def test_spaced_ngram_observed_feature_compression_matches_direct_cross_covariance() -> None:
    pytest.importorskip('gpytorch')
    pytest.importorskip('scipy')
    x1 = torch.tensor([[0, 1, 0, 1, 2, 1], [2, 1, 2, 0, 1, 0], [1, 1, 0, 2, 0, 2]], dtype=torch.double)
    x2 = torch.tensor([[0, 1, 2, 1, 0, 1], [2, 0, 2, 1, 1, 0]], dtype=torch.double)
    direct = _forced_spaced_ngram_kernel(compressed=False)(x1, x2).to_dense()
    compressed = _forced_spaced_ngram_kernel(compressed=True)(x1, x2).to_dense()
    assert torch.equal(compressed, direct)

def _capture_fit_policy(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    botorch_fit = pytest.importorskip('botorch.fit')
    pytest.importorskip('gpytorch')
    captured: dict[str, object] = {}

    def fake_fit(mll: object) -> object:
        model = mll.model
        captured['trainable'] = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        captured['outputscale'] = float(model.covar_module.outputscale.detach().cpu().item())
        captured['model'] = model
        return mll
    monkeypatch.setattr(botorch_fit, 'fit_gpytorch_mll', fake_fit)
    return captured

@pytest.mark.parametrize('kernel', ['onehot', 'spaced', 'additive'])
def test_fitted_mean_and_noise_are_the_only_trainable_gp_parameters(monkeypatch: pytest.MonkeyPatch, kernel: str) -> None:
    captured = _capture_fit_policy(monkeypatch)
    common = {'noise_variance': None, 'outputscale': 1.0, 'learn_outputscale': False, 'fit_mean_and_noise_only': True}
    x = np.asarray([[0, 1, 2, 0], [0, 1, 1, 0], [2, 1, 0, 2], [1, 2, 0, 1]], dtype=np.float64)
    y = np.asarray([0.0, 0.25, 1.0, 0.5], dtype=np.float64)
    if kernel == 'onehot':
        regressor = TanimotoGPRegressor(**common)
    elif kernel == 'spaced':
        regressor = SpacedNGramGPRegressor(instance=SimpleNamespace(spacings=torch.tensor([[0, 1]])), seq_len=4, vocab_size=3, offsets=[[0, 1]], **common)
    else:
        regressor = AdditiveSpacedNGramGPRegressor(instance=SimpleNamespace(), seq_len=4, vocab_size=3, offset_groups=[[[0, 1]], [[0, 1, 2]]], weights=[0.5, 0.5], **common)
    regressor.fit(x, y)
    assert captured['trainable'] == ['likelihood.noise_covar.raw_noise', 'mean_module.raw_constant']
    assert captured['outputscale'] == 1.0
    model = captured['model']
    assert all((parameter.requires_grad for parameter in model.mean_module.parameters()))
    assert not any((parameter.requires_grad for parameter in model.covar_module.parameters()))

def test_gp_archive_refresh_keeps_hyperparameters_and_uses_all_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_fit_policy(monkeypatch)
    regressor = TanimotoGPRegressor(outputscale=1.0, learn_outputscale=False, fit_mean_and_noise_only=True).fit(np.asarray([[1.0, 0.0], [0.0, 1.0]]), np.asarray([0.0, 1.0]))
    mean_before = regressor.model_.mean_module.constant.detach().clone()
    noise_before = regressor.model_.likelihood.noise.detach().clone()
    transform_mean = regressor.model_.outcome_transform.means.detach().clone()
    transform_std = regressor.model_.outcome_transform.stdvs.detach().clone()
    updated_y = np.asarray([0.0, 1.0, 0.5])
    regressor.refresh_training_data(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), updated_y)
    assert regressor.model_.train_inputs[0].shape == (3, 2)
    assert torch.equal(regressor.model_.mean_module.constant, mean_before)
    assert torch.equal(regressor.model_.likelihood.noise, noise_before)
    expected_targets = (torch.as_tensor(updated_y, dtype=torch.double).view(-1, 1) - transform_mean) / transform_std
    assert torch.allclose(regressor.model_.train_targets, expected_targets.squeeze(-1))
