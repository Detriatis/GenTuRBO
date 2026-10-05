"""Active-learning helpers for interpolant search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from surrogate.features import (
    AdditiveSpacedNGramGPRegressor,
    EhrlichDiffusionGPRegressor,
    SpacedNGramGPRegressor,
    TanimotoGPRegressor,
)

DEFAULT_PROTEIN_ONEHOT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(slots=True)
class ActiveLearningConfig:
    enabled: bool = False
    initial_batch_matches_buffer_size: bool = False
    oracle_batch_size: int = 32
    oracle_best_budget_mode: str = "physical"
    sequential_selection: bool = False
    local_penalization: bool = False
    local_penalization_strength: float = 1.0
    local_penalization_power: float = 2.0
    acquisition: str = "expected_improvement"
    qei_mc_samples: int = 128
    explore_batch_size: int = 0
    exploit_batch_size: int | None = None
    wei_lambda: float = 0.0
    wei_standardize: bool = False
    wei_eps: float = 1e-12
    thompson_above_incumbent_only: bool = False
    ei_threshold: float | None = None
    debug_random_sample_count: int = 0
    max_resample_batches: int = 0
    buffer_size: int = 256
    buffer_mode: str = "recent"
    morgan_radius: int = 2
    morgan_bits: int = 1024
    morgan_representation: str = "bit"
    onehot_max_len: int | None = None
    onehot_alphabet: str = DEFAULT_PROTEIN_ONEHOT_ALPHABET
    gp_device: str = "cpu"
    gp_dtype: str = "double"
    gp_jitter: float | None = None
    gp_noise_variance: float | None = None
    gp_outputscale: float | None = None
    gp_learn_outputscale: bool = True
    gp_fit_mean_and_noise_only: bool = False
    gp_hyperparameter_refit_interval: int = 1
    gp_lengthscale: float | None = None
    gp_learn_lengthscale: bool = False
    min_train_size: int = 2
    ei_xi: float = 0.0
    buffer_diversity_max_similarity: float | None = None
    ehrlich_black_box_feasibility: bool = False


@dataclass(slots=True)
class ActiveLearningSelection:
    selected_indices: list[int]
    smiles: list[Optional[str]]
    surrogate_mean: list[Optional[float]]
    surrogate_std: list[Optional[float]]
    expected_improvement: list[Optional[float]]
    acquisition_value: list[Optional[float]]
    local_kernel_log_prob: list[Optional[float]]
    max_expected_improvement: Optional[float] = None
    num_candidates_above_threshold: int = 0
    threshold_exhausted: bool = False


@dataclass(slots=True)
class ActiveLearningBufferStats:
    pre_size: int = 0
    valid_rows: int = 0
    admitted: int = 0
    replaced: int = 0
    rejected_similar: int = 0
    post_size: int = 0


@dataclass(slots=True)
class ActiveLearningSimilarityDiagnostics:
    in_buffer: Optional[bool] = None
    max_similarity: Optional[float] = None
    mean_topk_similarity: Optional[float] = None


@dataclass(slots=True)
class _BufferEntry:
    smiles: str
    sequence_key: bytes
    fingerprint: np.ndarray
    score: float
    seen_at: int


def _fingerprint_key(fingerprint: np.ndarray) -> bytes:
    fp = np.asarray(fingerprint, dtype=np.float32).reshape(-1)
    return fp.tobytes()


def _tanimoto_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_np = np.asarray(a, dtype=np.float64).reshape(-1)
    b_np = np.asarray(b, dtype=np.float64).reshape(-1)
    numerator = float(np.dot(a_np, b_np))
    denom = float(np.dot(a_np, a_np) + np.dot(b_np, b_np) - numerator)
    if denom <= 0.0:
        return 1.0
    return numerator / denom


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    best: float,
    xi: float,
) -> np.ndarray:
    mean_t = torch.as_tensor(mean, dtype=torch.double)
    std_t = torch.as_tensor(std, dtype=torch.double).clamp_min(0.0)
    improvement = mean_t - float(best) - float(xi)
    safe_std = torch.where(std_t > 0.0, std_t, torch.ones_like(std_t))
    z = improvement / safe_std
    normal_pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    normal_cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    ei = improvement * normal_cdf + std_t * normal_pdf
    ei = torch.where(std_t > 0.0, ei, torch.clamp(improvement, min=0.0))
    return ei.detach().cpu().numpy()


def _normalize_acquisition(name: str) -> str:
    value = str(name).strip().lower()
    if value in {"ei", "expected_improvement"}:
        return "expected_improvement"
    if value in {"wei", "weighted_ei", "weighted_expected_improvement"}:
        return "weighted_expected_improvement"
    if value in {"qei", "batch_ei", "batched_ei", "batched_expected_improvement"}:
        return "batched_expected_improvement"
    if value in {
        "explore_then_qei",
        "explore_then_batch_ei",
        "explore_then_batched_ei",
        "explore_then_batched_expected_improvement",
    }:
        return "explore_then_batched_expected_improvement"
    if value in {
        "wqei",
        "weighted_qei",
        "weighted_batch_ei",
        "weighted_batched_ei",
        "weighted_batched_expected_improvement",
    }:
        return "weighted_batched_expected_improvement"
    if value in {"random", "uniform_random", "random_selection"}:
        return "random"
    if value in {
        "oracle",
        "oracle_best",
        "oracle_greedy",
        "oracle_selection",
        "oracle_upper_bound",
    }:
        return "oracle_best"
    if value in {"thompson", "thompson_sampling", "ts"}:
        return "thompson"
    if value in {
        "slotwise_thompson",
        "slotwise_thompson_sampling",
        "per_slot_thompson",
        "independent_thompson",
        "independent_slot_thompson",
    }:
        return "slotwise_thompson"
    raise ValueError(
        "active_learning acquisition must be one of: expected_improvement, "
        "weighted_expected_improvement, batched_expected_improvement, "
        "weighted_batched_expected_improvement, explore_then_batched_expected_improvement, "
        "random, oracle_best, thompson, slotwise_thompson"
    )


def _normalize_morgan_representation(name: str) -> str:
    value = str(name).strip().lower()
    if value in {"bit", "bits", "binary"}:
        return "bit"
    if value in {"count", "counts"}:
        return "count"
    raise ValueError("active_learning morgan_representation must be one of: bit, count")


def _normalize_onehot_alphabet(alphabet: str | None) -> str:
    raw = DEFAULT_PROTEIN_ONEHOT_ALPHABET if alphabet is None else str(alphabet)
    seen: set[str] = set()
    out: list[str] = []
    for token in raw.upper():
        if token.isspace() or token in seen:
            continue
        seen.add(token)
        out.append(token)
    if not out:
        raise ValueError("active_learning onehot_alphabet must not be empty")
    return "".join(out)


def make_sequence_onehot_features(
    sequences: list[Optional[str]],
    *,
    max_len: int,
    alphabet: str | None = DEFAULT_PROTEIN_ONEHOT_ALPHABET,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode decoded sequences as fixed-length flattened one-hot vectors."""
    max_len_int = int(max_len)
    if max_len_int <= 0:
        raise ValueError("active_learning onehot_max_len must be positive")
    alphabet_clean = _normalize_onehot_alphabet(alphabet)
    aa_to_col = {aa: idx for idx, aa in enumerate(alphabet_clean)}
    valid_mask = np.zeros((len(sequences),), dtype=bool)
    rows: list[np.ndarray] = []
    feature_dim = max_len_int * len(alphabet_clean)
    for row_idx, value in enumerate(sequences):
        if value is None:
            continue
        seq = "".join(str(value).upper().split())
        if not seq or len(seq) > max_len_int:
            continue
        if any(aa not in aa_to_col for aa in seq):
            continue
        features = np.zeros((max_len_int, len(alphabet_clean)), dtype=np.float64)
        for pos, aa in enumerate(seq):
            features[pos, aa_to_col[aa]] = 1.0
        rows.append(features.reshape(feature_dim))
        valid_mask[row_idx] = True
    if not rows:
        return np.zeros((0, feature_dim), dtype=np.float64), valid_mask
    return np.stack(rows, axis=0), valid_mask


def _local_penalization_factor(
    similarity: float,
    *,
    strength: float,
    power: float,
) -> float:
    sim = max(0.0, min(1.0, float(similarity)))
    return max(0.0, 1.0 - float(strength) * (sim ** float(power)))


def _zscore(values: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if not np.any(finite):
        return out
    finite_values = arr[finite]
    std = float(np.std(finite_values))
    if not math.isfinite(std) or std <= 0.0:
        out[finite] = 0.0
        return out
    out[finite] = (finite_values - float(np.mean(finite_values))) / (std + float(eps))
    return out


class SequenceExpectedImprovementSelector:
    """Select a fixed oracle batch using a Tanimoto GP over Morgan fingerprints."""

    def __init__(
        self,
        *,
        reward_fn: Any,
        config: ActiveLearningConfig,
    ) -> None:
        decode_batch = getattr(reward_fn, "decode_batch", None)
        if not callable(decode_batch):
            raise TypeError(
                "active_learning requires a reward_fn with decode_batch(...), for example InVirtuoGenSmilesReward"
            )
        self.reward_fn = reward_fn
        self.config = config
        self._acquisition = _normalize_acquisition(config.acquisition)
        self._morgan_representation = _normalize_morgan_representation(config.morgan_representation)
        self._local_penalization = bool(config.local_penalization)
        self._local_penalization_strength = float(config.local_penalization_strength)
        self._local_penalization_power = float(config.local_penalization_power)
        if self._local_penalization_strength < 0.0:
            raise ValueError("active_learning local_penalization_strength must be >= 0")
        if self._local_penalization_power <= 0.0:
            raise ValueError("active_learning local_penalization_power must be > 0")
        if int(config.qei_mc_samples) < 1:
            raise ValueError("active_learning qei_mc_samples must be >= 1")
        if int(config.gp_hyperparameter_refit_interval) < 1:
            raise ValueError("active_learning gp_hyperparameter_refit_interval must be >= 1")
        if int(config.explore_batch_size) < 0:
            raise ValueError("active_learning explore_batch_size must be >= 0")
        if config.exploit_batch_size is not None and int(config.exploit_batch_size) < 0:
            raise ValueError("active_learning exploit_batch_size must be >= 0")
        if self._acquisition == "explore_then_batched_expected_improvement":
            explore_batch_size = int(config.explore_batch_size)
            exploit_batch_size = config.exploit_batch_size
            oracle_batch_size = max(0, int(config.oracle_batch_size))
            if explore_batch_size > oracle_batch_size:
                raise ValueError("active_learning explore_batch_size must be <= oracle_batch_size")
            if exploit_batch_size is not None and explore_batch_size + int(exploit_batch_size) > oracle_batch_size:
                raise ValueError("active_learning explore_batch_size + exploit_batch_size must be <= oracle_batch_size")
        if float(config.wei_lambda) < 0.0:
            raise ValueError("active_learning wei_lambda must be >= 0")
        if float(config.wei_eps) <= 0.0:
            raise ValueError("active_learning wei_eps must be > 0")
        self._entries: dict[bytes, _BufferEntry] = {}
        self._score_cache: dict[bytes, float] = {}
        self._best_entry: Optional[_BufferEntry] = None
        self._seen_counter = 0
        self._last_buffer_stats = ActiveLearningBufferStats()
        self._fitted_gp_cache: tuple[int, Any, np.ndarray] | None = None
        self._gp_archive_generation = -1
        self._last_hyperparameter_refit_generation: int | None = None

    def _make_fingerprint_features(
        self,
        smiles: list[Optional[str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Choose a sequence or protein feature selector")

    @property
    def buffer_size(self) -> int:
        return len(self._entries)

    def _feature_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return _tanimoto_similarity(a, b)

    @property
    def last_buffer_stats(self) -> ActiveLearningBufferStats:
        return self._last_buffer_stats

    def _next_seen_at(self) -> int:
        self._seen_counter += 1
        return self._seen_counter

    def _prune(self) -> None:
        max_size = max(1, int(self.config.buffer_size))
        mode = str(self.config.buffer_mode).strip().lower()
        entries = list(self._entries.values())
        if mode == "top":
            ordered = sorted(entries, key=lambda item: (float(item.score), int(item.seen_at)), reverse=True)
        elif mode == "recent":
            ordered = sorted(entries, key=lambda item: int(item.seen_at), reverse=True)
        else:
            raise ValueError("active_learning buffer_mode must be one of: recent, top")
        threshold = self.config.buffer_diversity_max_similarity
        if threshold is not None:
            threshold = float(threshold)
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError("active_learning buffer_diversity_max_similarity must be in [0, 1]")
        keep: list[_BufferEntry] = []
        must_keep = self._best_entry
        if must_keep is not None:
            keep.append(must_keep)
        must_keep_key = None if must_keep is None else _fingerprint_key(must_keep.fingerprint)
        for item in ordered:
            if len(keep) >= max_size:
                break
            if must_keep_key is not None and _fingerprint_key(item.fingerprint) == must_keep_key:
                continue
            if threshold is not None:
                too_similar = any(
                    self._feature_similarity(item.fingerprint, kept.fingerprint) >= threshold for kept in keep
                )
                if too_similar:
                    continue
            keep.append(item)
        self._entries = {_fingerprint_key(item.fingerprint): item for item in keep}

    def _closest_entry(self, fingerprint: np.ndarray) -> tuple[Optional[bytes], Optional[_BufferEntry], float]:
        if not self._entries:
            return None, None, float("-inf")
        best_key: Optional[bytes] = None
        best_entry: Optional[_BufferEntry] = None
        best_similarity = float("-inf")
        for key, entry in self._entries.items():
            similarity = self._feature_similarity(fingerprint, entry.fingerprint)
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key
                best_entry = entry
        return best_key, best_entry, best_similarity

    def similarity_diagnostics(
        self,
        smiles: list[Optional[str]],
        *,
        top_k: int = 5,
    ) -> list[ActiveLearningSimilarityDiagnostics]:
        diagnostics = [ActiveLearningSimilarityDiagnostics() for _ in smiles]
        if not smiles or not self._entries:
            return diagnostics
        k = max(1, min(int(top_k), len(self._entries)))
        try:
            fingerprints, valid_mask = self._make_fingerprint_features(smiles)
        except ValueError:
            return diagnostics

        valid_rows = np.flatnonzero(valid_mask)
        entries = list(self._entries.values())
        for fp_row, batch_idx in enumerate(valid_rows.tolist()):
            fingerprint = np.asarray(fingerprints[fp_row], dtype=np.float64)
            fp_key = _fingerprint_key(fingerprint)
            similarities = sorted(
                (self._feature_similarity(fingerprint, entry.fingerprint) for entry in entries),
                reverse=True,
            )
            if not similarities:
                diagnostics[batch_idx] = ActiveLearningSimilarityDiagnostics(
                    in_buffer=(fp_key in self._entries),
                )
                continue
            diagnostics[batch_idx] = ActiveLearningSimilarityDiagnostics(
                in_buffer=(fp_key in self._entries),
                max_similarity=float(similarities[0]),
                mean_topk_similarity=float(np.mean(similarities[:k])),
            )
        return diagnostics

    def observe_oracle_batch(self, batch: torch.Tensor, scores: torch.Tensor) -> None:
        if batch.ndim != 2:
            raise ValueError(f"observe_oracle_batch expects [B, L], got {tuple(batch.shape)}")
        score_values = torch.as_tensor(scores, dtype=torch.float).view(-1).detach().cpu().tolist()
        if len(score_values) != int(batch.shape[0]):
            raise ValueError("scores must align with batch rows")

        smiles = self.reward_fn.decode_batch(batch)
        try:
            fingerprints, valid_mask = self._make_fingerprint_features(smiles)
        except ValueError:
            return

        threshold = self.config.buffer_diversity_max_similarity
        if threshold is not None:
            threshold = float(threshold)
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError("active_learning buffer_diversity_max_similarity must be in [0, 1]")
        stats = ActiveLearningBufferStats(pre_size=len(self._entries), valid_rows=int(np.count_nonzero(valid_mask)))
        valid_rows = np.flatnonzero(valid_mask)
        for fp_row, batch_idx in enumerate(valid_rows.tolist()):
            smiles_value = smiles[batch_idx]
            if smiles_value is None:
                continue
            score_value = float(score_values[batch_idx])
            if not math.isfinite(score_value):
                continue
            seen_at = self._next_seen_at()
            sequence_key = batch[batch_idx].detach().to("cpu").contiguous().numpy().tobytes()
            fingerprint = np.asarray(fingerprints[fp_row], dtype=np.float64)
            fp_key = _fingerprint_key(fingerprint)
            candidate_entry = _BufferEntry(
                smiles=str(smiles_value),
                sequence_key=sequence_key,
                fingerprint=fingerprint,
                score=score_value,
                seen_at=seen_at,
            )
            self._score_cache[sequence_key] = score_value
            if self._best_entry is None or score_value > float(self._best_entry.score):
                self._best_entry = candidate_entry
            existing = self._entries.get(fp_key)
            if existing is not None:
                if score_value > float(existing.score):
                    self._entries[fp_key] = candidate_entry
                    stats.replaced += 1
                else:
                    stats.rejected_similar += 1
                continue

            if threshold is not None:
                closest_key, closest_entry, closest_similarity = self._closest_entry(fingerprint)
                if closest_entry is not None and closest_similarity >= threshold:
                    if score_value > float(closest_entry.score):
                        assert closest_key is not None
                        del self._entries[closest_key]
                        self._entries[fp_key] = candidate_entry
                        stats.replaced += 1
                    else:
                        stats.rejected_similar += 1
                    continue

            self._entries[fp_key] = candidate_entry
            stats.admitted += 1
        self._prune()
        stats.post_size = len(self._entries)
        self._last_buffer_stats = stats

    def cached_scores(self, batch: torch.Tensor) -> list[Optional[float]]:
        if batch.ndim != 2:
            raise ValueError(f"cached_scores expects [B, L], got {tuple(batch.shape)}")
        cached: list[Optional[float]] = []
        batch_cpu = batch.detach().to("cpu").contiguous()
        for row in batch_cpu:
            key = row.numpy().tobytes()
            cached.append(self._score_cache.get(key))
        return cached

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        entries = list(self._entries.values())
        if not entries:
            raise ValueError("active learning buffer is empty")
        x = np.stack([entry.fingerprint for entry in entries], axis=0)
        y = np.asarray([entry.score for entry in entries], dtype=np.float64)
        return x, y

    def _make_gp_regressor(self) -> TanimotoGPRegressor:
        """Construct the surrogate used by this selector.

        Kept as a method so categorical tasks can reuse the mature acquisition
        and buffering logic with a task-specific positive-semidefinite kernel.
        """

        return TanimotoGPRegressor(
            device=str(self.config.gp_device),
            dtype=str(self.config.gp_dtype),
            jitter=self.config.gp_jitter,
            noise_variance=self.config.gp_noise_variance,
            outputscale=self.config.gp_outputscale,
            learn_outputscale=bool(self.config.gp_learn_outputscale),
            fit_mean_and_noise_only=bool(self.config.gp_fit_mean_and_noise_only),
        )

    def _fitted_gp(self) -> tuple[Any, np.ndarray]:
        """Refresh the posterior each archive update and refit parameters by cadence."""

        cached = self._fitted_gp_cache
        if cached is not None and cached[0] == self._seen_counter:
            return cached[1], cached[2]
        x_train, y_train = self._training_arrays()
        self._gp_archive_generation += 1
        interval = int(self.config.gp_hyperparameter_refit_interval)
        should_refit = (
            cached is None
            or self._last_hyperparameter_refit_generation is None
            or self._gp_archive_generation - self._last_hyperparameter_refit_generation >= interval
        )
        if should_refit:
            gp = self._make_gp_regressor()
            gp.fit(x_train, y_train)
            self._last_hyperparameter_refit_generation = self._gp_archive_generation
        else:
            gp = cached[1]
            refresh = getattr(gp, "refresh_training_data", None)
            if not callable(refresh):
                raise TypeError(
                    "gp_hyperparameter_refit_interval > 1 requires a regressor with refresh_training_data(...)"
                )
            refresh(x_train, y_train)
        self._fitted_gp_cache = (self._seen_counter, gp, y_train)
        return gp, y_train

    def fit_surrogate_moments_predictor(self) -> Any:
        """Return aligned posterior mean and standard-deviation predictions."""

        if self.buffer_size < max(1, int(self.config.min_train_size)):
            raise ValueError("active learning buffer is too small to fit a surrogate")
        gp, _ = self._fitted_gp()

        def predict(batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
            sequences = self.reward_fn.decode_batch(batch)
            features, valid = self._make_fingerprint_features(sequences)
            mean = np.full((int(batch.shape[0]),), -np.inf, dtype=np.float64)
            std = np.zeros_like(mean)
            if features.shape[0] > 0:
                valid_rows = np.flatnonzero(valid)
                valid_mean, valid_std = self._predict_gp_mean_std(gp, features)
                mean[valid_rows] = valid_mean
                std[valid_rows] = valid_std
            return mean, std

        return predict

    def fit_surrogate_predictor(self) -> Any:
        """Return mean and EI predictions for proposal optimizers."""

        predict_moments = self.fit_surrogate_moments_predictor()
        _, y_train = self._fitted_gp()
        incumbent = float(np.max(y_train))

        def predict(batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
            mean, std = predict_moments(batch)
            acquisition = np.full_like(mean, -np.inf)
            valid_rows = np.flatnonzero(np.isfinite(mean))
            if valid_rows.size > 0:
                acquisition[valid_rows] = _expected_improvement(
                    mean[valid_rows],
                    std[valid_rows],
                    best=incumbent,
                    xi=float(self.config.ei_xi),
                )
            return mean, acquisition

        return predict

    @staticmethod
    def _predict_gp_mean_std(gp: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predict_joint = getattr(gp, "predict_mean_std", None)
        if callable(predict_joint):
            mean, std = predict_joint(features)
        else:
            mean, std = gp.predict(features), gp.predict_std(features)
        return (
            np.asarray(mean, dtype=np.float64).reshape(-1),
            np.asarray(std, dtype=np.float64).reshape(-1),
        )

    def _fallback_indices(self, smiles: list[Optional[str]], budget: int) -> list[int]:
        selected: list[int] = []
        seen_smiles: set[str] = set()
        for idx, value in enumerate(smiles):
            if value is None:
                continue
            key = str(value)
            if key in seen_smiles:
                continue
            seen_smiles.add(key)
            selected.append(idx)
            if len(selected) >= budget:
                return selected
        for idx in range(len(smiles)):
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= budget:
                break
        return selected

    def select(
        self,
        batch: torch.Tensor,
        *,
        budget: int | None = None,
        local_kernel_log_probs: Optional[list[Optional[float]] | np.ndarray | torch.Tensor] = None,
    ) -> ActiveLearningSelection:
        if batch.ndim != 2:
            raise ValueError(f"select expects [B, L], got {tuple(batch.shape)}")
        num_rows = int(batch.shape[0])
        local_kernel_values: list[Optional[float]] = [None] * num_rows
        local_kernel_array = np.zeros((num_rows,), dtype=np.float64)
        if local_kernel_log_probs is not None:
            if isinstance(local_kernel_log_probs, torch.Tensor):
                raw_kernel = local_kernel_log_probs.detach().cpu().view(-1).tolist()
            elif isinstance(local_kernel_log_probs, np.ndarray):
                raw_kernel = np.asarray(local_kernel_log_probs).reshape(-1).tolist()
            else:
                raw_kernel = list(local_kernel_log_probs)
            if len(raw_kernel) != num_rows:
                raise ValueError("local_kernel_log_probs must align with batch rows")
            for idx, value in enumerate(raw_kernel):
                if value is None:
                    continue
                try:
                    float_value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(float_value):
                    local_kernel_values[idx] = float_value
                    local_kernel_array[idx] = float_value
        select_budget = int(self.config.oracle_batch_size) if budget is None else int(budget)
        budget = max(0, min(select_budget, num_rows))
        smiles = self.reward_fn.decode_batch(batch)
        surrogate_mean: list[Optional[float]] = [None] * num_rows
        surrogate_std: list[Optional[float]] = [None] * num_rows
        expected_improvement: list[Optional[float]] = [None] * num_rows
        acquisition_value: list[Optional[float]] = [None] * num_rows
        local_kernel_log_prob: list[Optional[float]] = list(local_kernel_values)
        max_expected_improvement: Optional[float] = None
        num_candidates_above_threshold = 0
        ei_threshold = float(self.config.ei_threshold) if self.config.ei_threshold is not None else None
        if ei_threshold is not None and ei_threshold <= 0.0:
            ei_threshold = None
        mixed_explore_then_qei = self._acquisition == "explore_then_batched_expected_improvement"
        explore_slots = 0
        if mixed_explore_then_qei:
            explore_slots = max(0, int(self.config.explore_batch_size))
            if self.config.exploit_batch_size is not None:
                requested_budget = explore_slots + max(0, int(self.config.exploit_batch_size))
                budget = min(budget, requested_budget)
            explore_slots = min(explore_slots, budget)

        fallback = self._fallback_indices(smiles, budget)
        try:
            x_candidate_valid, valid_mask = self._make_fingerprint_features(smiles)
        except ValueError:
            return ActiveLearningSelection(
                selected_indices=fallback,
                smiles=smiles,
                surrogate_mean=surrogate_mean,
                surrogate_std=surrogate_std,
                expected_improvement=expected_improvement,
                acquisition_value=acquisition_value,
                local_kernel_log_prob=local_kernel_log_prob,
                max_expected_improvement=max_expected_improvement,
                num_candidates_above_threshold=num_candidates_above_threshold,
                threshold_exhausted=False,
            )

        valid_rows = np.flatnonzero(valid_mask)
        representative_rows: list[int] = []
        representative_fp_keys: list[bytes] = []
        seen_candidate_fingerprints: set[bytes] = set()
        for batch_idx in valid_rows.tolist():
            fp_row = int(np.sum(valid_mask[:batch_idx]))
            fp_key = _fingerprint_key(x_candidate_valid[fp_row])
            if fp_key in seen_candidate_fingerprints:
                continue
            seen_candidate_fingerprints.add(fp_key)
            representative_rows.append(batch_idx)
            representative_fp_keys.append(fp_key)

        observed_fingerprints = set(self._entries.keys())
        paired_rows = zip(representative_rows, representative_fp_keys, strict=True)
        unobserved_rows = [idx for idx, fp_key in paired_rows if fp_key not in observed_fingerprints]
        observed_rows = [
            idx
            for idx, fp_key in zip(representative_rows, representative_fp_keys, strict=True)
            if fp_key in observed_fingerprints
        ]

        if self._acquisition == "random":
            eligible_rows = list(unobserved_rows)
            eligible_rows.extend(observed_rows)
            if not eligible_rows:
                eligible_rows = list(fallback)
            if not eligible_rows or budget <= 0:
                selected: list[int] = []
            else:
                order = torch.randperm(len(eligible_rows)).detach().cpu().tolist()
                selected = [int(eligible_rows[int(pos)]) for pos in order[:budget]]
            return ActiveLearningSelection(
                selected_indices=selected,
                smiles=smiles,
                surrogate_mean=surrogate_mean,
                surrogate_std=surrogate_std,
                expected_improvement=expected_improvement,
                acquisition_value=acquisition_value,
                local_kernel_log_prob=local_kernel_log_prob,
                max_expected_improvement=max_expected_improvement,
                num_candidates_above_threshold=num_candidates_above_threshold,
                threshold_exhausted=False,
            )

        if self.buffer_size < max(1, int(self.config.min_train_size)):
            return ActiveLearningSelection(
                selected_indices=fallback,
                smiles=smiles,
                surrogate_mean=surrogate_mean,
                surrogate_std=surrogate_std,
                expected_improvement=expected_improvement,
                acquisition_value=acquisition_value,
                local_kernel_log_prob=local_kernel_log_prob,
                max_expected_improvement=max_expected_improvement,
                num_candidates_above_threshold=num_candidates_above_threshold,
                threshold_exhausted=False,
            )

        gp, y_train = self._fitted_gp()
        mean_valid, std_valid = self._predict_gp_mean_std(gp, x_candidate_valid)
        ei_valid = _expected_improvement(
            mean_valid,
            std_valid,
            best=float(np.max(y_train)),
            xi=float(self.config.ei_xi),
        )
        valid_rows = np.flatnonzero(valid_mask)
        local_kernel_valid = np.asarray(
            [local_kernel_array[int(batch_idx)] for batch_idx in valid_rows.tolist()],
            dtype=np.float64,
        )
        local_kernel_valid = np.where(np.isfinite(local_kernel_valid), local_kernel_valid, 0.0)
        use_weighted_ei = self._acquisition in {
            "weighted_expected_improvement",
            "weighted_batched_expected_improvement",
        }
        wei_lambda = float(self.config.wei_lambda)
        wei_eps = float(self.config.wei_eps)

        def _weighted_log_score(values: np.ndarray, kernel_values: np.ndarray) -> np.ndarray:
            values_np = np.asarray(values, dtype=np.float64).reshape(-1)
            kernel_np = np.asarray(kernel_values, dtype=np.float64).reshape(-1)
            if kernel_np.shape[0] != values_np.shape[0]:
                raise ValueError("kernel_values must align with acquisition values")
            log_values = np.log(np.clip(values_np, wei_eps, None))
            kernel_np = np.where(np.isfinite(kernel_np), kernel_np, 0.0)
            if bool(self.config.wei_standardize):
                return _zscore(log_values) + wei_lambda * _zscore(kernel_np)
            return log_values + wei_lambda * kernel_np

        if self._acquisition == "thompson":
            acquisition_valid = np.asarray(
                gp.sample_posterior(x_candidate_valid, num_samples=1), dtype=np.float64
            ).reshape(-1)
        elif use_weighted_ei:
            acquisition_valid = _weighted_log_score(ei_valid, local_kernel_valid)
        else:
            # Slotwise Thompson has no single acquisition value for the whole
            # candidate pool. Selected rows are overwritten below with the
            # per-slot posterior draw value that actually selected them.
            acquisition_valid = np.asarray(ei_valid, dtype=np.float64).reshape(-1)
        incumbent_score = float(self._best_entry.score) if self._best_entry is not None else float(np.max(y_train))

        for fp_row, batch_idx in enumerate(valid_rows.tolist()):
            surrogate_mean[batch_idx] = float(mean_valid[fp_row])
            surrogate_std[batch_idx] = float(std_valid[fp_row])
            expected_improvement[batch_idx] = float(ei_valid[fp_row])
            acquisition_value[batch_idx] = float(acquisition_valid[fp_row])
        if ei_valid.size > 0:
            max_expected_improvement = float(np.max(ei_valid))

        thompson_gate_active = self._acquisition in {"thompson", "slotwise_thompson"} and bool(
            self.config.thompson_above_incumbent_only
        )

        fp_row_by_batch_idx = {batch_idx: fp_row for fp_row, batch_idx in enumerate(valid_rows.tolist())}

        def _acquisition_key(idx: int) -> tuple[float, int]:
            fp_row = fp_row_by_batch_idx.get(int(idx))
            value = None if fp_row is None else acquisition_valid[fp_row]
            return (float("-inf") if value is None else float(value), -int(idx))

        ordered = sorted(unobserved_rows, key=_acquisition_key, reverse=True)
        ordered.extend(sorted(observed_rows, key=_acquisition_key, reverse=True))
        eligible_rows = list(ordered)
        if thompson_gate_active and self._acquisition == "thompson":
            eligible_rows = [
                idx
                for idx in eligible_rows
                if fp_row_by_batch_idx.get(int(idx)) is not None
                and float(acquisition_valid[fp_row_by_batch_idx[int(idx)]]) > incumbent_score
            ]
        if ei_threshold is not None:
            eligible_rows = [
                idx
                for idx in eligible_rows
                if expected_improvement[idx] is not None and float(expected_improvement[idx]) > ei_threshold
            ]
        if thompson_gate_active and self._acquisition == "thompson":
            num_candidates_above_threshold = len(eligible_rows)
        elif ei_threshold is not None:
            num_candidates_above_threshold = len(eligible_rows)

        selected: list[int] = []
        use_batched_ei = (
            self._acquisition
            in {
                "batched_expected_improvement",
                "weighted_batched_expected_improvement",
                "explore_then_batched_expected_improvement",
            }
            and budget > 0
            and len(eligible_rows) > 0
            and (mixed_explore_then_qei or (budget > 1 and len(eligible_rows) > 1))
        )
        use_slotwise_thompson = self._acquisition == "slotwise_thompson" and budget > 1 and len(eligible_rows) > 1
        use_local_penalization = (
            self._local_penalization
            and not use_batched_ei
            and not use_slotwise_thompson
            and budget > 1
            and len(eligible_rows) > 1
        )
        if use_batched_ei:
            remaining_unobserved = [int(idx) for idx in eligible_rows if int(idx) in set(unobserved_rows)]
            remaining_observed = [int(idx) for idx in eligible_rows if int(idx) not in set(unobserved_rows)]
            while len(selected) < budget and (remaining_unobserved or remaining_observed):
                pool = remaining_unobserved if remaining_unobserved else remaining_observed
                if not pool:
                    break
                if mixed_explore_then_qei and len(selected) < explore_slots:
                    chosen_idx = max(
                        pool,
                        key=lambda idx: (
                            float(std_valid[fp_row_by_batch_idx[int(idx)]]),
                            _acquisition_key(int(idx))[0],
                            -int(idx),
                        ),
                    )
                    selected.append(chosen_idx)
                    if chosen_idx in remaining_unobserved:
                        remaining_unobserved.remove(chosen_idx)
                    if chosen_idx in remaining_observed:
                        remaining_observed.remove(chosen_idx)
                    continue
                qei_inputs: list[np.ndarray] = []
                candidate_ids: list[int] = []
                for idx in pool:
                    fp_rows = [fp_row_by_batch_idx[int(chosen)] for chosen in selected]
                    fp_rows.append(fp_row_by_batch_idx[int(idx)])
                    qei_inputs.append(
                        np.stack(
                            [np.asarray(x_candidate_valid[fp_row], dtype=np.float64) for fp_row in fp_rows], axis=0
                        )
                    )
                    candidate_ids.append(int(idx))
                qei_values = np.asarray(
                    gp.q_expected_improvement(
                        np.stack(qei_inputs, axis=0),
                        best=incumbent_score,
                        num_samples=int(self.config.qei_mc_samples),
                    ),
                    dtype=np.float64,
                ).reshape(-1)
                qei_values = np.where(np.isfinite(qei_values), qei_values, float("-inf"))
                qei_priority = qei_values
                if self._acquisition == "weighted_batched_expected_improvement":
                    kernel_pool = np.asarray(
                        [
                            local_kernel_valid[fp_row_by_batch_idx[int(candidate_idx)]]
                            for candidate_idx in candidate_ids
                        ],
                        dtype=np.float64,
                    )
                    qei_priority = _weighted_log_score(qei_values, kernel_pool)
                best_pos = max(
                    range(len(candidate_ids)),
                    key=lambda pos: (
                        float(qei_priority[pos]),
                        _acquisition_key(candidate_ids[pos])[0],
                        -int(candidate_ids[pos]),
                    ),
                )
                chosen_idx = int(candidate_ids[best_pos])
                selected.append(chosen_idx)
                if chosen_idx in remaining_unobserved:
                    remaining_unobserved.remove(chosen_idx)
                if chosen_idx in remaining_observed:
                    remaining_observed.remove(chosen_idx)
        elif use_slotwise_thompson:
            remaining_unobserved = [int(idx) for idx in eligible_rows if int(idx) in set(unobserved_rows)]
            remaining_observed = [int(idx) for idx in eligible_rows if int(idx) not in set(unobserved_rows)]
            while len(selected) < budget and (remaining_unobserved or remaining_observed):
                pool = remaining_unobserved if remaining_unobserved else remaining_observed
                if not pool:
                    break
                slot_sample = np.asarray(
                    gp.sample_posterior(x_candidate_valid, num_samples=1),
                    dtype=np.float64,
                ).reshape(-1)
                best_idx: Optional[int] = None
                best_key: tuple[float, int] = (float("-inf"), 0)
                for idx in pool:
                    fp_row = fp_row_by_batch_idx.get(int(idx))
                    if fp_row is None:
                        continue
                    value = float(slot_sample[fp_row])
                    if thompson_gate_active and value <= incumbent_score:
                        continue
                    key = (value, -int(idx))
                    if best_idx is None or key > best_key:
                        best_idx = int(idx)
                        best_key = key
                if best_idx is None:
                    if remaining_unobserved:
                        remaining_unobserved.clear()
                        continue
                    break
                selected.append(best_idx)
                fp_row = fp_row_by_batch_idx.get(best_idx)
                if fp_row is not None:
                    acquisition_valid[fp_row] = float(best_key[0])
                    acquisition_value[best_idx] = float(best_key[0])
                if best_idx in remaining_unobserved:
                    remaining_unobserved.remove(best_idx)
                if best_idx in remaining_observed:
                    remaining_observed.remove(best_idx)
            if thompson_gate_active:
                num_candidates_above_threshold = len(selected)
        elif use_local_penalization:
            acquisition_priority = np.asarray(acquisition_valid, dtype=np.float64).reshape(-1)
            if acquisition_priority.size > 0 and np.min(acquisition_priority) < 0.0:
                acquisition_priority = acquisition_priority - float(np.min(acquisition_priority))
            priority_scale = float(np.max(acquisition_priority)) if acquisition_priority.size > 0 else 0.0
            if not math.isfinite(priority_scale) or priority_scale <= 0.0:
                acquisition_priority = np.ones_like(acquisition_priority, dtype=np.float64)
            else:
                acquisition_priority = acquisition_priority / priority_scale
            penalized_priority = {
                int(idx): float(acquisition_priority[fp_row_by_batch_idx[int(idx)]]) for idx in eligible_rows
            }
            unobserved_set = set(unobserved_rows)
            remaining = list(eligible_rows)
            while remaining and len(selected) < budget:
                best_idx = max(
                    remaining,
                    key=lambda idx: (
                        1 if int(idx) in unobserved_set else 0,
                        penalized_priority.get(int(idx), float("-inf")),
                        _acquisition_key(int(idx))[0],
                        -int(idx),
                    ),
                )
                selected.append(int(best_idx))
                remaining.remove(best_idx)
                chosen_fp = np.asarray(
                    x_candidate_valid[fp_row_by_batch_idx[int(best_idx)]],
                    dtype=np.float64,
                )
                for idx in remaining:
                    fp_row = fp_row_by_batch_idx[int(idx)]
                    similarity = self._feature_similarity(
                        np.asarray(x_candidate_valid[fp_row], dtype=np.float64),
                        chosen_fp,
                    )
                    factor = _local_penalization_factor(
                        similarity,
                        strength=self._local_penalization_strength,
                        power=self._local_penalization_power,
                    )
                    penalized_priority[int(idx)] = float(penalized_priority[int(idx)] * factor)
        else:
            for idx in eligible_rows:
                selected.append(idx)
                if len(selected) >= budget:
                    break
        gate_active = thompson_gate_active or ei_threshold is not None
        if not gate_active and len(selected) < budget:
            for idx in fallback:
                if idx in selected:
                    continue
                selected.append(idx)
                if len(selected) >= budget:
                    break

        return ActiveLearningSelection(
            selected_indices=selected,
            smiles=smiles,
            surrogate_mean=surrogate_mean,
            surrogate_std=surrogate_std,
            expected_improvement=expected_improvement,
            acquisition_value=acquisition_value,
            local_kernel_log_prob=local_kernel_log_prob,
            max_expected_improvement=max_expected_improvement,
            num_candidates_above_threshold=num_candidates_above_threshold,
            threshold_exhausted=(gate_active and len(selected) == 0),
        )


class OneHotTanimotoExpectedImprovementSelector(SequenceExpectedImprovementSelector):
    """Select a fixed oracle batch using a Tanimoto GP over sequence one-hots."""

    def __init__(
        self,
        *,
        reward_fn: Any,
        config: ActiveLearningConfig,
    ) -> None:
        if config.onehot_max_len is None:
            raise ValueError("onehot_tanimoto active learning requires onehot_max_len")
        config.onehot_max_len = int(config.onehot_max_len)
        if config.onehot_max_len <= 0:
            raise ValueError("active_learning onehot_max_len must be positive")
        config.onehot_alphabet = _normalize_onehot_alphabet(config.onehot_alphabet)
        if config.gp_outputscale is None:
            config.gp_outputscale = 1.0
        config.gp_learn_outputscale = False
        super().__init__(reward_fn=reward_fn, config=config)

    def _make_fingerprint_features(
        self,
        smiles: list[Optional[str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        return make_sequence_onehot_features(
            smiles,
            max_len=int(self.config.onehot_max_len or 0),
            alphabet=str(self.config.onehot_alphabet),
        )


def _categorical_display_alphabet(vocab_size: int) -> str:
    """Return a deterministic one-character alphabet for categorical tokens."""

    vocab_size = int(vocab_size)
    if vocab_size < 2:
        raise ValueError("categorical vocab_size must be at least 2")
    readable = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&()*+,-./:;<=>?@[]^_{|}~"
    if vocab_size <= len(readable):
        return readable[:vocab_size]
    # Private-use code points are stable under ``str.upper`` and JSON encoding.
    extra = "".join(chr(0xE000 + idx) for idx in range(vocab_size - len(readable)))
    return readable + extra


class _EhrlichCategoricalDecoder:
    """Selector-local, lossless token display adapter.

    This deliberately does not add ``decode_batch`` to ``EhrlichReward``:
    the evaluator feature-detects that method, so doing so would change legacy
    non-active-learning oracle accounting before golden replays are captured.
    """

    def __init__(
        self,
        reward_fn: Any,
        *,
        alphabet: str,
        enforce_transition_feasibility: bool = True,
    ) -> None:
        instance = getattr(reward_fn, "instance", None)
        if instance is None:
            raise TypeError("Ehrlich active learning requires reward_fn.instance")
        self.instance = instance
        self.alphabet = str(alphabet)
        self.max_len = int(instance.cfg.seq_len)
        self.vocab_size = int(instance.cfg.vocab_size)
        self.enforce_transition_feasibility = bool(enforce_transition_feasibility)

    def decode_batch(self, batch: torch.Tensor) -> list[Optional[str]]:
        if batch.ndim == 1:
            batch = batch.view(1, -1)
        if batch.ndim != 2 or int(batch.shape[1]) != self.max_len:
            raise ValueError(f"Ehrlich categorical sequences must have shape [B, {self.max_len}]")
        if batch.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("Ehrlich categorical sequences must use an integer dtype")
        if bool(((batch < 0) | (batch >= self.vocab_size)).any().item()):
            raise ValueError(f"Ehrlich tokens must be in [0, {self.vocab_size})")
        rows = batch.detach().to(device="cpu", dtype=torch.long)
        decoded = ["".join(self.alphabet[int(token)] for token in row.tolist()) for row in rows]
        if not self.enforce_transition_feasibility:
            return decoded

        from ehrlich.ehrlich_score import is_feasible

        feasible = torch.as_tensor(
            is_feasible(batch, self.instance.transition_mask.to(batch.device)),
            dtype=torch.bool,
        ).view(-1)
        return [value if bool(feasible[idx].item()) else None for idx, value in enumerate(decoded)]


class EhrlichOneHotExpectedImprovementSelector(OneHotTanimotoExpectedImprovementSelector):
    """Tanimoto-GP baseline over raw Ehrlich categorical one-hot features."""

    def __init__(self, *, reward_fn: Any, config: ActiveLearningConfig) -> None:
        instance = getattr(reward_fn, "instance", None)
        if instance is None:
            raise TypeError("Ehrlich active learning requires reward_fn.instance")
        self.ehrlich_reward_fn = reward_fn
        self.ehrlich_instance = instance
        alphabet = _categorical_display_alphabet(int(instance.cfg.vocab_size))
        config.onehot_max_len = int(instance.cfg.seq_len)
        config.onehot_alphabet = alphabet
        if config.gp_fit_mean_and_noise_only:
            if config.gp_noise_variance is not None:
                raise ValueError("gp_fit_mean_and_noise_only requires fitted likelihood noise")
            config.gp_outputscale = 1.0
            config.gp_learn_outputscale = False
        super().__init__(
            reward_fn=_EhrlichCategoricalDecoder(
                reward_fn,
                alphabet=alphabet,
                enforce_transition_feasibility=not bool(config.ehrlich_black_box_feasibility),
            ),
            config=config,
        )

    def _make_gp_regressor(self) -> TanimotoGPRegressor:
        return TanimotoGPRegressor(
            device=str(self.config.gp_device),
            dtype=str(self.config.gp_dtype),
            jitter=self.config.gp_jitter,
            noise_variance=self.config.gp_noise_variance,
            outputscale=self.config.gp_outputscale,
            learn_outputscale=bool(self.config.gp_learn_outputscale),
            fit_mean_and_noise_only=bool(self.config.gp_fit_mean_and_noise_only),
        )

    def decode_candidate_batch(self, batch: torch.Tensor) -> list[str | None]:
        """Decode evaluator candidates while preserving Ehrlich feasibility semantics."""

        return list(self.reward_fn.decode_batch(batch))

    def observe_oracle_batch(self, batch: torch.Tensor, scores: torch.Tensor) -> None:
        """Observe paid invalid Ehrlich queries as zero-valued GP examples."""

        if not self.config.ehrlich_black_box_feasibility:
            super().observe_oracle_batch(batch, scores)
            return
        gp_scores = torch.as_tensor(scores).clone()
        gp_scores[torch.isneginf(gp_scores)] = 0.0
        super().observe_oracle_batch(batch, gp_scores)

    def _fallback_indices(self, sequences: list[Optional[str]], budget: int) -> list[int]:
        # The legacy decoder exposes transition-invalid rows as None. In
        # black-box mode every in-alphabet row is represented by a string.
        selected: list[int] = []
        seen: set[str] = set()
        for idx, value in enumerate(sequences):
            if value is None or value in seen:
                continue
            seen.add(value)
            selected.append(idx)
            if len(selected) >= budget:
                break
        return selected


class EhrlichDiffusionExpectedImprovementSelector(EhrlichOneHotExpectedImprovementSelector):
    """Active-learning selector using the Ehrlich transition-diffusion GP."""

    def __init__(
        self,
        *,
        reward_fn: Any,
        config: ActiveLearningConfig,
        tau: float = 1.0,
        combine: str = "mean",
        use_transition_weights: bool = True,
        normalized_laplacian: bool = True,
    ) -> None:
        self.diffusion_tau = float(tau)
        self.diffusion_combine = str(combine).strip().lower()
        self.diffusion_use_transition_weights = bool(use_transition_weights)
        self.diffusion_normalized_laplacian = bool(normalized_laplacian)
        strict_gp_policy = bool(config.gp_fit_mean_and_noise_only)
        requested_outputscale = config.gp_outputscale
        requested_learn_outputscale = bool(config.gp_learn_outputscale)
        super().__init__(reward_fn=reward_fn, config=config)
        if not strict_gp_policy:
            # Preserve the caller's normal scale policy outside the explicitly
            # scoped Ehrlich kernel study policy.
            self.config.gp_outputscale = requested_outputscale
            self.config.gp_learn_outputscale = requested_learn_outputscale

    def _make_gp_regressor(self) -> EhrlichDiffusionGPRegressor:
        return EhrlichDiffusionGPRegressor(
            instance=self.ehrlich_instance,
            seq_len=int(self.ehrlich_instance.cfg.seq_len),
            vocab_size=int(self.ehrlich_instance.cfg.vocab_size),
            tau=self.diffusion_tau,
            combine=self.diffusion_combine,
            use_transition_weights=self.diffusion_use_transition_weights,
            normalized_laplacian=self.diffusion_normalized_laplacian,
            device=str(self.config.gp_device),
            dtype=str(self.config.gp_dtype),
            jitter=self.config.gp_jitter,
            noise_variance=self.config.gp_noise_variance,
            outputscale=self.config.gp_outputscale,
            learn_outputscale=bool(self.config.gp_learn_outputscale),
            fit_mean_and_noise_only=bool(self.config.gp_fit_mean_and_noise_only),
        )


class EhrlichSpacedNGramExpectedImprovementSelector(EhrlichOneHotExpectedImprovementSelector):
    """Active-learning selector using sparse counts of spaced categorical n-grams."""

    def __init__(
        self,
        *,
        reward_fn: Any,
        config: ActiveLearningConfig,
        offsets: Any | None = None,
        normalize: bool = True,
    ) -> None:
        self.spaced_ngram_offsets = (
            torch.as_tensor(reward_fn.instance.spacings, dtype=torch.long)
            if offsets is None
            else torch.as_tensor(offsets, dtype=torch.long)
        )
        self.spaced_ngram_normalize = bool(normalize)
        strict_gp_policy = bool(config.gp_fit_mean_and_noise_only)
        requested_outputscale = config.gp_outputscale
        requested_learn_outputscale = bool(config.gp_learn_outputscale)
        super().__init__(reward_fn=reward_fn, config=config)
        if not strict_gp_policy:
            self.config.gp_outputscale = requested_outputscale
            self.config.gp_learn_outputscale = requested_learn_outputscale

    def _make_gp_regressor(self) -> SpacedNGramGPRegressor:
        return SpacedNGramGPRegressor(
            instance=self.ehrlich_instance,
            seq_len=int(self.ehrlich_instance.cfg.seq_len),
            vocab_size=int(self.ehrlich_instance.cfg.vocab_size),
            offsets=self.spaced_ngram_offsets,
            normalize=self.spaced_ngram_normalize,
            device=str(self.config.gp_device),
            dtype=str(self.config.gp_dtype),
            jitter=self.config.gp_jitter,
            noise_variance=self.config.gp_noise_variance,
            outputscale=self.config.gp_outputscale,
            learn_outputscale=bool(self.config.gp_learn_outputscale),
            fit_mean_and_noise_only=bool(self.config.gp_fit_mean_and_noise_only),
        )


class EhrlichAdditiveSpacedNGramExpectedImprovementSelector(EhrlichOneHotExpectedImprovementSelector):
    """Active-learning selector using a fixed sum of spaced n-gram kernels."""

    def __init__(
        self,
        *,
        reward_fn: Any,
        config: ActiveLearningConfig,
        offset_groups: list[Any] | tuple[Any, ...],
        weights: Any | None = None,
        normalize: bool = True,
    ) -> None:
        self.spaced_ngram_offset_groups = tuple(torch.as_tensor(group, dtype=torch.long) for group in offset_groups)
        self.spaced_ngram_weights = np.asarray(
            [1.0] * len(self.spaced_ngram_offset_groups) if weights is None else weights,
            dtype=np.float64,
        )
        self.spaced_ngram_weights = self.spaced_ngram_weights / self.spaced_ngram_weights.sum()
        self.spaced_ngram_normalize = bool(normalize)
        strict_gp_policy = bool(config.gp_fit_mean_and_noise_only)
        requested_outputscale = config.gp_outputscale
        requested_learn_outputscale = bool(config.gp_learn_outputscale)
        super().__init__(reward_fn=reward_fn, config=config)
        if not strict_gp_policy:
            self.config.gp_outputscale = requested_outputscale
            self.config.gp_learn_outputscale = requested_learn_outputscale

    def _make_gp_regressor(self) -> AdditiveSpacedNGramGPRegressor:
        return AdditiveSpacedNGramGPRegressor(
            instance=self.ehrlich_instance,
            seq_len=int(self.ehrlich_instance.cfg.seq_len),
            vocab_size=int(self.ehrlich_instance.cfg.vocab_size),
            offset_groups=list(self.spaced_ngram_offset_groups),
            weights=self.spaced_ngram_weights,
            normalize=self.spaced_ngram_normalize,
            device=str(self.config.gp_device),
            dtype=str(self.config.gp_dtype),
            jitter=self.config.gp_jitter,
            noise_variance=self.config.gp_noise_variance,
            outputscale=self.config.gp_outputscale,
            learn_outputscale=bool(self.config.gp_learn_outputscale),
            fit_mean_and_noise_only=bool(self.config.gp_fit_mean_and_noise_only),
        )


__all__ = [
    "ActiveLearningConfig",
    "ActiveLearningSelection",
    "ActiveLearningSimilarityDiagnostics",
    "DEFAULT_PROTEIN_ONEHOT_ALPHABET",
    "EhrlichDiffusionExpectedImprovementSelector",
    "EhrlichAdditiveSpacedNGramExpectedImprovementSelector",
    "EhrlichOneHotExpectedImprovementSelector",
    "EhrlichSpacedNGramExpectedImprovementSelector",
    "SequenceExpectedImprovementSelector",
    "OneHotTanimotoExpectedImprovementSelector",
    "make_sequence_onehot_features",
]
