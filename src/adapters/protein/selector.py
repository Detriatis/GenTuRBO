"""Active-learning selector using ESM embeddings for protein sequences."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np
import torch

from adapters.protein.esm import ESM2Featurizer
from adapters.protein.gp import EmbeddingGPRegressor
from adapters.protein.validation import validate_sequence
from search.interpolant.active_learning import (
    ActiveLearningBufferStats,
    ActiveLearningConfig,
    ActiveLearningSelection,
    ActiveLearningSimilarityDiagnostics,
    _expected_improvement,
    _local_penalization_factor,
    _normalize_acquisition,
    _zscore,
)


@dataclass(slots=True)
class _ProteinBufferEntry:
    seq: str
    sequence_key: bytes
    embedding: np.ndarray
    score: float
    seen_at: int


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_np = np.asarray(a, dtype=np.float64).reshape(-1)
    b_np = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a_np) * np.linalg.norm(b_np))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)


class ProteinEmbeddingExpectedImprovementSelector:
    """Select oracle batches with a GP trained on ESM sequence embeddings."""

    def __init__(
        self,
        *,
        reward_fn: Any,
        config: ActiveLearningConfig,
        featurizer: Optional[ESM2Featurizer] = None,
        gp_factory: Any = EmbeddingGPRegressor,
    ) -> None:
        decode_batch = getattr(reward_fn, "decode_batch", None)
        if not callable(decode_batch):
            raise TypeError("protein active learning requires reward_fn.decode_batch(...)")
        self.reward_fn = reward_fn
        self.config = config
        self.featurizer = featurizer or ESM2Featurizer(device=str(config.gp_device))
        self.gp_factory = gp_factory
        self._acquisition = _normalize_acquisition(config.acquisition)
        self._entries: dict[str, _ProteinBufferEntry] = {}
        self._score_cache: dict[bytes, float] = {}
        self._score_by_seq: dict[str, float] = {}
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._best_entry: Optional[_ProteinBufferEntry] = None
        self._seen_counter = 0
        self._last_buffer_stats = ActiveLearningBufferStats()

    @property
    def buffer_size(self) -> int:
        return len(self._entries)

    @property
    def last_buffer_stats(self) -> ActiveLearningBufferStats:
        return self._last_buffer_stats

    def _next_seen_at(self) -> int:
        self._seen_counter += 1
        return self._seen_counter

    def _valid_decoded(self, decoded: list[Optional[str]]) -> list[Optional[str]]:
        out: list[Optional[str]] = []
        for value in decoded:
            if value is None:
                out.append(None)
                continue
            try:
                out.append(validate_sequence(value))
            except (TypeError, ValueError):
                out.append(None)
        return out

    def _embeddings_for_valid(self, seqs: list[str]) -> np.ndarray:
        missing = [seq for seq in seqs if seq not in self._embedding_cache]
        if missing:
            z = self.featurizer.featurize(missing).detach().cpu().numpy().astype(np.float64)
            for seq, row in zip(missing, z):
                self._embedding_cache[seq] = np.asarray(row, dtype=np.float64)
        return np.stack([self._embedding_cache[seq] for seq in seqs], axis=0)

    def _make_embedding_features(self, seqs: list[Optional[str]]) -> tuple[np.ndarray, np.ndarray]:
        valid_mask = np.asarray([seq is not None for seq in seqs], dtype=bool)
        valid = [str(seq) for seq in seqs if seq is not None]
        if not valid:
            raise ValueError("No valid protein sequences were available for ESM embedding features")
        return self._embeddings_for_valid(valid), valid_mask

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
        keep: list[_ProteinBufferEntry] = []
        if self._best_entry is not None:
            keep.append(self._best_entry)
        for item in ordered:
            if len(keep) >= max_size:
                break
            if any(item.seq == kept.seq for kept in keep):
                continue
            keep.append(item)
        self._entries = {item.seq: item for item in keep}

    def observe_oracle_batch(self, batch: torch.Tensor, scores: torch.Tensor) -> None:
        if batch.ndim != 2:
            raise ValueError(f"observe_oracle_batch expects [B, L], got {tuple(batch.shape)}")
        score_values = torch.as_tensor(scores, dtype=torch.float).view(-1).detach().cpu().tolist()
        if len(score_values) != int(batch.shape[0]):
            raise ValueError("scores must align with batch rows")
        seqs = self._valid_decoded(self.reward_fn.decode_batch(batch))
        stats = ActiveLearningBufferStats(pre_size=len(self._entries), valid_rows=sum(seq is not None for seq in seqs))
        valid_rows = [idx for idx, seq in enumerate(seqs) if seq is not None]
        if not valid_rows:
            self._last_buffer_stats = stats
            return
        embeddings = self._embeddings_for_valid([str(seqs[idx]) for idx in valid_rows])
        batch_cpu = batch.detach().to("cpu").contiguous()
        for emb_row, batch_idx in enumerate(valid_rows):
            seq = str(seqs[batch_idx])
            score_value = float(score_values[batch_idx])
            if not math.isfinite(score_value):
                continue
            sequence_key = batch_cpu[batch_idx].numpy().tobytes()
            embedding = np.asarray(embeddings[emb_row], dtype=np.float64)
            entry = _ProteinBufferEntry(
                seq=seq,
                sequence_key=sequence_key,
                embedding=embedding,
                score=score_value,
                seen_at=self._next_seen_at(),
            )
            self._score_cache[sequence_key] = score_value
            self._score_by_seq[seq] = score_value
            if self._best_entry is None or score_value > float(self._best_entry.score):
                self._best_entry = entry
            existing = self._entries.get(seq)
            if existing is None:
                self._entries[seq] = entry
                stats.admitted += 1
            elif score_value > float(existing.score):
                self._entries[seq] = entry
                stats.replaced += 1
            else:
                stats.rejected_similar += 1
        self._prune()
        stats.post_size = len(self._entries)
        self._last_buffer_stats = stats

    def cached_scores(self, batch: torch.Tensor) -> list[Optional[float]]:
        if batch.ndim != 2:
            raise ValueError(f"cached_scores expects [B, L], got {tuple(batch.shape)}")
        seqs = self._valid_decoded(self.reward_fn.decode_batch(batch))
        cached: list[Optional[float]] = []
        batch_cpu = batch.detach().to("cpu").contiguous()
        for row_idx, row in enumerate(batch_cpu):
            key = row.numpy().tobytes()
            cached_score = self._score_cache.get(key)
            if cached_score is None and seqs[row_idx] is not None:
                cached_score = self._score_by_seq.get(str(seqs[row_idx]))
            cached.append(cached_score)
        return cached

    def similarity_diagnostics(
        self,
        smiles: list[Optional[str]],
        *,
        top_k: int = 5,
    ) -> list[ActiveLearningSimilarityDiagnostics]:
        seqs = self._valid_decoded(smiles)
        diagnostics = [ActiveLearningSimilarityDiagnostics() for _ in seqs]
        if not seqs or not self._entries:
            return diagnostics
        valid_rows = [idx for idx, seq in enumerate(seqs) if seq is not None]
        if not valid_rows:
            return diagnostics
        try:
            embeddings = self._embeddings_for_valid([str(seqs[idx]) for idx in valid_rows])
        except ValueError:
            return diagnostics
        entries = list(self._entries.values())
        k = max(1, min(int(top_k), len(entries)))
        for emb_row, batch_idx in enumerate(valid_rows):
            seq = str(seqs[batch_idx])
            similarities = sorted(
                (_cosine_similarity(embeddings[emb_row], entry.embedding) for entry in entries),
                reverse=True,
            )
            diagnostics[batch_idx] = ActiveLearningSimilarityDiagnostics(
                in_buffer=(seq in self._entries),
                max_similarity=float(similarities[0]) if similarities else None,
                mean_topk_similarity=float(np.mean(similarities[:k])) if similarities else None,
            )
        return diagnostics

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        entries = list(self._entries.values())
        if not entries:
            raise ValueError("active learning buffer is empty")
        x = np.stack([entry.embedding for entry in entries], axis=0)
        y = np.asarray([entry.score for entry in entries], dtype=np.float64)
        return x, y

    def _fallback_indices(self, seqs: list[Optional[str]], budget: int) -> list[int]:
        selected: list[int] = []
        seen: set[str] = set()
        for idx, value in enumerate(seqs):
            if value is None or value in seen:
                continue
            seen.add(str(value))
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
        select_budget = int(self.config.oracle_batch_size) if budget is None else int(budget)
        budget = max(0, min(select_budget, num_rows))
        seqs = self._valid_decoded(self.reward_fn.decode_batch(batch))
        surrogate_mean: list[Optional[float]] = [None] * num_rows
        surrogate_std: list[Optional[float]] = [None] * num_rows
        expected_improvement: list[Optional[float]] = [None] * num_rows
        acquisition_value: list[Optional[float]] = [None] * num_rows
        local_kernel_log_prob: list[Optional[float]] = [None] * num_rows
        local_kernel_array = np.zeros((num_rows,), dtype=np.float64)
        if local_kernel_log_probs is not None:
            raw_kernel = (
                local_kernel_log_probs.detach().cpu().view(-1).tolist()
                if isinstance(local_kernel_log_probs, torch.Tensor)
                else np.asarray(local_kernel_log_probs, dtype=object).reshape(-1).tolist()
            )
            if len(raw_kernel) != num_rows:
                raise ValueError("local_kernel_log_probs must align with batch rows")
            for idx, value in enumerate(raw_kernel):
                if value is None:
                    continue
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(val):
                    local_kernel_log_prob[idx] = val
                    local_kernel_array[idx] = val

        fallback = self._fallback_indices(seqs, budget)
        try:
            x_candidate_valid, valid_mask = self._make_embedding_features(seqs)
        except ValueError:
            return ActiveLearningSelection(
                selected_indices=fallback,
                smiles=seqs,
                surrogate_mean=surrogate_mean,
                surrogate_std=surrogate_std,
                expected_improvement=expected_improvement,
                acquisition_value=acquisition_value,
                local_kernel_log_prob=local_kernel_log_prob,
            )
        valid_rows = np.flatnonzero(valid_mask)
        representative_rows: list[int] = []
        seen_seqs: set[str] = set()
        for idx in valid_rows.tolist():
            seq = str(seqs[idx])
            if seq in seen_seqs:
                continue
            seen_seqs.add(seq)
            representative_rows.append(int(idx))
        observed = set(self._entries.keys())
        unobserved_rows = [idx for idx in representative_rows if str(seqs[idx]) not in observed]
        observed_rows = [idx for idx in representative_rows if str(seqs[idx]) in observed]

        if self._acquisition == "random":
            eligible = list(unobserved_rows) + list(observed_rows)
            if not eligible:
                eligible = fallback
            order = torch.randperm(len(eligible)).detach().cpu().tolist() if eligible and budget > 0 else []
            selected = [int(eligible[int(pos)]) for pos in order[:budget]]
            return ActiveLearningSelection(
                selected_indices=selected,
                smiles=seqs,
                surrogate_mean=surrogate_mean,
                surrogate_std=surrogate_std,
                expected_improvement=expected_improvement,
                acquisition_value=acquisition_value,
                local_kernel_log_prob=local_kernel_log_prob,
            )

        if self.buffer_size < max(1, int(self.config.min_train_size)):
            return ActiveLearningSelection(
                selected_indices=fallback,
                smiles=seqs,
                surrogate_mean=surrogate_mean,
                surrogate_std=surrogate_std,
                expected_improvement=expected_improvement,
                acquisition_value=acquisition_value,
                local_kernel_log_prob=local_kernel_log_prob,
            )

        x_train, y_train = self._training_arrays()
        gp = self.gp_factory(
            device=str(self.config.gp_device),
            dtype=str(self.config.gp_dtype),
            jitter=self.config.gp_jitter,
            noise_variance=self.config.gp_noise_variance,
            outputscale=self.config.gp_outputscale,
            learn_outputscale=bool(self.config.gp_learn_outputscale),
            lengthscale=self.config.gp_lengthscale,
            learn_lengthscale=bool(self.config.gp_learn_lengthscale),
        )
        gp.fit(x_train, y_train)
        mean_valid = gp.predict(x_candidate_valid)
        std_valid = gp.predict_std(x_candidate_valid)
        incumbent = float(self._best_entry.score) if self._best_entry is not None else float(np.max(y_train))
        ei_valid = _expected_improvement(mean_valid, std_valid, best=incumbent, xi=float(self.config.ei_xi))
        use_weighted = self._acquisition in {"weighted_expected_improvement", "weighted_batched_expected_improvement"}
        if self._acquisition == "thompson":
            acquisition_valid = np.asarray(gp.sample_posterior(x_candidate_valid, num_samples=1), dtype=np.float64).reshape(-1)
        else:
            acquisition_valid = np.asarray(ei_valid, dtype=np.float64).reshape(-1)
        local_kernel_valid = np.asarray([local_kernel_array[int(idx)] for idx in valid_rows.tolist()], dtype=np.float64)

        def _weighted_log_score(values: np.ndarray, kernel_values: np.ndarray) -> np.ndarray:
            eps = float(self.config.wei_eps)
            base = np.log(np.clip(np.asarray(values, dtype=np.float64), eps, None))
            kernel = np.where(np.isfinite(kernel_values), kernel_values, 0.0)
            if bool(self.config.wei_standardize):
                return _zscore(base) + float(self.config.wei_lambda) * _zscore(kernel)
            return base + float(self.config.wei_lambda) * kernel

        def _weighted_from_log_score(log_values: np.ndarray, kernel_values: np.ndarray) -> np.ndarray:
            base = np.asarray(log_values, dtype=np.float64)
            kernel = np.where(np.isfinite(kernel_values), kernel_values, 0.0)
            if bool(self.config.wei_standardize):
                return _zscore(base) + float(self.config.wei_lambda) * _zscore(kernel)
            return base + float(self.config.wei_lambda) * kernel

        if use_weighted:
            acquisition_valid = _weighted_log_score(acquisition_valid, local_kernel_valid)

        fp_row_by_batch_idx = {int(batch_idx): int(row_idx) for row_idx, batch_idx in enumerate(valid_rows.tolist())}
        for row_idx, batch_idx in enumerate(valid_rows.tolist()):
            surrogate_mean[int(batch_idx)] = float(mean_valid[row_idx])
            surrogate_std[int(batch_idx)] = float(std_valid[row_idx])
            expected_improvement[int(batch_idx)] = float(ei_valid[row_idx])
            acquisition_value[int(batch_idx)] = float(acquisition_valid[row_idx])

        max_expected_improvement = float(np.max(ei_valid)) if ei_valid.size > 0 else None
        ei_threshold = float(self.config.ei_threshold) if self.config.ei_threshold is not None else None
        if ei_threshold is not None and ei_threshold <= 0.0:
            ei_threshold = None

        def _acq_key(idx: int) -> tuple[float, int]:
            row = fp_row_by_batch_idx.get(int(idx))
            value = float("-inf") if row is None else float(acquisition_valid[row])
            return value, -int(idx)

        eligible_rows = sorted(unobserved_rows, key=_acq_key, reverse=True)
        eligible_rows.extend(sorted(observed_rows, key=_acq_key, reverse=True))
        thompson_gate_active = (
            self._acquisition in {"thompson", "slotwise_thompson"}
            and bool(self.config.thompson_above_incumbent_only)
        )
        if thompson_gate_active and self._acquisition == "thompson":
            eligible_rows = [
                idx
                for idx in eligible_rows
                if fp_row_by_batch_idx.get(int(idx)) is not None
                and float(acquisition_valid[fp_row_by_batch_idx[int(idx)]]) > incumbent
            ]
        if ei_threshold is not None:
            eligible_rows = [
                idx
                for idx in eligible_rows
                if expected_improvement[idx] is not None and float(expected_improvement[idx]) > ei_threshold
            ]
        num_above_threshold = len(eligible_rows) if (ei_threshold is not None or thompson_gate_active) else 0
        selected: list[int] = []
        use_batched_ei = (
            self._acquisition in {"batched_expected_improvement", "weighted_batched_expected_improvement"}
            and budget > 1
            and len(eligible_rows) > 1
        )
        use_slotwise_thompson = self._acquisition == "slotwise_thompson" and budget > 1 and len(eligible_rows) > 1
        if use_batched_ei:
            remaining = list(eligible_rows)
            while remaining and len(selected) < budget:
                qei_inputs = []
                candidate_ids = []
                for idx in remaining:
                    rows = [fp_row_by_batch_idx[int(chosen)] for chosen in selected]
                    rows.append(fp_row_by_batch_idx[int(idx)])
                    qei_inputs.append(np.stack([x_candidate_valid[row] for row in rows], axis=0))
                    candidate_ids.append(int(idx))
                qei_batch = np.stack(qei_inputs, axis=0)
                q_log_ei_fn = getattr(gp, "q_log_expected_improvement", None)
                if callable(q_log_ei_fn):
                    priority = np.asarray(
                        q_log_ei_fn(
                            qei_batch,
                            best=incumbent,
                            num_samples=int(self.config.qei_mc_samples),
                        ),
                        dtype=np.float64,
                    )
                    qei_is_log = True
                else:
                    priority = np.asarray(
                        gp.q_expected_improvement(
                            qei_batch,
                            best=incumbent,
                            num_samples=int(self.config.qei_mc_samples),
                        ),
                        dtype=np.float64,
                    )
                    qei_is_log = False
                if self._acquisition == "weighted_batched_expected_improvement":
                    kernel_pool = np.asarray([local_kernel_valid[fp_row_by_batch_idx[idx]] for idx in candidate_ids])
                    priority = (
                        _weighted_from_log_score(priority, kernel_pool)
                        if qei_is_log
                        else _weighted_log_score(priority, kernel_pool)
                    )
                best_pos = max(range(len(candidate_ids)), key=lambda pos: (float(priority[pos]), _acq_key(candidate_ids[pos])[0]))
                chosen = int(candidate_ids[best_pos])
                selected.append(chosen)
                remaining.remove(chosen)
        elif use_slotwise_thompson:
            remaining = list(eligible_rows)
            while remaining and len(selected) < budget:
                slot_sample = np.asarray(gp.sample_posterior(x_candidate_valid, num_samples=1), dtype=np.float64).reshape(-1)
                best_idx: Optional[int] = None
                best_key: tuple[float, int] = (float("-inf"), 0)
                for idx in remaining:
                    row = fp_row_by_batch_idx.get(int(idx))
                    if row is None:
                        continue
                    value = float(slot_sample[row])
                    if thompson_gate_active and value <= incumbent:
                        continue
                    key = (value, -int(idx))
                    if best_idx is None or key > best_key:
                        best_idx = int(idx)
                        best_key = key
                if best_idx is None:
                    break
                selected.append(best_idx)
                row = fp_row_by_batch_idx.get(best_idx)
                if row is not None:
                    acquisition_valid[row] = float(best_key[0])
                    acquisition_value[best_idx] = float(best_key[0])
                remaining.remove(best_idx)
            if thompson_gate_active:
                num_above_threshold = len(selected)
        elif bool(self.config.local_penalization) and budget > 1 and len(eligible_rows) > 1:
            remaining = list(eligible_rows)
            priority = {int(idx): _acq_key(int(idx))[0] for idx in remaining}
            while remaining and len(selected) < budget:
                chosen = max(remaining, key=lambda idx: (priority.get(int(idx), float("-inf")), -int(idx)))
                selected.append(int(chosen))
                remaining.remove(chosen)
                chosen_row = fp_row_by_batch_idx[int(chosen)]
                chosen_emb = x_candidate_valid[chosen_row]
                for idx in remaining:
                    row = fp_row_by_batch_idx[int(idx)]
                    sim = max(0.0, _cosine_similarity(x_candidate_valid[row], chosen_emb))
                    priority[int(idx)] *= _local_penalization_factor(
                        sim,
                        strength=float(self.config.local_penalization_strength),
                        power=float(self.config.local_penalization_power),
                    )
        else:
            selected = [int(idx) for idx in eligible_rows[:budget]]

        gate_active = ei_threshold is not None or thompson_gate_active
        if not gate_active and len(selected) < budget:
            for idx in fallback:
                if idx in selected:
                    continue
                selected.append(idx)
                if len(selected) >= budget:
                    break

        return ActiveLearningSelection(
            selected_indices=selected,
            smiles=seqs,
            surrogate_mean=surrogate_mean,
            surrogate_std=surrogate_std,
            expected_improvement=expected_improvement,
            acquisition_value=acquisition_value,
            local_kernel_log_prob=local_kernel_log_prob,
            max_expected_improvement=max_expected_improvement,
            num_candidates_above_threshold=num_above_threshold,
            threshold_exhausted=(gate_active and len(selected) == 0),
        )


__all__ = ["ProteinEmbeddingExpectedImprovementSelector"]
