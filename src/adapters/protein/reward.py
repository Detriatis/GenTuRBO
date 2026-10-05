"""Reward adapter for tokenized protein sequences."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from adapters.protein.fitness_benchmarks import KirjnerCNNFitnessOracle, OracleNormalizationStats
from adapters.protein.gfp_aav_benchmark import (
    get_oracle_checkpoint_path,
    load_fitness_stats,
)
from adapters.protein.validation import CANONICAL_AA, validate_sequence


class ProteinGFPAVCNNOracleReward:
    """Decode tokenized protein sequences and score them with the Kirjner/GGS CNN oracle."""

    def __init__(
        self,
        *,
        protein: str,
        benchmark_root: str | Path = "benchmarks/gfp_aav",
        model_path: str | Path | None = None,
        stats_path: str | Path | None = None,
        normalize: bool = True,
        decode_alphabet: str = CANONICAL_AA,
        oracle_alphabet: str = "ARNDCQEGHILKMFPSTWYV",
        invalid_score: float = float("-inf"),
        pad_token_id: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        min_len: int = 1,
        max_len: Optional[int] = None,
        sequence_length: Optional[int] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        protein_name = str(protein).upper()
        root = Path(benchmark_root)
        stats = load_fitness_stats(protein_name, root=root) if stats_path is None else _load_stats(Path(stats_path))
        y_min = float(stats["score_min"])
        y_max = float(stats["score_max"])
        seq_len = int(sequence_length or stats["sequence_length"])
        oracle_path = Path(model_path) if model_path is not None else get_oracle_checkpoint_path(protein_name, root=root)
        self.oracle = KirjnerCNNFitnessOracle(
            model_path=oracle_path,
            stats=OracleNormalizationStats(
                protein=protein_name,
                full_dms_size=int(stats["num_rows"]),
                y_min=y_min,
                y_max=y_max,
            ),
            sequence_length=seq_len,
            alphabet=str(oracle_alphabet),
            device=device,
        )
        self.protein = protein_name
        self.normalize = bool(normalize)
        self.decode_alphabet = str(decode_alphabet)
        self.invalid_score = float(invalid_score)
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.mask_token_id = None if mask_token_id is None else int(mask_token_id)
        self.min_len = int(min_len)
        self.max_len = int(max_len) if max_len is not None else seq_len
        self.sequence_length = seq_len
        self._score_cache: dict[str, float] = {}

    def _decode_row(self, row: Tensor) -> Optional[str]:
        residues: list[str] = []
        for raw in row.detach().cpu().view(-1).tolist():
            token = int(raw)
            if self.pad_token_id is not None and token == self.pad_token_id:
                continue
            if self.mask_token_id is not None and token == self.mask_token_id:
                return None
            if token < 0 or token >= len(self.decode_alphabet):
                return None
            residues.append(self.decode_alphabet[token])
        if not residues:
            return None
        seq = "".join(residues)
        try:
            return validate_sequence(seq, min_len=self.min_len, max_len=self.max_len, alphabet=self.decode_alphabet)
        except ValueError:
            return None

    def decode_batch(self, x: Tensor) -> list[Optional[str]]:
        if x.ndim == 1:
            return [self._decode_row(x)]
        if x.ndim != 2:
            raise ValueError(f"decode_batch expects [L] or [B, L], got shape {tuple(x.shape)}")
        return [self._decode_row(row) for row in x]

    def score_sequences(self, seqs: Sequence[Optional[str]]) -> list[float]:
        valid: list[str] = []
        valid_indices: list[int] = []
        scores = [self.invalid_score for _ in seqs]
        for idx, seq in enumerate(seqs):
            if seq is None:
                continue
            if seq in self._score_cache:
                scores[idx] = self._score_cache[seq]
                continue
            valid.append(seq)
            valid_indices.append(idx)
        if valid:
            raw_scores = self.oracle.normalized_fitness(valid) if self.normalize else self.oracle.predict(valid)
            for idx, seq, score in zip(valid_indices, valid, raw_scores):
                value = float(score)
                if not math.isfinite(value):
                    value = self.invalid_score
                self._score_cache[seq] = value
                scores[idx] = value
        return scores

    def score_batch(self, x: Tensor) -> Tensor:
        squeeze = x.ndim == 1
        seqs = self.decode_batch(x)
        scores = torch.as_tensor(self.score_sequences(seqs), dtype=torch.float, device=x.device)
        return scores[0] if squeeze else scores

    def __call__(self, x: Tensor) -> float:
        return float(self.score_batch(x).item())


def _load_stats(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def build_gfp_aav_cnn_oracle_reward(**cfg: Any) -> ProteinGFPAVCNNOracleReward:
    return ProteinGFPAVCNNOracleReward(**cfg)


__all__ = [
    "ProteinGFPAVCNNOracleReward",
    "build_gfp_aav_cnn_oracle_reward",
]
