"""Kermut's structure + sequence covariance for fixed-length substitutions.

Implements Groth et al., NeurIPS 2024, Eqs. (1)--(4):
https://arxiv.org/abs/2407.00002 . The GP retains GenTuRBO's constant mean and
likelihood policy; this is a kernel comparison, not the zero-shot-mean model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel, RBFKernel

from adapters.protein.esm import ESM2Featurizer
from adapters.protein.validation import CANONICAL_AA, validate_sequence
from search.interpolant.active_learning import ActiveLearningConfig, SequenceExpectedImprovementSelector
from surrogate.features import TanimotoGPRegressor


@dataclass(frozen=True)
class KermutResources:
    """Reference-aligned ProteinMPNN probabilities and C-alpha coordinates.

    ``alphabet`` records the column order of ``conditional_probs``. Rows of
    both arrays must correspond exactly to ``wild_type`` (not PDB numbering).
    Coordinates are in Angstrom. No missing residues or indels are supported.
    """

    wild_type: str
    alphabet: str
    conditional_probs: np.ndarray
    coords: np.ndarray
    embedding_prefix: str = ""
    embedding_suffix: str = ""

    def __post_init__(self) -> None:
        wild_type = validate_sequence(self.wild_type)
        embedding_prefix = validate_sequence(self.embedding_prefix, min_len=0)
        embedding_suffix = validate_sequence(self.embedding_suffix, min_len=0)
        # ESM sees the full construct, whereas the structural rows remain
        # aligned only to the fixed-length optimisation window.
        validate_sequence(embedding_prefix + wild_type + embedding_suffix)
        alphabet = str(self.alphabet).strip().upper()
        if len(alphabet) != 20 or set(alphabet) != set(CANONICAL_AA):
            raise ValueError("Kermut alphabet must contain each of the 20 canonical amino acids exactly once")
        probs = np.array(self.conditional_probs, dtype=np.float64, copy=True)
        coords = np.array(self.coords, dtype=np.float64, copy=True)
        if probs.shape != (len(wild_type), 20):
            raise ValueError("Kermut conditional_probs must have shape [len(wild_type), 20]")
        if not np.isfinite(probs).all() or np.any(probs <= 0) or np.any(probs > 1):
            raise ValueError("Kermut conditional_probs must be finite probabilities in (0, 1]")
        if coords.shape != (len(wild_type), 3) or not np.isfinite(coords).all():
            raise ValueError("Kermut coords must be finite C-alpha coordinates with shape [len(wild_type), 3]")
        # Upstream averages log-probabilities before exponentiating and drops
        # the noncanonical X column; its processed rows need not sum to one.
        object.__setattr__(self, "wild_type", wild_type)
        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "conditional_probs", probs)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "embedding_prefix", embedding_prefix)
        object.__setattr__(self, "embedding_suffix", embedding_suffix)

    @classmethod
    def load(cls, path: str | Path, *, expected_length: int | None = None) -> KermutResources:
        """Validate a non-pickled NPZ without loading ESM or evaluating an oracle."""
        with np.load(path, allow_pickle=False) as data:
            required = {"wild_type", "alphabet", "conditional_probs", "coords"}
            if not required.issubset(data.files):
                raise ValueError(f"Kermut resource NPZ requires keys {sorted(required)}")
            result = cls(
                wild_type=str(data["wild_type"].item()),
                alphabet=str(data["alphabet"].item()),
                conditional_probs=data["conditional_probs"],
                coords=data["coords"],
                embedding_prefix=(
                    str(data["embedding_prefix"].item()) if "embedding_prefix" in data.files else ""
                ),
                embedding_suffix=(
                    str(data["embedding_suffix"].item()) if "embedding_suffix" in data.files else ""
                ),
            )
        if expected_length is not None and len(result.wild_type) != int(expected_length):
            raise ValueError(
                f"Kermut wild_type length {len(result.wild_type)} does not match task length {expected_length}"
            )
        return result


class KermutKernel(Kernel):
    """Weighted structure + RBF kernel on packed ``[residue IDs, ESM]`` rows.

    The structure term sums over mutations relative to the reference, including
    all within-variant cross terms. Wild type has zero structure covariance.
    Unique mutation types and sparse incidence matrices avoid constructing a
    quadratic matrix over every mutation occurrence in a 2,000-label archive.
    """

    def __init__(self, resources: KermutResources) -> None:
        super().__init__()
        self.seq_len = len(resources.wild_type)
        wt = [resources.alphabet.index(aa) for aa in resources.wild_type]
        probs = torch.tensor(resources.conditional_probs, dtype=torch.double)
        coords = torch.tensor(resources.coords, dtype=torch.double)
        self.register_buffer("wild_type_tokens", torch.tensor(wt, dtype=torch.long))
        self.register_buffer("log_probs", probs.log().reshape(-1))
        self.register_buffer(
            "hellinger",
            torch.cdist(probs.sqrt(), probs.sqrt(), compute_mode="donot_use_mm_for_euclid_dist") / (2.0**0.5),
        )
        self.register_buffer("distances", torch.cdist(coords, coords, compute_mode="donot_use_mm_for_euclid_dist"))
        self.register_parameter("raw_decays", torch.nn.Parameter(torch.zeros(3, dtype=torch.double)))
        self.register_constraint("raw_decays", Positive())
        self.initialize(raw_decays=self.raw_decays_constraint.inverse_transform(torch.ones(3, dtype=torch.double)))
        self.register_parameter("raw_mixture", torch.nn.Parameter(torch.zeros((), dtype=torch.double)))
        self.register_parameter("raw_structure_scale", torch.nn.Parameter(torch.zeros((), dtype=torch.double)))
        self.register_constraint("raw_structure_scale", Positive())
        self.initialize(raw_structure_scale=self.raw_structure_scale_constraint.inverse_transform(torch.tensor(1.0)))
        self.sequence_kernel = RBFKernel()
        self.sequence_kernel.lengthscale = 1.0

    @property
    def decays(self) -> torch.Tensor:
        return self.raw_decays_constraint.transform(self.raw_decays)

    @property
    def mixture(self) -> torch.Tensor:
        return self.raw_mixture.sigmoid()

    @property
    def structure_scale(self) -> torch.Tensor:
        return self.raw_structure_scale_constraint.transform(self.raw_structure_scale)

    def _mutation_incidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x[..., : self.seq_len].long()
        rows, sites = torch.where(tokens != self.wild_type_tokens)
        mutation_ids = sites * 20 + tokens[rows, sites]
        unique_ids, inverse = mutation_ids.unique(sorted=True, return_inverse=True)
        incidence = torch.sparse_coo_tensor(
            torch.stack((rows, inverse)),
            torch.ones(rows.shape, dtype=x.dtype, device=x.device),
            (x.shape[0], unique_ids.numel()),
            check_invariants=False,
        ).coalesce()
        return unique_ids, incidence

    def _structure_matrix(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        ids1, incidence1 = self._mutation_incidence(x1)
        ids2, incidence2 = self._mutation_incidence(x2)
        if not ids1.numel() or not ids2.numel():
            return x1.new_zeros(x1.shape[0], x2.shape[0])
        sites1, sites2 = ids1 // 20, ids2 // 20
        h, p, d = self.decays.unbind()
        # Positive products of exponential distance kernels, then additive
        # lifting via incidence1 @ mutation_kernel @ incidence2.T.
        exponent = h * self.hellinger[sites1[:, None], sites2[None, :]]
        exponent = exponent + d * self.distances[sites1[:, None], sites2[None, :]]
        exponent = exponent + p * (self.log_probs[ids1, None] - self.log_probs[None, ids2]).abs()
        mutation_kernel = (-exponent).exp()
        left = torch.sparse.mm(incidence1, mutation_kernel)
        return torch.sparse.mm(incidence2, left.transpose(-1, -2)).transpose(-1, -2)

    def _padded_mutations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = x[:, : self.seq_len].long()
        mutated = tokens != self.wild_type_tokens
        counts = mutated.sum(dim=-1)
        rows, sites = torch.where(mutated)
        width = int(counts.max().item())
        ids = tokens.new_zeros((tokens.shape[0], width))
        offsets = counts.cumsum(0) - counts
        slots = torch.arange(rows.numel(), device=x.device) - offsets[rows]
        ids[rows, slots] = sites * 20 + tokens[rows, sites]
        valid = torch.arange(width, device=x.device)[None, :] < counts[:, None]
        return ids, valid

    def _structure_diagonal(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Sum mutation pairs only within aligned rows, never across N rows."""
        if x1.shape[0] == 0:
            return x1.new_empty((0,))
        h, p, d = self.decays.unbind()
        chunks: list[torch.Tensor] = []
        # Bound padded mutation-pair memory even for heavily mutated variants.
        for start in range(0, x1.shape[0], 128):
            ids1, valid1 = self._padded_mutations(x1[start : start + 128])
            ids2, valid2 = self._padded_mutations(x2[start : start + 128])
            sites1, sites2 = ids1[:, :, None] // 20, ids2[:, None, :] // 20
            exponent = h * self.hellinger[sites1, sites2] + d * self.distances[sites1, sites2]
            exponent = exponent + p * (self.log_probs[ids1, None] - self.log_probs[ids2][:, None, :]).abs()
            pairs = (-exponent).exp() * (valid1[:, :, None] & valid2[:, None, :])
            chunks.append(pairs.sum(dim=(-2, -1)))
        return torch.cat(chunks)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params: Any) -> torch.Tensor:
        if params.get("last_dim_is_batch", False):
            raise ValueError("Kermut does not treat individual packed features as batches")
        if x1.shape[-1] <= self.seq_len or x2.shape[-1] != x1.shape[-1]:
            raise ValueError("Kermut inputs must pack residue IDs followed by nonempty sequence embeddings")
        batch_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
        a = x1.expand(*batch_shape, *x1.shape[-2:])
        b = x2.expand(*batch_shape, *x2.shape[-2:])
        if diag:
            if a.shape[-2] != b.shape[-2]:
                raise ValueError("Kermut diagonal inputs must have the same number of rows")
            structure = self._structure_diagonal(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]))
            structure = structure.reshape(*batch_shape, a.shape[-2])
        elif batch_shape:
            matrices = [
                self._structure_matrix(left, right)
                for left, right in zip(a.reshape(-1, *a.shape[-2:]), b.reshape(-1, *b.shape[-2:]), strict=True)
            ]
            structure = torch.stack(matrices).reshape(*batch_shape, a.shape[-2], b.shape[-2])
        else:
            structure = self._structure_matrix(a, b)
        sequence = self.sequence_kernel(x1[..., self.seq_len :], x2[..., self.seq_len :], diag=diag).to_dense()
        return self.mixture * self.structure_scale * structure + (1.0 - self.mixture) * sequence


class KermutGPRegressor(TanimotoGPRegressor):
    """Existing exact-GP fit, posterior, refresh and qEI API with Kermut covariance."""

    def __init__(self, *, resources: KermutResources, lengthscale: float | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("outputscale", 1.0)
        kwargs.setdefault("learn_outputscale", False)
        if kwargs.get("fit_mean_and_noise_only", False):
            raise ValueError("Kermut requires learnable kernel hyperparameters; disable gp_fit_mean_and_noise_only")
        super().__init__(**kwargs)
        self.resources = resources
        self.lengthscale = 1.0 if lengthscale is None else float(lengthscale)
        if not np.isfinite(self.lengthscale) or self.lengthscale <= 0:
            raise ValueError("Kermut initial sequence lengthscale must be finite and positive")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {**super().get_params(deep=deep), "resources": self.resources, "lengthscale": self.lengthscale}

    def _make_base_kernel(self) -> Kernel:
        kernel = KermutKernel(self.resources)
        kernel.sequence_kernel.lengthscale = self.lengthscale
        return kernel

    def _validate_features(self, x: np.ndarray) -> np.ndarray:
        length = len(self.resources.wild_type)
        if x.shape[-1] <= length or not np.isfinite(x).all():
            raise ValueError("Kermut features require finite residue IDs and nonempty ESM embeddings")
        tokens = x[..., :length]
        if np.any(tokens != np.floor(tokens)) or np.any(tokens < 0) or np.any(tokens >= 20):
            raise ValueError("Kermut residue IDs must be integers in [0, 20), in the resource alphabet order")
        return x

    def _coerce_x(self, x: Any) -> np.ndarray:
        return self._validate_features(super()._coerce_x(x))

    def _coerce_x_batch(self, x: Any) -> np.ndarray:
        return self._validate_features(super()._coerce_x_batch(x))


class KermutExpectedImprovementSelector(SequenceExpectedImprovementSelector):
    """Reuse archive/acquisition behavior, with cached ESM features and Kermut GP."""

    def __init__(
        self,
        *,
        reward_fn: Any,
        config: ActiveLearningConfig,
        resources: KermutResources,
        featurizer: ESM2Featurizer,
    ) -> None:
        if config.gp_fit_mean_and_noise_only:
            raise ValueError("Kermut requires learnable kernel hyperparameters; disable al_gp_fit_mean_and_noise_only")
        if config.gp_outputscale not in {None, 1.0}:
            raise ValueError("Kermut uses its own structure scale and mixture; al_gp_outputscale must be 1.0")
        config.gp_outputscale = 1.0
        config.gp_learn_outputscale = False
        config.gp_learn_lengthscale = True
        self.resources = resources
        self.featurizer = featurizer
        self._embedding_cache: dict[str, np.ndarray] = {}
        super().__init__(reward_fn=reward_fn, config=config)

    def _make_fingerprint_features(self, smiles: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
        length = len(self.resources.wild_type)
        sequences: list[str] = []
        valid = np.zeros(len(smiles), dtype=bool)
        for idx, value in enumerate(smiles):
            if value is None:
                continue
            try:
                sequence = validate_sequence(value)
            except (ValueError, TypeError):
                continue
            if len(sequence) == length:
                valid[idx] = True
                sequences.append(sequence)
        if not sequences:
            raise ValueError("Kermut requires fixed-length canonical substitution sequences aligned to wild_type")
        embedding_sequences = [
            self.resources.embedding_prefix + sequence + self.resources.embedding_suffix
            for sequence in sequences
        ]
        missing = list(dict.fromkeys(seq for seq in embedding_sequences if seq not in self._embedding_cache))
        if missing:
            try:
                embeddings = self.featurizer.featurize(missing).detach().cpu().numpy()
            except Exception as exc:
                # The shared selector treats ValueError as an invalid candidate
                # batch. A failed embedding model must instead stop the run.
                raise RuntimeError("Kermut could not compute required ESM embeddings") from exc
            if embeddings.ndim != 2 or embeddings.shape[0] != len(missing) or not np.isfinite(embeddings).all():
                raise RuntimeError("Kermut ESM featurizer returned invalid embeddings")
            self._embedding_cache.update(zip(missing, embeddings, strict=True))
        tokens = np.array([[self.resources.alphabet.index(aa) for aa in seq] for seq in sequences], dtype=np.float64)
        embeddings = np.stack([self._embedding_cache[seq] for seq in embedding_sequences])
        return np.concatenate((tokens, embeddings), axis=1), valid

    def _feature_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # Preserve the protein one-hot Tanimoto diversity/diagnostic metric;
        # packed token IDs and signed ESM embeddings are not fingerprints.
        length = len(self.resources.wild_type)
        matches = int(np.count_nonzero(a[:length] == b[:length]))
        return matches / (2 * length - matches)

    def _make_gp_regressor(self) -> KermutGPRegressor:
        return KermutGPRegressor(
            resources=self.resources,
            lengthscale=self.config.gp_lengthscale,
            device=str(self.config.gp_device),
            dtype=str(self.config.gp_dtype),
            jitter=self.config.gp_jitter,
            noise_variance=self.config.gp_noise_variance,
            outputscale=self.config.gp_outputscale,
            learn_outputscale=bool(self.config.gp_learn_outputscale),
            fit_mean_and_noise_only=bool(self.config.gp_fit_mean_and_noise_only),
        )
