"""Helpers for pool-based surrogate experiments on discrete sequence datasets."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import Tensor

from search.interpolant.output import open_jsonl, resolve_jsonl_path

try:
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.kernels import Kernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ConstantMean
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError:  # pragma: no cover - optional dependency for notebook use
    MultivariateNormal = None
    Kernel = object  # type: ignore[assignment]
    ScaleKernel = None
    GaussianLikelihood = None
    ConstantMean = None
    ExactMarginalLogLikelihood = None

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:  # pragma: no cover - optional dependency for notebook use
    sp = None
    spla = None

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:  # pragma: no cover - optional dependency for notebook use
    RandomForestRegressor = None


ScalarAcquisitionFactory = Callable[[Any, Tensor], Any]
ModelInitializer = Callable[..., tuple[Any, Any]]
ModelFitter = Callable[[Any], Any]


@dataclass
class TrialSummary:
    """Stacked BO and random best-so-far trajectories across trials."""

    best_observed_all: Tensor
    best_random_all: Tensor

    @property
    def observed_mean(self) -> Tensor:
        return self.best_observed_all.mean(dim=0)

    @property
    def observed_std(self) -> Tensor:
        return self.best_observed_all.std(dim=0)

    @property
    def random_mean(self) -> Tensor:
        return self.best_random_all.mean(dim=0)

    @property
    def random_std(self) -> Tensor:
        return self.best_random_all.std(dim=0)


def fit_random_forest_regressor(
    train_x: np.ndarray | Tensor,
    train_y: np.ndarray | Tensor,
    *,
    n_estimators: int = 256,
    min_samples_leaf: int = 1,
    max_features: str | int | float | None = "sqrt",
    bootstrap: bool = True,
    n_jobs: int | None = -1,
    random_state: Optional[int] = None,
) -> Any:
    """Fit a random forest regressor on a discrete pool."""
    if RandomForestRegressor is None:
        raise ImportError("scikit-learn is required to fit a random forest surrogate")

    x_np = np.asarray(train_x)
    y_np = np.asarray(train_y, dtype=np.float64).reshape(-1)
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        bootstrap=bool(bootstrap),
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(x_np, y_np)
    return model


def random_forest_predict_mean_std(
    model: Any,
    x: np.ndarray | Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate predictive mean and std from per-tree random forest predictions."""
    x_np = np.asarray(x)
    if x_np.ndim != 2:
        raise ValueError(f"x must have shape [N, D], got {tuple(x_np.shape)}")
    estimators = getattr(model, "estimators_", None)
    if not estimators:
        raise ValueError("Model does not expose fitted estimators_ for ensemble uncertainty")
    tree_preds = np.stack([est.predict(x_np) for est in estimators], axis=0)
    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0, ddof=0)
    return mean, std


def build_sequence_knn_graph(
    sequences: np.ndarray | Tensor,
    *,
    graph_k: int = 16,
    sigma: Optional[float] = None,
    block_size: int = 32,
    add_self_loops: bool = True,
    self_loop_weight: float = 1.0,
    symmetrize: str = "max",
    eps: float = 1e-8,
) -> tuple[Any, dict[str, float]]:
    """Build a sparse sequence graph using kNN in normalized Hamming distance."""
    if sp is None:
        raise ImportError("scipy is required to build a sparse sequence kNN graph")

    x = np.asarray(sequences, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError(f"sequences must have shape [N, L], got {tuple(x.shape)}")
    n, seq_len = x.shape
    if n < 2:
        raise ValueError(f"Need at least 2 sequences to build a graph, got N={n}")

    k = max(1, min(int(graph_k), n - 1))
    block = max(1, int(block_size))
    nn_idx = np.empty((n, k), dtype=np.int64)
    nn_dist = np.empty((n, k), dtype=np.float64)

    for start in range(0, n, block):
        stop = min(start + block, n)
        x_block = x[start:stop]
        dist = np.mean(x_block[:, None, :] != x[None, :, :], axis=2, dtype=np.float64)
        row_ids = np.arange(start, stop)
        dist[np.arange(stop - start), row_ids] = np.inf

        part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(dist, part, axis=1)
        order = np.argsort(part_dist, axis=1)
        nn_idx[start:stop] = np.take_along_axis(part, order, axis=1)
        nn_dist[start:stop] = np.take_along_axis(part_dist, order, axis=1)

    finite_dists = nn_dist[np.isfinite(nn_dist)]
    positive_dists = finite_dists[finite_dists > 0.0]
    sigma_used = float(sigma) if sigma is not None else float(np.median(positive_dists)) if positive_dists.size > 0 else 1.0 / max(seq_len, 1)
    sigma_used = max(float(sigma_used), float(eps))

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = nn_idx.reshape(-1)
    dists = nn_dist.reshape(-1)
    weights = np.exp(-0.5 * np.square(dists / sigma_used))

    adjacency = sp.coo_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    sym = str(symmetrize).strip().lower()
    if sym == "max":
        adjacency = adjacency.maximum(adjacency.T)
    elif sym == "mean":
        adjacency = 0.5 * (adjacency + adjacency.T)
    else:
        raise ValueError(f"Unsupported symmetrize mode '{symmetrize}'; expected 'max' or 'mean'")

    if add_self_loops:
        adjacency = adjacency + sp.eye(n, dtype=np.float64, format="csr") * float(self_loop_weight)

    meta = {
        "graph_k": float(k),
        "sigma": float(sigma_used),
        "num_nodes": float(n),
        "seq_len": float(seq_len),
        "nnz": float(adjacency.nnz),
    }
    return adjacency, meta


def build_sequence_graph_diffusion_embeddings(
    sequences: np.ndarray | Tensor,
    *,
    tau: float = 1.0,
    graph_k: int = 16,
    sigma: Optional[float] = None,
    num_eigs: int = 128,
    block_size: int = 32,
    normalized_laplacian: bool = True,
    add_self_loops: bool = True,
    self_loop_weight: float = 1.0,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, float]]:
    """Approximate a pool sequence-graph heat kernel via truncated spectral embeddings."""
    if sp is None:
        raise ImportError("scipy is required to build sequence-graph diffusion embeddings")
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")

    adjacency, meta = build_sequence_knn_graph(
        sequences,
        graph_k=graph_k,
        sigma=sigma,
        block_size=block_size,
        add_self_loops=add_self_loops,
        self_loop_weight=self_loop_weight,
        eps=eps,
    )
    n = int(adjacency.shape[0])
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if normalized_laplacian:
        inv_sqrt_deg = 1.0 / np.sqrt(np.maximum(degrees, float(eps)))
        d_inv_sqrt = sp.diags(inv_sqrt_deg, offsets=0, format="csr")
        laplacian = sp.eye(n, dtype=np.float64, format="csr") - d_inv_sqrt @ adjacency @ d_inv_sqrt
    else:
        laplacian = sp.diags(degrees, offsets=0, format="csr") - adjacency

    eig_count = max(1, min(int(num_eigs), n))
    if eig_count >= n or n <= 512 or spla is None:
        lap_dense = laplacian.toarray()
        eigvals, eigvecs = np.linalg.eigh(lap_dense)
        eigvals = eigvals[:eig_count]
        eigvecs = eigvecs[:, :eig_count]
    else:
        eigvals, eigvecs = spla.eigsh(laplacian, k=eig_count, which="SM")
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

    scales = np.exp(-0.5 * float(tau) * eigvals)
    embeddings = eigvecs * scales[None, :]
    meta.update(
        {
            "tau": float(tau),
            "num_eigs": float(eig_count),
        }
    )
    return torch.as_tensor(embeddings, dtype=torch.double), meta


def build_token_diffusion_similarity(
    transition_probs: Tensor,
    *,
    tau: float = 1.0,
    transition_mask: Optional[Tensor] = None,
    use_transition_weights: bool = True,
    normalized_laplacian: bool = True,
    add_self_loops: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Build a token-level heat kernel from an Ehrlich transition graph."""
    probs = transition_probs.detach().to(dtype=torch.double)
    if probs.ndim != 2 or probs.shape[0] != probs.shape[1]:
        raise ValueError(f"transition_probs must have shape [V, V], got {tuple(probs.shape)}")
    if not np.isfinite(float(tau)) or tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    if not np.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not bool(torch.isfinite(probs).all()):
        raise ValueError("transition_probs must contain only finite values")
    if bool((probs < 0).any()):
        raise ValueError("transition_probs must be nonnegative")

    mask: Optional[Tensor] = None
    if transition_mask is not None:
        mask = transition_mask.detach().to(device=probs.device, dtype=torch.double)
        if mask.shape != probs.shape:
            raise ValueError(
                f"transition_mask must have shape {tuple(probs.shape)}, got {tuple(mask.shape)}"
            )
        if not bool(torch.isfinite(mask).all()):
            raise ValueError("transition_mask must contain only finite values")
        if bool((mask < 0).any()):
            raise ValueError("transition_mask must be nonnegative")

    if use_transition_weights:
        # The Ehrlich transition graph is directed. Apply its directed support
        # before projecting the weights onto an undirected graph suitable for a
        # self-adjoint graph Laplacian.
        directed_adjacency = probs if mask is None else probs * mask
        adjacency = 0.5 * (directed_adjacency + directed_adjacency.transpose(0, 1))
    else:
        if mask is None:
            adjacency = (probs > 0).to(dtype=torch.double)
        else:
            adjacency = mask
        adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))

    if add_self_loops:
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(1.0)

    if not bool(torch.isfinite(adjacency).all()):
        raise ValueError("Transition graph adjacency must contain only finite values")
    if bool((adjacency < 0).any()):
        raise ValueError("Transition graph adjacency must be nonnegative")
    if not torch.equal(adjacency, adjacency.transpose(0, 1)):
        raise RuntimeError("Transition graph adjacency must be exactly symmetric before eigendecomposition")

    degree = adjacency.sum(dim=1)
    if normalized_laplacian:
        inv_sqrt_degree = (degree + float(eps)).rsqrt()
        laplacian = torch.eye(adjacency.shape[0], dtype=torch.double, device=adjacency.device)
        laplacian = laplacian - (
            inv_sqrt_degree.view(-1, 1) * adjacency * inv_sqrt_degree.view(1, -1)
        )
    else:
        laplacian = torch.diag(degree) - adjacency

    # Preserve exact symmetry even when the normalized scaling operations use
    # different floating-point multiplication orders across the two triangles.
    laplacian = 0.5 * (laplacian + laplacian.transpose(0, 1))
    if not bool(torch.isfinite(laplacian).all()):
        raise RuntimeError("Transition graph Laplacian contains non-finite values")
    if not torch.equal(laplacian, laplacian.transpose(0, 1)):
        raise RuntimeError("Transition graph Laplacian must be exactly symmetric before eigendecomposition")

    eigvals, eigvecs = torch.linalg.eigh(laplacian)
    heat = eigvecs @ torch.diag(torch.exp(-float(tau) * eigvals)) @ eigvecs.transpose(0, 1)
    heat = 0.5 * (heat + heat.transpose(0, 1))
    diag = torch.diag(heat).clamp_min(float(eps))
    heat = heat / torch.sqrt(diag.view(-1, 1) * diag.view(1, -1))
    return heat.to(dtype=transition_probs.dtype, device=transition_probs.device)


def sequence_diffusion_similarity(
    x1: Tensor,
    x2: Tensor,
    *,
    token_similarity: Tensor,
    combine: str = "mean",
    position_weights: Optional[Tensor] = None,
    diag: bool = False,
) -> Tensor:
    """Evaluate a sequence kernel by aggregating token-level similarities per position."""
    if x1.ndim == 1:
        x1 = x1.view(1, -1)
    if x2.ndim == 1:
        x2 = x2.view(1, -1)
    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError(f"x1 and x2 must have shape [..., N, L], got {tuple(x1.shape)} and {tuple(x2.shape)}")
    if x1.shape[-1] != x2.shape[-1]:
        raise ValueError(f"x1 and x2 must share sequence length, got {x1.shape[-1]} and {x2.shape[-1]}")

    token_similarity = token_similarity.to(device=x1.device)
    vocab_size = int(token_similarity.shape[0])
    x1_idx = x1.to(dtype=torch.long)
    x2_idx = x2.to(dtype=torch.long)
    if (x1_idx < 0).any() or (x2_idx < 0).any() or (x1_idx >= vocab_size).any() or (x2_idx >= vocab_size).any():
        raise ValueError("Sequence tokens are out of range for the provided token similarity matrix")

    combine_name = combine.strip().lower()
    if combine_name not in {"mean", "sum", "product"}:
        raise ValueError(f"Unsupported combine mode '{combine}'; expected 'mean', 'sum', or 'product'")

    batch_shape = torch.broadcast_shapes(x1_idx.shape[:-2], x2_idx.shape[:-2])
    x1_idx = torch.broadcast_to(x1_idx, batch_shape + x1_idx.shape[-2:])
    x2_idx = torch.broadcast_to(x2_idx, batch_shape + x2_idx.shape[-2:])

    n1 = int(x1_idx.shape[-2])
    n2 = int(x2_idx.shape[-2])
    num_pos = int(x1_idx.shape[-1])
    if position_weights is None:
        weights = torch.full((num_pos,), 1.0 / max(num_pos, 1), dtype=token_similarity.dtype, device=x1.device)
    else:
        weights = position_weights.to(device=x1.device, dtype=token_similarity.dtype).view(-1)
        if weights.numel() != num_pos:
            raise ValueError(f"position_weights must have length {num_pos}, got {weights.numel()}")
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)

    if combine_name == "product":
        if diag:
            if n1 != n2:
                raise ValueError(f"diag=True requires matching set sizes, got {n1} and {n2}")
            out = torch.ones(batch_shape + (n1,), dtype=token_similarity.dtype, device=x1.device)
        else:
            out = torch.ones(batch_shape + (n1, n2), dtype=token_similarity.dtype, device=x1.device)
    else:
        if diag:
            if n1 != n2:
                raise ValueError(f"diag=True requires matching set sizes, got {n1} and {n2}")
            out = torch.zeros(batch_shape + (n1,), dtype=token_similarity.dtype, device=x1.device)
        else:
            out = torch.zeros(batch_shape + (n1, n2), dtype=token_similarity.dtype, device=x1.device)

    for pos in range(num_pos):
        if diag:
            token_vals = token_similarity[x1_idx[..., :, pos], x2_idx[..., :, pos]]
        else:
            token_vals = token_similarity[
                x1_idx[..., :, pos].unsqueeze(-1),
                x2_idx[..., :, pos].unsqueeze(-2),
            ]

        if combine_name == "product":
            out = out * token_vals.clamp_min(torch.finfo(token_vals.dtype).eps)
        elif combine_name == "sum":
            out = out + token_vals
        else:
            out = out + weights[pos] * token_vals

    return out


class DiscreteSequenceDiffusionKernel(Kernel):
    """GPyTorch kernel for fixed-length token sequences using a token heat kernel."""

    has_lengthscale = False
    is_stationary = False

    def __init__(
        self,
        *,
        token_similarity: Tensor,
        combine: str = "mean",
        position_weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        if MultivariateNormal is None:
            raise ImportError("gpytorch is required to use DiscreteSequenceDiffusionKernel")
        super().__init__(**kwargs)
        self.register_buffer("token_similarity", token_similarity.detach().clone())
        if position_weights is not None:
            self.register_buffer("position_weights", position_weights.detach().clone())
        else:
            self.position_weights = None  # type: ignore[assignment]
        self.combine = str(combine)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        if last_dim_is_batch:
            raise NotImplementedError("DiscreteSequenceDiffusionKernel does not support last_dim_is_batch")
        _ = params
        weights = getattr(self, "position_weights", None)
        return sequence_diffusion_similarity(
            x1,
            x2,
            token_similarity=self.token_similarity,
            combine=self.combine,
            position_weights=weights,
            diag=diag,
        )


def make_local_spaced_ngram_offsets(n: int, window_size: int = 8) -> Tensor:
    """Return every anchored n-token pattern inside a local sequence window."""

    order = int(n)
    window = int(window_size)
    if order < 2:
        raise ValueError("local spaced n-gram order must be at least 2")
    if window < order:
        raise ValueError("local spaced n-gram window_size must be at least n")
    return torch.tensor(
        [[0, *offsets] for offsets in combinations(range(1, window), order - 1)],
        dtype=torch.long,
    )


def make_local_spaced_pair_offsets(window_size: int = 8) -> Tensor:
    """Compatibility wrapper for all local two-token offset patterns."""

    return make_local_spaced_ngram_offsets(2, window_size)


def make_local_spaced_trigram_offsets(window_size: int = 8) -> Tensor:
    """Compatibility wrapper for all local three-token offset patterns."""

    return make_local_spaced_ngram_offsets(3, window_size)


class SpacedNGramKernel(Kernel):
    """Normalized count kernel over categorical n-grams at fixed offsets.

    Each offset row defines one gapped pattern, for example ``[0, 2, 5]``.
    A sequence is represented by counts of every token tuple observed while
    sliding that pattern along the sequence.  The kernel is the cosine dot
    product between those sparse count vectors, so it is positive semidefinite
    and has unit diagonal for every non-empty sequence.
    """

    has_lengthscale = False
    is_stationary = False
    _MAX_DIRECT_SPARSE_FEATURES = 10_000_000

    def __init__(
        self,
        *,
        offsets: Tensor | np.ndarray,
        vocab_size: int,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        if MultivariateNormal is None:
            raise ImportError("gpytorch is required to use SpacedNGramKernel")
        if sp is None:
            raise ImportError("scipy is required to use SpacedNGramKernel")
        super().__init__(**kwargs)
        offsets_tensor = torch.as_tensor(offsets, dtype=torch.long)
        if offsets_tensor.ndim == 1:
            offsets_tensor = offsets_tensor.view(1, -1)
        if offsets_tensor.ndim != 2 or offsets_tensor.shape[0] < 1 or offsets_tensor.shape[1] < 1:
            raise ValueError("offsets must have shape [num_patterns, n] with non-empty dimensions")
        if bool((offsets_tensor < 0).any().item()):
            raise ValueError("spaced n-gram offsets must be non-negative")
        if bool((offsets_tensor[:, 1:] <= offsets_tensor[:, :-1]).any().item()):
            raise ValueError("each spaced n-gram offset pattern must be strictly increasing")
        normalized_offsets = offsets_tensor - offsets_tensor[:, :1]
        if not torch.equal(offsets_tensor, normalized_offsets):
            raise ValueError("each spaced n-gram offset pattern must start at zero")
        vocab_size_int = int(vocab_size)
        if vocab_size_int < 2:
            raise ValueError("vocab_size must be at least 2")

        n = int(offsets_tensor.shape[1])
        features_per_pattern = vocab_size_int**n
        if features_per_pattern > np.iinfo(np.int64).max // int(offsets_tensor.shape[0]):
            raise ValueError("spaced n-gram feature space is too large for 64-bit sparse indices")
        self.register_buffer("offsets", offsets_tensor)
        self.vocab_size = vocab_size_int
        self.normalize = bool(normalize)
        self.n = n
        self.features_per_pattern = int(features_per_pattern)
        self.num_features = int(offsets_tensor.shape[0]) * self.features_per_pattern
        self._row_cache: dict[tuple[int, ...], Any] = {}

    def _validate_and_flatten(self, x: Tensor) -> tuple[np.ndarray, torch.Size]:
        if x.ndim < 2:
            raise ValueError(f"x must have shape [..., N, L], got {tuple(x.shape)}")
        if torch.is_floating_point(x):
            rounded = x.round()
            if not bool(torch.allclose(x, rounded, rtol=0.0, atol=1e-8)):
                raise ValueError("spaced n-gram inputs must contain integer-valued tokens")
            x = rounded
        x_long = x.detach().to(device="cpu", dtype=torch.long)
        if x_long.numel() > 0 and bool(((x_long < 0) | (x_long >= self.vocab_size)).any().item()):
            raise ValueError(f"spaced n-gram tokens must be in [0, {self.vocab_size})")
        if int(self.offsets.max().item()) >= int(x_long.shape[-1]):
            raise ValueError(
                f"spaced n-gram maximum offset {int(self.offsets.max().item())} "
                f"does not fit sequence length {int(x_long.shape[-1])}"
            )
        return x_long.reshape(-1, x_long.shape[-1]).numpy(), x_long.shape[:-1]

    def _feature_row(self, sequence: np.ndarray) -> Any:
        key = tuple(int(value) for value in sequence.tolist())
        cached = self._row_cache.get(key)
        if cached is not None:
            return cached

        columns: list[int] = []
        values: list[float] = []
        offsets_np = self.offsets.detach().cpu().numpy()
        for pattern_idx, pattern in enumerate(offsets_np):
            num_starts = int(sequence.shape[0]) - int(pattern[-1])
            counts: dict[int, int] = {}
            for start in range(num_starts):
                encoded = 0
                for offset in pattern:
                    encoded = encoded * self.vocab_size + int(sequence[start + int(offset)])
                feature_idx = pattern_idx * self.features_per_pattern + encoded
                counts[feature_idx] = counts.get(feature_idx, 0) + 1
            for feature_idx, count in counts.items():
                columns.append(int(feature_idx))
                values.append(float(count))

        if self.normalize:
            norm = float(np.linalg.norm(np.asarray(values, dtype=np.float64)))
            if norm > 0.0:
                values = [value / norm for value in values]
        row = sp.csr_matrix(
            (
                np.asarray(values, dtype=np.float64),
                np.asarray(columns, dtype=np.int64),
                np.asarray([0, len(columns)], dtype=np.int64),
            ),
            shape=(1, self.num_features),
        )
        self._row_cache[key] = row
        return row

    def _features(self, sequences: np.ndarray) -> Any:
        return sp.vstack([self._feature_row(row) for row in sequences], format="csr")

    @staticmethod
    def _compressed_feature_matrices(
        rows1: list[Any],
        rows2: list[Any],
    ) -> tuple[Any, Any]:
        """Materialize sparse feature sets over their observed columns only.

        The nominal feature space grows as ``vocab_size ** n`` and can contain
        tens of billions of columns even though a sequence observes only a few
        hundred of them.  Remapping the union of observed feature identifiers
        to compact local columns preserves dot products exactly while avoiding
        sparse multiplication work arrays proportional to the nominal width.
        """

        all_rows = [*rows1, *rows2]
        observed_parts = [row.indices for row in all_rows if row.indices.size]
        observed = (
            np.unique(np.concatenate(observed_parts))
            if observed_parts
            else np.empty((0,), dtype=np.int64)
        )

        def materialize(rows: list[Any]) -> Any:
            data_parts: list[np.ndarray] = []
            index_parts: list[np.ndarray] = []
            indptr = np.zeros(len(rows) + 1, dtype=np.int64)
            for row_idx, row in enumerate(rows):
                data = np.asarray(row.data, dtype=np.float64)
                columns = np.asarray(row.indices, dtype=np.int64)
                data_parts.append(data)
                index_parts.append(np.searchsorted(observed, columns).astype(np.int64, copy=False))
                indptr[row_idx + 1] = indptr[row_idx] + data.size

            data = np.concatenate(data_parts) if data_parts else np.empty((0,), dtype=np.float64)
            indices = (
                np.concatenate(index_parts)
                if index_parts
                else np.empty((0,), dtype=np.int64)
            )
            return sp.csr_matrix(
                (data, indices, indptr),
                shape=(len(rows), int(observed.size)),
            )

        return materialize(rows1), materialize(rows2)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        if last_dim_is_batch:
            raise NotImplementedError("SpacedNGramKernel does not support last_dim_is_batch")
        _ = params
        if x1.shape[-1] != x2.shape[-1]:
            raise ValueError(f"x1 and x2 must share sequence length, got {x1.shape[-1]} and {x2.shape[-1]}")

        flat1, shape1 = self._validate_and_flatten(x1)
        flat2, shape2 = self._validate_and_flatten(x2)
        batch_shape = torch.broadcast_shapes(shape1[:-1], shape2[:-1])
        n1 = int(shape1[-1])
        n2 = int(shape2[-1])
        array1 = np.broadcast_to(flat1.reshape(tuple(shape1[:-1]) + (n1, x1.shape[-1])), tuple(batch_shape) + (n1, x1.shape[-1]))
        array2 = np.broadcast_to(flat2.reshape(tuple(shape2[:-1]) + (n2, x2.shape[-1])), tuple(batch_shape) + (n2, x2.shape[-1]))

        matrices: list[np.ndarray] = []
        flat_batch1 = array1.reshape(-1, n1, x1.shape[-1])
        flat_batch2 = array2.reshape(-1, n2, x2.shape[-1])
        for batch_x1, batch_x2 in zip(flat_batch1, flat_batch2, strict=True):
            if self.num_features <= self._MAX_DIRECT_SPARSE_FEATURES:
                features1 = self._features(batch_x1)
                features2 = (
                    features1
                    if np.array_equal(batch_x1, batch_x2)
                    else self._features(batch_x2)
                )
            else:
                rows1 = [self._feature_row(row) for row in batch_x1]
                rows2 = (
                    rows1
                    if np.array_equal(batch_x1, batch_x2)
                    else [self._feature_row(row) for row in batch_x2]
                )
                features1, features2 = self._compressed_feature_matrices(rows1, rows2)
            if diag:
                if n1 != n2:
                    raise ValueError(f"diag=True requires matching set sizes, got {n1} and {n2}")
                values = np.asarray(features1.multiply(features2).sum(axis=1)).reshape(-1)
            else:
                values = (features1 @ features2.T).toarray()
            matrices.append(np.asarray(values, dtype=np.float64))

        result_shape = tuple(batch_shape) + ((n1,) if diag else (n1, n2))
        result = np.stack(matrices, axis=0).reshape(result_shape)
        result_dtype = x1.dtype if torch.is_floating_point(x1) else torch.get_default_dtype()
        return torch.as_tensor(result, device=x1.device, dtype=result_dtype)


class AdditiveSpacedNGramKernel(Kernel):
    """Fixed convex combination of spaced n-gram kernels of different orders.

    Each component is independently normalized before combination.  The
    positive weights are normalized to sum to one, preserving a unit prior
    diagonal and keeping the mixture separate from the GP output scale.
    """

    has_lengthscale = False
    is_stationary = False

    def __init__(
        self,
        *,
        offset_groups: list[Tensor | np.ndarray] | tuple[Tensor | np.ndarray, ...],
        vocab_size: int,
        weights: Tensor | np.ndarray | list[float] | tuple[float, ...] | None = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        if MultivariateNormal is None:
            raise ImportError("gpytorch is required to use AdditiveSpacedNGramKernel")
        super().__init__(**kwargs)
        if len(offset_groups) < 2:
            raise ValueError("offset_groups must contain at least two spaced n-gram components")
        self.components = torch.nn.ModuleList(
            [
                SpacedNGramKernel(
                    offsets=offsets,
                    vocab_size=int(vocab_size),
                    normalize=bool(normalize),
                )
                for offsets in offset_groups
            ]
        )
        raw_weights = torch.ones(len(self.components), dtype=torch.double)
        if weights is not None:
            raw_weights = torch.as_tensor(weights, dtype=torch.double).reshape(-1)
        if raw_weights.numel() != len(self.components):
            raise ValueError("weights must contain one value per offset group")
        if not torch.isfinite(raw_weights).all() or bool((raw_weights <= 0).any()):
            raise ValueError("additive spaced n-gram weights must be finite and positive")
        self.register_buffer("mixture_weights", raw_weights / raw_weights.sum())

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        if last_dim_is_batch:
            raise NotImplementedError(
                "AdditiveSpacedNGramKernel does not support last_dim_is_batch"
            )
        values: Tensor | None = None
        for weight, component in zip(self.mixture_weights, self.components, strict=True):
            component_value = component.forward(x1, x2, diag=diag, **params)
            weighted = weight.to(
                device=component_value.device,
                dtype=component_value.dtype,
            ) * component_value
            values = weighted if values is None else values + weighted
        if values is None:  # pragma: no cover - constructor prevents this
            raise RuntimeError("additive spaced n-gram kernel has no components")
        return values


class PoolSequenceGraphKernel(Kernel):
    """GPyTorch kernel backed by a finite pool graph-diffusion embedding."""

    has_lengthscale = False
    is_stationary = False

    def __init__(
        self,
        *,
        pool_sequences: np.ndarray | Tensor,
        diffusion_embeddings: Tensor,
        **kwargs: Any,
    ) -> None:
        if MultivariateNormal is None:
            raise ImportError("gpytorch is required to use PoolSequenceGraphKernel")
        super().__init__(**kwargs)
        pool_x = torch.as_tensor(pool_sequences, dtype=torch.long).contiguous()
        if pool_x.ndim != 2:
            raise ValueError(f"pool_sequences must have shape [N, L], got {tuple(pool_x.shape)}")
        emb = torch.as_tensor(diffusion_embeddings, dtype=torch.double)
        if emb.ndim != 2 or emb.shape[0] != pool_x.shape[0]:
            raise ValueError(
                f"diffusion_embeddings must have shape [N, R] aligned with pool_sequences, got {tuple(emb.shape)}"
            )

        self.register_buffer("pool_sequences", pool_x)
        self.register_buffer("diffusion_embeddings", emb)
        self._seq_to_index = {
            tuple(int(v) for v in row.tolist()): idx for idx, row in enumerate(pool_x.cpu())
        }

    def _lookup_indices(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.view(1, -1)
        if x.ndim < 2:
            raise ValueError(f"x must have shape [..., N, L], got {tuple(x.shape)}")
        flat = x.detach().to(device="cpu", dtype=torch.long).reshape(-1, x.shape[-1])
        idxs: list[int] = []
        for row in flat:
            key = tuple(int(v) for v in row.tolist())
            idx = self._seq_to_index.get(key)
            if idx is None:
                raise KeyError(
                    "PoolSequenceGraphKernel received a sequence that is not in the fixed pool graph. "
                    "This kernel only supports pool-based BO over the provided sequence set."
                )
            idxs.append(int(idx))
        return torch.tensor(idxs, dtype=torch.long, device=x.device).view(x.shape[:-1])

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        if last_dim_is_batch:
            raise NotImplementedError("PoolSequenceGraphKernel does not support last_dim_is_batch")
        _ = params

        idx1 = self._lookup_indices(x1)
        idx2 = self._lookup_indices(x2)
        batch_shape = torch.broadcast_shapes(idx1.shape[:-1], idx2.shape[:-1])
        idx1 = torch.broadcast_to(idx1, batch_shape + idx1.shape[-1:])
        idx2 = torch.broadcast_to(idx2, batch_shape + idx2.shape[-1:])

        emb = self.diffusion_embeddings.to(device=x1.device, dtype=x1.dtype)
        emb1 = emb[idx1]
        emb2 = emb[idx2]
        if diag:
            if emb1.shape[-2] != emb2.shape[-2]:
                raise ValueError(
                    f"diag=True requires matching set sizes, got {emb1.shape[-2]} and {emb2.shape[-2]}"
                )
            return (emb1 * emb2).sum(dim=-1)
        return emb1 @ emb2.transpose(-1, -2)


def load_sequence_pool_from_jsonl(
    path: str | Path,
    *,
    seq_key: str = "seq",
    score_key: str = "score",
    deduplicate: bool = True,
    drop_nonfinite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a JSONL run output into a discrete pool of integer sequences and scores."""
    path = resolve_jsonl_path(Path(path).expanduser())
    if not path.exists():
        raise FileNotFoundError(f"Pool path does not exist: {path}")

    seqs: list[np.ndarray] = []
    scores: list[float] = []
    with open_jsonl(path, "rt") as fp:
        for line in fp:
            rec = json.loads(line)
            seq = rec.get(seq_key)
            score = rec.get(score_key)
            if seq is None or score is None:
                continue
            score_val = float(score)
            if drop_nonfinite and not np.isfinite(score_val):
                continue
            seqs.append(np.asarray(seq, dtype=np.int64))
            scores.append(score_val)

    if not seqs:
        raise ValueError(f"No valid sequence/score records found in {path}")

    x = np.stack(seqs, axis=0)
    y = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    if deduplicate:
        x, y = deduplicate_sequence_pool(x, y)
    return x, y


def deduplicate_sequence_pool(
    x: np.ndarray,
    y: np.ndarray,
    *,
    reduction: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate sequences to one row, aggregating scores per sequence."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"x must have shape [N, L], got {tuple(x.shape)}")
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"x and y must agree on N, got {x.shape[0]} vs {y.shape[0]}")

    reduction_name = reduction.strip().lower()
    if reduction_name not in {"max", "mean"}:
        raise ValueError(f"Unsupported reduction '{reduction}'; expected 'max' or 'mean'")

    buckets: dict[tuple[int, ...], list[float]] = {}
    for row, score in zip(x, y[:, 0], strict=True):
        key = tuple(int(v) for v in row.tolist())
        buckets.setdefault(key, []).append(float(score))

    agg_rows: list[np.ndarray] = []
    agg_scores: list[float] = []
    for key, bucket in buckets.items():
        agg_rows.append(np.asarray(key, dtype=np.int64))
        if reduction_name == "max":
            agg_scores.append(float(np.max(bucket)))
        else:
            agg_scores.append(float(np.mean(bucket)))

    return np.stack(agg_rows, axis=0), np.asarray(agg_scores, dtype=np.float64).reshape(-1, 1)


def split_pool_for_trial(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_init: int,
    trial_seed: int,
    exclude_global_best_from_initial: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a fixed-size initial design and heldout pool from a discrete dataset."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n = int(x.shape[0])
    if x.ndim != 2:
        raise ValueError(f"x must have shape [N, L], got {tuple(x.shape)}")
    if y.shape[0] != n:
        raise ValueError(f"x and y must agree on N, got {n} vs {y.shape[0]}")
    if n_init <= 0 or n_init >= n:
        raise ValueError(f"n_init must be in [1, N-1], got {n_init} for N={n}")

    rng = np.random.default_rng(int(trial_seed))
    all_indices = np.arange(n, dtype=np.int64)
    selected: np.ndarray

    if exclude_global_best_from_initial:
        best_value = float(np.nanmax(y[:, 0]))
        best_mask = np.isclose(y[:, 0], best_value)
        non_best_indices = all_indices[~best_mask]
        if non_best_indices.shape[0] >= n_init:
            selected = rng.choice(non_best_indices, size=n_init, replace=False)
        else:
            selected = rng.choice(all_indices, size=n_init, replace=False)
    else:
        selected = rng.choice(all_indices, size=n_init, replace=False)

    selected = np.sort(selected)
    heldout_mask = np.ones(n, dtype=bool)
    heldout_mask[selected] = False

    return x[selected], x[heldout_mask], y[selected], y[heldout_mask]


def select_best_pool_observation(
    acq_func: Any,
    heldout_inputs: Tensor,
    heldout_outputs: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate an acquisition function on a finite heldout pool and take the best point."""
    if heldout_inputs.ndim != 2:
        raise ValueError(f"heldout_inputs must have shape [N, D], got {tuple(heldout_inputs.shape)}")
    if heldout_outputs.ndim not in {1, 2}:
        raise ValueError(f"heldout_outputs must have shape [N] or [N, 1], got {tuple(heldout_outputs.shape)}")
    if heldout_inputs.shape[0] == 0:
        raise ValueError("heldout pool is empty")

    candidate_batch = heldout_inputs.unsqueeze(-2)
    try:
        acq_vals = torch.as_tensor(acq_func(candidate_batch), dtype=torch.float, device=heldout_inputs.device)
    except Exception:
        values: list[Tensor] = []
        for row in heldout_inputs:
            raw = acq_func(row.view(1, 1, -1))
            val = torch.as_tensor(raw, dtype=torch.float, device=heldout_inputs.device).view(-1)[0]
            values.append(val)
        acq_vals = torch.stack(values, dim=0)

    acq_vals = acq_vals.flatten()
    best_idx = int(torch.argmax(acq_vals).item())
    return take_pool_observation(heldout_inputs, heldout_outputs, best_idx)


def take_pool_observation(
    heldout_inputs: Tensor,
    heldout_outputs: Tensor,
    index: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Take one point from a heldout pool and return the updated pool."""
    if index < 0 or index >= int(heldout_inputs.shape[0]):
        raise IndexError(f"index {index} is out of bounds for pool of size {heldout_inputs.shape[0]}")

    new_x = heldout_inputs[index : index + 1]
    if heldout_outputs.ndim == 1:
        new_y = heldout_outputs[index : index + 1].view(1, 1)
    else:
        new_y = heldout_outputs[index : index + 1]

    keep = torch.ones(int(heldout_inputs.shape[0]), dtype=torch.bool, device=heldout_inputs.device)
    keep[index] = False
    return new_x, new_y, heldout_inputs[keep], heldout_outputs[keep]


def _take_pool_observation_numpy(
    heldout_inputs: np.ndarray,
    heldout_outputs: np.ndarray,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if index < 0 or index >= int(heldout_inputs.shape[0]):
        raise IndexError(f"index {index} is out of bounds for pool of size {heldout_inputs.shape[0]}")
    new_x = heldout_inputs[index : index + 1]
    new_y = heldout_outputs[index : index + 1]
    keep = np.ones(int(heldout_inputs.shape[0]), dtype=bool)
    keep[index] = False
    return new_x, new_y, heldout_inputs[keep], heldout_outputs[keep]


def update_random_observations(
    best_random: list[Tensor],
    heldout_inputs: Tensor,
    heldout_outputs: Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[list[Tensor], Tensor, Tensor]:
    """Advance the random baseline by drawing one uniformly random heldout point."""
    if heldout_inputs.shape[0] == 0:
        return best_random, heldout_inputs, heldout_outputs
    index = int(torch.randint(int(heldout_inputs.shape[0]), (1,), generator=generator, device=heldout_inputs.device).item())
    _, new_y, heldout_inputs, heldout_outputs = take_pool_observation(heldout_inputs, heldout_outputs, index)
    next_random_best = torch.maximum(best_random[-1].view(()), new_y.view(()))
    best_random.append(next_random_best)
    return best_random, heldout_inputs, heldout_outputs


def _update_random_observations_numpy(
    best_random: list[float],
    heldout_inputs: np.ndarray,
    heldout_outputs: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Advance the random baseline by drawing one uniformly random heldout point."""
    if heldout_inputs.shape[0] == 0:
        best_random.append(float(best_random[-1]))
        return best_random, heldout_inputs, heldout_outputs

    index = int(rng.integers(int(heldout_inputs.shape[0])))
    _, new_y, heldout_inputs, heldout_outputs = _take_pool_observation_numpy(heldout_inputs, heldout_outputs, index)
    next_random_best = max(float(best_random[-1]), float(np.asarray(new_y).reshape(-1)[0]))
    best_random.append(float(next_random_best))
    return best_random, heldout_inputs, heldout_outputs


def run_random_forest_trials(
    x: np.ndarray | Tensor,
    y: np.ndarray | Tensor,
    *,
    n_trials: int = 20,
    n_init: int = 10,
    n_iters: int = 20,
    beta: float = 2.0,
    n_estimators: int = 256,
    min_samples_leaf: int = 1,
    max_features: str | int | float | None = "sqrt",
    bootstrap: bool = True,
    n_jobs: int | None = -1,
    exclude_global_best_from_initial: bool = True,
    trial_seed_offset: int = 1,
    verbose: bool = False,
) -> TrialSummary:
    """Run a random-forest UCB benchmark on a finite pool."""
    x_np = np.asarray(x)
    y_np = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if x_np.ndim != 2:
        raise ValueError(f"x must have shape [N, D], got {tuple(x_np.shape)}")
    if y_np.shape[0] != x_np.shape[0]:
        raise ValueError(f"x and y must agree on N, got {x_np.shape[0]} vs {y_np.shape[0]}")

    best_observed_all: list[Tensor] = []
    best_random_all: list[Tensor] = []

    for trial in range(1, int(n_trials) + 1):
        if verbose:
            print(f"\nTrial {trial:>2} of {n_trials} ", end="")

        seed = int(trial_seed_offset + trial - 1)
        rng = np.random.default_rng(seed)
        train_x, heldout_x, train_y, heldout_y = split_pool_for_trial(
            x_np,
            y_np,
            n_init=int(n_init),
            trial_seed=seed,
            exclude_global_best_from_initial=bool(exclude_global_best_from_initial),
        )
        heldout_x_random = heldout_x.copy()
        heldout_y_random = heldout_y.copy()

        best_observed = [float(np.max(train_y))]
        best_random = [best_observed[0]]

        for iteration in range(int(n_iters)):
            if heldout_x.shape[0] == 0:
                best_observed.append(float(best_observed[-1]))
                best_random, heldout_x_random, heldout_y_random = _update_random_observations_numpy(
                    best_random,
                    heldout_x_random,
                    heldout_y_random,
                    rng=rng,
                )
                if verbose:
                    print(".", end="")
                continue

            model = fit_random_forest_regressor(
                train_x,
                train_y,
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                n_jobs=n_jobs,
                random_state=seed + iteration,
            )
            mean, std = random_forest_predict_mean_std(model, heldout_x)
            acq = mean + float(beta) * std
            best_idx = int(np.argmax(acq))
            new_x, new_y, heldout_x, heldout_y = _take_pool_observation_numpy(heldout_x, heldout_y, best_idx)
            train_x = np.concatenate([train_x, new_x], axis=0)
            train_y = np.concatenate([train_y, new_y], axis=0)

            best_random, heldout_x_random, heldout_y_random = _update_random_observations_numpy(
                best_random,
                heldout_x_random,
                heldout_y_random,
                rng=rng,
            )
            best_observed.append(max(float(best_observed[-1]), float(np.asarray(new_y).reshape(-1)[0])))
            if verbose:
                print(".", end="")

        best_observed_all.append(torch.as_tensor(best_observed, dtype=torch.double))
        best_random_all.append(torch.as_tensor(best_random, dtype=torch.double))

    return TrialSummary(
        best_observed_all=torch.stack(best_observed_all),
        best_random_all=torch.stack(best_random_all),
    )


def run_discrete_bo_trials(
    x: np.ndarray | Tensor,
    y: np.ndarray | Tensor,
    *,
    initialize_model: ModelInitializer,
    fit_model: ModelFitter,
    acquisition_factory: ScalarAcquisitionFactory,
    n_trials: int = 20,
    n_init: int = 10,
    n_iters: int = 20,
    dtype: torch.dtype = torch.double,
    device: str | torch.device = "cpu",
    exclude_global_best_from_initial: bool = True,
    trial_seed_offset: int = 1,
    verbose: bool = False,
) -> TrialSummary:
    """Run a discrete BO-vs-random benchmark on a finite pool."""
    x_np = np.asarray(x, dtype=np.int64)
    y_np = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    device = torch.device(device)

    best_observed_all: list[Tensor] = []
    best_random_all: list[Tensor] = []

    for trial in range(1, int(n_trials) + 1):
        if verbose:
            print(f"\nTrial {trial:>2} of {n_trials} ", end="")

        train_x_np, heldout_x_np, train_y_np, heldout_y_np = split_pool_for_trial(
            x_np,
            y_np,
            n_init=int(n_init),
            trial_seed=int(trial_seed_offset + trial - 1),
            exclude_global_best_from_initial=bool(exclude_global_best_from_initial),
        )

        train_x = torch.as_tensor(train_x_np, dtype=dtype, device=device)
        heldout_x = torch.as_tensor(heldout_x_np, dtype=dtype, device=device)
        train_y = torch.as_tensor(train_y_np, dtype=dtype, device=device)
        heldout_y = torch.as_tensor(heldout_y_np, dtype=dtype, device=device)

        heldout_x_random = heldout_x.clone()
        heldout_y_random = heldout_y.clone()
        random_gen = torch.Generator(device=device.type if device.type != "mps" else "cpu")
        random_gen.manual_seed(int(trial_seed_offset + trial - 1))

        best_observed = [train_y.max().view(())]
        best_random = [best_observed[0].clone()]
        mll, model = initialize_model(train_x, train_y)

        for _ in range(int(n_iters)):
            fit_model(mll)
            if hasattr(model, "eval"):
                model.eval()
            likelihood = getattr(model, "likelihood", None)
            if likelihood is not None and hasattr(likelihood, "eval"):
                likelihood.eval()
            acq_func = acquisition_factory(model, train_y)
            new_x, new_y, heldout_x, heldout_y = select_best_pool_observation(acq_func, heldout_x, heldout_y)

            train_x = torch.cat([train_x, new_x], dim=0)
            train_y = torch.cat([train_y, new_y], dim=0)

            best_random, heldout_x_random, heldout_y_random = update_random_observations(
                best_random,
                heldout_x_random,
                heldout_y_random,
                generator=random_gen,
            )
            best_observed.append(torch.maximum(best_observed[-1], new_y.view(())))

            prev_state_dict = model.state_dict() if hasattr(model, "state_dict") else None
            mll, model = _reinitialize_model(initialize_model, train_x, train_y, prev_state_dict)
            if verbose:
                print(".", end="")

        best_observed_all.append(torch.stack(best_observed))
        best_random_all.append(torch.stack(best_random))

    return TrialSummary(
        best_observed_all=torch.stack(best_observed_all),
        best_random_all=torch.stack(best_random_all),
    )


def _reinitialize_model(
    initialize_model: ModelInitializer,
    train_x: Tensor,
    train_y: Tensor,
    state_dict: Optional[dict[str, Any]],
) -> tuple[Any, Any]:
    try:
        signature = inspect.signature(initialize_model)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and len(signature.parameters) >= 3:
        return initialize_model(train_x, train_y, state_dict)
    return initialize_model(train_x, train_y)


def make_ehrlich_diffusion_gp_initializer(
    *,
    instance: Optional[Any] = None,
    instance_path: Optional[str | Path] = None,
    tau: float = 1.0,
    combine: str = "mean",
    use_transition_weights: bool = True,
    normalized_laplacian: bool = True,
    standardize_outcomes: bool = True,
    position_weights: Optional[Tensor] = None,
) -> tuple[ModelInitializer, Tensor]:
    """Build a GP initializer using a diffusion kernel over the Ehrlich token transition graph."""
    if MultivariateNormal is None or GaussianLikelihood is None or ConstantMean is None or ExactMarginalLogLikelihood is None:
        raise ImportError("gpytorch is required to build an Ehrlich diffusion GP initializer")
    try:
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.transforms import Standardize
    except ImportError as exc:
        raise ImportError("botorch is required to build an Ehrlich diffusion GP initializer") from exc

    if instance is None and instance_path is None:
        raise ValueError("Provide either instance or instance_path")

    if instance is None:
        from ehrlich.ehrlich_instance import EhrlichInstance

        loaded = torch.load(Path(instance_path).expanduser(), map_location="cpu", weights_only=False)
        if not isinstance(loaded, EhrlichInstance):
            raise TypeError("instance_path did not contain an EhrlichInstance")
        instance = loaded

    token_similarity = build_token_diffusion_similarity(
        instance.transition_probs,
        tau=float(tau),
        transition_mask=instance.transition_mask,
        use_transition_weights=bool(use_transition_weights),
        normalized_laplacian=bool(normalized_laplacian),
    )

    class EhrlichDiffusionGP(SingleTaskGP):
        def __init__(self, train_X: Tensor, train_Y: Tensor):
            mean_module = ConstantMean()
            model_kwargs: dict[str, Any] = {"likelihood": GaussianLikelihood()}
            if standardize_outcomes:
                model_kwargs["outcome_transform"] = Standardize(m=int(train_Y.shape[-1] if train_Y.ndim > 1 else 1))
            base_kernel = DiscreteSequenceDiffusionKernel(
                token_similarity=token_similarity.to(device=train_X.device, dtype=train_X.dtype),
                combine=combine,
                position_weights=position_weights.to(device=train_X.device, dtype=train_X.dtype)
                if position_weights is not None
                else None,
            )
            covar_module = ScaleKernel(base_kernel) if ScaleKernel is not None else base_kernel
            model_kwargs["mean_module"] = mean_module
            model_kwargs["covar_module"] = covar_module
            super().__init__(train_X, train_Y, **model_kwargs)
            self.to(train_X)

        def forward(self, x: Tensor) -> Any:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    def initialize_model(train_x: Tensor, train_y: Tensor, state_dict: Optional[dict[str, Any]] = None) -> tuple[Any, Any]:
        model = EhrlichDiffusionGP(train_x, train_y).to(train_x)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    return initialize_model, token_similarity


def make_spaced_ngram_gp_initializer(
    *,
    offsets: Tensor | np.ndarray,
    vocab_size: int,
    normalize: bool = True,
    standardize_outcomes: bool = True,
) -> ModelInitializer:
    """Build an exact GP initializer using a sparse spaced n-gram kernel."""
    if MultivariateNormal is None or GaussianLikelihood is None or ConstantMean is None or ExactMarginalLogLikelihood is None:
        raise ImportError("gpytorch is required to build a spaced n-gram GP initializer")
    try:
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.transforms import Standardize
    except ImportError as exc:
        raise ImportError("botorch is required to build a spaced n-gram GP initializer") from exc

    offsets_tensor = torch.as_tensor(offsets, dtype=torch.long)

    class SpacedNGramGP(SingleTaskGP):
        def __init__(self, train_X: Tensor, train_Y: Tensor):
            model_kwargs: dict[str, Any] = {"likelihood": GaussianLikelihood()}
            if standardize_outcomes:
                model_kwargs["outcome_transform"] = Standardize(m=int(train_Y.shape[-1] if train_Y.ndim > 1 else 1))
            base_kernel = SpacedNGramKernel(
                offsets=offsets_tensor,
                vocab_size=int(vocab_size),
                normalize=bool(normalize),
            )
            model_kwargs["mean_module"] = ConstantMean()
            model_kwargs["covar_module"] = ScaleKernel(base_kernel) if ScaleKernel is not None else base_kernel
            super().__init__(train_X, train_Y, **model_kwargs)
            self.to(train_X)

        def forward(self, x: Tensor) -> Any:
            return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def initialize_model(
        train_x: Tensor,
        train_y: Tensor,
        state_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, Any]:
        model = SpacedNGramGP(train_x, train_y).to(train_x)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return ExactMarginalLogLikelihood(model.likelihood, model), model

    return initialize_model


def make_additive_spaced_ngram_gp_initializer(
    *,
    offset_groups: list[Tensor | np.ndarray] | tuple[Tensor | np.ndarray, ...],
    vocab_size: int,
    weights: Tensor | np.ndarray | list[float] | tuple[float, ...] | None = None,
    normalize: bool = True,
    standardize_outcomes: bool = True,
) -> ModelInitializer:
    """Build an exact GP using a fixed convex sum of spaced n-gram kernels."""

    if (
        MultivariateNormal is None
        or GaussianLikelihood is None
        or ConstantMean is None
        or ExactMarginalLogLikelihood is None
    ):
        raise ImportError("gpytorch is required to build an additive spaced n-gram GP initializer")
    try:
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.transforms import Standardize
    except ImportError as exc:
        raise ImportError(
            "botorch is required to build an additive spaced n-gram GP initializer"
        ) from exc

    group_tensors = tuple(torch.as_tensor(group, dtype=torch.long) for group in offset_groups)
    weights_tensor = None if weights is None else torch.as_tensor(weights, dtype=torch.double)

    class AdditiveSpacedNGramGP(SingleTaskGP):
        def __init__(self, train_X: Tensor, train_Y: Tensor):
            model_kwargs: dict[str, Any] = {"likelihood": GaussianLikelihood()}
            if standardize_outcomes:
                model_kwargs["outcome_transform"] = Standardize(
                    m=int(train_Y.shape[-1] if train_Y.ndim > 1 else 1)
                )
            base_kernel = AdditiveSpacedNGramKernel(
                offset_groups=group_tensors,
                vocab_size=int(vocab_size),
                weights=weights_tensor,
                normalize=bool(normalize),
            )
            model_kwargs["mean_module"] = ConstantMean()
            model_kwargs["covar_module"] = (
                ScaleKernel(base_kernel) if ScaleKernel is not None else base_kernel
            )
            super().__init__(train_X, train_Y, **model_kwargs)
            self.to(train_X)

        def forward(self, x: Tensor) -> Any:
            return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def initialize_model(
        train_x: Tensor,
        train_y: Tensor,
        state_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, Any]:
        model = AdditiveSpacedNGramGP(train_x, train_y).to(train_x)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return ExactMarginalLogLikelihood(model.likelihood, model), model

    return initialize_model


def make_pool_sequence_graph_gp_initializer(
    *,
    pool_sequences: np.ndarray | Tensor,
    tau: float = 1.0,
    graph_k: int = 16,
    sigma: Optional[float] = None,
    num_eigs: int = 128,
    block_size: int = 32,
    normalized_laplacian: bool = True,
    standardize_outcomes: bool = True,
) -> tuple[ModelInitializer, dict[str, Any]]:
    """Build a GP initializer from a heat kernel on a finite sequence graph."""
    if MultivariateNormal is None or GaussianLikelihood is None or ConstantMean is None or ExactMarginalLogLikelihood is None:
        raise ImportError("gpytorch is required to build a pool sequence-graph GP initializer")
    try:
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.transforms import Standardize
    except ImportError as exc:
        raise ImportError("botorch is required to build a pool sequence-graph GP initializer") from exc

    pool_x = np.asarray(pool_sequences, dtype=np.int64)
    diffusion_embeddings, graph_meta = build_sequence_graph_diffusion_embeddings(
        pool_x,
        tau=tau,
        graph_k=graph_k,
        sigma=sigma,
        num_eigs=num_eigs,
        block_size=block_size,
        normalized_laplacian=normalized_laplacian,
    )

    class PoolSequenceGraphGP(SingleTaskGP):
        def __init__(self, train_X: Tensor, train_Y: Tensor):
            mean_module = ConstantMean()
            model_kwargs: dict[str, Any] = {"likelihood": GaussianLikelihood()}
            if standardize_outcomes:
                model_kwargs["outcome_transform"] = Standardize(m=int(train_Y.shape[-1] if train_Y.ndim > 1 else 1))
            base_kernel = PoolSequenceGraphKernel(
                pool_sequences=pool_x,
                diffusion_embeddings=diffusion_embeddings.to(device=train_X.device, dtype=train_X.dtype),
            )
            covar_module = ScaleKernel(base_kernel) if ScaleKernel is not None else base_kernel
            model_kwargs["mean_module"] = mean_module
            model_kwargs["covar_module"] = covar_module
            super().__init__(train_X, train_Y, **model_kwargs)
            self.to(train_X)

        def forward(self, x: Tensor) -> Any:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    def initialize_model(train_x: Tensor, train_y: Tensor, state_dict: Optional[dict[str, Any]] = None) -> tuple[Any, Any]:
        model = PoolSequenceGraphGP(train_x, train_y).to(train_x)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    meta = {
        "pool_size": int(pool_x.shape[0]),
        "seq_len": int(pool_x.shape[1]),
        "diffusion_embeddings": diffusion_embeddings,
        **graph_meta,
    }
    return initialize_model, meta


__all__ = [
    "AdditiveSpacedNGramKernel",
    "TrialSummary",
    "DiscreteSequenceDiffusionKernel",
    "PoolSequenceGraphKernel",
    "SpacedNGramKernel",
    "build_sequence_graph_diffusion_embeddings",
    "build_sequence_knn_graph",
    "build_token_diffusion_similarity",
    "deduplicate_sequence_pool",
    "fit_random_forest_regressor",
    "load_sequence_pool_from_jsonl",
    "make_ehrlich_diffusion_gp_initializer",
    "make_additive_spaced_ngram_gp_initializer",
    "make_pool_sequence_graph_gp_initializer",
    "make_local_spaced_pair_offsets",
    "make_local_spaced_ngram_offsets",
    "make_local_spaced_trigram_offsets",
    "make_spaced_ngram_gp_initializer",
    "run_discrete_bo_trials",
    "run_random_forest_trials",
    "random_forest_predict_mean_std",
    "sequence_diffusion_similarity",
    "select_best_pool_observation",
    "split_pool_for_trial",
    "take_pool_observation",
    "update_random_observations",
]
