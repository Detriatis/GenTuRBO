"""Exact Gaussian-process regressors for discrete sequence features."""

from __future__ import annotations

import inspect
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from search.interpolant.output import open_jsonl


def _configure_exact_gp_parameters(
    model: Any,
    *,
    noise_variance: float | None,
    outputscale: float | None,
    learn_outputscale: bool,
    fit_mean_and_noise_only: bool,
) -> None:
    """Apply either the legacy GP policy or the Ehrlich study policy.

    ``fit_mean_and_noise_only`` is explicit rather than inferred from a missing
    fixed-noise value.  Existing fixed-noise and generic GP callers remain
    backward compatible, while the Ehrlich study can expose its exact
    trainability contract on the model itself.
    """

    covar_module = getattr(model, "covar_module", None)
    if fit_mean_and_noise_only:
        if noise_variance is not None:
            raise ValueError("fit_mean_and_noise_only requires fitted likelihood noise")
        if learn_outputscale:
            raise ValueError("fit_mean_and_noise_only forbids outputscale learning")
        if outputscale is not None and not math.isclose(float(outputscale), 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("fit_mean_and_noise_only requires outputscale=1.0")
        if not hasattr(covar_module, "outputscale"):
            raise TypeError("fit_mean_and_noise_only requires a scaled covariance module")
        covar_module.outputscale = 1.0

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        raw_noise = getattr(getattr(model.likelihood, "noise_covar", None), "raw_noise", None)
        if raw_noise is None:
            raise TypeError("fit_mean_and_noise_only requires a trainable Gaussian likelihood noise")
        raw_mean = getattr(getattr(model, "mean_module", None), "raw_constant", None)
        if raw_mean is None:
            raise TypeError("fit_mean_and_noise_only requires a trainable constant mean")
        raw_mean.requires_grad_(True)
        raw_noise.requires_grad_(True)
        return

    fixed_outputscale = 1.0 if outputscale is None and not learn_outputscale else outputscale
    if fixed_outputscale is not None and hasattr(covar_module, "outputscale"):
        covar_module.outputscale = float(fixed_outputscale)
    if not learn_outputscale and hasattr(covar_module, "raw_outputscale"):
        covar_module.raw_outputscale.requires_grad_(False)
    if noise_variance is not None:
        model.likelihood.noise = float(noise_variance)
        model.likelihood.noise_covar.raw_noise.requires_grad_(False)


@dataclass
class FeatureSet:
    """Container for a regression-ready feature matrix and aligned labels."""

    name: str
    x: Any
    y: np.ndarray
    row_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_rows(self) -> int:
        return int(self.y.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])


class TanimotoGPRegressor:
    """Exact GP regressor over dense fingerprint features with a Tanimoto kernel."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        dtype: str = "double",
        standardize_outcomes: bool = True,
        jitter: float | None = None,
        noise_variance: float | None = None,
        outputscale: float | None = None,
        learn_outputscale: bool = True,
        fit_mean_and_noise_only: bool = False,
    ) -> None:
        self.device = str(device)
        self.dtype = str(dtype)
        self.standardize_outcomes = bool(standardize_outcomes)
        self.jitter = None if jitter is None else float(jitter)
        self.noise_variance = None if noise_variance is None else float(noise_variance)
        if self.noise_variance is not None and self.noise_variance <= 0.0:
            raise ValueError("noise_variance must be positive when provided")
        self.outputscale = None if outputscale is None else float(outputscale)
        self.learn_outputscale = bool(learn_outputscale)
        self.fit_mean_and_noise_only = bool(fit_mean_and_noise_only)
        if self.fit_mean_and_noise_only:
            if self.noise_variance is not None:
                raise ValueError("fit_mean_and_noise_only requires fitted likelihood noise")
            if self.learn_outputscale:
                raise ValueError("fit_mean_and_noise_only forbids outputscale learning")
            if self.outputscale is not None and not math.isclose(self.outputscale, 1.0, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("fit_mean_and_noise_only requires outputscale=1.0")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        _ = deep
        return {
            "device": self.device,
            "dtype": self.dtype,
            "standardize_outcomes": self.standardize_outcomes,
            "jitter": self.jitter,
            "noise_variance": self.noise_variance,
            "outputscale": self.outputscale,
            "learn_outputscale": self.learn_outputscale,
            "fit_mean_and_noise_only": self.fit_mean_and_noise_only,
        }

    def set_params(self, **params: Any) -> "TanimotoGPRegressor":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _torch_dtype(self) -> torch.dtype:
        name = self.dtype.strip().lower()
        if name in {"double", "float64", "torch.double"}:
            return torch.double
        if name in {"float", "float32", "torch.float"}:
            return torch.float
        raise ValueError(f"Unsupported dtype '{self.dtype}' for TanimotoGPRegressor")

    def _coerce_x(self, x: Any) -> np.ndarray:
        if _feature_matrix_is_sparse(x):
            x = x.toarray()
        x_np = np.asarray(x, dtype=np.float64)
        if x_np.ndim != 2:
            raise ValueError(f"x must have shape [N, D], got {tuple(x_np.shape)}")
        return x_np

    def _coerce_y(self, y: Any) -> np.ndarray:
        y_np = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if y_np.ndim != 2 or y_np.shape[1] != 1:
            raise ValueError(f"y must be one-dimensional or shaped [N, 1], got {tuple(y_np.shape)}")
        return y_np

    def _coerce_x_batch(self, x: Any) -> np.ndarray:
        if _feature_matrix_is_sparse(x):
            x = x.toarray()
        x_np = np.asarray(x, dtype=np.float64)
        if x_np.ndim == 2:
            x_np = x_np[:, None, :]
        if x_np.ndim != 3:
            raise ValueError(f"x_batch must have shape [N, q, D] or [N, D], got {tuple(x_np.shape)}")
        return x_np

    def _jitter_context(self) -> Any:
        if self.jitter is None or self.jitter <= 0.0:
            return nullcontext()
        try:
            from gpytorch.settings import cholesky_jitter
        except ImportError as exc:
            raise ImportError("gpytorch is required to use TanimotoGPRegressor") from exc
        try:
            return cholesky_jitter(
                float_value=self.jitter,
                double_value=self.jitter,
                half_value=self.jitter,
            )
        except TypeError:
            return cholesky_jitter(
                float=self.jitter,
                double=self.jitter,
                half=self.jitter,
            )

    def _make_base_kernel(self) -> Any:
        from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

        return TanimotoKernel()

    def fit(self, x: Any, y: Any) -> "TanimotoGPRegressor":
        try:
            from botorch.fit import fit_gpytorch_mll
            from botorch.models.gp_regression import SingleTaskGP
            from botorch.models.transforms import Standardize
            from gpytorch.distributions import MultivariateNormal
            from gpytorch.kernels import ScaleKernel
            from gpytorch.likelihoods import GaussianLikelihood
            from gpytorch.means import ConstantMean
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError as exc:
            raise ImportError("botorch and gpytorch are required to use exact GP regressors") from exc

        x_np = self._coerce_x(x)
        y_np = self._coerce_y(y)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"x and y must agree on N, got {x_np.shape[0]} vs {y_np.shape[0]}")

        device = torch.device(self.device)
        dtype = self._torch_dtype()
        train_x = torch.as_tensor(x_np, dtype=dtype, device=device)
        train_y = torch.as_tensor(y_np, dtype=dtype, device=device)

        standardize_outcomes = self.standardize_outcomes
        noise_variance = self.noise_variance
        outputscale = self.outputscale
        learn_outputscale = self.learn_outputscale
        fit_mean_and_noise_only = self.fit_mean_and_noise_only
        base_kernel = self._make_base_kernel()

        class _TanimotoExactGP(SingleTaskGP):
            def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor):
                model_kwargs: dict[str, Any] = {"likelihood": GaussianLikelihood()}
                if standardize_outcomes:
                    model_kwargs["outcome_transform"] = Standardize(m=int(train_Y.shape[-1]))
                super().__init__(train_X, train_Y, **model_kwargs)
                self.mean_module = ConstantMean()
                self.covar_module = ScaleKernel(base_kernel=base_kernel)
                self.to(train_X)
                _configure_exact_gp_parameters(
                    self,
                    noise_variance=noise_variance,
                    outputscale=outputscale,
                    learn_outputscale=learn_outputscale,
                    fit_mean_and_noise_only=fit_mean_and_noise_only,
                )

            def forward(self, inputs: torch.Tensor) -> MultivariateNormal:
                mean_x = self.mean_module(inputs)
                covar_x = self.covar_module(inputs)
                return MultivariateNormal(mean_x, covar_x)

        self.model_ = _TanimotoExactGP(train_x, train_y)
        self.mll_ = ExactMarginalLogLikelihood(self.model_.likelihood, self.model_)
        with self._jitter_context():
            fit_gpytorch_mll(self.mll_)
        self.model_.eval()
        self.model_.likelihood.eval()
        self.n_features_in_ = int(train_x.shape[1])
        self.train_x_ = train_x
        self.train_y_ = train_y
        return self

    def refresh_training_data(self, x: Any, y: Any) -> TanimotoGPRegressor:
        """Condition the fitted GP on a new archive without refitting hyperparameters.

        The outcome transform, constant mean, likelihood noise, and fixed
        kernel parameters remain exactly as they were at the last full fit.
        Only the exact GP training inputs and targets are replaced, so the
        posterior still uses every observation in the current archive.
        """

        if not hasattr(self, "model_"):
            raise RuntimeError("TanimotoGPRegressor must be fit before refreshing training data")
        x_np = self._coerce_x(x)
        y_np = self._coerce_y(y)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"x and y must agree on N, got {x_np.shape[0]} vs {y_np.shape[0]}")
        if x_np.shape[1] != self.n_features_in_:
            raise ValueError(f"x has {x_np.shape[1]} features, but this model was fit on {self.n_features_in_}")

        train_x = torch.as_tensor(
            x_np,
            dtype=self._torch_dtype(),
            device=torch.device(self.device),
        )
        train_y = torch.as_tensor(
            y_np,
            dtype=self._torch_dtype(),
            device=torch.device(self.device),
        )
        outcome_transform = getattr(self.model_, "outcome_transform", None)
        if outcome_transform is not None:
            outcome_transform.eval()
            with torch.no_grad():
                transformed_y, _ = outcome_transform(train_y, X=train_x)
        else:
            transformed_y = train_y
        self.model_.set_train_data(
            inputs=train_x,
            targets=transformed_y.squeeze(-1),
            strict=False,
        )
        self.model_.eval()
        self.model_.likelihood.eval()
        self.train_x_ = train_x
        self.train_y_ = train_y
        return self

    def _posterior(self, x: Any) -> Any:
        if not hasattr(self, "model_"):
            raise RuntimeError("TanimotoGPRegressor must be fit before calling predict")
        x_np = self._coerce_x(x)
        if x_np.shape[1] != self.n_features_in_:
            raise ValueError(f"x has {x_np.shape[1]} features, but this model was fit on {self.n_features_in_}")
        x_tensor = torch.as_tensor(
            x_np,
            dtype=self._torch_dtype(),
            device=torch.device(self.device),
        )
        with torch.no_grad():
            with self._jitter_context():
                return self.model_.posterior(x_tensor)

    def predict(self, x: Any) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_std(self, x: Any) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std

    def predict_mean_std(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior moments from one exact-GP posterior evaluation."""

        posterior = self._posterior(x)
        mean = posterior.mean.detach().cpu().view(-1).numpy()
        std = posterior.variance.clamp_min(0.0).sqrt().detach().cpu().view(-1).numpy()
        return mean, std

    def sample_posterior(self, x: Any, *, num_samples: int = 1) -> np.ndarray:
        if int(num_samples) < 1:
            raise ValueError("num_samples must be >= 1")
        posterior = self._posterior(x)
        with torch.no_grad():
            samples = posterior.rsample(sample_shape=torch.Size([int(num_samples)]))
        samples_np = samples.detach().cpu().numpy()
        if int(num_samples) == 1:
            return np.asarray(samples_np[0]).reshape(-1)
        return np.asarray(samples_np).reshape(int(num_samples), -1)

    def q_expected_improvement(
        self,
        x_batch: Any,
        *,
        best: float,
        num_samples: int = 128,
    ) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TanimotoGPRegressor must be fit before calling q_expected_improvement")
        if int(num_samples) < 1:
            raise ValueError("num_samples must be >= 1")
        try:
            from botorch.acquisition.monte_carlo import qExpectedImprovement

            try:
                from botorch.sampling.normal import SobolQMCNormalSampler
            except ImportError:
                from botorch.sampling.samplers import SobolQMCNormalSampler
        except ImportError as exc:
            raise ImportError("botorch is required to evaluate batch qEI") from exc

        x_np = self._coerce_x_batch(x_batch)
        if x_np.shape[-1] != self.n_features_in_:
            raise ValueError(f"x_batch has {x_np.shape[-1]} features, but this model was fit on {self.n_features_in_}")
        x_tensor = torch.as_tensor(
            x_np,
            dtype=self._torch_dtype(),
            device=torch.device(self.device),
        )

        sampler_kwargs = {}
        if "sample_shape" in inspect.signature(SobolQMCNormalSampler).parameters:
            sampler_kwargs["sample_shape"] = torch.Size([int(num_samples)])
        else:
            sampler_kwargs["num_samples"] = int(num_samples)
        sampler = SobolQMCNormalSampler(**sampler_kwargs)
        acqf = qExpectedImprovement(
            model=self.model_,
            best_f=float(best),
            sampler=sampler,
        )
        with torch.no_grad():
            with self._jitter_context():
                values = acqf(x_tensor)
        return values.detach().cpu().reshape(-1).numpy()


class EhrlichTokenGPRegressor(TanimotoGPRegressor):
    """Shared input validation for exact GPs over Ehrlich token sequences."""

    def __init__(
        self,
        *,
        instance: Any,
        seq_len: int,
        vocab_size: int,
        device: str = "cpu",
        dtype: str = "double",
        standardize_outcomes: bool = True,
        jitter: float | None = None,
        noise_variance: float | None = None,
        outputscale: float | None = None,
        learn_outputscale: bool = True,
        fit_mean_and_noise_only: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype,
            standardize_outcomes=standardize_outcomes,
            jitter=jitter,
            noise_variance=noise_variance,
            outputscale=outputscale,
            learn_outputscale=learn_outputscale,
            fit_mean_and_noise_only=fit_mean_and_noise_only,
        )
        self.instance = instance
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update(
            {
                "instance": self.instance,
                "seq_len": self.seq_len,
                "vocab_size": self.vocab_size,
            }
        )
        return params

    def _onehot_to_tokens(self, values: np.ndarray) -> np.ndarray:
        expected = self.seq_len * self.vocab_size
        if values.shape[-1] == self.seq_len:
            rounded = np.rint(values)
            if not np.allclose(values, rounded, atol=1e-8, rtol=0.0):
                raise ValueError("raw Ehrlich token features must be integer-valued")
            tokens = rounded.astype(np.int64)
        elif values.shape[-1] == expected:
            onehot = values.reshape(values.shape[:-1] + (self.seq_len, self.vocab_size))
            if not np.isfinite(onehot).all() or np.any(onehot < -1e-8):
                raise ValueError("Ehrlich one-hot features must be finite and non-negative")
            row_sums = onehot.sum(axis=-1)
            maxima = onehot.max(axis=-1)
            if not np.allclose(row_sums, 1.0, atol=1e-8, rtol=0.0) or np.any(maxima < 1.0 - 1e-8):
                raise ValueError("Ehrlich one-hot features must contain exactly one active token per position")
            tokens = onehot.argmax(axis=-1).astype(np.int64)
        else:
            raise ValueError(
                f"Ehrlich features must have final dimension {self.seq_len} (tokens) "
                f"or {expected} (one-hot), got {values.shape[-1]}"
            )
        if np.any(tokens < 0) or np.any(tokens >= self.vocab_size):
            raise ValueError(f"Ehrlich tokens must be in [0, {self.vocab_size})")
        return tokens

    def _coerce_x(self, x: Any) -> np.ndarray:
        if _feature_matrix_is_sparse(x):
            x = x.toarray()
        x_np = np.asarray(x, dtype=np.float64)
        if x_np.ndim != 2:
            raise ValueError(f"x must have shape [N, D], got {tuple(x_np.shape)}")
        return self._onehot_to_tokens(x_np)

    def _coerce_x_batch(self, x: Any) -> np.ndarray:
        if _feature_matrix_is_sparse(x):
            x = x.toarray()
        x_np = np.asarray(x, dtype=np.float64)
        if x_np.ndim == 2:
            x_np = x_np[:, None, :]
        if x_np.ndim != 3:
            raise ValueError(f"x_batch must have shape [N, q, D] or [N, D], got {tuple(x_np.shape)}")
        return self._onehot_to_tokens(x_np)


class EhrlichDiffusionGPRegressor(EhrlichTokenGPRegressor):
    """Exact GP over Ehrlich token sequences with a transition heat kernel.

    The live active-learning selector stores fixed-position one-hot features for
    deduplication and diversity diagnostics. This regressor losslessly converts
    those features back to categorical token sequences before fitting the
    Ehrlich-specific kernel.
    """

    def __init__(
        self,
        *,
        instance: Any,
        seq_len: int,
        vocab_size: int,
        tau: float = 1.0,
        combine: str = "mean",
        use_transition_weights: bool = True,
        normalized_laplacian: bool = True,
        device: str = "cpu",
        dtype: str = "double",
        standardize_outcomes: bool = True,
        jitter: float | None = None,
        noise_variance: float | None = None,
        outputscale: float | None = None,
        learn_outputscale: bool = True,
        fit_mean_and_noise_only: bool = False,
    ) -> None:
        super().__init__(
            instance=instance,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
            standardize_outcomes=standardize_outcomes,
            jitter=jitter,
            noise_variance=noise_variance,
            outputscale=outputscale,
            learn_outputscale=learn_outputscale,
            fit_mean_and_noise_only=fit_mean_and_noise_only,
        )
        self.tau = float(tau)
        self.combine = str(combine).strip().lower()
        self.use_transition_weights = bool(use_transition_weights)
        self.normalized_laplacian = bool(normalized_laplacian)
        if self.tau <= 0.0 or not np.isfinite(self.tau):
            raise ValueError("tau must be finite and positive")
        if self.combine not in {"mean", "sum", "product"}:
            raise ValueError("combine must be one of: mean, sum, product")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update(
            {
                "tau": self.tau,
                "combine": self.combine,
                "use_transition_weights": self.use_transition_weights,
                "normalized_laplacian": self.normalized_laplacian,
            }
        )
        return params

    def fit(self, x: Any, y: Any) -> EhrlichDiffusionGPRegressor:
        try:
            from botorch.fit import fit_gpytorch_mll
        except ImportError as exc:
            raise ImportError("botorch is required to use EhrlichDiffusionGPRegressor") from exc
        from surrogate.model import make_ehrlich_diffusion_gp_initializer

        x_np = self._coerce_x(x)
        y_np = self._coerce_y(y)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"x and y must agree on N, got {x_np.shape[0]} vs {y_np.shape[0]}")

        device = torch.device(self.device)
        dtype = self._torch_dtype()
        train_x = torch.as_tensor(x_np, dtype=dtype, device=device)
        train_y = torch.as_tensor(y_np, dtype=dtype, device=device)
        initialize_model, _ = make_ehrlich_diffusion_gp_initializer(
            instance=self.instance,
            tau=self.tau,
            combine=self.combine,
            use_transition_weights=self.use_transition_weights,
            normalized_laplacian=self.normalized_laplacian,
            standardize_outcomes=self.standardize_outcomes,
        )
        self.mll_, self.model_ = initialize_model(train_x, train_y)
        _configure_exact_gp_parameters(
            self.model_,
            noise_variance=self.noise_variance,
            outputscale=self.outputscale,
            learn_outputscale=self.learn_outputscale,
            fit_mean_and_noise_only=self.fit_mean_and_noise_only,
        )
        with self._jitter_context():
            fit_gpytorch_mll(self.mll_)
        self.model_.eval()
        self.model_.likelihood.eval()
        self.n_features_in_ = self.seq_len
        self.train_x_ = train_x
        self.train_y_ = train_y
        return self


class SpacedNGramGPRegressor(EhrlichTokenGPRegressor):
    """Exact GP over sparse counts of categorical n-grams at fixed gaps."""

    def __init__(
        self,
        *,
        instance: Any,
        seq_len: int,
        vocab_size: int,
        offsets: Any | None = None,
        normalize: bool = True,
        device: str = "cpu",
        dtype: str = "double",
        standardize_outcomes: bool = True,
        jitter: float | None = None,
        noise_variance: float | None = None,
        outputscale: float | None = None,
        learn_outputscale: bool = True,
        fit_mean_and_noise_only: bool = False,
    ) -> None:
        super().__init__(
            instance=instance,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
            standardize_outcomes=standardize_outcomes,
            jitter=jitter,
            noise_variance=noise_variance,
            outputscale=outputscale,
            learn_outputscale=learn_outputscale,
            fit_mean_and_noise_only=fit_mean_and_noise_only,
        )
        raw_offsets = getattr(instance, "spacings", None) if offsets is None else offsets
        if raw_offsets is None:
            raise ValueError("offsets are required when instance does not expose spacings")
        offsets_np = np.asarray(torch.as_tensor(raw_offsets, dtype=torch.long).cpu())
        if offsets_np.ndim == 1:
            offsets_np = offsets_np.reshape(1, -1)
        if offsets_np.ndim != 2 or offsets_np.shape[0] < 1 or offsets_np.shape[1] < 1:
            raise ValueError("offsets must have shape [num_patterns, n]")
        self.offsets = offsets_np.astype(np.int64, copy=True)
        self.normalize = bool(normalize)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update({"offsets": self.offsets.copy(), "normalize": self.normalize})
        return params

    def fit(self, x: Any, y: Any) -> SpacedNGramGPRegressor:
        try:
            from botorch.fit import fit_gpytorch_mll
        except ImportError as exc:
            raise ImportError("botorch is required to use SpacedNGramGPRegressor") from exc
        from surrogate.model import make_spaced_ngram_gp_initializer

        x_np = self._coerce_x(x)
        y_np = self._coerce_y(y)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"x and y must agree on N, got {x_np.shape[0]} vs {y_np.shape[0]}")

        device = torch.device(self.device)
        dtype = self._torch_dtype()
        train_x = torch.as_tensor(x_np, dtype=dtype, device=device)
        train_y = torch.as_tensor(y_np, dtype=dtype, device=device)
        initialize_model = make_spaced_ngram_gp_initializer(
            offsets=self.offsets,
            vocab_size=self.vocab_size,
            normalize=self.normalize,
            standardize_outcomes=self.standardize_outcomes,
        )
        self.mll_, self.model_ = initialize_model(train_x, train_y)
        _configure_exact_gp_parameters(
            self.model_,
            noise_variance=self.noise_variance,
            outputscale=self.outputscale,
            learn_outputscale=self.learn_outputscale,
            fit_mean_and_noise_only=self.fit_mean_and_noise_only,
        )
        with self._jitter_context():
            fit_gpytorch_mll(self.mll_)
        self.model_.eval()
        self.model_.likelihood.eval()
        self.n_features_in_ = self.seq_len
        self.train_x_ = train_x
        self.train_y_ = train_y
        return self


class AdditiveSpacedNGramGPRegressor(EhrlichTokenGPRegressor):
    """Exact GP over a fixed convex combination of spaced n-gram kernels."""

    def __init__(
        self,
        *,
        instance: Any,
        seq_len: int,
        vocab_size: int,
        offset_groups: list[Any] | tuple[Any, ...],
        weights: Any | None = None,
        normalize: bool = True,
        device: str = "cpu",
        dtype: str = "double",
        standardize_outcomes: bool = True,
        jitter: float | None = None,
        noise_variance: float | None = None,
        outputscale: float | None = None,
        learn_outputscale: bool = True,
        fit_mean_and_noise_only: bool = False,
    ) -> None:
        super().__init__(
            instance=instance,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
            standardize_outcomes=standardize_outcomes,
            jitter=jitter,
            noise_variance=noise_variance,
            outputscale=outputscale,
            learn_outputscale=learn_outputscale,
            fit_mean_and_noise_only=fit_mean_and_noise_only,
        )
        if len(offset_groups) < 2:
            raise ValueError("offset_groups must contain at least two spaced n-gram components")
        self.offset_groups: tuple[np.ndarray, ...] = tuple(
            np.asarray(torch.as_tensor(group, dtype=torch.long).cpu(), dtype=np.int64) for group in offset_groups
        )
        if any(group.ndim != 2 or group.shape[0] < 1 or group.shape[1] < 1 for group in self.offset_groups):
            raise ValueError("every offset group must have shape [num_patterns, n]")
        raw_weights = np.ones(len(self.offset_groups), dtype=np.float64)
        if weights is not None:
            raw_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if raw_weights.size != len(self.offset_groups):
            raise ValueError("weights must contain one value per offset group")
        if not np.isfinite(raw_weights).all() or np.any(raw_weights <= 0.0):
            raise ValueError("additive spaced n-gram weights must be finite and positive")
        self.weights = raw_weights / raw_weights.sum()
        self.normalize = bool(normalize)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update(
            {
                "offset_groups": tuple(group.copy() for group in self.offset_groups),
                "weights": self.weights.copy(),
                "normalize": self.normalize,
            }
        )
        return params

    def fit(self, x: Any, y: Any) -> AdditiveSpacedNGramGPRegressor:
        try:
            from botorch.fit import fit_gpytorch_mll
        except ImportError as exc:
            raise ImportError("botorch is required to use AdditiveSpacedNGramGPRegressor") from exc
        from surrogate.model import make_additive_spaced_ngram_gp_initializer

        x_np = self._coerce_x(x)
        y_np = self._coerce_y(y)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"x and y must agree on N, got {x_np.shape[0]} vs {y_np.shape[0]}")

        device = torch.device(self.device)
        dtype = self._torch_dtype()
        train_x = torch.as_tensor(x_np, dtype=dtype, device=device)
        train_y = torch.as_tensor(y_np, dtype=dtype, device=device)
        initialize_model = make_additive_spaced_ngram_gp_initializer(
            offset_groups=list(self.offset_groups),
            vocab_size=self.vocab_size,
            weights=self.weights,
            normalize=self.normalize,
            standardize_outcomes=self.standardize_outcomes,
        )
        self.mll_, self.model_ = initialize_model(train_x, train_y)
        _configure_exact_gp_parameters(
            self.model_,
            noise_variance=self.noise_variance,
            outputscale=self.outputscale,
            learn_outputscale=self.learn_outputscale,
            fit_mean_and_noise_only=self.fit_mean_and_noise_only,
        )
        with self._jitter_context():
            fit_gpytorch_mll(self.mll_)
        self.model_.eval()
        self.model_.likelihood.eval()
        self.n_features_in_ = self.seq_len
        self.train_x_ = train_x
        self.train_y_ = train_y
        return self



def _feature_matrix_is_sparse(x: Any) -> bool:
    try:
        from scipy import sparse
    except ImportError:
        return False
    return bool(sparse.issparse(x))
