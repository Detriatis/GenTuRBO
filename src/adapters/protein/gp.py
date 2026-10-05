"""Gaussian-process regression on dense protein embeddings."""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import math
from typing import Any

import numpy as np
import torch


class EmbeddingGPRegressor:
    """Exact GP regressor over dense ESM embedding vectors."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        dtype: str = "double",
        standardize_inputs: bool = True,
        standardize_outcomes: bool = True,
        jitter: float | None = None,
        noise_variance: float | None = None,
        outputscale: float | None = None,
        learn_outputscale: bool = True,
        lengthscale: float | None = None,
        learn_lengthscale: bool = False,
    ) -> None:
        self.device = str(device)
        self.dtype = str(dtype)
        self.standardize_inputs = bool(standardize_inputs)
        self.standardize_outcomes = bool(standardize_outcomes)
        self.jitter = None if jitter is None else float(jitter)
        self.noise_variance = None if noise_variance is None else float(noise_variance)
        if self.noise_variance is not None and self.noise_variance <= 0.0:
            raise ValueError("noise_variance must be positive when provided")
        self.outputscale = None if outputscale is None else float(outputscale)
        self.learn_outputscale = bool(learn_outputscale)
        self.lengthscale = None if lengthscale is None else float(lengthscale)
        if self.lengthscale is not None and self.lengthscale <= 0.0:
            raise ValueError("lengthscale must be positive when provided")
        self.learn_lengthscale = bool(learn_lengthscale)

    def _torch_dtype(self) -> torch.dtype:
        name = self.dtype.strip().lower()
        if name in {"double", "float64", "torch.double"}:
            return torch.double
        if name in {"float", "float32", "torch.float"}:
            return torch.float
        raise ValueError(f"Unsupported dtype '{self.dtype}' for EmbeddingGPRegressor")

    @staticmethod
    def _coerce_x(x: Any) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float64)
        if x_np.ndim != 2:
            raise ValueError(f"x must have shape [N, D], got {tuple(x_np.shape)}")
        return x_np

    @staticmethod
    def _coerce_y(y: Any) -> np.ndarray:
        y_np = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if y_np.ndim != 2 or y_np.shape[1] != 1:
            raise ValueError(f"y must be one-dimensional or shaped [N, 1], got {tuple(y_np.shape)}")
        return y_np

    @staticmethod
    def _coerce_x_batch(x: Any) -> np.ndarray:
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
            raise ImportError("gpytorch is required to use EmbeddingGPRegressor") from exc
        try:
            return cholesky_jitter(float_value=self.jitter, double_value=self.jitter, half_value=self.jitter)
        except TypeError:
            return cholesky_jitter(float=self.jitter, double=self.jitter, half=self.jitter)

    @staticmethod
    def _default_lengthscale(n_features: int) -> float:
        n = int(n_features)
        if n <= 0:
            raise ValueError("n_features must be positive")
        return math.sqrt(float(n))

    @staticmethod
    def _lengthscale_kernel(covar_module: Any) -> Any | None:
        if covar_module is None:
            return None
        stack = [covar_module]
        seen: set[int] = set()
        while stack:
            module = stack.pop()
            if module is None:
                continue
            ident = id(module)
            if ident in seen:
                continue
            seen.add(ident)
            if hasattr(module, "lengthscale") and hasattr(module, "raw_lengthscale"):
                return module
            for attr in ("base_kernel", "data_covar_module", "covar_module"):
                child = getattr(module, attr, None)
                if child is not None:
                    stack.append(child)
            kernels = getattr(module, "kernels", None)
            if kernels is not None:
                stack.extend(list(kernels))
        return None

    def _resolve_lengthscale(self, n_features: int) -> float:
        if self.lengthscale is not None:
            return float(self.lengthscale)
        return self._default_lengthscale(n_features)

    def _set_kernel_lengthscale(
        self,
        covar_module: Any,
        *,
        n_features: int,
    ) -> float | None:
        kernel = self._lengthscale_kernel(covar_module)
        if kernel is None:
            return None
        value = self._resolve_lengthscale(n_features)
        current = getattr(kernel, "lengthscale", None)
        if isinstance(current, torch.Tensor):
            kernel.lengthscale = torch.full_like(current.detach(), float(value))
        else:
            kernel.lengthscale = float(value)
        raw_lengthscale = getattr(kernel, "raw_lengthscale", None)
        if not self.learn_lengthscale and raw_lengthscale is not None:
            raw_lengthscale.requires_grad_(False)
        return value

    def _normalize_train_x(self, x: np.ndarray) -> np.ndarray:
        if not self.standardize_inputs:
            self.x_min_ = np.zeros((1, x.shape[1]), dtype=np.float64)
            self.x_range_ = np.ones((1, x.shape[1]), dtype=np.float64)
            return x
        self.x_min_ = x.min(axis=0, keepdims=True)
        x_max = x.max(axis=0, keepdims=True)
        self.x_range_ = x_max - self.x_min_
        self.x_range_ = np.where(self.x_range_ > 1e-12, self.x_range_, 1.0)
        return (x - self.x_min_) / self.x_range_

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "x_min_") or not hasattr(self, "x_range_"):
            raise RuntimeError("EmbeddingGPRegressor must be fit before prediction")
        return (x - self.x_min_) / self.x_range_

    def fit(self, x: Any, y: Any) -> "EmbeddingGPRegressor":
        try:
            from botorch.fit import fit_gpytorch_mll
            from botorch.models.gp_regression import SingleTaskGP
            from botorch.models.transforms import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError as exc:
            raise ImportError("botorch and gpytorch are required to use EmbeddingGPRegressor") from exc

        x_np = self._coerce_x(x)
        y_np = self._coerce_y(y)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"x and y must agree on N, got {x_np.shape[0]} vs {y_np.shape[0]}")
        x_np = self._normalize_train_x(x_np)
        device = torch.device(self.device)
        dtype = self._torch_dtype()
        train_x = torch.as_tensor(x_np, dtype=dtype, device=device)
        train_y = torch.as_tensor(y_np, dtype=dtype, device=device)
        train_yvar = (
            torch.full_like(train_y, float(self.noise_variance))
            if self.noise_variance is not None
            else None
        )
        kwargs: dict[str, Any] = {}
        if self.standardize_outcomes:
            kwargs["outcome_transform"] = Standardize(m=int(train_y.shape[-1]))
        if train_yvar is not None:
            kwargs["train_Yvar"] = train_yvar
        self.model_ = SingleTaskGP(train_x, train_y, **kwargs)
        self.n_features_in_ = int(train_x.shape[1])
        fixed_outputscale = 1.0 if self.outputscale is None and not self.learn_outputscale else self.outputscale
        covar_module = getattr(self.model_, "covar_module", None)
        self.lengthscale_used_ = self._set_kernel_lengthscale(
            covar_module,
            n_features=self.n_features_in_,
        )
        if fixed_outputscale is not None and hasattr(covar_module, "outputscale"):
            covar_module.outputscale = float(fixed_outputscale)
        raw_outputscale = getattr(covar_module, "raw_outputscale", None)
        if not self.learn_outputscale and raw_outputscale is not None:
            raw_outputscale.requires_grad_(False)
        self.mll_ = ExactMarginalLogLikelihood(self.model_.likelihood, self.model_)
        with self._jitter_context():
            fit_gpytorch_mll(self.mll_)
        self.model_.eval()
        self.model_.likelihood.eval()
        return self

    def _posterior(self, x: Any) -> Any:
        if not hasattr(self, "model_"):
            raise RuntimeError("EmbeddingGPRegressor must be fit before calling predict")
        x_np = self._coerce_x(x)
        if x_np.shape[1] != self.n_features_in_:
            raise ValueError(f"x has {x_np.shape[1]} features, but this model was fit on {self.n_features_in_}")
        x_np = self._normalize_x(x_np)
        x_tensor = torch.as_tensor(x_np, dtype=self._torch_dtype(), device=torch.device(self.device))
        with torch.no_grad():
            with self._jitter_context():
                return self.model_.posterior(x_tensor)

    def predict(self, x: Any) -> np.ndarray:
        posterior = self._posterior(x)
        return posterior.mean.detach().cpu().view(-1).numpy()

    def predict_std(self, x: Any) -> np.ndarray:
        posterior = self._posterior(x)
        return posterior.variance.clamp_min(0.0).sqrt().detach().cpu().view(-1).numpy()

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

    def _q_expected_improvement_values(
        self,
        x_batch: Any,
        *,
        best: float,
        num_samples: int,
        return_log: bool,
    ) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("EmbeddingGPRegressor must be fit before calling q_expected_improvement")
        returns_log = False
        try:
            try:
                from botorch.acquisition.logei import qLogExpectedImprovement as QEI
                returns_log = True
            except ImportError:
                from botorch.acquisition.monte_carlo import qExpectedImprovement as QEI
            try:
                from botorch.sampling.normal import SobolQMCNormalSampler
            except ImportError:
                from botorch.sampling.samplers import SobolQMCNormalSampler
        except ImportError as exc:
            raise ImportError("botorch is required to evaluate batch qEI") from exc

        x_np = self._coerce_x_batch(x_batch)
        if x_np.shape[-1] != self.n_features_in_:
            raise ValueError(
                f"x_batch has {x_np.shape[-1]} features, but this model was fit on {self.n_features_in_}"
            )
        flat = x_np.reshape(-1, x_np.shape[-1])
        flat = self._normalize_x(flat)
        x_norm = flat.reshape(x_np.shape)
        x_tensor = torch.as_tensor(x_norm, dtype=self._torch_dtype(), device=torch.device(self.device))
        sampler_kwargs = {}
        if "sample_shape" in inspect.signature(SobolQMCNormalSampler).parameters:
            sampler_kwargs["sample_shape"] = torch.Size([int(num_samples)])
        else:
            sampler_kwargs["num_samples"] = int(num_samples)
        sampler = SobolQMCNormalSampler(**sampler_kwargs)
        acqf = QEI(model=self.model_, best_f=float(best), sampler=sampler)
        with torch.no_grad():
            with self._jitter_context():
                values = acqf(x_tensor)
        values_np = values.detach().cpu().reshape(-1).numpy().astype(np.float64)
        if return_log:
            if returns_log:
                return values_np
            return np.log(np.clip(values_np, np.finfo(np.float64).tiny, None))
        if returns_log:
            return np.exp(np.clip(values_np, -745.0, 700.0))
        return values_np

    def q_expected_improvement(self, x_batch: Any, *, best: float, num_samples: int = 128) -> np.ndarray:
        return self._q_expected_improvement_values(
            x_batch,
            best=best,
            num_samples=int(num_samples),
            return_log=False,
        )

    def q_log_expected_improvement(self, x_batch: Any, *, best: float, num_samples: int = 128) -> np.ndarray:
        return self._q_expected_improvement_values(
            x_batch,
            best=best,
            num_samples=int(num_samples),
            return_log=True,
        )


__all__ = ["EmbeddingGPRegressor"]
