"""Native EvoDiff-D3PM adapter for GenTuRBO interpolant search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from adapters.protein.validation import CANONICAL_AA
from core.registry import register
from models.dfm import DFMConfig

BackendLoader = Callable[[str], Any]


_D3PM_LOADER_BY_KEY = {
    "uniform_38m": "D3PM_UNIFORM_38M",
    "uniform_640m": "D3PM_UNIFORM_640M",
    "blosum_38m": "D3PM_BLOSUM_38M",
    "blosum_640m": "D3PM_BLOSUM_640M",
}


@dataclass(slots=True)
class _D3PMBackend:
    model: Any
    tokenizer: Any
    scheme: str
    timesteps: int
    q_bar: Tensor
    q: Tensor


def _as_token_tensor(raw: Any) -> Tensor:
    if isinstance(raw, Tensor):
        return raw.detach().to(dtype=torch.long).view(-1)
    if isinstance(raw, int):
        return torch.tensor([raw], dtype=torch.long)
    return torch.as_tensor(list(raw), dtype=torch.long).view(-1)


def _normalize_transition(name: str) -> str:
    value = str(name).strip().lower()
    if value in {"uniform", "d3pm_uniform"}:
        return "uniform"
    if value in {"blosum", "blossom", "d3pm_blosum", "d3pm_blossom"}:
        return "blosum"
    raise ValueError("d3pm_transition must be one of: uniform, blosum")


def _normalize_size(name: str) -> str:
    value = str(name).strip().lower().replace("_", "")
    if value in {"38m", "38"}:
        return "38m"
    if value in {"640m", "640"}:
        return "640m"
    raise ValueError("model_size must be one of: 38M, 640M")


@register(
    "model",
    "protein_evodiff_d3pm",
    aliases=("evodiff_d3pm",),
)
class EvoDiffD3PMDenoiser(nn.Module):
    """Expose native EvoDiff-D3PM sampling to the existing search loop.

    The native path uses EvoDiff's ``D3PM_*`` loaders with ``return_all=True``
    and performs the reverse diffusion update with EvoDiff's ``Q`` and
    ``Q_bar`` matrices. The normal GenTuRBO interpolant machinery still chooses
    parent sequences, trust-region times, and BO oracle batches.
    """

    def __init__(
        self,
        *,
        model_type: str = "d3pm_uniform_38m",
        d3pm_transition: str = "uniform",
        model_size: str = "38M",
        seq_len: int = 32,
        vocab_size: int = len(CANONICAL_AA),
        mask_id: Optional[int] = None,
        device: str | torch.device = "cpu",
        time_conditioning: str = "scaled",
        backend_loader: Optional[BackendLoader] = None,
        skip_genturbo_checkpoint_load: bool = True,
        subsample_rollout_steps: bool = False,
    ) -> None:
        super().__init__()
        self.model_type = str(model_type)
        self.d3pm_transition = _normalize_transition(d3pm_transition)
        self.model_size = _normalize_size(model_size)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        if self.vocab_size != len(CANONICAL_AA):
            raise ValueError(f"EvoDiff-D3PM adapter expects vocab_size={len(CANONICAL_AA)}")
        self.mask_id = int(mask_id) if mask_id is not None else self.vocab_size
        self.pad_token_id: Optional[int] = None
        self.device = torch.device(device)
        self.time_conditioning = str(time_conditioning).strip().lower()
        if self.time_conditioning not in {"zero", "scaled"}:
            raise ValueError("time_conditioning must be one of: zero, scaled")
        self.backend_loader = backend_loader
        self.subsample_rollout_steps = bool(subsample_rollout_steps)
        self._backend: Optional[_D3PMBackend] = None
        self._repo_to_backend_token: Optional[Tensor] = None
        self._backend_to_repo_token: Optional[Tensor] = None
        self._backend_aa_tokens: Optional[Tensor] = None
        self._skip_genturbo_checkpoint_load = bool(skip_genturbo_checkpoint_load)
        self.cfg = DFMConfig(vocab_size=self.vocab_size, seq_len=self.seq_len, mask_id=self.mask_id)

    def _loader_name(self) -> str:
        key = self.model_type.strip().lower()
        key = key.replace("-", "_")
        key = key.removeprefix("evodiff_").removeprefix("protein_")
        aliases = {
            "d3pm": f"{self.d3pm_transition}_{self.model_size}",
            "d3pm_38m": f"{self.d3pm_transition}_38m",
            "d3pm_640m": f"{self.d3pm_transition}_640m",
            "d3pm_uniform": f"uniform_{self.model_size}",
            "d3pm_blosum": f"blosum_{self.model_size}",
            "d3pm_blossom": f"blosum_{self.model_size}",
            "evodiff_d3pm": f"{self.d3pm_transition}_{self.model_size}",
        }
        key = aliases.get(key, key)
        key = key.removeprefix("d3pm_")
        key = key.replace("blossom", "blosum")
        if key not in _D3PM_LOADER_BY_KEY:
            supported = ", ".join(sorted(_D3PM_LOADER_BY_KEY))
            raise ValueError(f"Unsupported EvoDiff-D3PM model_type='{self.model_type}'. Supported: {supported}")
        return _D3PM_LOADER_BY_KEY[key]

    def _load_backend(self) -> _D3PMBackend:
        if self._backend is not None:
            return self._backend
        if self.backend_loader is not None:
            loaded = self.backend_loader(self.model_type)
        else:
            try:
                import evodiff.pretrained as pretrained
            except ImportError as exc:
                raise ImportError(
                    "evodiff is required for protein_evodiff_d3pm; install it in the runtime environment."
                ) from exc
            loader_name = self._loader_name()
            if loader_name != "D3PM_UNIFORM_38M":
                raise ValueError("This reproduction release pins only EvoDiff D3PM uniform 38M weights")
            from genturbo.model_weights import prepare_weights
            prepare_weights(download=False, names=("evodiff",))
            loader = getattr(pretrained, loader_name)
            loaded = loader(return_all=True)

        backend = self._normalize_loaded_backend(loaded)
        if hasattr(backend.model, "eval"):
            backend.model = backend.model.eval()
        if hasattr(backend.model, "to"):
            backend.model = backend.model.to(self.device)
        backend.q = torch.as_tensor(backend.q, dtype=torch.float64, device=self.device)
        backend.q_bar = torch.as_tensor(backend.q_bar, dtype=torch.float64, device=self.device)
        self._backend = backend
        self._build_token_maps(backend.tokenizer)
        return backend

    def _normalize_loaded_backend(self, loaded: Any) -> _D3PMBackend:
        if isinstance(loaded, _D3PMBackend):
            return loaded
        if not isinstance(loaded, (tuple, list)):
            raise TypeError("EvoDiff-D3PM backend loader must return a tuple")
        if len(loaded) == 7:
            model, _collater, tokenizer, scheme, timesteps, q_bar, q = loaded
        elif len(loaded) == 6:
            model, tokenizer, scheme, timesteps, q_bar, q = loaded
        else:
            raise TypeError("EvoDiff-D3PM backend loader must return 6 or 7 values")
        return _D3PMBackend(
            model=model,
            tokenizer=tokenizer,
            scheme=str(scheme),
            timesteps=int(timesteps),
            q_bar=torch.as_tensor(q_bar, dtype=torch.float),
            q=torch.as_tensor(q, dtype=torch.float),
        )

    def _tokenizer_k(self, tokenizer: Any) -> int:
        value = getattr(tokenizer, "K", None)
        if value is not None:
            return int(value)
        all_aas = str(getattr(tokenizer, "all_aas", CANONICAL_AA))
        return max(len(all_aas), len(CANONICAL_AA))

    def _standard_token_count(self, backend: _D3PMBackend) -> int:
        k = self._tokenizer_k(backend.tokenizer)
        return min(self.vocab_size, max(1, k - 6))

    def _tokenize_residue(self, tokenizer: Any, residue: str) -> int:
        raw: Any
        if callable(getattr(tokenizer, "tokenize", None)):
            raw = tokenizer.tokenize(residue)
        elif callable(getattr(tokenizer, "encode", None)):
            raw = tokenizer.encode(residue)
        else:
            all_aas = str(getattr(tokenizer, "all_aas", CANONICAL_AA))
            if residue not in all_aas:
                raise ValueError(f"EvoDiff tokenizer alphabet does not contain residue {residue!r}")
            return int(all_aas.index(residue))
        tokens = _as_token_tensor(raw)
        if int(tokens.numel()) == 1:
            return int(tokens[0].item())
        if callable(getattr(tokenizer, "tokenize", None)):
            doubled = _as_token_tensor(tokenizer.tokenize(residue * 2))
        elif callable(getattr(tokenizer, "encode", None)):
            doubled = _as_token_tensor(tokenizer.encode(residue * 2))
        else:
            doubled = torch.empty((0,), dtype=torch.long)
        for token in tokens.tolist():
            token_id = int(token)
            if int((doubled == token_id).sum().item()) > int((tokens == token_id).sum().item()):
                return token_id
        all_aas = str(getattr(tokenizer, "all_aas", CANONICAL_AA))
        if residue in all_aas:
            return int(all_aas.index(residue))
        raise ValueError(f"Could not identify EvoDiff token id for residue {residue!r}")

    def _build_token_maps(self, tokenizer: Any) -> None:
        aa_tokens = torch.tensor(
            [self._tokenize_residue(tokenizer, aa) for aa in CANONICAL_AA],
            dtype=torch.long,
        )
        backend_vocab = max(int(aa_tokens.max().item()) + 1, self._tokenizer_k(tokenizer))
        repo_to_backend = torch.empty((self.mask_id + 1,), dtype=torch.long)
        repo_to_backend[: self.vocab_size] = aa_tokens
        repo_to_backend[self.mask_id] = int(getattr(tokenizer, "mask_id", self.mask_id))
        backend_to_repo = torch.full((backend_vocab,), -1, dtype=torch.long)
        for repo_id, backend_id in enumerate(aa_tokens.tolist()):
            if 0 <= int(backend_id) < backend_vocab:
                backend_to_repo[int(backend_id)] = int(repo_id)
        self._repo_to_backend_token = repo_to_backend
        self._backend_to_repo_token = backend_to_repo
        self._backend_aa_tokens = aa_tokens

    def _data_time_to_d3pm_step(self, t_value: float, backend: _D3PMBackend) -> int:
        t_data = min(max(float(t_value), 0.0), 1.0)
        max_step = max(0, int(backend.timesteps) - 1)
        return int(round((1.0 - t_data) * float(max_step)))

    def _backend_timesteps(self, timesteps: Tensor, backend: _D3PMBackend) -> Tensor:
        if self.time_conditioning == "zero":
            return torch.zeros((timesteps.shape[0],), dtype=torch.long, device=timesteps.device)
        values = [
            self._data_time_to_d3pm_step(float(value), backend)
            for value in timesteps.detach().cpu().view(-1).tolist()
        ]
        return torch.as_tensor(values, dtype=torch.long, device=timesteps.device)

    def _repo_to_backend(self, tokens: Tensor, *, device: torch.device) -> Tensor:
        if self._repo_to_backend_token is None:
            raise RuntimeError("EvoDiff token map has not been initialised")
        token_map = self._repo_to_backend_token.to(device=device)
        safe = tokens.long().clamp(min=0, max=int(token_map.numel()) - 1)
        return token_map[safe]

    def _state_to_backend(self, tokens: Tensor, backend: _D3PMBackend, *, device: torch.device) -> Tensor:
        tok = tokens.to(device=device, dtype=torch.long)
        if tok.numel() and int(tok.max().item()) >= self.vocab_size:
            k = self._tokenizer_k(backend.tokenizer)
            return tok.clamp(min=0, max=max(0, k - 1))
        return self._repo_to_backend(tok, device=device)

    def _backend_to_repo(self, tokens: Tensor) -> Tensor:
        if self._backend_to_repo_token is None:
            raise RuntimeError("EvoDiff token map has not been initialised")
        token_map = self._backend_to_repo_token.to(device=tokens.device)
        safe = tokens.long().clamp(min=0, max=int(token_map.numel()) - 1)
        mapped = token_map[safe]
        return torch.where(mapped >= 0, mapped, torch.zeros_like(mapped))

    def _extract_logits(self, output: Any, *, seq_len: int) -> Tensor:
        logits = output
        if isinstance(output, dict):
            logits = output.get("logits", output.get("out", output))
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        if not isinstance(logits, Tensor):
            logits = torch.as_tensor(logits)
        if logits.ndim != 3:
            raise ValueError(f"EvoDiff model must return logits [B, L, V], got shape {tuple(logits.shape)}")
        if logits.shape[1] != seq_len and logits.shape[2] == seq_len:
            logits = logits.transpose(1, 2)
        if logits.shape[1] != seq_len:
            raise ValueError(f"EvoDiff logits sequence length {logits.shape[1]} does not match input {seq_len}")
        return logits

    def _residue_logits_from_backend(self, backend_logits: Tensor) -> Tensor:
        if self._backend_aa_tokens is None:
            raise RuntimeError("EvoDiff token map has not been initialised")
        gather_ids = self._backend_aa_tokens.to(device=backend_logits.device)
        if int(gather_ids.max().item()) >= int(backend_logits.shape[-1]):
            if backend_logits.shape[-1] < self.vocab_size:
                raise ValueError(
                    f"EvoDiff logits last dimension {backend_logits.shape[-1]} is smaller than {self.vocab_size}"
                )
            gather_ids = torch.arange(self.vocab_size, device=backend_logits.device, dtype=torch.long)
        return backend_logits.index_select(dim=-1, index=gather_ids)

    def forward(self, tokens: Tensor, *, timesteps: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("EvoDiffD3PMDenoiser expects tokens with shape [batch, seq_len]")
        if int(tokens.shape[1]) != self.seq_len:
            raise ValueError(f"Token sequence length {tokens.shape[1]} does not match seq_len={self.seq_len}")
        if timesteps.ndim != 1 or int(timesteps.shape[0]) != int(tokens.shape[0]):
            raise ValueError("timesteps must have shape [batch]")

        backend = self._load_backend()
        if hasattr(backend.model, "to"):
            backend.model = backend.model.to(tokens.device)
        backend_tokens = self._state_to_backend(tokens, backend, device=tokens.device)
        backend_t = self._backend_timesteps(timesteps.to(tokens.device), backend)
        backend_logits = self._extract_logits(backend.model(backend_tokens, backend_t), seq_len=int(tokens.shape[1]))
        return self._residue_logits_from_backend(backend_logits).to(device=tokens.device)

    def sample_native(
        self,
        *,
        num_samples: int,
        seq_len: int,
        device: torch.device,
        steps: Optional[int] = None,
        max_time: float = 1.0,
    ) -> Tensor:
        _ = (steps, max_time)
        if int(seq_len) != self.seq_len:
            raise ValueError("sample_native seq_len does not match EvoDiff adapter seq_len")
        backend = self._load_backend()
        k = self._tokenizer_k(backend.tokenizer)
        start = torch.randint(0, k, (int(num_samples), self.seq_len), dtype=torch.long, device=device)
        states = self._reverse_d3pm(
            start,
            start_step=max(0, backend.timesteps - 1),
            update_mask=None,
            denoise_steps=None,
            sample_is_backend=True,
        )
        return states[-1].to(device=device)

    def sample_interpolant(
        self,
        *,
        base_sample: Tensor,
        t_tensor: Tensor,
        elite_probs: Optional[Tensor] = None,
        transition_mask: Optional[Tensor] = None,
        mixing_mode: str = "random",
    ) -> Tensor:
        _ = (elite_probs, transition_mask, mixing_mode)
        if base_sample.ndim != 2:
            raise ValueError("sample_interpolant expects base_sample [B, L]")
        backend = self._load_backend()
        base = base_sample.to(device=self.device, dtype=torch.long)
        backend_base = self._repo_to_backend(base, device=self.device)
        t_values = t_tensor.detach().cpu().view(-1).tolist()
        if len(t_values) != int(base.shape[0]):
            raise ValueError("t_tensor must align with base_sample rows")
        out = backend_base.clone()
        q_bar = backend.q_bar.to(device=self.device, dtype=torch.float64)
        for row_idx, t_value in enumerate(t_values):
            step = self._data_time_to_d3pm_step(float(t_value), backend)
            if step <= 0:
                continue
            probs = q_bar[step][backend_base[row_idx]].clamp_min(0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(probs, num_samples=1).view(-1)
            out[row_idx] = sampled
        return out.to(device=base_sample.device)

    def rollout_from_batch_native(
        self,
        x_t: Tensor,
        *,
        t_start: float,
        update_mask: Optional[Tensor] = None,
        denoise_steps: Optional[int] = None,
        max_time: float = 1.0,
        ctmc_steps: int = 1,
    ) -> list[Tensor]:
        _ = (max_time, ctmc_steps)
        backend = self._load_backend()
        start_step = self._data_time_to_d3pm_step(float(t_start), backend)
        return self._reverse_d3pm(
            x_t.to(device=self.device, dtype=torch.long),
            start_step=start_step,
            update_mask=update_mask,
            denoise_steps=denoise_steps if self.subsample_rollout_steps else None,
            sample_is_backend=True,
        )

    def corrupt_mask(self, base_sample: Tensor, xt: Tensor) -> Tensor:
        """Compare a clean GenTuRBO sequence with a D3PM-corrupted proposal."""
        backend = self._load_backend()
        device = self.device
        base_backend = self._state_to_backend(base_sample, backend, device=device)
        k = self._tokenizer_k(backend.tokenizer)
        xt_backend = xt.to(device=device, dtype=torch.long).clamp(min=0, max=max(0, k - 1))
        return (base_backend != xt_backend).to(device=xt.device)

    def _reverse_schedule(self, start_step: int, denoise_steps: Optional[int]) -> list[int]:
        start = max(0, int(start_step))
        if start <= 0:
            return []
        if denoise_steps is None or int(denoise_steps) >= start:
            return list(range(start, 0, -1))
        n = max(1, int(denoise_steps))
        points = torch.linspace(float(start), 1.0, steps=n).round().to(dtype=torch.long).tolist()
        schedule: list[int] = []
        seen: set[int] = set()
        for value in points:
            step = int(max(1, min(start, int(value))))
            if step in seen:
                continue
            seen.add(step)
            schedule.append(step)
        return schedule

    def _reverse_d3pm(
        self,
        sample: Tensor,
        *,
        start_step: int,
        update_mask: Optional[Tensor],
        denoise_steps: Optional[int],
        sample_is_backend: bool = False,
    ) -> list[Tensor]:
        backend = self._load_backend()
        if sample_is_backend:
            current = sample.to(device=self.device, dtype=torch.long).clone()
        else:
            current = self._state_to_backend(sample, backend, device=self.device).clone()
        states: list[Tensor] = [current.detach().clone()]
        if update_mask is not None:
            update_mask = update_mask.to(device=self.device, dtype=torch.bool)
            frozen_reference = current.clone()
        else:
            frozen_reference = None
        k = self._tokenizer_k(backend.tokenizer)
        standard_k = self._standard_token_count(backend)
        q = backend.q.to(device=self.device, dtype=torch.float64)
        q_bar = backend.q_bar.to(device=self.device, dtype=torch.float64)
        with torch.no_grad():
            for step in self._reverse_schedule(start_step, denoise_steps):
                timestep = torch.full((current.shape[0],), int(step), dtype=torch.long, device=self.device)
                pred = self._extract_logits(backend.model(current, timestep), seq_len=int(current.shape[1]))
                p = F.softmax(pred[:, :, :k], dim=-1).to(dtype=torch.float64)

                a = q[int(step)].transpose(0, 1)[current.clamp(min=0, max=k - 1)]
                b_pred = p.unsqueeze(-1) * q_bar[int(step) - 1].view(1, 1, k, k)
                q_t = a.unsqueeze(-2) * b_pred
                marginal = torch.matmul(q_t.transpose(-2, -1), p.unsqueeze(-1)).squeeze(-1)
                marginal = marginal.clamp_min(0.0)
                if int(step) == 1:
                    marginal = marginal[:, :, :standard_k]
                marginal = marginal / marginal.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                sampled = torch.multinomial(marginal.view(-1, marginal.shape[-1]), num_samples=1).view_as(current)
                if update_mask is not None and frozen_reference is not None:
                    sampled = torch.where(update_mask, sampled, frozen_reference)
                current = sampled
                states.append(current.detach().clone())
        states[-1] = self._backend_to_repo(states[-1]).to(device=sample.device)
        return [state.to(device=sample.device) for state in states]

    def sample_initial_state(
        self,
        *,
        num_samples: int,
        seq_len: int,
        device: torch.device,
        vocab_size: int,
        interpolant: str,
    ) -> Optional[Tensor]:
        _ = (num_samples, seq_len, device, vocab_size, interpolant)
        return None

    def sequence_edit_distance(self, a: Tensor, b: Tensor) -> int:
        a1 = torch.as_tensor(a, dtype=torch.long).view(-1)
        b1 = torch.as_tensor(b, dtype=torch.long).view(-1)
        if a1.shape != b1.shape:
            raise ValueError("Cannot compute sequence distance for tensors with different shapes")
        return int((a1 != b1).sum().item())


__all__ = ["EvoDiffD3PMDenoiser"]
