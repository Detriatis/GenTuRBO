"""Minimal uniform categorical denoiser for constrained discrete benchmarks."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from core.registry import register
from models.dfm import DFMConfig


@register("model", "uniform_categorical_denoiser", aliases=("uniform_denoiser", "protein_uniform_denoiser"))
class UniformCategoricalDenoiser(nn.Module):
    """Denoiser that assigns equal logits to every token.

    This is useful for constrained lookup benchmarks, where the search loop and
    trust-region policy are being tested independently of a trained generator.
    """

    _skip_genturbo_checkpoint_load = True

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        mask_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        fixed_sequence: Optional[str | Sequence[int]] = None,
        fixed_sequence_alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
        mutable_positions: Optional[Sequence[int]] = None,
        mutable_positions_1_indexed: Optional[Sequence[int]] = None,
        sample_interpolant_mode: str = "uniform",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.mask_id = int(mask_id) if mask_id is not None else int(vocab_size)
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.sample_interpolant_mode = str(sample_interpolant_mode or "uniform").strip().lower()
        if self.sample_interpolant_mode not in {"uniform", "mask"}:
            raise ValueError("sample_interpolant_mode must be 'uniform' or 'mask'")
        self.cfg = DFMConfig(vocab_size=self.vocab_size, mask_id=self.mask_id, seq_len=self.seq_len)
        fixed_tokens = self._coerce_fixed_sequence(
            fixed_sequence,
            alphabet=fixed_sequence_alphabet,
        )
        mutable_mask = self._coerce_mutable_mask(
            mutable_positions=mutable_positions,
            mutable_positions_1_indexed=mutable_positions_1_indexed,
        )
        if fixed_tokens is None:
            fixed_tokens = torch.empty(0, dtype=torch.long)
        self.register_buffer("_dummy", torch.empty(0), persistent=False)
        self.register_buffer("_fixed_tokens", fixed_tokens, persistent=False)
        self.register_buffer("_mutable_positions_mask", mutable_mask, persistent=False)
        self.to(torch.device(device))

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def forward(self, tokens: Tensor, *, timesteps: Tensor) -> Tensor:
        _ = timesteps
        if tokens.ndim != 2:
            raise ValueError(f"UniformCategoricalDenoiser expects [B, L], got {tuple(tokens.shape)}")
        return torch.zeros((*tokens.shape, self.vocab_size), dtype=torch.float, device=tokens.device)

    def _coerce_fixed_sequence(
        self,
        value: Optional[str | Sequence[int]],
        *,
        alphabet: str,
    ) -> Optional[Tensor]:
        if value is None:
            return None
        if isinstance(value, str):
            clean = "".join(value.upper().split())
            if len(clean) != self.seq_len:
                raise ValueError(f"fixed_sequence length {len(clean)} does not match seq_len={self.seq_len}")
            aa_to_id = {aa: idx for idx, aa in enumerate(str(alphabet))}
            try:
                tokens = [aa_to_id[aa] for aa in clean]
            except KeyError as exc:
                raise ValueError(f"fixed_sequence contains token outside fixed_sequence_alphabet: {exc}") from exc
        else:
            tokens = [int(token) for token in value]
            if len(tokens) != self.seq_len:
                raise ValueError(f"fixed_sequence length {len(tokens)} does not match seq_len={self.seq_len}")
        token_tensor = torch.as_tensor(tokens, dtype=torch.long)
        if bool(((token_tensor < 0) | (token_tensor >= self.vocab_size)).any().item()):
            raise ValueError("fixed_sequence tokens must be in [0, vocab_size)")
        return token_tensor

    def _coerce_mutable_mask(
        self,
        *,
        mutable_positions: Optional[Sequence[int]],
        mutable_positions_1_indexed: Optional[Sequence[int]],
    ) -> Tensor:
        if mutable_positions is not None and mutable_positions_1_indexed is not None:
            raise ValueError("Use either mutable_positions or mutable_positions_1_indexed, not both")
        mask = torch.ones((self.seq_len,), dtype=torch.bool)
        positions = mutable_positions
        if mutable_positions_1_indexed is not None:
            positions = [int(pos) - 1 for pos in mutable_positions_1_indexed]
        if positions is None:
            return mask
        mask = torch.zeros((self.seq_len,), dtype=torch.bool)
        for raw_pos in positions:
            pos = int(raw_pos)
            if pos < 0 or pos >= self.seq_len:
                raise ValueError(f"mutable position {pos} is outside [0, {self.seq_len})")
            mask[pos] = True
        if not bool(mask.any().item()):
            raise ValueError("At least one mutable position is required")
        return mask

    def _fixed_context_enabled(self) -> bool:
        return int(self._fixed_tokens.numel()) == int(self.seq_len)

    def mutable_mask(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            batch_shape = (1, self.seq_len)
        elif x.ndim == 2:
            batch_shape = (int(x.shape[0]), self.seq_len)
        else:
            raise ValueError("mutable_mask expects [L] or [B, L]")
        if x.shape[-1] != self.seq_len:
            raise ValueError(f"mutable_mask expected length {self.seq_len}, got {x.shape[-1]}")
        return self._mutable_positions_mask.to(device=x.device).view(1, -1).expand(batch_shape)

    def trust_region_seq_len(self, sample_batch: Tensor) -> int:
        mask = self.mutable_mask(sample_batch)
        return int(mask[0].sum().item())

    def _sample_noise(self, shape: torch.Size | tuple[int, ...], *, device: torch.device) -> Tensor:
        return torch.randint(0, self.vocab_size, tuple(shape), dtype=torch.long, device=device)

    def _fixed_context_batch(self, *, num_samples: int, device: torch.device) -> Optional[Tensor]:
        if not self._fixed_context_enabled():
            return None
        return self._fixed_tokens.to(device=device).view(1, -1).expand(int(num_samples), -1).clone()

    def sample_initial_state(
        self,
        *,
        num_samples: int,
        seq_len: int,
        device: str | torch.device,
        vocab_size: Optional[int] = None,
        interpolant: str = "uniform",
    ) -> Tensor:
        _ = interpolant
        vocab = int(vocab_size) if vocab_size is not None else self.vocab_size
        if int(seq_len) != self.seq_len:
            raise ValueError(f"seq_len={seq_len} does not match model seq_len={self.seq_len}")
        device_t = torch.device(device)
        base = self._fixed_context_batch(num_samples=int(num_samples), device=device_t)
        if base is None:
            return torch.randint(0, vocab, (int(num_samples), int(seq_len)), dtype=torch.long, device=device_t)
        noise = torch.randint(0, vocab, base.shape, dtype=torch.long, device=device_t)
        mutable = self.mutable_mask(base)
        return torch.where(mutable, noise, base)

    def sample_interpolant(
        self,
        *,
        base_sample: Tensor,
        t_tensor: Tensor,
        elite_probs: Optional[Tensor] = None,
        transition_mask: Optional[Tensor] = None,
        mixing_mode: str = "random",
    ) -> Tensor:
        _ = transition_mask
        if base_sample.ndim != 2:
            raise ValueError("sample_interpolant expects base_sample [B, L]")
        if base_sample.shape[1] != self.seq_len:
            raise ValueError(f"sample_interpolant expected length {self.seq_len}, got {base_sample.shape[1]}")
        if t_tensor.ndim != 1 or t_tensor.shape[0] != base_sample.shape[0]:
            raise ValueError("t_tensor must align with base_sample rows")
        if str(mixing_mode).strip().lower() == "invalid":
            raise ValueError("UniformCategoricalDenoiser fixed-context sampling does not support mixing_mode=invalid")
        base = base_sample.to(dtype=torch.long)
        keep = torch.bernoulli(t_tensor.to(device=base.device, dtype=torch.float)[:, None].expand_as(base)).bool()
        if self.sample_interpolant_mode == "mask":
            mixed = torch.full_like(base, int(self.mask_id))
        elif elite_probs is not None:
            probs = elite_probs.to(device=base.device, dtype=torch.float)
            if probs.ndim != 1 or probs.numel() != self.vocab_size:
                raise ValueError("elite_probs must have shape [vocab_size]")
            weights = probs.clamp_min(0.0)
            if float(weights.sum().item()) <= 0.0:
                mixed = self._sample_noise(base.shape, device=base.device)
            else:
                sampled = torch.multinomial(
                    weights / weights.sum().clamp_min(1e-12),
                    num_samples=base.numel(),
                    replacement=True,
                ).view_as(base)
                mixed = sampled.to(dtype=torch.long)
        else:
            mixed = self._sample_noise(base.shape, device=base.device)
        out = torch.where(keep, base, mixed)
        mutable = self.mutable_mask(base)
        out = torch.where(mutable, out, base)
        if self._fixed_context_enabled():
            fixed = self._fixed_tokens.to(device=base.device).view(1, -1).expand_as(base)
            out = torch.where(mutable, out, fixed)
        return out

    def corrupt_mask(self, base_sample: Tensor, xt: Tensor) -> Tensor:
        if base_sample.shape != xt.shape:
            raise ValueError("corrupt_mask expects base_sample and xt with the same shape")
        return (base_sample != xt) & self.mutable_mask(base_sample)


__all__ = ["UniformCategoricalDenoiser"]
