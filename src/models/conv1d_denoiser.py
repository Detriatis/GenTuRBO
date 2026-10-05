"""Minimal Conv1D denoiser for discrete sequence flows."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from core.registry import register
from models.dfm import TimeEmbedding

__all__ = ["Conv1DConfig", "Conv1DDenoiser"]


@dataclass
class Conv1DConfig:
    vocab_size: int
    seq_len: int
    mask_id: int | None = None
    model_dim: int = 256
    num_layers: int = 6
    kernel_size: int = 4
    dilation: int = 1
    dropout: float = 0.1
    pos_emb: str = "none"  # none | learned | sin

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.dilation <= 0:
            raise ValueError("dilation must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.pos_emb not in {"none", "learned", "sin"}:
            raise ValueError("pos_emb must be one of: none, learned, sin")
        if self.mask_id is None:
            self.mask_id = int(self.vocab_size)
        elif self.mask_id < self.vocab_size:
            raise ValueError("mask_id must be >= vocab_size so it does not collide with data tokens")


def _make_sinusoidal_positions(seq_len: int, dim: int, *, max_period: float = 10000.0) -> Tensor:
    positions = torch.arange(seq_len, dtype=torch.float)
    half_dim = dim // 2
    freqs = torch.exp(torch.linspace(0.0, math.log(max_period), half_dim))
    args = positions[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class ConvBlock1D(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        eff = dilation * (kernel_size - 1) + 1
        pad = eff // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, padding=pad)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        if y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]
        elif y.shape[-1] < x.shape[-1]:
            y = F.pad(y, (0, x.shape[-1] - y.shape[-1]))
        y = self.act(y)
        y = self.dropout(y)
        return x + y


class Conv1DDenoiser(nn.Module):
    """Conv1D denoiser producing token logits per position."""

    def __init__(self, cfg: Conv1DConfig):
        super().__init__()
        self.cfg = cfg
        num_embeddings = max(cfg.vocab_size, int(cfg.mask_id) + 1)
        self.token_emb = nn.Embedding(num_embeddings, cfg.model_dim)
        self.pos_emb: nn.Module | None = None
        if cfg.pos_emb == "learned":
            self.pos_emb = nn.Embedding(cfg.seq_len, cfg.model_dim)
            self.register_buffer("_pos_idx", torch.arange(cfg.seq_len), persistent=False)
        elif cfg.pos_emb == "sin":
            pos = _make_sinusoidal_positions(cfg.seq_len, cfg.model_dim)
            self.register_buffer("_pos_emb", pos, persistent=False)
        self.time_mlp = nn.Sequential(
            TimeEmbedding(cfg.model_dim),
            nn.Linear(cfg.model_dim, cfg.model_dim),
            nn.SiLU(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ConvBlock1D(cfg.model_dim, cfg.kernel_size, cfg.dilation, cfg.dropout)
                for _ in range(cfg.num_layers)
            ]
        )
        self.out = nn.Conv1d(cfg.model_dim, cfg.vocab_size, kernel_size=1)

    def forward(self, tokens: Tensor, *, timesteps: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError("Conv1DDenoiser expects tokens with shape [batch, seq_len]")
        batch, seq_len = tokens.shape
        if seq_len != self.cfg.seq_len and self.cfg.pos_emb == "learned":
            raise ValueError(
                f"Token sequence length ({seq_len}) does not match configured seq_len ({self.cfg.seq_len})"
            )
        if timesteps.shape[0] != batch:
            raise ValueError("timesteps batch dimension must match tokens batch size")

        x_emb = self.token_emb(tokens)
        t_emb = self.time_mlp(timesteps).to(dtype=x_emb.dtype)[:, None, :]
        h = x_emb + t_emb
        if self.pos_emb is not None:
            if isinstance(self.pos_emb, nn.Embedding):
                pos_emb = self.pos_emb(self._pos_idx).to(dtype=h.dtype)
            else:
                if seq_len != self.cfg.seq_len:
                    pos_emb = _make_sinusoidal_positions(seq_len, self.cfg.model_dim).to(
                        device=h.device, dtype=h.dtype
                    )
                else:
                    pos_emb = self._pos_emb.to(device=h.device, dtype=h.dtype)
            h = h + pos_emb[None, :, :]

        h = h.transpose(1, 2)  # [B, D, L]
        for block in self.blocks:
            h = block(h)
        logits = self.out(h).transpose(1, 2)  # [B, L, vocab]
        return logits


@register("model", "conv1d_denoiser")
def build_conv1d_denoiser(**cfg: object) -> Conv1DDenoiser:
    """Registry entry for Conv1DDenoiser."""
    return Conv1DDenoiser(Conv1DConfig(**cfg))
