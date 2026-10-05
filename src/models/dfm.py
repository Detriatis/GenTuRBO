"""Discrete Flow Model (DFM) denoiser for masking CTMC flows.

This module implements the small Transformer-based denoiser used by the
masking interpolant formulation of Campbell et al. (2024).  The denoiser
predicts the clean token distribution ``p_theta(x_1 | x_t, t)`` which drives
both the flow-matching loss and the CTMC rate matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["DFMConfig", "TimeEmbedding", "DFMDenoiser"]


@dataclass
class DFMConfig:
    """Lightweight configuration for the Transformer denoiser.

    Args:
        vocab_size: Number of *data* tokens ``S`` (mask token excluded).
        mask_id: Integer id used for the mask token.  Must be ``>= vocab_size``
            so the embeddings cover both the data vocabulary and the mask.
        seq_len: Sequence length ``D`` for the discrete process.
        model_dim: Hidden size of the Transformer encoder.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of self-attention heads.
    """

    vocab_size: int
    mask_id: int
    seq_len: int
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    pos_emb: str = "none"
    attention_window: int | None = None
    ff_dim: int = 2048
    dropout: float = 0.1
    attn_bias: str = "none"
    attn_sigma_init: float = 2.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.mask_id < self.vocab_size:
            raise ValueError("mask_id must be >= vocab_size so it does not collide with data tokens")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.model_dim <= 0 or self.model_dim % 2 != 0:
            raise ValueError("model_dim must be a positive even integer (required by the time embedding)")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.pos_emb not in {"none", "learned", "sin"}:
            raise ValueError("pos_emb must be one of: none, learned, sin")
        if self.attention_window is not None and self.attention_window < 0:
            raise ValueError("attention_window must be >= 0 or None")
        if self.ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.attn_bias not in {"none", "gaussian"}:
            raise ValueError("attn_bias must be one of: none, gaussian")
        if self.attn_sigma_init <= 0:
            raise ValueError("attn_sigma_init must be positive")


def _make_sinusoidal_positions(seq_len: int, dim: int, *, max_period: float = 10000.0) -> Tensor:
    positions = torch.arange(seq_len, dtype=torch.float)
    half_dim = dim // 2
    freqs = torch.exp(torch.linspace(0.0, math.log(max_period), half_dim))
    args = positions[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


def _make_local_attn_mask(seq_len: int, window: int) -> Tensor:
    positions = torch.arange(seq_len)
    dist = (positions[None, :] - positions[:, None]).abs()
    return dist > int(window)


def _init_log_sigma(sigma_init: float) -> Tensor:
    base = torch.tensor(float(sigma_init))
    return torch.log(torch.expm1(base))


class LocalSelfAttention(nn.Module):
    """Sliding-window self-attention with optional Gaussian bias."""

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        if cfg.model_dim % cfg.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if cfg.attention_window is None:
            raise ValueError("LocalSelfAttention requires attention_window to be set")
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.model_dim // cfg.num_heads
        self.scale = self.head_dim**-0.5
        self.window = int(cfg.attention_window)
        self.qkv = nn.Linear(cfg.model_dim, 3 * cfg.model_dim)
        self.out_proj = nn.Linear(cfg.model_dim, cfg.model_dim)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.proj_dropout = nn.Dropout(cfg.dropout)
        self.use_gaussian = cfg.attn_bias == "gaussian"
        if self.use_gaussian:
            self.log_sigma = nn.Parameter(_init_log_sigma(cfg.attn_sigma_init).repeat(cfg.num_heads))
            offsets = torch.arange(-self.window, self.window + 1).float()
            dist2 = offsets**2
            self.register_buffer("_dist2", dist2, persistent=False)

    def forward(self, x: Tensor, *, attn_mask: Tensor | None = None) -> Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        w = self.window
        k_pad = F.pad(k, (0, 0, w, w))
        v_pad = F.pad(v, (0, 0, w, w))
        k_win = k_pad.unfold(dimension=2, size=2 * w + 1, step=1)
        v_win = v_pad.unfold(dimension=2, size=2 * w + 1, step=1)
        # unfold appends the window dimension at the end: [B, H, L, D, W]
        k_win = k_win.permute(0, 1, 2, 4, 3)
        v_win = v_win.permute(0, 1, 2, 4, 3)

        attn_logits = (q.unsqueeze(-2) * k_win).sum(dim=-1) * self.scale
        if self.use_gaussian:
            sigma = F.softplus(self.log_sigma) + 1e-6
            bias = -self._dist2[None, :] / (2.0 * sigma[:, None] ** 2)
            attn_logits = attn_logits + bias[None, :, None, :]
        if attn_mask is not None and attn_mask.shape == attn_logits.shape:
            attn_logits = attn_logits.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn.unsqueeze(-1) * v_win).sum(dim=-2)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj_dropout(self.out_proj(out))


class LocalEncoderLayer(nn.Module):
    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.self_attn = LocalSelfAttention(cfg)
        self.linear1 = nn.Linear(cfg.model_dim, cfg.ff_dim)
        self.linear2 = nn.Linear(cfg.ff_dim, cfg.model_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.model_dim)
        self.norm2 = nn.LayerNorm(cfg.model_dim)

    def forward(self, x: Tensor, *, attn_mask: Tensor | None = None) -> Tensor:
        attn_out = self.self_attn(x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x


class GaussianBiasSelfAttention(nn.Module):
    def __init__(self, cfg: DFMConfig):
        super().__init__()
        if cfg.model_dim % cfg.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.model_dim // cfg.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(cfg.model_dim, 3 * cfg.model_dim)
        self.out_proj = nn.Linear(cfg.model_dim, cfg.model_dim)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.proj_dropout = nn.Dropout(cfg.dropout)
        self.log_sigma = nn.Parameter(_init_log_sigma(cfg.attn_sigma_init).repeat(cfg.num_heads))
        dist = torch.arange(cfg.seq_len).float()
        dist2 = (dist[None, :] - dist[:, None]) ** 2
        self.register_buffer("_dist2", dist2, persistent=False)

    def forward(self, x: Tensor, *, attn_mask: Tensor | None = None) -> Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        sigma = F.softplus(self.log_sigma) + 1e-6
        bias = -self._dist2[None, :, :] / (2.0 * sigma[:, None, None] ** 2)
        attn_logits = attn_logits + bias[None, :, :, :]
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask[None, None, :, :], float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj_dropout(self.out_proj(out))


class GaussianBiasEncoderLayer(nn.Module):
    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.self_attn = GaussianBiasSelfAttention(cfg)
        self.linear1 = nn.Linear(cfg.model_dim, cfg.ff_dim)
        self.linear2 = nn.Linear(cfg.ff_dim, cfg.model_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.model_dim)
        self.norm2 = nn.LayerNorm(cfg.model_dim)

    def forward(self, x: Tensor, *, attn_mask: Tensor | None = None) -> Tensor:
        attn_out = self.self_attn(x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar timesteps ``t in [0, 1]``."""

    def __init__(self, dim: int, max_period: float = 1000.0):
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("TimeEmbedding expects a positive even dimension")
        self.dim = dim
        self.max_period = float(max_period)

    def forward(self, timesteps: Tensor) -> Tensor:
        t = timesteps.view(-1).float()
        half_dim = self.dim // 2
        device = t.device
        freqs = torch.exp(
            torch.linspace(0.0, math.log(self.max_period), half_dim, device=device, dtype=t.dtype)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class DFMDenoiser(nn.Module):
    """Transformer denoiser modelling ``p_theta(x_1 | x_t, t)``.

    The module mirrors the sketch from the user instructions: tokens are embedded,
    a sinusoidal time embedding is added, and the mixture is processed by a
    Transformer encoder before projecting back onto the data vocabulary.
    """

    def __init__(self, cfg: DFMConfig):
        super().__init__()
        self.cfg = cfg
        num_embeddings = max(cfg.vocab_size, cfg.mask_id + 1)
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
        if cfg.attention_window is not None:
            self.encoder = nn.ModuleList([LocalEncoderLayer(cfg) for _ in range(cfg.num_layers)])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.model_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.ff_dim,
                dropout=cfg.dropout,
                batch_first=True,
            )
            if cfg.attn_bias == "gaussian":
                self.encoder = nn.ModuleList([GaussianBiasEncoderLayer(cfg) for _ in range(cfg.num_layers)])
            else:
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.out = nn.Linear(cfg.model_dim, cfg.vocab_size)

    def forward(self, tokens: Tensor, *, timesteps: Tensor) -> Tensor:
        """Return logits over the data vocabulary.

        Args:
            tokens: Integer tensor ``[B, D]`` in ``{0, ..., mask_id}``.
            timesteps: Noise levels ``[B]`` in ``[0, 1]``.

        Returns:
            ``logits`` tensor with shape ``[B, D, vocab_size]``.
        """

        if tokens.ndim != 2:
            raise ValueError("DFMDenoiser expects tokens with shape [batch, seq_len]")
        batch, seq_len = tokens.shape
        requires_fixed_len = self.cfg.pos_emb == "learned" or (
            self.cfg.attn_bias == "gaussian" and self.cfg.attention_window is None
        )
        if seq_len != self.cfg.seq_len and requires_fixed_len:
            raise ValueError(
                f"Token sequence length ({seq_len}) does not match configured seq_len ({self.cfg.seq_len})"
            )
        if timesteps.shape[0] != batch:
            raise ValueError("timesteps batch dimension must match tokens batch size")

        x_emb = self.token_emb(tokens)
        t_emb = self.time_mlp(timesteps)
        t_emb = t_emb[:, None, :].to(x_emb.dtype)
        t_emb = t_emb.expand(batch, seq_len, -1)

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
        attn_mask = getattr(self, "_attn_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=h.device)
        if isinstance(self.encoder, nn.ModuleList):
            for layer in self.encoder:
                h = layer(h, attn_mask=attn_mask)
        else:
            h = self.encoder(h, mask=attn_mask)
        logits = self.out(h)
        return logits
