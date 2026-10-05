"""Independent implementation of the architecture in arXiv:2608.15958.

Uses the published implementation's sinusoid convention and pre-normalized
attention blocks. Public time follows GenTuRBO (0=noise, 1=clean); the embedding
receives the paper's noise timestep T*(1-t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from core.registry import register


@dataclass
class SokobanTransformerConfig:
    vocab_size: int = 7
    mask_id: int = 7
    seq_len: int = 100
    height: int = 10
    width: int = 10
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    num_timesteps: int = 100

    def __post_init__(self):
        if (self.vocab_size, self.mask_id, self.height, self.width, self.seq_len, self.num_timesteps) != (7, 7, 10, 10, 100, 100):
            raise ValueError("expected the 10x10, seven-tile, T=100 Boxoban contract")
        if self.model_dim < 4 or self.model_dim % 2 or self.num_heads < 1 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be even and divisible by positive num_heads")
        if self.num_layers < 1 or self.ff_dim < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid depth, feedforward width or dropout")


class GridAttentionBlock(nn.Module):
    def __init__(self, cfg: SokobanTransformerConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(cfg.model_dim)
        self.attention = nn.MultiheadAttention(cfg.model_dim, cfg.num_heads, cfg.dropout, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(cfg.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(cfg.model_dim, cfg.ff_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.model_dim), nn.Dropout(cfg.dropout),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        normal = self.attention_norm(hidden)
        hidden = hidden + self.attention(normal, normal, normal, need_weights=False)[0]
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class SokobanTransformer(nn.Module):
    def __init__(self, cfg: SokobanTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size + 1, cfg.model_dim)
        self.row_emb = nn.Embedding(cfg.height, cfg.model_dim)
        self.col_emb = nn.Embedding(cfg.width, cfg.model_dim)
        self.register_buffer("rows", torch.arange(cfg.seq_len) // cfg.width, persistent=False)
        self.register_buffer("cols", torch.arange(cfg.seq_len) % cfg.width, persistent=False)
        self.register_buffer("frequencies", torch.exp(
            -math.log(10000.0) * torch.arange(cfg.model_dim // 2) / (cfg.model_dim // 2 - 1)
        ), persistent=False)
        self.time_mlp = nn.Sequential(nn.Linear(cfg.model_dim, cfg.model_dim), nn.GELU(),
                                      nn.Linear(cfg.model_dim, cfg.model_dim))
        self.blocks = nn.ModuleList(GridAttentionBlock(cfg) for _ in range(cfg.num_layers))
        self.final_norm = nn.LayerNorm(cfg.model_dim)
        self.out = nn.Linear(cfg.model_dim, cfg.vocab_size)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor, *, timesteps: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.cfg.seq_len or timesteps.shape != (tokens.shape[0],):
            raise ValueError("expected tokens [B,100] and timesteps [B]")
        phase = (self.cfg.num_timesteps * (1 - timesteps.float()))[:, None] * self.frequencies[None, :]
        time_features = self.time_mlp(torch.cat((phase.sin(), phase.cos()), dim=-1))
        hidden = self.token_emb(tokens) + self.row_emb(self.rows) + self.col_emb(self.cols)
        hidden = hidden + time_features[:, None, :]
        for block in self.blocks:
            hidden = block(hidden)
        return self.out(self.final_norm(hidden))


@register("model", "sokoban_transformer")
def build_sokoban_transformer(**cfg):
    return SokobanTransformer(SokobanTransformerConfig(**cfg))
