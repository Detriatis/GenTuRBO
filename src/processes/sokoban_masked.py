"""Paper-style masked diffusion loss and random-order single-cell sampling."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from core.registry import register
from processes.dfm_mask import MaskingCTMCFlowProcess


def weighted_masked_loss(logits, clean, masked, noise_steps, *, num_timesteps=100, weight_cap=10.0):
    """Mean across boards of min(T/k, cap) * mean CE over each board's masks.

    Bernoulli corruption occasionally hides zero cells; such a board contributes
    zero, with no resampling or training on visible tokens. CE stays float32
    under AMP. The timestep k is noise level, in [1,T].
    """
    token_loss = F.cross_entropy(logits.float().transpose(1, 2), clean, reduction="none")
    per_board = (token_loss * masked).sum(1) / masked.sum(1).clamp_min(1)
    weights = (num_timesteps / noise_steps.float()).clamp_max(weight_cap)
    return (per_board * weights).mean(), per_board.mean()


@torch.no_grad()
def sample_one_cell(model, initial, *, temperature=1.0, generator=None, on_step=None):
    """Uniformly pick one remaining cell, then sample its categorical value.

    All committed cells remain fixed. Conditional starts use the actual number
    of hidden cells to select t. The order and categorical draws have the same
    distribution as the paper, without drawing values for unused positions.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if initial.ndim != 2 or initial.shape[1] != 100 or initial.dtype != torch.long:
        raise ValueError("expected integer tokens [B,100]")
    if torch.any((initial < 0) | (initial > 7)):
        raise ValueError("invalid token")
    board = initial.clone()
    was_training = model.training
    model.eval()
    try:
        for _ in range(100):
            masked = board == 7
            counts = masked.sum(1)
            active = counts > 0
            if not bool(active.any()):
                break
            time = 1.0 - counts.float() / 100
            logits = model(board, timesteps=time)
            priorities = torch.rand(board.shape, device=board.device, generator=generator).masked_fill(~masked, -1)
            chosen = priorities.argmax(1)
            rows = torch.arange(len(board), device=board.device)
            probs = (logits[rows, chosen].float() / temperature).softmax(-1)
            values = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            board[rows[active], chosen[active]] = values[active]
            if on_step is not None:
                on_step(board.clone(), time.clone())
    finally:
        model.train(was_training)
    return board


class SokobanMaskedProcess(MaskingCTMCFlowProcess):
    def __init__(self, model, *, vocab_size=7, seq_len=100, mask_id=7, num_timesteps=100,
                 weight_cap=10.0, temperature=1.0, device=None):
        if (vocab_size, seq_len, mask_id, num_timesteps) != (7, 100, 7, 100):
            raise ValueError("expected seven tiles, 100 cells and 100 noise steps")
        if not math.isfinite(weight_cap) or weight_cap <= 0:
            raise ValueError("weight_cap must be positive and finite")
        super().__init__(model, vocab_size=vocab_size, seq_len=seq_len, mask_id=mask_id,
                         interpolant="mask", mask_loss_only=True, temperature=temperature, device=device)
        self.num_timesteps = num_timesteps
        self.weight_cap = weight_cap

    def training_step(self, batch):
        clean = batch["x"].to(self.device, dtype=torch.long)
        noise_steps = torch.randint(1, self.num_timesteps + 1, (len(clean),), device=self.device)
        probability = noise_steps.float() / self.num_timesteps
        masked = torch.rand(clean.shape, device=self.device) < probability[:, None]
        logits = self.model(clean.masked_fill(masked, self.cfg.mask_id), timesteps=1 - probability)
        loss, unweighted = weighted_masked_loss(logits, clean, masked, noise_steps,
                                                num_timesteps=self.num_timesteps, weight_cap=self.weight_cap)
        return loss, {"loss": loss.detach(), "unweighted_masked_ce": unweighted.detach(),
                      "mask_fraction": masked.float().mean().detach()}

    def sample(self, num_samples, *, steps=None, device=None, show_progress=False,
               prior_invalid=None, prior_invalid_mode=None, **kwargs):
        # The common sample runner passes these optional uniform-prior fields
        # even for masking processes. Only an actual request is unsupported.
        if prior_invalid:
            raise ValueError("single-cell masking starts from masks, not an invalid-token prior")
        if steps not in (None, 100) or kwargs:
            raise ValueError("single-cell sampling uses exactly 100 steps; only steps/device/show_progress are supported")
        target_device = torch.device(device) if device is not None else self.device
        if target_device != self.device:
            raise ValueError("sample on the process device")
        initial = self._custom_initial_state(num_samples, target_device)
        if initial is None:
            initial = torch.full((num_samples, 100), 7, dtype=torch.long, device=target_device)
        return sample_one_cell(self.model, initial, temperature=self.temperature)


@register("process", "sokoban_masked_diffusion")
def build_sokoban_masked(model, **cfg):
    return SokobanMaskedProcess(model, **cfg)
