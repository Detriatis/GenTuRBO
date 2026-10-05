"""Adapter to use a discrete CTMC flow process with the ancestor-tree search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional

import torch

from core.utils import resolve_device
from processes.dfm_mask import (
    MaskingCTMCFlowProcess,
    ctmc_euler_step,
    ctmc_rate_masking,
    ctmc_rate_uniform,
)
from toy_flow import AbstractFlowProcess


@dataclass
class CTMCFlowAdapter(AbstractFlowProcess):
    """
    Wraps MaskingCTMCFlowProcess so it can drive ancestor_tree_search.

    Exposes a discrete horizon T (number of CTMC steps) and returns full
    trajectories for rollout/rollout_from.
    """

    process: MaskingCTMCFlowProcess
    steps: int = 100
    max_time: float = 1.0
    device: torch.device | str = "cpu"
    stop_when_filled: bool = False
    temperature: float = 1.0
    time_grid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        self.process.model.eval()
        self.T = int(self.steps)
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.max_time <= 0:
            raise ValueError("max_time must be positive")
        self.temperature = float(self.temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.time_grid is not None:
            t_grid = torch.as_tensor(self.time_grid, dtype=torch.float)
            if t_grid.ndim != 1 or t_grid.shape[0] != self.steps + 1:
                raise ValueError("time_grid must have shape [steps + 1]")
            if not torch.all(t_grid[1:] > t_grid[:-1]):
                raise ValueError("time_grid must be strictly increasing")
            if abs(float(t_grid[0].item())) > 1e-6:
                raise ValueError("time_grid must start at 0")
            if abs(float(t_grid[-1].item()) - float(self.max_time)) > 1e-6:
                raise ValueError("time_grid must end at max_time")
            self.time_grid = t_grid

    def sample_prior(self) -> torch.Tensor:
        cfg = self.process.cfg
        custom_sampler = getattr(self.process, "_custom_initial_state", None)
        if callable(custom_sampler):
            custom_prior = custom_sampler(1, self.device)
            if isinstance(custom_prior, torch.Tensor):
                return custom_prior[0].to(self.device)
        if getattr(self.process, "interpolant", "mask") == "mask":
            return torch.full((cfg.seq_len,), cfg.mask_id, dtype=torch.long, device=self.device)
        return torch.randint(0, cfg.vocab_size, (cfg.seq_len,), device=self.device, dtype=torch.long)

    def rollout(self, x0: torch.Tensor) -> List[torch.Tensor]:
        return self.rollout_from(x0, 0)

    def rollout_from(self, x_t: torch.Tensor, t: int) -> List[torch.Tensor]:
        if x_t.ndim != 1:
            raise ValueError("rollout_from expects a 1D tensor [seq_len]")
        traj = self.rollout_from_batch(x_t.unsqueeze(0), t)
        return [s[0].clone() for s in traj]

    def _normalize_update_mask(self, update_mask: Optional[torch.Tensor], batch_size: int, seq_len: int) -> Optional[torch.Tensor]:
        if update_mask is None:
            return None
        mask = torch.as_tensor(update_mask, device=self.device, dtype=torch.bool)
        if mask.ndim == 1:
            if mask.numel() != seq_len:
                raise ValueError(f"1D update_mask must have length {seq_len}")
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        elif mask.ndim == 2:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(f"2D update_mask must have shape ({batch_size}, {seq_len})")
        else:
            raise ValueError("update_mask must be 1D [seq_len] or 2D [batch, seq_len]")
        return mask

    def rollout_from_batch(
        self,
        x_t: torch.Tensor,
        t: int | float,
        *,
        update_mask: Optional[torch.Tensor] = None,
        denoise_steps: Optional[int] = None,
        t_is_time: bool = False,
    ) -> List[torch.Tensor]:
        def time_to_step_coord(t_scalar: float) -> float:
            t_clamped = min(max(float(t_scalar), 0.0), float(self.max_time))
            if self.time_grid is None:
                return (t_clamped / float(self.max_time)) * float(self.steps)
            t_grid = self.time_grid
            if t_clamped <= float(t_grid[0].item()):
                return 0.0
            if t_clamped >= float(t_grid[-1].item()):
                return float(self.steps)
            t_tensor = torch.tensor(t_clamped, device=t_grid.device, dtype=t_grid.dtype)
            idx_hi = int(torch.searchsorted(t_grid, t_tensor, right=False).item())
            idx_hi = max(1, min(idx_hi, self.steps))
            idx_lo = idx_hi - 1
            t_lo = float(t_grid[idx_lo].item())
            t_hi = float(t_grid[idx_hi].item())
            alpha = (t_clamped - t_lo) / max(t_hi - t_lo, 1e-12)
            alpha = min(max(alpha, 0.0), 1.0)
            return float(idx_lo) + alpha

        if t_is_time:
            t_start_step = time_to_step_coord(float(t))
        else:
            t_start_step = float(t)
        if t_start_step < 0.0 or t_start_step > float(self.T):
            raise ValueError(f"t must be in [0, {self.T}]")
        if x_t.ndim != 2 or x_t.shape[1] != self.process.cfg.seq_len:
            raise ValueError(f"Expected x_t shape (batch, {self.process.cfg.seq_len})")
        interpolant = getattr(self.process, "interpolant", "mask")
        cfg = self.process.cfg
        model = self.process.model
        model.eval()
        current = x_t.clone().to(self.device)
        states = [current.clone()]
        update_mask_tensor = self._normalize_update_mask(update_mask, current.shape[0], current.shape[1])
        pad_token_id = getattr(model, "pad_token_id", None)
        if pad_token_id is not None:
            pad_frozen_mask = current == int(pad_token_id)
            if update_mask_tensor is None:
                update_mask_tensor = ~pad_frozen_mask
            else:
                update_mask_tensor = update_mask_tensor & (~pad_frozen_mask)
        frozen_reference = current.clone() if update_mask_tensor is not None else None

        def step_coord_to_time(step_coord: float) -> float:
            s = min(max(float(step_coord), 0.0), float(self.steps))
            if self.time_grid is None:
                return (s / float(max(1, self.steps))) * float(self.max_time)
            lo = int(math.floor(s))
            hi = min(self.steps, lo + 1)
            if hi == lo:
                return float(self.time_grid[lo].item())
            alpha = s - float(lo)
            t_lo = float(self.time_grid[lo].item())
            t_hi = float(self.time_grid[hi].item())
            return (1.0 - alpha) * t_lo + alpha * t_hi

        def iter_time_steps() -> Iterable[tuple[float, float]]:
            if denoise_steps is None:
                s = min(max(float(t_start_step), 0.0), float(self.steps))
                while s < float(self.steps):
                    s_next = min(float(self.steps), float(math.floor(s) + 1.0))
                    if s_next <= s:
                        break
                    t0 = step_coord_to_time(s)
                    t1 = step_coord_to_time(s_next)
                    yield t0, (t1 - t0)
                    s = s_next
                return

            n_steps = int(denoise_steps)
            if n_steps <= 0:
                raise ValueError("denoise_steps must be > 0 when provided")
            remaining = float(self.steps) - t_start_step
            if remaining <= 0.0:
                return

            # Subsample the remaining interval by evenly spaced step coordinates.
            # When denoise_steps is provided, treat it as the exact number of
            # integration substeps over [t_start, max_time], regardless of how
            # much of the suffix remains.
            step_points = [t_start_step + (remaining * float(k) / float(n_steps)) for k in range(n_steps + 1)]
            step_points[-1] = float(self.steps)

            for i in range(n_steps):
                s0 = float(step_points[i])
                s1 = float(step_points[i + 1])
                if s1 <= s0:
                    continue
                t0 = step_coord_to_time(s0)
                t1 = step_coord_to_time(s1)
                yield t0, (t1 - t0)

        custom_rollout = getattr(model, "rollout_from_batch_native", None)
        if callable(custom_rollout):
            start_time = step_coord_to_time(t_start_step)
            start_time_norm = start_time / float(self.max_time)
            return custom_rollout(
                current,
                t_start=start_time_norm,
                update_mask=update_mask_tensor,
                denoise_steps=denoise_steps,
                max_time=self.max_time,
                ctmc_steps=self.steps,
            )

        with torch.no_grad():
            for t_scalar, dt in iter_time_steps():
                if dt <= 0:
                    raise ValueError("Non-positive dt encountered in time_grid")
                t_batch = torch.full((current.shape[0],), t_scalar, device=self.device)
                logits = model(current, timesteps=t_batch)
                if interpolant == "mask":
                    rates = ctmc_rate_masking(
                        current,
                        t_scalar=t_scalar,
                        logits=logits,
                        mask_id=cfg.mask_id,
                        temperature=self.temperature,
                    )
                    next_state = ctmc_euler_step(
                        current,
                        rates,
                        dt,
                        vocab_size=cfg.vocab_size,
                        mask_id=cfg.mask_id,
                    )
                else:
                    rates = ctmc_rate_uniform(
                        current,
                        t_scalar=t_scalar,
                        logits=logits,
                        temperature=self.temperature,
                    )
                    next_state = ctmc_euler_step(
                        current,
                        rates,
                        dt,
                        vocab_size=cfg.vocab_size,
                        mask_id=None,
                    )
                if update_mask_tensor is not None:
                    # Freeze non-masked positions to their original x_t values.
                    next_state = torch.where(update_mask_tensor, next_state, frozen_reference)
                states.append(next_state.clone())
                current = next_state
                if self.stop_when_filled and interpolant == "mask":
                    if update_mask_tensor is None:
                        if not (current == cfg.mask_id).any():
                            break
                    else:
                        mutable_masks = (current == cfg.mask_id) & update_mask_tensor
                        if not mutable_masks.any():
                            break
        return states
