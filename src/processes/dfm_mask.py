"""CMDP/CTMC-style discrete flow matching with the masking interpolant.

The implementation follows the user-provided sketch verbatim and maps each of
the mathematical components from Campbell et al. (2024) to concrete helpers:

* ``mask_interpolant_sample`` implements Eq. (6) for sampling ``x_t``.
* ``MaskingCTMCFlowProcess.training_step`` minimises the cross-entropy loss in
  Eq. (8) using the helper loss registry.
* ``ctmc_rate_masking`` computes the conditional rate matrix for masking (Table
  1) and ``ctmc_euler_step`` performs the Euler CTMC update from Eq. (3).
* ``dfm_sample_ctmc`` stitches everything together to produce Algorithm 1.

The process class exposes a high-level training/sampling interface so that the
flow can be used like the other registry-backed processes in GenTuRBO.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from core.losses import LossFn, make_loss
from core.registry import register
from core.types import Batch
from core.utils import batch_to_device, resolve_device
from core.time_sampling import make_time_sampler, TimeSampler
from models.dfm import DFMConfig, DFMDenoiser

__all__ = [
    "mask_interpolant_sample",
    "uniform_interpolant_sample",
    "uniform_interpolant_sample_invalid",
    "corrupt_positions_invalid",
    "corrupt_positions_uniform",
    "sample_invalid_prior",
    "ctmc_rate_masking",
    "ctmc_rate_uniform",
    "ctmc_euler_step",
    "dfm_sample_ctmc",
    "MaskingCTMCFlowProcess",
    "build_flow_mask_ctmc",
]


def mask_interpolant_sample(x1: Tensor, t: Tensor, mask_id: int) -> Tensor:
    """Sample ``x_t`` from the masking interpolant ``p^{mask}_{t|1}`` (Eq. 6).

    ``x1`` and the returned ``x_t`` have shape ``[B, D]`` with entries in
    ``{0, ..., mask_id}``.  Each token independently chooses between staying at
    the clean token (probability ``t``) or being replaced by ``mask_id``.
    """

    if x1.ndim != 2:
        raise ValueError("mask_interpolant_sample expects x1 with shape [batch, seq_len]")
    if t.ndim != 1 or t.shape[0] != x1.shape[0]:
        raise ValueError("t must have shape [batch] to match x1")
    if not torch.is_floating_point(t):
        t = t.float()
    t_broadcast = t[:, None].expand_as(x1)
    keep_data = torch.bernoulli(t_broadcast).bool()
    xt = torch.full_like(x1, fill_value=mask_id)
    xt[keep_data] = x1[keep_data]
    return xt


def uniform_interpolant_sample(x1: Tensor, t: Tensor, vocab_size: int) -> Tensor:
    """Sample x_t by mixing x1 with uniform noise over the vocabulary."""
    if x1.ndim != 2:
        raise ValueError("uniform_interpolant_sample expects x1 with shape [batch, seq_len]")
    if t.ndim != 1 or t.shape[0] != x1.shape[0]:
        raise ValueError("t must have shape [batch] to match x1")
    if not torch.is_floating_point(t):
        t = t.float()
    keep = torch.bernoulli(t[:, None].expand_as(x1)).bool()
    noise = torch.randint(0, vocab_size, x1.shape, device=x1.device, dtype=torch.long)
    xt = torch.where(keep, x1, noise)
    return xt


def corrupt_positions_invalid(
    x1: Tensor,
    corrupt_mask: Tensor,
    transition_mask: Tensor,
    *,
    mode: str = "sequential_both",
) -> Tensor:
    """Corrupt specified positions to invalid tokens (sequential, left-to-right)."""
    if x1.ndim != 2:
        raise ValueError("corrupt_positions_invalid expects x1 with shape [batch, seq_len]")
    if corrupt_mask.shape != x1.shape:
        raise ValueError("corrupt_mask must have the same shape as x1")
    if transition_mask.ndim != 2 or transition_mask.shape[0] != transition_mask.shape[1]:
        raise ValueError("transition_mask must have shape [V, V]")
    if not mode.startswith("sequential"):
        raise ValueError("corrupt_positions_invalid only supports sequential modes")
    base_mode = mode.split("_", 1)[1] if "_" in mode else "both"
    if base_mode not in {"prev", "next", "any", "both"}:
        raise ValueError("sequential mode must be prev, next, any, or both")

    device = x1.device
    transition_mask = transition_mask.to(device)
    B, L = x1.shape
    V = transition_mask.shape[0]
    xt = x1.clone()

    def _combine(
        invalid_prev: Tensor | None,
        invalid_next: Tensor | None,
        how: str,
    ) -> Tensor | None:
        if how == "prev":
            return invalid_prev
        if how == "next":
            return invalid_next
        if invalid_prev is None:
            return invalid_next
        if invalid_next is None:
            return invalid_prev
        if how == "both":
            return invalid_prev & invalid_next
        return invalid_prev | invalid_next

    for pos in range(L):
        mask_pos = corrupt_mask[:, pos]
        if not mask_pos.any():
            continue
        invalid_prev = None
        invalid_next = None
        if pos > 0:
            prev = xt[:, pos - 1]
            invalid_prev = ~transition_mask[prev]
        if pos + 1 < L:
            nxt = xt[:, pos + 1]
            invalid_next = ~transition_mask.transpose(0, 1)[nxt]
        invalid_mask = _combine(invalid_prev, invalid_next, base_mode)
        if invalid_mask is None:
            invalid_mask = torch.zeros((B, V), dtype=torch.bool, device=device)
        weights = invalid_mask.float()
        counts = weights.sum(dim=-1, keepdim=True)
        fallback = counts == 0
        if fallback.any() and base_mode == "both":
            any_mask = _combine(invalid_prev, invalid_next, "any")
            if any_mask is not None:
                any_weights = any_mask.float()
                any_counts = any_weights.sum(dim=-1, keepdim=True)
                use_any = fallback & (any_counts > 0)
                weights = torch.where(use_any, any_weights, weights)
                counts = weights.sum(dim=-1, keepdim=True)
                fallback = counts == 0
        if fallback.any():
            weights = torch.where(fallback, torch.ones_like(weights), weights)
            counts = weights.sum(dim=-1, keepdim=True)
        probs = weights / counts.clamp_min(1.0)
        noise = torch.multinomial(probs, 1).view(-1)
        xt[mask_pos, pos] = noise[mask_pos]
    return xt


def corrupt_positions_uniform(x1: Tensor, corrupt_mask: Tensor, vocab_size: int) -> Tensor:
    """Corrupt specified positions to uniform random tokens."""
    if x1.ndim != 2:
        raise ValueError("corrupt_positions_uniform expects x1 with shape [batch, seq_len]")
    if corrupt_mask.shape != x1.shape:
        raise ValueError("corrupt_mask must have the same shape as x1")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    xt = x1.clone()
    if corrupt_mask.any():
        num_mut = int(corrupt_mask.sum().item())
        noise = torch.randint(0, vocab_size, (num_mut,), device=x1.device, dtype=x1.dtype)
        xt[corrupt_mask] = noise
    return xt


def uniform_interpolant_sample_invalid(
    x1: Tensor,
    t: Tensor,
    transition_mask: Tensor,
    *,
    mode: str = "any",
) -> Tensor:
    """Sample x_t by mixing x1 with tokens that violate local transitions."""
    if x1.ndim != 2:
        raise ValueError("uniform_interpolant_sample_invalid expects x1 with shape [batch, seq_len]")
    if t.ndim != 1 or t.shape[0] != x1.shape[0]:
        raise ValueError("t must have shape [batch] to match x1")
    if mode not in {
        "prev",
        "next",
        "any",
        "both",
        "sequential",
        "sequential_prev",
        "sequential_next",
        "sequential_any",
        "sequential_both",
    }:
        raise ValueError(
            "mode must be one of: prev, next, any, both, sequential, "
            "sequential_prev, sequential_next, sequential_any, sequential_both"
        )
    if transition_mask.ndim != 2 or transition_mask.shape[0] != transition_mask.shape[1]:
        raise ValueError("transition_mask must have shape [V, V]")
    if not torch.is_floating_point(t):
        t = t.float()
    keep = torch.bernoulli(t[:, None].expand_as(x1)).bool()
    device = x1.device
    transition_mask = transition_mask.to(device)
    B, L = x1.shape
    V = transition_mask.shape[0]

    if mode.startswith("sequential"):
        return corrupt_positions_invalid(x1, ~keep, transition_mask, mode=mode)

    invalid_prev = torch.zeros((B, L, V), dtype=torch.bool, device=device)
    if L > 1:
        prev = x1[:, :-1]
        invalid_prev[:, 1:, :] = ~transition_mask[prev]

    invalid_next = torch.zeros((B, L, V), dtype=torch.bool, device=device)
    if L > 1:
        nxt = x1[:, 1:]
        allowed_prev = transition_mask.transpose(0, 1)[nxt]
        invalid_next[:, :-1, :] = ~allowed_prev

    if mode == "prev":
        invalid_mask = invalid_prev
    elif mode == "next":
        invalid_mask = invalid_next
    elif mode == "both":
        invalid_mask = invalid_prev & invalid_next
    else:
        invalid_mask = invalid_prev | invalid_next

    weights = invalid_mask.float()
    counts = weights.sum(dim=-1, keepdim=True)
    fallback = counts == 0
    if fallback.any() and mode == "both":
        any_weights = (invalid_prev | invalid_next).float()
        any_counts = any_weights.sum(dim=-1, keepdim=True)
        use_any = fallback & (any_counts > 0)
        weights = torch.where(use_any, any_weights, weights)
        counts = weights.sum(dim=-1, keepdim=True)
        fallback = counts == 0
    if fallback.any():
        weights = torch.where(fallback, torch.ones_like(weights), weights)
        counts = weights.sum(dim=-1, keepdim=True)
    probs = weights / counts.clamp_min(1.0)
    noise = torch.multinomial(probs.view(-1, V), 1).view(B, L)
    xt = torch.where(keep, x1, noise)
    return xt


def sample_invalid_prior(
    num_samples: int,
    seq_len: int,
    transition_mask: Tensor,
    *,
    mode: str = "sequential_both",
    device: Optional[torch.device] = None,
) -> Tensor:
    """Sample a prior sequence by enforcing invalid transitions."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if transition_mask.ndim != 2 or transition_mask.shape[0] != transition_mask.shape[1]:
        raise ValueError("transition_mask must have shape [V, V]")
    device = device or transition_mask.device
    vocab_size = int(transition_mask.shape[0])
    x0 = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    t0 = torch.zeros((num_samples,), device=device)
    return uniform_interpolant_sample_invalid(x0, t0, transition_mask.to(device), mode=mode)


def ctmc_rate_masking(
    xt: Tensor,
    t_scalar: float,
    logits: Tensor,
    mask_id: int,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Conditional CTMC rate matrix for masking (Table 1 + Eq. 7).

    Args:
        xt: Integer tensor ``[B, D]`` with the current CTMC state.
        t_scalar: Scalar time ``t`` in ``(0, 1)``.
        logits: Model logits ``[B, D, S]`` over the data vocabulary.
        mask_id: Mask token id ``M``.

    Returns:
        Off-diagonal rates ``R_t^\theta(x_t, j)`` with shape ``[B, D, S]``.
    """

    if not (0.0 <= t_scalar < 1.0):
        raise ValueError("t must lie in [0, 1) for the masking rate")
    # Clamp away from 1.0 to avoid division by zero when t_scalar is extremely close.
    t_scalar = min(float(t_scalar), 1.0 - 1e-6)
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    is_mask = (xt == mask_id).unsqueeze(-1)
    rates = torch.where(is_mask, probs / (1.0 - t_scalar), torch.zeros_like(probs))
    return rates


def ctmc_rate_uniform(
    xt: Tensor,
    t_scalar: float,
    logits: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """CTMC off-diagonal rates for uniform interpolant: allow moves from any token."""
    if not (0.0 <= t_scalar < 1.0):
        raise ValueError("t must lie in [0, 1) for the uniform rate")
    t_scalar = min(float(t_scalar), 1.0 - 1e-6)
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    num_classes = probs.size(-1)
    self_mask = F.one_hot(xt, num_classes=num_classes).bool()
    off_diag = probs.masked_fill(self_mask, 0.0)
    return off_diag / (1.0 - t_scalar)


def ctmc_euler_step(
    xt: Tensor,
    R_off: Tensor,
    dt: float,
    *,
    vocab_size: int,
    mask_id: Optional[int],
) -> Tensor:
    """Single Euler step of the CTMC transition (Eq. 3).

    ``R_off`` represents the off-diagonal transition rates to data tokens and
    has shape ``[B, D, S]``.  The diagonal term is chosen implicitly so rows
    sum to one when multiplied by ``dt``.  Positions that are not currently the
    mask never move because their rate rows are identically zero.  The
    underlying categorical support includes both the data tokens and the mask
    index ``mask_id`` (if ``mask_id == vocab_size`` this simply adds one extra
    entry for the masked state).
    """

    if xt.ndim != 2 or R_off.shape[:2] != xt.shape:
        raise ValueError("ctmc_euler_step expects R_off to match xt in the first two dims")
    if R_off.shape[-1] != vocab_size:
        raise ValueError("R_off last dimension must equal vocab_size (data tokens only)")
    p_data = (R_off * dt).clamp_min(0.0)
    out_mass = p_data.sum(dim=-1, keepdim=True)
    stay_prob = (1.0 - out_mass).clamp_min(0.0).squeeze(-1)

    total_states = vocab_size if mask_id is None else max(vocab_size, mask_id + 1)
    probs = torch.zeros((*xt.shape, total_states), device=xt.device, dtype=p_data.dtype)
    probs[:, :, :vocab_size] = p_data
    flat = xt.view(-1)
    probs_flat = probs.view(-1, total_states)
    probs_flat[torch.arange(flat.numel(), device=xt.device), flat] += stay_prob.view(-1)
    probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sampled = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
    return sampled.view_as(xt)


def _build_time_grid(
    steps: int,
    max_time: float,
    *,
    schedule: str = "linear",
    schedule_cfg: Optional[dict[str, object]] = None,
) -> Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive for time grid")
    schedule_cfg = schedule_cfg or {}
    s = torch.linspace(0.0, 1.0, steps + 1)
    if schedule == "linear":
        t = s
    elif schedule == "power":
        power = float(schedule_cfg.get("power", 2.0))
        if power <= 0:
            raise ValueError("power must be positive")
        bias = str(schedule_cfg.get("bias", "end"))
        if bias == "end":
            t = 1.0 - (1.0 - s) ** power
        elif bias == "start":
            t = s**power
        else:
            raise ValueError("bias must be 'start' or 'end'")
    elif schedule == "log":
        min_frac = float(schedule_cfg.get("min_frac", schedule_cfg.get("min", 1e-3)))
        if not (0.0 < min_frac < 1.0):
            raise ValueError("min_frac must be in (0, 1)")
        bias = str(schedule_cfg.get("bias", "end"))
        log_min = math.log10(min_frac)
        log_space = torch.logspace(log_min, 0.0, steps + 1)
        if bias == "start":
            t = log_space
        elif bias == "end":
            t = 1.0 - log_space.flip(0) + min_frac
        else:
            raise ValueError("bias must be 'start' or 'end'")
    elif schedule == "cosine":
        t = 0.5 - 0.5 * torch.cos(torch.pi * s)
    else:
        raise ValueError(f"Unknown time schedule '{schedule}'")
    return t * float(max_time)


@torch.no_grad()
def dfm_sample_ctmc(
    model: nn.Module,
    num_samples: int,
    cfg: DFMConfig,
    *,
    dt: float,
    max_time: float = 1.0,
    device: torch.device,
    show_progress: bool = False,
    interpolant: str = "mask",
    temperature: float = 1.0,
    stop_when_filled: bool = True,
    time_schedule: str = "linear",
    time_schedule_cfg: Optional[dict[str, object]] = None,
    prior_invalid: bool = False,
    prior_invalid_mode: str = "sequential_both",
    transition_mask: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
) -> Tensor:
    """Euler CTMC sampler mirroring Algorithm 1 in the paper."""

    if dt <= 0:
        raise ValueError("dt must be positive")
    steps_est = max(1, int(math.ceil(max_time / dt)))
    time_grid = _build_time_grid(steps_est, max_time, schedule=time_schedule, schedule_cfg=time_schedule_cfg)

    iterator = range(steps_est)
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="Sampling CTMC", leave=False)
        except Exception:
            pass  # tqdm is optional; fall back silently
    model.eval()
    seq_len = cfg.seq_len
    vocab_size = cfg.vocab_size
    if initial_state is not None:
        x = torch.as_tensor(initial_state, device=device, dtype=torch.long)
        if x.shape != (num_samples, seq_len):
            raise ValueError(f"initial_state must have shape ({num_samples}, {seq_len})")
        mask_id = cfg.mask_id if interpolant == "mask" else None
    elif interpolant == "mask":
        x = torch.full((num_samples, seq_len), cfg.mask_id, dtype=torch.long, device=device)
        mask_id = cfg.mask_id
    elif interpolant == "uniform":
        if prior_invalid:
            if transition_mask is None:
                raise ValueError("prior_invalid requested but no transition_mask provided")
            x = sample_invalid_prior(
                num_samples,
                seq_len,
                transition_mask,
                mode=prior_invalid_mode,
                device=device,
            )
        else:
            x = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
        mask_id = None
    else:
        raise ValueError(f"Unknown interpolant '{interpolant}'")
    pad_token_id = getattr(model, "pad_token_id", None)
    pad_frozen_mask = None
    pad_frozen_reference = None
    if initial_state is not None and pad_token_id is not None:
        pad_frozen_mask = x == int(pad_token_id)
        if bool(pad_frozen_mask.any().item()):
            pad_frozen_reference = x.clone()
    for idx in iterator:
        if idx >= steps_est:
            break
        t = float(time_grid[idx].item())
        dt_step = float(time_grid[idx + 1].item() - t)
        if dt_step <= 0:
            continue
        t_batch = torch.full((num_samples,), t, device=device)
        logits = model(x, timesteps=t_batch)
        if interpolant == "mask":
            R_off = ctmc_rate_masking(
                x,
                t_scalar=t,
                logits=logits,
                mask_id=cfg.mask_id,
                temperature=temperature,
            )
        else:
            R_off = ctmc_rate_uniform(
                x,
                t_scalar=t,
                logits=logits,
                temperature=temperature,
            )
        x = ctmc_euler_step(x, R_off, dt_step, vocab_size=vocab_size, mask_id=mask_id)
        if pad_frozen_mask is not None and pad_frozen_reference is not None:
            x = torch.where(pad_frozen_mask, pad_frozen_reference, x)
        if stop_when_filled and mask_id is not None and not (x == mask_id).any():
            break  # all tokens are filled; further steps would be no-ops
    return x


@dataclass
class MaskingCTMCFlowProcess:
    """Flow-matching process for discrete data using the masking interpolant."""

    model: nn.Module
    cfg: DFMConfig
    loss_fn: LossFn
    device: torch.device
    interpolant: str
    time_sampler: TimeSampler
    mask_loss_only: bool
    temperature: float
    stop_when_filled: bool
    corrupt_invalid: bool
    corrupt_invalid_mode: str
    transition_mask: Optional[Tensor]
    _warned_missing_transition_mask: bool
    prior_invalid: bool
    prior_invalid_mode: str

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        *,
        vocab_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        mask_id: Optional[int] = None,
        dt: float = 1e-2,
        loss: str = "flow_matching_discrete",
        device: Optional[str | torch.device] = None,
        denoiser_cfg: Optional[dict[str, object]] = None,
        interpolant: str = "mask",
        time_sampler: str = "uniform",
        time_sampler_cfg: Optional[dict[str, object]] = None,
        mask_loss_only: bool = False,
        temperature: float = 1.0,
        stop_when_filled: bool = True,
        corrupt_invalid: bool = False,
        corrupt_invalid_mode: str = "any",
        transition_mask: Optional[Tensor] = None,
        transition_mask_path: Optional[str] = None,
        sample_time_schedule: str = "linear",
        sample_time_schedule_cfg: Optional[dict[str, object]] = None,
        prior_invalid: bool = False,
        prior_invalid_mode: str = "sequential_both",
    ):
        cfg_from_model = getattr(model, "cfg", None) if model is not None else None
        if isinstance(cfg_from_model, DFMConfig):
            cfg = cfg_from_model
        else:
            if model is not None:
                if vocab_size is None:
                    inferred_vocab = getattr(model, "vocab_size", None)
                    if inferred_vocab is not None:
                        vocab_size = int(inferred_vocab)
                if seq_len is None:
                    inferred_seq_len = getattr(model, "seq_len", None)
                    if inferred_seq_len is not None:
                        seq_len = int(inferred_seq_len)
                if mask_id is None:
                    inferred_mask = getattr(model, "mask_id", None)
                    if inferred_mask is not None:
                        mask_id = int(inferred_mask)
            if vocab_size is None or seq_len is None:
                raise ValueError("vocab_size and seq_len must be provided when model has no cfg")
            vocab_size = int(vocab_size)
            seq_len = int(seq_len)
            mask_value = int(mask_id) if mask_id is not None else vocab_size
            cfg_kwargs: dict[str, Any] = {
                "vocab_size": vocab_size,
                "seq_len": seq_len,
                "mask_id": mask_value,
            }
            if denoiser_cfg:
                cfg_kwargs.update(denoiser_cfg)
            cfg = DFMConfig(**cfg_kwargs)
            if model is None:
                model = DFMDenoiser(cfg)

        if model is None:
            raise RuntimeError("DFM model construction failed; please provide a model or denoiser config")

        self.model = model
        self.cfg = cfg
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        self.loss_fn = make_loss(loss)
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.interpolant = interpolant
        self.time_sampler = make_time_sampler(time_sampler, **(time_sampler_cfg or {}))
        self.mask_loss_only = bool(mask_loss_only)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.stop_when_filled = bool(stop_when_filled)
        self._data_shape: Optional[tuple[int, ...]] = (self.cfg.seq_len,)
        self.corrupt_invalid = bool(corrupt_invalid)
        self.corrupt_invalid_mode = corrupt_invalid_mode
        self.transition_mask = None
        if transition_mask is not None:
            self.transition_mask = transition_mask.detach().clone()
        elif transition_mask_path is not None:
            from pathlib import Path

            loaded = torch.load(Path(transition_mask_path).expanduser(), map_location="cpu", weights_only=False)
            if isinstance(loaded, Tensor):
                self.transition_mask = loaded
            elif isinstance(loaded, dict) and "transition_mask" in loaded:
                self.transition_mask = loaded["transition_mask"]
            else:
                mask = getattr(loaded, "transition_mask", None)
                if mask is None:
                    raise TypeError("transition_mask_path did not contain a transition_mask")
                self.transition_mask = mask
        self._warned_missing_transition_mask = False
        self.sample_time_schedule = sample_time_schedule
        self.sample_time_schedule_cfg = sample_time_schedule_cfg or {}
        self.prior_invalid = bool(prior_invalid)
        self.prior_invalid_mode = prior_invalid_mode

    def _custom_initial_state(self, num_samples: int, device: torch.device) -> Optional[Tensor]:
        sampler = getattr(self.model, "sample_initial_state", None)
        if not callable(sampler):
            return None
        initial = sampler(
            num_samples=num_samples,
            seq_len=self.cfg.seq_len,
            device=device,
            vocab_size=self.cfg.vocab_size,
            interpolant=self.interpolant,
        )
        if initial is None:
            return None
        initial_tensor = torch.as_tensor(initial, device=device, dtype=torch.long)
        if initial_tensor.shape != (num_samples, self.cfg.seq_len):
            raise ValueError(
                f"custom initial state must have shape ({num_samples}, {self.cfg.seq_len})"
            )
        return initial_tensor

    def _record_shape(self, x: Tensor) -> None:
        if self._data_shape is None:
            self._data_shape = tuple(x.shape[1:])

    def training_step(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        batch = batch_to_device(batch, self.device)
        x1 = batch["x"].long()
        self._record_shape(x1)
        bsz = x1.shape[0]
        t = self.time_sampler(bsz, self.device)
        if self.interpolant == "mask":
            xt = mask_interpolant_sample(x1, t, self.cfg.mask_id)
        elif self.interpolant == "uniform":
            if self.corrupt_invalid:
                transition_mask = batch.get("transition_mask")
                if transition_mask is None:
                    instance = batch.get("instance")
                    transition_mask = getattr(instance, "transition_mask", None)
                if transition_mask is None:
                    transition_mask = self.transition_mask
                if transition_mask is None:
                    if not self._warned_missing_transition_mask:
                        print("[warn] corrupt_invalid enabled but no transition_mask provided; using uniform noise.")
                        self._warned_missing_transition_mask = True
                    xt = uniform_interpolant_sample(x1, t, self.cfg.vocab_size)
                else:
                    xt = uniform_interpolant_sample_invalid(
                        x1,
                        t,
                        transition_mask,
                        mode=self.corrupt_invalid_mode,
                    )
            else:
                xt = uniform_interpolant_sample(x1, t, self.cfg.vocab_size)
        else:
            raise ValueError(f"Unknown interpolant '{self.interpolant}'")
        logits = self.model(xt, timesteps=t)
        logits_flat = logits.reshape(-1, self.cfg.vocab_size)
        targets_flat = x1.view(-1)
        target_batch: Batch = {**batch, "target_class": targets_flat}
        mask_flat = None
        if self.interpolant == "mask":
            mask_flat = (xt == self.cfg.mask_id).view(-1)
        elif self.interpolant == "uniform":
            mask_flat = (xt.view(-1) != targets_flat)

        if self.mask_loss_only and mask_flat is not None and mask_flat.any():
            masked_targets = targets_flat[mask_flat]
            loss = self.loss_fn(logits_flat[mask_flat], {**batch, "target_class": masked_targets})
        else:
            loss = self.loss_fn(logits_flat, target_batch)
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            accuracy = (pred == x1).float().mean()
            metrics = {
                "loss": loss.detach(),
                "t_mean": t.mean().detach(),
                "token_accuracy": accuracy.detach(),
            }
            if mask_flat is not None:
                if self.interpolant == "mask":
                    frac_mask = (xt == self.cfg.mask_id).float().mean()
                    metrics["mask_fraction"] = frac_mask.detach()
                else:
                    frac_changed = (xt != x1).float().mean()
                    metrics["changed_fraction"] = frac_changed.detach()
                if mask_flat.any():
                    masked_acc = (pred.view(-1)[mask_flat] == targets_flat[mask_flat]).float().mean()
                    metrics["masked_token_accuracy"] = masked_acc.detach()
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        *,
        steps: Optional[int] = None,
        dt: Optional[float] = None,
        max_time: float = 1.0,
        device: Optional[str | torch.device] = None,
        show_progress: bool = True,
        time_schedule: Optional[str] = None,
        time_schedule_cfg: Optional[dict[str, object]] = None,
        prior_invalid: Optional[bool] = None,
        prior_invalid_mode: Optional[str] = None,
    ) -> Tensor:
        if self._data_shape is None:
            raise RuntimeError("sample() called before observing any data")
        device = resolve_device(device) if device is not None else self.device
        if steps is not None:
            if steps <= 0:
                raise ValueError("steps must be positive when provided")
            integrator_dt = float(max_time) / float(steps)
        else:
            integrator_dt = float(dt) if dt is not None else self.dt
        initial_state = self._custom_initial_state(num_samples, device)
        native_sampler = getattr(self.model, "sample_native", None)
        if callable(native_sampler):
            return native_sampler(
                num_samples=num_samples,
                seq_len=self.cfg.seq_len,
                device=device,
                steps=steps,
                max_time=max_time,
            )
        return dfm_sample_ctmc(
            self.model,
            num_samples,
            self.cfg,
            dt=integrator_dt,
            max_time=max_time,
            device=device,
            show_progress=show_progress,
            interpolant=self.interpolant,
            temperature=self.temperature,
            stop_when_filled=self.stop_when_filled,
            time_schedule=time_schedule or self.sample_time_schedule,
            time_schedule_cfg=time_schedule_cfg or self.sample_time_schedule_cfg,
            prior_invalid=self.prior_invalid if prior_invalid is None else prior_invalid,
            prior_invalid_mode=self.prior_invalid_mode if prior_invalid_mode is None else prior_invalid_mode,
            transition_mask=self.transition_mask,
            initial_state=initial_state,
        )


@register("process", "flow_mask_ctmc")
def build_flow_mask_ctmc(model: Optional[nn.Module] = None, **cfg: Any) -> MaskingCTMCFlowProcess:
    return MaskingCTMCFlowProcess(model=model, **cfg)
