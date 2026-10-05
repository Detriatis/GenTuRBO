"""Trust-region strategy implementations for interpolant search."""

from __future__ import annotations

import math
from typing import Optional

import torch
from search.interpolant.types import TrustRegionConfig
from search.interpolant.utils import edits_to_time, time_to_edits


def _clamp_edits(edits: float, *, low: float, high: float) -> float:
    return float(min(max(float(edits), float(low)), float(high)))


def _clamp_time(t_value: float) -> float:
    return float(min(max(float(t_value), 0.0), 1.0))


def _domain(config: TrustRegionConfig) -> str:
    return str(getattr(config, "domain", "edits") or "edits").lower()


def _apply_alpha_to_time(config: TrustRegionConfig, t_base: float) -> float:
    if _domain(config) == "time":
        t = float(t_base) * float(config.alpha)
        return _clamp_time_to_bounds(config, t)
    return _clamp_time(t_base)


def _time_bounds(config: TrustRegionConfig) -> tuple[float, float]:
    low = _clamp_time(getattr(config, "time_min", 0.0))
    high = _clamp_time(getattr(config, "time_max", 1.0))
    if high < low:
        raise RuntimeError("trust_region time bounds are invalid")
    return low, high


def _clamp_time_to_bounds(config: TrustRegionConfig, t_value: float) -> float:
    low, high = _time_bounds(config)
    return float(min(max(float(t_value), low), high))


def _time_to_noise_radius(t_value: float) -> float:
    return 1.0 - _clamp_time(t_value)


def _noise_radius_to_time(radius: float) -> float:
    return 1.0 - _clamp_time(radius)


def _region_expand_scalar(config: TrustRegionConfig) -> float:
    value = config.region_expand_scalar if config.region_expand_scalar is not None else config.region_scalar
    scale = float(value)
    if scale <= 1.0:
        raise RuntimeError("region_expand_scalar must be > 1.0")
    return scale


def _region_contract_scalar(config: TrustRegionConfig) -> float:
    value = config.region_contract_scalar if config.region_contract_scalar is not None else config.region_scalar
    scale = float(value)
    if scale <= 1.0:
        raise RuntimeError("region_contract_scalar must be > 1.0")
    return scale


def _region_shrink_scalar(config: TrustRegionConfig) -> float:
    if config.region_shrink_scalar is not None:
        shrink = float(config.region_shrink_scalar)
        if not (0.0 < shrink < 1.0):
            raise RuntimeError("region_shrink_scalar must be in (0, 1)")
        if config.region_contract_scalar is not None:
            contract = _region_contract_scalar(config)
            if abs(contract - (1.0 / shrink)) > 1e-12:
                raise RuntimeError("region_contract_scalar must equal 1 / region_shrink_scalar when both are set")
        return shrink
    return 1.0 / _region_contract_scalar(config)


def _scale_time(
    config: TrustRegionConfig,
    current_time: float,
    *,
    improved: bool,
) -> float:
    # The model/writeout convention is t=0 pure noise and t=1 data. Apply
    # trust-region expansion/shrinkage in noise-radius space so success moves
    # toward pure noise and failure moves toward the data state.
    cur_radius = _time_to_noise_radius(current_time)
    updated_radius = (
        cur_radius * _region_expand_scalar(config)
        if improved
        else cur_radius * _region_shrink_scalar(config)
    )
    return _clamp_time_to_bounds(config, _noise_radius_to_time(updated_radius))


def _target_edits_for_seq_len(config: TrustRegionConfig, *, seq_len: int) -> float:
    if config.mean_edits is None:
        raise RuntimeError("trust_region requires mean_edits")
    effective_len = int(max(1, seq_len))
    return float(min(max(float(config.mean_edits), 0.0), float(effective_len)))


def _target_time_for_seq_len(config: TrustRegionConfig, *, seq_len: int) -> float:
    if _domain(config) == "time":
        if config.mean_time is None:
            raise RuntimeError("trust_region requires mean_time")
        return _clamp_time(config.mean_time)
    mean_edits = _target_edits_for_seq_len(config, seq_len=seq_len)
    return edits_to_time(mean_edits, seq_len=seq_len)


def _update_time_alpha_from_error(
    config: TrustRegionConfig,
    *,
    mean_edit_error: Optional[float],
) -> None:
    if _domain(config) != "time" or mean_edit_error is None or float(config.alpha_k) <= 0.0:
        return
    # mean_edit_error is expected to already be normalized by the effective
    # editable sequence length for each proposal.
    scale = math.exp(-float(config.alpha_k) * float(mean_edit_error))
    new_alpha = float(config.alpha) * scale
    config.alpha = float(min(max(new_alpha, float(config.alpha_min)), float(config.alpha_max)))


def _update_mean_edits_from_error(
    config: TrustRegionConfig,
    *,
    mean_edit_error: Optional[float],
    low_edits: float,
    high_edits: float,
) -> None:
    if _domain(config) != "edits" or mean_edit_error is None or float(config.alpha_k) <= 0.0:
        return
    if config.mean_edits is None:
        return
    # Positive error means observed edits were lower than requested, so increase
    # the edit target directly. Negative error makes the next request more local.
    scale = math.exp(float(config.alpha_k) * float(mean_edit_error))
    new_edits = float(config.mean_edits) * scale
    config.mean_edits = _clamp_edits(new_edits, low=low_edits, high=high_edits)


class AdditiveTrustRegion:
    """Maintain a scalar trust-region state and update it additively on improve/fail."""

    def __init__(self, *, config: TrustRegionConfig, seq_len: int, device: torch.device) -> None:
        _ = device
        self.config = config
        self.seq_len = int(max(1, seq_len))
        self.domain = _domain(config)
        self.low_edits = float(max(0.0, config.edits_min))
        max_from_cfg = float(config.edits_max) if config.edits_max is not None else float(self.seq_len)
        self.high_edits = float(min(float(self.seq_len), max_from_cfg))
        if self.domain == "time":
            self.low_time, self.high_time = _time_bounds(config)
            if self.config.mean_time is None:
                raise RuntimeError("trust_region requires mean_time")
            self.config.mean_time = _clamp_time(self.config.mean_time)
            if self.config.time_init is None:
                self.config.time_init = float(self.config.mean_time)
            else:
                self.config.time_init = _clamp_time(self.config.time_init)
        else:
            if self.config.mean_edits is None:
                raise RuntimeError("trust_region requires mean_edits")
            self.config.mean_edits = _clamp_edits(self.config.mean_edits, low=self.low_edits, high=self.high_edits)

    def sample_times(
        self,
        *,
        num_candidates: int,
        iteration: int,
        seq_len: Optional[int] = None,
    ) -> list[float]:
        effective_seq_len = int(max(1, seq_len if seq_len is not None else self.seq_len))
        t_base = _target_time_for_seq_len(self.config, seq_len=effective_seq_len)
        t = _apply_alpha_to_time(self.config, t_base)
        if self.config.log_every and (iteration % int(self.config.log_every) == 0):
            if self.domain == "time":
                print(
                    f"[tr] it={iteration:04d} type=AdditiveTrustRegion "
                    f"mean_time={t_base:.3f} alpha={self.config.alpha:.3f} t={t:.3f}"
                )
            else:
                mean_edits = _target_edits_for_seq_len(self.config, seq_len=effective_seq_len)
                print(
                    f"[tr] it={iteration:04d} type=AdditiveTrustRegion "
                    f"mean_edits={mean_edits:.2f} seq_len={effective_seq_len:d} "
                    f"t={t:.3f} cal_k={self.config.alpha_k:.3f}"
                )
        return [t for _ in range(max(1, int(num_candidates)))]

    def expand_once(self) -> tuple[Optional[float], Optional[float]]:
        edits_delta = float(self.config.edits_delta)
        if self.domain == "time":
            if self.config.mean_time is None:
                raise RuntimeError("trust_region state is incomplete")
            old = _clamp_time(self.config.mean_time)
            if edits_delta > 0.0:
                new = _clamp_time_to_bounds(self.config, old - edits_delta)
            else:
                new = _scale_time(self.config, old, improved=True)
            self.config.mean_time = float(new)
            return old, new

        if self.config.mean_edits is None:
            raise RuntimeError("trust_region state is incomplete")
        old = _clamp_edits(self.config.mean_edits, low=self.low_edits, high=self.high_edits)
        if edits_delta > 0.0:
            new = _clamp_edits(old + edits_delta, low=self.low_edits, high=self.high_edits)
        else:
            new = _clamp_edits(old * _region_expand_scalar(self.config), low=self.low_edits, high=self.high_edits)
        self.config.mean_edits = float(new)
        return old, new

    def update(
        self,
        *,
        best_time: Optional[float],
        best_edits: Optional[float] = None,
        improved: bool,
        mean_edit_error: Optional[float] = None,
    ) -> None:
        edits_delta = float(self.config.edits_delta)
        if self.domain == "time":
            _update_time_alpha_from_error(self.config, mean_edit_error=mean_edit_error)
            if self.config.mean_time is None:
                raise RuntimeError("trust_region state is incomplete")
            if edits_delta > 0.0:
                direction = -1.0 if improved else 1.0
                self.config.mean_time = _clamp_time_to_bounds(
                    self.config,
                    self.config.mean_time + (direction * edits_delta),
                )
                return
            if improved and best_time is not None:
                self.config.mean_time = _clamp_time_to_bounds(self.config, best_time)
            return

        if self.config.mean_edits is None:
            raise RuntimeError("trust_region state is incomplete")
        _update_mean_edits_from_error(
            self.config,
            mean_edit_error=mean_edit_error,
            low_edits=self.low_edits,
            high_edits=self.high_edits,
        )
        if edits_delta > 0.0:
            direction = 1.0 if improved else -1.0
            current = _clamp_edits(self.config.mean_edits, low=self.low_edits, high=self.high_edits)
            self.config.mean_edits = _clamp_edits(
                current + (direction * edits_delta),
                low=self.low_edits,
                high=self.high_edits,
            )
            return
        if improved and best_edits is not None:
            self.config.mean_edits = _clamp_edits(
                float(best_edits),
                low=self.low_edits,
                high=self.high_edits,
            )
            return
        if improved and best_time is not None:
            self.config.mean_edits = _clamp_edits(
                time_to_edits(_clamp_time(best_time), seq_len=self.seq_len),
                low=self.low_edits,
                high=self.high_edits,
            )


class ScalarTrustRegion:
    """Thresholded scalar trust region with multiplicative expansion/shrinkage."""

    def __init__(self, *, config: TrustRegionConfig, seq_len: int, device: torch.device) -> None:
        _ = device
        self.config = config
        self.seq_len = int(max(1, seq_len))
        self.domain = _domain(config)
        self.low_edits = float(max(0.0, config.edits_min))
        max_from_cfg = float(config.edits_max) if config.edits_max is not None else float(self.seq_len)
        self.high_edits = float(min(float(self.seq_len), max_from_cfg))

        if self.config.tau_success < 1:
            raise ValueError("tau_success must be >= 1")
        if self.config.tau_failure < 1:
            raise ValueError("tau_failure must be >= 1")
        _region_expand_scalar(self.config)
        _region_shrink_scalar(self.config)

        if self.domain == "time":
            self.low_time, self.high_time = _time_bounds(config)
            if self.config.mean_time is None:
                raise RuntimeError("trust_region requires mean_time")
            self.config.time_init = (
                _clamp_time(self.config.time_init)
                if self.config.time_init is not None
                else _clamp_time(self.config.mean_time)
            )
            self.config.mean_time = float(self.config.time_init)
        else:
            if self.config.mean_edits is None:
                raise RuntimeError("trust_region requires mean_edits")
            self.config.edits_init = (
                _clamp_edits(self.config.edits_init, low=self.low_edits, high=self.high_edits)
                if self.config.edits_init is not None
                else _clamp_edits(self.config.mean_edits, low=self.low_edits, high=self.high_edits)
            )
            self.config.mean_edits = float(self.config.edits_init)

        self.success_streak = 0
        self.failure_streak = 0

    def sample_times(
        self,
        *,
        num_candidates: int,
        iteration: int,
        seq_len: Optional[int] = None,
    ) -> list[float]:
        effective_seq_len = int(max(1, seq_len if seq_len is not None else self.seq_len))
        t_base = _target_time_for_seq_len(self.config, seq_len=effective_seq_len)
        t = _apply_alpha_to_time(self.config, t_base)
        if self.config.log_every and (iteration % int(self.config.log_every) == 0):
            if self.domain == "time":
                print(
                    f"[tr] it={iteration:04d} type=ScalarTrustRegion "
                    f"mean_time={t_base:.3f} alpha={self.config.alpha:.3f} "
                    f"t={t:.3f} succ={self.success_streak} fail={self.failure_streak}"
                )
            else:
                mean_edits = _target_edits_for_seq_len(self.config, seq_len=effective_seq_len)
                print(
                    f"[tr] it={iteration:04d} type=ScalarTrustRegion mean_edits={mean_edits:.2f} "
                    f"seq_len={effective_seq_len:d} t={t:.3f} cal_k={self.config.alpha_k:.3f} "
                    f"succ={self.success_streak} fail={self.failure_streak}"
                )
        return [t for _ in range(max(1, int(num_candidates)))]

    def expand_once(self) -> tuple[Optional[float], Optional[float]]:
        self.success_streak = 0
        self.failure_streak = 0
        if self.domain == "time":
            if self.config.mean_time is None:
                raise RuntimeError("trust_region state is incomplete")
            old = _clamp_time(self.config.mean_time)
            new = _scale_time(self.config, old, improved=True)
            self.config.mean_time = float(new)
            return old, new

        if self.config.mean_edits is None:
            raise RuntimeError("trust_region state is incomplete")
        old = _clamp_edits(self.config.mean_edits, low=self.low_edits, high=self.high_edits)
        new = _clamp_edits(old * _region_expand_scalar(self.config), low=self.low_edits, high=self.high_edits)
        self.config.mean_edits = float(new)
        return old, new

    def update(
        self,
        *,
        best_time: Optional[float],
        best_edits: Optional[float] = None,
        improved: bool,
        mean_edit_error: Optional[float] = None,
    ) -> None:
        _ = best_time
        _ = best_edits
        if self.domain == "time":
            _update_time_alpha_from_error(self.config, mean_edit_error=mean_edit_error)
            if self.config.mean_time is None:
                raise RuntimeError("trust_region state is incomplete")
            cur_time = _clamp_time(self.config.mean_time)
            if improved:
                self.success_streak += 1
                self.failure_streak = 0
                if self.success_streak >= int(self.config.tau_success):
                    self.config.mean_time = _scale_time(
                        self.config,
                        cur_time,
                        improved=True,
                    )
                    self.success_streak = 0
                    self.failure_streak = 0
                return

            self.failure_streak += 1
            self.success_streak = 0
            if self.failure_streak >= int(self.config.tau_failure):
                if cur_time >= (self.high_time - 1e-12):
                    self.config.mean_time = _clamp_time_to_bounds(
                        self.config,
                        float(self.config.time_init if self.config.time_init is not None else self.low_time),
                    )
                else:
                    self.config.mean_time = _scale_time(
                        self.config,
                        cur_time,
                        improved=False,
                    )
                self.success_streak = 0
                self.failure_streak = 0
            return

        if self.config.mean_edits is None:
            raise RuntimeError("trust_region state is incomplete")
        _update_mean_edits_from_error(
            self.config,
            mean_edit_error=mean_edit_error,
            low_edits=self.low_edits,
            high_edits=self.high_edits,
        )
        cur = _clamp_edits(self.config.mean_edits, low=self.low_edits, high=self.high_edits)

        if improved:
            self.success_streak += 1
            self.failure_streak = 0
            if self.success_streak >= int(self.config.tau_success):
                cur = min(self.high_edits, cur * _region_expand_scalar(self.config))
                self.config.mean_edits = float(cur)
                self.success_streak = 0
                self.failure_streak = 0
            return

        self.failure_streak += 1
        self.success_streak = 0
        if self.failure_streak >= int(self.config.tau_failure):
            if cur <= (self.low_edits + 1e-12):
                cur = float(self.config.edits_init if self.config.edits_init is not None else self.low_edits)
            else:
                cur = max(self.low_edits, cur * _region_shrink_scalar(self.config))
            self.config.mean_edits = _clamp_edits(cur, low=self.low_edits, high=self.high_edits)
            self.success_streak = 0
            self.failure_streak = 0


def make_trust_region_strategy(
    *,
    config: TrustRegionConfig,
    seq_len: int,
    device: torch.device,
) -> AdditiveTrustRegion | ScalarTrustRegion:
    kind = str(config.kind or "AdditiveTrustRegion")
    if kind == "AdditiveTrustRegion":
        return AdditiveTrustRegion(config=config, seq_len=seq_len, device=device)
    if kind == "ScalarTrustRegion":
        return ScalarTrustRegion(config=config, seq_len=seq_len, device=device)
    raise ValueError("tr_type must be one of: AdditiveTrustRegion, ScalarTrustRegion")


__all__ = [
    "AdditiveTrustRegion",
    "ScalarTrustRegion",
    "make_trust_region_strategy",
]
