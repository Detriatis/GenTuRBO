"""Candidate/trust-region config resolution for interpolant search."""

from __future__ import annotations

import argparse
from typing import Any

from search.interpolant.types import CandidateConfig, TrustRegionConfig
from search.interpolant.utils import time_to_edits


def _clamp_edits(edits: float, seq_len: int) -> float:
    return min(max(float(edits), 0.0), float(max(1, seq_len)))


def _mean_edits_from_inputs(
    args: argparse.Namespace,
    *,
    seq_len: int,
) -> float:
    if args.tr_edits is not None:
        edits = float(args.tr_edits)
        if edits < 0.0:
            raise ValueError("tr-edits must be >= 0")
        return _clamp_edits(edits, seq_len)

    if args.tr_mean is None:
        raise ValueError("trust_region requires one of: tr-mean or tr-edits")
    t_mean = float(args.tr_mean)
    if not (0.0 <= t_mean <= 1.0):
        raise ValueError("tr-mean must be in [0, 1]")
    return _clamp_edits(time_to_edits(t_mean, seq_len=seq_len), seq_len)


def _mean_time_from_inputs(args: argparse.Namespace) -> float:
    if args.tr_mean is None:
        raise ValueError("time-domain trust_region requires tr-mean")
    t_mean = float(args.tr_mean)
    if not (0.0 <= t_mean <= 1.0):
        raise ValueError("tr-mean must be in [0, 1]")
    return t_mean


def _optional_float_arg(args: argparse.Namespace, name: str) -> float | None:
    value = getattr(args, name, None)
    return float(value) if value is not None else None


def _region_contract_scalar_from_inputs(args: argparse.Namespace) -> float | None:
    contract_scalar = _optional_float_arg(args, "region_contract_scalar")
    shrink_scalar = _optional_float_arg(args, "region_shrink_scalar")
    if contract_scalar is not None and shrink_scalar is not None:
        if shrink_scalar <= 0.0:
            raise ValueError("region_shrink_scalar must be > 0")
        if abs(contract_scalar - (1.0 / shrink_scalar)) > 1e-12:
            raise ValueError("region_contract_scalar must equal 1 / region_shrink_scalar when both are set")
    return contract_scalar


def _region_shrink_scalar_from_inputs(args: argparse.Namespace) -> float | None:
    return _optional_float_arg(args, "region_shrink_scalar")


def _effective_region_scalar(tr: TrustRegionConfig, *, kind: str) -> float:
    if kind == "expand":
        value = tr.region_expand_scalar
    elif kind == "contract":
        value = tr.region_contract_scalar
    else:
        raise ValueError(f"Unknown region scalar kind: {kind}")
    return float(value) if value is not None else float(tr.region_scalar)


def _effective_region_shrink_scalar(tr: TrustRegionConfig) -> float:
    if tr.region_shrink_scalar is not None:
        return float(tr.region_shrink_scalar)
    contract_scalar = tr.region_contract_scalar if tr.region_contract_scalar is not None else tr.region_scalar
    return 1.0 / float(contract_scalar)


def resolve_candidate_config(args: argparse.Namespace, runtime: Any) -> tuple[CandidateConfig, TrustRegionConfig]:
    strategy = (args.candidate_strategy or "uniform").lower()
    if strategy not in {"uniform", "random", "trust_region"}:
        raise ValueError("candidate_strategy must be 'uniform', 'random', or 'trust_region'")

    source = (args.candidate_source or "best").lower()
    if source not in {"best", "random", "topk"}:
        raise ValueError("candidate_source must be 'best', 'random', or 'topk'")

    mixing_mode = (args.mixing_mode or "random").lower()
    if mixing_mode not in {"random", "invalid"}:
        raise ValueError("mixing_mode must be 'random' or 'invalid'")

    num_times = args.num_times if args.num_times is not None else args.num_candidates
    num_cand = max(1, int(num_times or 5))
    mixes_per_time = max(1, int(args.mixes_per_time or 1))
    seq_len = int(runtime.seq_len)

    edits_delta = float(args.tr_edits_delta) if args.tr_edits_delta is not None else 0.0

    sampling_mode = args.tr_sampling
    if sampling_mode is None:
        sampling_mode = "fixed_budget"
    tr_domain = str(getattr(args, "tr_domain", None) or "edits").lower()
    if tr_domain not in {"edits", "time"}:
        raise ValueError("tr_domain must be one of: edits, time")

    tr = TrustRegionConfig(
        kind=str(args.tr_type or "AdditiveTrustRegion"),
        eps=float(args.tr_eps),
        log_every=int(args.tr_log_every) if args.tr_log_every is not None else 1,
        domain=tr_domain,
        mean_edits=None,
        mean_time=None,
        edits_delta=edits_delta,
        edits_min=0.0,
        edits_max=None,
        edits_init=None,
        time_min=0.0,
        time_max=1.0,
        tau_success=int(args.tau_success) if args.tau_success is not None else 1,
        tau_failure=int(args.tau_failure) if args.tau_failure is not None else 1,
        region_scalar=float(args.region_scalar) if args.region_scalar is not None else 2.0,
        region_expand_scalar=_optional_float_arg(args, "region_expand_scalar"),
        region_contract_scalar=_region_contract_scalar_from_inputs(args),
        region_shrink_scalar=_region_shrink_scalar_from_inputs(args),
        alpha=float(args.tr_alpha) if args.tr_alpha is not None else 1.0,
        alpha_k=float(args.tr_alpha_k) if args.tr_alpha_k is not None else 0.0,
        alpha_min=float(args.tr_alpha_min) if args.tr_alpha_min is not None else 0.25,
        alpha_max=float(args.tr_alpha_max) if args.tr_alpha_max is not None else 2.0,
    )

    if strategy == "trust_region":
        if args.fixed_time is not None or (args.fixed_times is not None and len(args.fixed_times) > 0):
            raise ValueError("trust_region strategy cannot be combined with fixed_time(s)")
        if tr.edits_delta < 0.0:
            raise ValueError("tr-edits-delta must be >= 0")
        if tr.kind not in {"AdditiveTrustRegion", "ScalarTrustRegion"}:
            raise ValueError("tr_type must be one of: AdditiveTrustRegion, ScalarTrustRegion")

        edits_min_arg = args.edits_min if args.edits_min is not None else args.tr_edits_min
        if edits_min_arg is not None:
            edits_min = float(edits_min_arg)
            if edits_min < 0.0:
                raise ValueError("edits_min must be >= 0")
            if edits_min > float(max(1, seq_len)):
                raise ValueError("edits_min cannot exceed seq_len")
            tr.edits_min = edits_min

        if sampling_mode is not None and str(sampling_mode).lower() not in {"fixed_budget"}:
            raise ValueError("truncnorm has been removed; use tr_sampling=fixed_budget (or omit tr_sampling).")

        if tr.domain == "edits":
            tr.mean_edits = _mean_edits_from_inputs(
                args,
                seq_len=seq_len,
            )
            edits_max = float(args.edits_max) if args.edits_max is not None else float(max(1, seq_len))
            if edits_max <= 0.0:
                raise ValueError("edits_max must be > 0")
            if edits_max > float(max(1, seq_len)):
                edits_max = float(max(1, seq_len))
            if edits_max < float(tr.edits_min):
                raise ValueError("edits_max must be >= edits_min")
            tr.edits_max = edits_max

            tr.mean_edits = max(float(tr.edits_min), min(float(tr.mean_edits), float(edits_max)))
            tr.edits_init = float(args.edits_init) if args.edits_init is not None else float(tr.mean_edits)
            if tr.edits_init < float(tr.edits_min) or tr.edits_init > float(edits_max):
                raise ValueError("edits_init must lie within [edits_min, edits_max]")
        else:
            if args.tr_edits is not None:
                raise ValueError("time-domain trust_region expects tr-mean, not tr-edits")
            tr.mean_time = _mean_time_from_inputs(args)
            time_min = float(args.time_min) if args.time_min is not None else 0.0
            time_max = float(args.time_max) if args.time_max is not None else 1.0
            if not (0.0 <= time_min <= 1.0):
                raise ValueError("time_min must be in [0, 1]")
            if not (0.0 <= time_max <= 1.0):
                raise ValueError("time_max must be in [0, 1]")
            if time_max < time_min:
                raise ValueError("time_max must be >= time_min")
            if tr.mean_time < time_min or tr.mean_time > time_max:
                raise ValueError("tr_mean must lie within [time_min, time_max]")
            tr.time_min = time_min
            tr.time_max = time_max
            tr.time_init = float(tr.mean_time)

        if tr.tau_success < 1:
            raise ValueError("tau_success must be >= 1")
        if tr.tau_failure < 1:
            raise ValueError("tau_failure must be >= 1")
        if _effective_region_scalar(tr, kind="expand") <= 1.0:
            raise ValueError("region_expand_scalar must be > 1.0")
        if _effective_region_scalar(tr, kind="contract") <= 1.0:
            raise ValueError("region_contract_scalar must be > 1.0")
        if not (0.0 < _effective_region_shrink_scalar(tr) < 1.0):
            raise ValueError("region_shrink_scalar must be in (0, 1)")
        if tr.alpha <= 0.0:
            raise ValueError("tr_alpha must be > 0")
        if tr.alpha_k < 0.0:
            raise ValueError("tr_alpha_k must be >= 0")
        if tr.alpha_min <= 0.0:
            raise ValueError("tr_alpha_min must be > 0")
        if tr.alpha_max < tr.alpha_min:
            raise ValueError("tr_alpha_max must be >= tr_alpha_min")
        tr.alpha = max(tr.alpha_min, min(tr.alpha, tr.alpha_max))

    fixed_times: list[float] = []
    if args.fixed_time is not None:
        fixed_times.append(float(args.fixed_time))
    if args.fixed_times is not None:
        if isinstance(args.fixed_times, (list, tuple)):
            fixed_times.extend(float(t) for t in args.fixed_times)
        else:
            fixed_times.append(float(args.fixed_times))

    fixed_times_cfg = None
    if fixed_times:
        fixed_times_cfg = sorted({float(t) for t in fixed_times})
        for t in fixed_times_cfg:
            if not (0.0 <= t <= 1.0):
                raise ValueError("fixed_times must be within [0, 1]")

    candidate = CandidateConfig(
        strategy=strategy,
        source=source,
        mixing_mode=mixing_mode,
        num_candidates=num_cand,
        mixes_per_time=mixes_per_time,
        fixed_times=fixed_times_cfg,
    )
    return candidate, tr


__all__ = ["resolve_candidate_config"]
