"""AdaLead rollout using a frozen generator as its mutation operator."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _GeneratorStats:
    mutation_draw_count: int = 0
    rollout_batch_count: int = 0
    rollout_row_count: int = 0
    zero_mask_count: int = 0
    novelty_rejection_count: int = 0


class _MaskedGeneratorMutation:
    """Draw AdaLead sites and conditionally resample only those sites."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.runtime = policy.runtime
        self.model = getattr(getattr(self.runtime, "process", None), "model", None)
        self.adapter = getattr(self.runtime, "adapter", None)
        self.stats = _GeneratorStats()

        fixed_times = policy.candidate.fixed_times
        if fixed_times is None or len(fixed_times) != 1:
            raise ValueError(
                "adalead_evodiff requires exactly one fixed candidate time for generator noise"
            )
        self.noise_time = float(fixed_times[0])
        if not math.isfinite(self.noise_time) or not (0.0 <= self.noise_time < 1.0):
            raise ValueError("adalead_evodiff generator noise time must be finite and in [0, 1)")

        sample_interpolant = getattr(self.model, "sample_interpolant", None)
        if not callable(sample_interpolant):
            raise ValueError("adalead_evodiff requires a generator sample_interpolant method")
        self.sample_interpolant: Any = sample_interpolant
        rollout = getattr(self.adapter, "rollout_from_batch", None)
        if not callable(rollout):
            raise ValueError("adalead_evodiff requires adapter.rollout_from_batch")
        self.rollout: Any = rollout
        try:
            parameters = inspect.signature(self.rollout).parameters
        except (TypeError, ValueError) as exc:
            raise ValueError("could not inspect adalead_evodiff generator rollout") from exc
        if "update_mask" not in parameters or "t_is_time" not in parameters:
            raise ValueError(
                "adalead_evodiff requires rollout_from_batch(update_mask=..., t_is_time=...)"
            )
        self.supports_denoise_steps = "denoise_steps" in parameters
        raw_steps = getattr(policy.args, "rollout_denoise_steps", None)
        self.denoise_steps = None if raw_steps is None else int(raw_steps)
        if self.denoise_steps is not None:
            if self.denoise_steps <= 0:
                raise ValueError("adalead_evodiff rollout_denoise_steps must be > 0")
            if not self.supports_denoise_steps:
                raise ValueError("adalead_evodiff adapter does not support rollout_denoise_steps")

        length = int(self.runtime.seq_len)
        if length <= 0:
            raise ValueError("adalead_evodiff requires a non-empty sequence")
        self.mutation_probability = min(
            1.0,
            float(policy.baseline_adalead_mu) / float(length),
        )
        if self.mutation_probability <= 0.0:
            raise ValueError("adalead_evodiff requires baseline_adalead_mu > 0")
        set_eval = getattr(self.model, "eval", None)
        if callable(set_eval):
            set_eval()

    def _interpolant(self, batch: torch.Tensor, time: float) -> torch.Tensor:
        values = torch.full(
            (int(batch.shape[0]),),
            float(time),
            device=batch.device,
            dtype=torch.float,
        )
        sampled = self.sample_interpolant(
            base_sample=batch,
            t_tensor=values,
            elite_probs=None,
            transition_mask=self.runtime.transition_mask,
            mixing_mode=self.policy.candidate.mixing_mode,
        )
        result = torch.as_tensor(sampled, device=batch.device, dtype=torch.long)
        if result.shape != batch.shape:
            raise ValueError("adalead_evodiff generator interpolant returned the wrong shape")
        return result

    def mutate(self, nodes: list[torch.Tensor]) -> list[torch.Tensor]:
        if not nodes:
            return []
        batch = torch.cat(nodes, dim=0).to(device=self.runtime.device, dtype=torch.long)
        masks = torch.rand(batch.shape, device=batch.device) < self.mutation_probability
        self.stats.mutation_draw_count += int(batch.shape[0])
        active_indices = torch.nonzero(masks.any(dim=1), as_tuple=False).view(-1)
        self.stats.zero_mask_count += int(batch.shape[0] - active_indices.numel())
        children = batch.clone()
        if active_indices.numel() == 0:
            return [children[idx : idx + 1] for idx in range(int(children.shape[0]))]

        active_batch = batch.index_select(0, active_indices)
        active_mask = masks.index_select(0, active_indices)
        with torch.no_grad():
            # EvoDiff uses its native token space for x_t. A t=1 draw gives
            # the clean native encoding; selected sites are then taken from the
            # task's explicit one-edit-scale noisy draw. The update mask permits
            # changes only at those AdaLead-selected sites.
            clean_native = self._interpolant(active_batch, 1.0)
            noisy_native = self._interpolant(active_batch, self.noise_time)
            # As in official AdaLead's full-alphabet replacement, a selected
            # site may draw its current token. Novelty rejection remains the
            # caller's responsibility, and all unselected residues stay frozen.
            xt = torch.where(active_mask, noisy_native, clean_native)
            kwargs: dict[str, Any] = {
                "update_mask": active_mask,
                "t_is_time": True,
            }
            if self.denoise_steps is not None:
                kwargs["denoise_steps"] = self.denoise_steps
            trajectory = self.rollout(xt, self.noise_time, **kwargs)
        if not trajectory:
            raise RuntimeError("adalead_evodiff generator returned an empty trajectory")
        leaves = torch.as_tensor(trajectory[-1], device=batch.device, dtype=torch.long)
        if leaves.shape != active_batch.shape:
            raise ValueError("adalead_evodiff generator leaf batch has the wrong shape")
        if bool(((leaves != active_batch) & (~active_mask)).any().item()):
            raise RuntimeError("adalead_evodiff generator changed a frozen residue")
        children.index_copy_(0, active_indices, leaves)
        self.stats.rollout_batch_count += 1
        self.stats.rollout_row_count += int(active_indices.numel())
        return [children[idx : idx + 1] for idx in range(int(children.shape[0]))]


def propose_adalead_evodiff(policy: Any, target: int) -> Any:
    """Run official AdaLead search logic with masked generator mutations."""

    paid_batch_size = int(policy.baseline_adalead_paid_batch_size)
    if target != paid_batch_size:
        raise ValueError(
            "adalead_evodiff proposal target must equal its paid batch size "
            f"({paid_batch_size}), got {target}"
        )
    query_budget = int(policy.baseline_adalead_surrogate_queries)
    predictor = policy.active_learning_selector.fit_surrogate_predictor()
    mutation = _MaskedGeneratorMutation(policy)

    measured = list(policy._adalead_measured_rows)
    if not measured:
        raise RuntimeError("adalead_evodiff measured archive is empty")
    best = max(score for _, score in measured)
    sign = 1.0 if best > 0.0 else (-1.0 if best < 0.0 else 0.0)
    cutoff = best * (1.0 - sign * float(policy.baseline_fitness_threshold))
    retained = [row for row in measured if row[1] >= cutoff]
    if not retained:
        raise RuntimeError("adalead_evodiff threshold retained no parents")
    parents = [retained[idx % len(retained)] for idx in range(paid_batch_size)]

    measured_keys = set(policy._baseline_archive)
    generated: dict[bytes, tuple[float, torch.Tensor, float, torch.Tensor]] = {}
    queries = 0
    root_queries = 0
    mutant_queries = 0
    maximum_attempts = max(10_000, query_budget * 1_000)

    while queries < query_budget:
        for start in range(0, len(parents), int(policy.baseline_adalead_eval_batch_size)):
            remaining = query_budget - queries
            if remaining <= 0:
                break
            root_rows = parents[start : start + int(policy.baseline_adalead_eval_batch_size)]
            root_rows = root_rows[:remaining]
            root_batch = torch.cat([row[0] for row in root_rows], dim=0).to(
                device=policy.runtime.device,
                dtype=torch.long,
            )
            root_predictions = policy._adalead_means(predictor, root_batch, context="root")
            queries += len(root_rows)
            root_queries += len(root_rows)
            nodes = [
                (
                    idx,
                    root_batch[idx : idx + 1],
                    root_batch[idx : idx + 1],
                    float(root_rows[idx][1]),
                )
                for idx in range(len(root_rows))
            ]

            while nodes and queries + int(policy.baseline_adalead_eval_batch_size) < query_budget:
                # Keep FLEXS's nodes[-1], nodes[0..n-2] order, while drawing
                # all currently needed generator children in one GPU batch.
                ordered_nodes = [nodes[-1], *nodes[:-1]]
                accepted: list[torch.Tensor | None] = [None] * len(ordered_nodes)
                pending = list(range(len(ordered_nodes)))
                while pending:
                    candidates = mutation.mutate([ordered_nodes[idx][1] for idx in pending])
                    next_pending: list[int] = []
                    for slot, child in zip(pending, candidates, strict=True):
                        if mutation.stats.mutation_draw_count > maximum_attempts:
                            raise RuntimeError(
                                "adalead_evodiff could not draw a novel generator child "
                                "within its safety limit"
                            )
                        key = policy._sequence_key(child)
                        if key in measured_keys or key in generated:
                            mutation.stats.novelty_rejection_count += 1
                            next_pending.append(slot)
                        else:
                            accepted[slot] = child
                    pending = next_pending

                child_indices = [row[0] for row in ordered_nodes]
                child_rows = [child for child in accepted if child is not None]
                child_roots = [row[2] for row in ordered_nodes]
                child_root_scores = [row[3] for row in ordered_nodes]
                child_batch = torch.cat(child_rows, dim=0)
                child_predictions = policy._adalead_means(
                    predictor,
                    child_batch,
                    context="mutant",
                )
                queries += len(child_rows)
                mutant_queries += len(child_rows)
                nodes = []
                for idx, child, root, root_score, prediction in zip(
                    child_indices,
                    child_rows,
                    child_roots,
                    child_root_scores,
                    child_predictions.tolist(),
                    strict=True,
                ):
                    key = policy._sequence_key(child)
                    generated[key] = (float(prediction), root, root_score, child)
                    if float(prediction) >= float(root_predictions[idx]):
                        nodes.append((idx, child, root, root_score))

    if queries != query_budget:
        raise RuntimeError(
            f"adalead_evodiff used {queries} surrogate queries; expected exactly {query_budget}"
        )
    if len(generated) < paid_batch_size:
        raise RuntimeError(
            "adalead_evodiff produced fewer unique mutants than the paid batch size: "
            f"{len(generated)} < {paid_batch_size}"
        )

    ranked = sorted(generated.values(), key=lambda row: row[0], reverse=True)
    rows = [
        (root, root_score, child)
        for _prediction, root, root_score, child in ranked[:paid_batch_size]
    ]
    result = policy._direct_proposal_batch(rows)
    stats = mutation.stats
    result.generation_diagnostics = {
        "operator": "adalead_evodiff",
        "surrogate_query_count": queries,
        "surrogate_root_query_count": root_queries,
        "surrogate_mutant_query_count": mutant_queries,
        "paid_target_count": paid_batch_size,
        "unique_generated_count": len(generated),
        "mutation_operator": "evodiff_masked_conditional",
        "mutation_mask_probability": mutation.mutation_probability,
        "generator_noise_time": mutation.noise_time,
        "generator_mask_enforced": True,
        "generator_mutation_draw_count": stats.mutation_draw_count,
        "generator_rollout_batch_count": stats.rollout_batch_count,
        "generator_rollout_row_count": stats.rollout_row_count,
        "generator_novelty_rejection_count": stats.novelty_rejection_count,
        "generator_zero_mask_count": stats.zero_mask_count,
    }
    return result


__all__ = ["propose_adalead_evodiff"]
