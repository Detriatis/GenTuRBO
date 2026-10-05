"""Proposal-generation policy for interpolant search."""

from __future__ import annotations

import argparse
import hashlib
import math
from typing import Any, Optional

import torch

from processes.dfm_mask import (
    mask_interpolant_sample,
    uniform_interpolant_sample,
    uniform_interpolant_sample_invalid,
)
from search.interpolant.adalead_evodiff import propose_adalead_evodiff
from search.interpolant.time_policy import CandidateTimePolicy
from search.interpolant.types import (
    CandidateConfig,
    Proposal,
    ProposalBatch,
    TrustRegionConfig,
    WeightedArchiveDraw,
    WeightedArchiveEntry,
    WeightedArchiveSnapshot,
)
from search.interpolant.utils import (
    categorical_interpolant_sample,
    edits_to_time,
    select_topk_unique,
    time_to_edits,
)


class InterpolantProposalPolicy:
    """Generate xt proposals by composing anchor/time/token-mixing choices."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        runtime: Any,
        candidate: CandidateConfig,
        trust_region: TrustRegionConfig,
        active_learning_selector: Any = None,
    ) -> None:
        self.args = args
        self.runtime = runtime
        self.candidate = candidate
        self.time_policy = CandidateTimePolicy(
            runtime=runtime,
            candidate=candidate,
            trust_region=trust_region,
        )
        self.active_learning_selector = active_learning_selector
        self.proposal_operator = str(
            getattr(args, "proposal_operator", None) or "generator"
        ).strip().lower()
        if self.proposal_operator not in {
            "generator",
            "hamming",
            "adalead",
            "adalead_official",
            "adalead_evodiff",
            "bo_evo",
        }:
            raise ValueError(
                "proposal_operator must be one of: generator, hamming, adalead, "
                "adalead_official, adalead_evodiff, bo_evo"
            )
        exact_corruptions = getattr(args, "generator_exact_corruptions", None)
        self.generator_exact_corruptions = (
            None if exact_corruptions is None else int(exact_corruptions)
        )
        if self.generator_exact_corruptions is not None:
            if self.proposal_operator != "generator":
                raise ValueError(
                    "generator_exact_corruptions is only valid with proposal_operator=generator"
                )
            if self.generator_exact_corruptions <= 0:
                raise ValueError("generator_exact_corruptions must be > 0")
        self.baseline_population_size = max(
            1, int(getattr(args, "baseline_population_size", None) or 32)
        )
        self.baseline_generations = max(
            1, int(getattr(args, "baseline_generations", None) or 4)
        )
        fitness_threshold = getattr(args, "baseline_fitness_threshold", None)
        self.baseline_fitness_threshold = float(
            0.05 if fitness_threshold is None else fitness_threshold
        )
        self.baseline_mutation_edits = max(
            1, int(getattr(args, "baseline_mutation_edits", None) or 1)
        )
        adalead_queries = getattr(args, "baseline_adalead_surrogate_queries", None)
        self.baseline_adalead_surrogate_queries = int(
            128 if adalead_queries is None else adalead_queries
        )
        adalead_paid_batch_size = getattr(args, "baseline_adalead_paid_batch_size", None)
        adalead_mu = getattr(args, "baseline_adalead_mu", None)
        self.baseline_adalead_mu = float(1.0 if adalead_mu is None else adalead_mu)
        adalead_eval_batch_size = getattr(args, "baseline_adalead_eval_batch_size", None)
        self.baseline_adalead_eval_batch_size = int(
            20 if adalead_eval_batch_size is None else adalead_eval_batch_size
        )
        adalead_rho = getattr(args, "baseline_adalead_rho", None)
        self.baseline_adalead_rho = int(0 if adalead_rho is None else adalead_rho)
        adalead_recombination_rate = getattr(args, "baseline_adalead_recombination_rate", None)
        self.baseline_adalead_recombination_rate = float(
            0.0 if adalead_recombination_rate is None else adalead_recombination_rate
        )
        adalead_alphabet = getattr(args, "baseline_adalead_alphabet", None)
        self.baseline_adalead_alphabet = (
            None if adalead_alphabet is None else str(adalead_alphabet)
        )
        adalead_rng_backend = getattr(args, "baseline_adalead_rng_backend", None)
        self.baseline_adalead_rng_backend = (
            None if adalead_rng_backend is None else str(adalead_rng_backend)
        )
        parent_temperature = getattr(args, "baseline_parent_temperature", None)
        self.baseline_parent_temperature = float(
            10.0 if parent_temperature is None else parent_temperature
        )
        ucb_beta = getattr(args, "baseline_ucb_beta", None)
        self.baseline_ucb_beta = float(0.2 if ucb_beta is None else ucb_beta)
        children_per_selection = getattr(args, "baseline_children_per_selection", None)
        self.baseline_children_per_selection = int(
            8 if children_per_selection is None else children_per_selection
        )
        oracle_batch_size = getattr(args, "al_oracle_batch_size", None)
        self.baseline_oracle_batch_size = int(
            1 if oracle_batch_size is None else oracle_batch_size
        )
        self.baseline_adalead_paid_batch_size = int(
            self.baseline_oracle_batch_size
            if adalead_paid_batch_size is None
            else adalead_paid_batch_size
        )
        if not (0.0 <= self.baseline_fitness_threshold <= 1.0):
            raise ValueError("baseline_fitness_threshold must be in [0, 1]")
        if not math.isfinite(self.baseline_parent_temperature) or self.baseline_parent_temperature < 0.0:
            raise ValueError("baseline_parent_temperature must be >= 0")
        if not math.isfinite(self.baseline_ucb_beta) or self.baseline_ucb_beta < 0.0:
            raise ValueError("baseline_ucb_beta must be >= 0")
        if self.baseline_children_per_selection <= 0:
            raise ValueError("baseline_children_per_selection must be > 0")
        if self.baseline_oracle_batch_size <= 0:
            raise ValueError("al_oracle_batch_size must be > 0")
        if self.proposal_operator in {
            "adalead",
            "adalead_official",
            "adalead_evodiff",
        } and not callable(
            getattr(active_learning_selector, "fit_surrogate_predictor", None)
        ):
            raise ValueError(f"{self.proposal_operator} requires a fitted active-learning surrogate")
        if self.proposal_operator in {"adalead_official", "adalead_evodiff"}:
            if self.baseline_adalead_paid_batch_size != self.baseline_oracle_batch_size:
                raise ValueError(
                    "official AdaLead paid batch size must equal al_oracle_batch_size"
                )
            if self.baseline_adalead_surrogate_queries < self.baseline_oracle_batch_size:
                raise ValueError(
                    "official AdaLead surrogate-query budget must cover the paid batch size"
                )
            if not math.isfinite(self.baseline_adalead_mu) or self.baseline_adalead_mu < 0.0:
                raise ValueError("official AdaLead mu must be finite and >= 0")
            if self.baseline_adalead_eval_batch_size <= 0:
                raise ValueError("official AdaLead eval batch size must be > 0")
            if self.baseline_adalead_rho != 0 or self.baseline_adalead_recombination_rate != 0.0:
                raise ValueError("the locked protein AdaLead protocol disables crossover")
            if (
                self.baseline_adalead_alphabet is None
                or len(set(self.baseline_adalead_alphabet)) != len(self.baseline_adalead_alphabet)
                or len(self.baseline_adalead_alphabet) != int(self.runtime.vocab_size)
            ):
                raise ValueError(
                    "official AdaLead alphabet must explicitly match the runtime vocabulary"
                )
            runtime_alphabet = getattr(getattr(self.runtime, "reward_fn", None), "decode_alphabet", None)
            if runtime_alphabet is not None and str(runtime_alphabet) != self.baseline_adalead_alphabet:
                raise ValueError(
                    "official AdaLead alphabet must match the protein reward decoder ordering"
                )
            if self.baseline_adalead_rng_backend != "torch_global_seeded":
                raise ValueError(
                    "official AdaLead requires the explicit torch_global_seeded RNG adapter"
                )
        if self.proposal_operator == "bo_evo" and not callable(
            getattr(active_learning_selector, "fit_surrogate_moments_predictor", None)
        ):
            raise ValueError("bo_evo requires a fitted active-learning surrogate")
        default_k = int(getattr(args, "candidate_k", None) or 1)
        self.topk_use_archive = bool(args.topk_use_archive) if args.topk_use_archive is not None else True
        self.topk_archive_size = max(1, int(args.topk_archive_size) if args.topk_archive_size is not None else max(default_k, 32))
        self.topk_weight_n = float(args.topk_weight_n) if args.topk_weight_n is not None else 1.0
        self.topk_archive_d_min = int(args.topk_archive_d_min) if args.topk_archive_d_min is not None else 0
        if self.topk_weight_n < 0.0:
            raise ValueError("topk_weight_n must be >= 0")
        if self.topk_archive_d_min < 0:
            raise ValueError("topk_archive_d_min must be >= 0")

        self.topk_archive: list[WeightedArchiveEntry] = []
        self.last_archive_snapshot: Optional[WeightedArchiveSnapshot] = None
        self.last_archive_draws: list[WeightedArchiveDraw] = []
        self._baseline_archive: dict[bytes, tuple[torch.Tensor, float]] = {}
        self._adalead_measured_rows: list[tuple[torch.Tensor, float]] = []
        self._adalead_measurement_keys: set[bytes] = set()
        self._adalead_archive_initialized = False
        self._adalead_paid_keys: set[bytes] = set()
        self._ehrlich_adalead = self.proposal_operator == "adalead_official" and bool(
            getattr(args, "al_ehrlich_black_box_feasibility", False)
        )
        if self._ehrlich_adalead:
            if getattr(args, "al_acquisition", "random") != "random":
                raise ValueError("Ehrlich AdaLead requires pass-through acquisition")
            if getattr(args, "al_sequential_selection", False) or getattr(args, "al_local_penalization", False):
                raise ValueError("Ehrlich AdaLead uses its own top-mean batch selection")
            if self.baseline_adalead_surrogate_queries <= self.baseline_adalead_eval_batch_size + self.baseline_adalead_paid_batch_size:
                raise ValueError("AdaLead model-query budget must leave room for mutant predictions")
        self._bo_evo_parents: list[tuple[torch.Tensor, float]] | None = None

    def propose(
        self,
        *,
        iteration: int,
        batch: torch.Tensor,
        rewards: torch.Tensor,
        elite_probs: Optional[torch.Tensor],
        max_proposals: Optional[int] = None,
    ) -> ProposalBatch:
        top_idx = int(torch.argmax(rewards).item())
        top_reward = float(rewards[top_idx].item())

        self._update_topk_archive_from_batch(
            batch=batch,
            rewards=rewards,
            iteration=iteration,
        )
        if self.proposal_operator != "generator":
            self._remember_baseline_archive(batch, rewards)
            if self.proposal_operator == "bo_evo" and self._bo_evo_parents is None:
                self._bo_evo_parents = self._scored_rows(batch, rewards)
                self._remember_baseline_archive(
                    torch.cat([row[0] for row in self._bo_evo_parents], dim=0),
                    torch.as_tensor([row[1] for row in self._bo_evo_parents]),
                )
            return self._propose_sequence_baseline(
                iteration=iteration,
                batch=batch,
                rewards=rewards,
                max_proposals=max_proposals,
            )
        base_indices, base_samples, base_rewards = self._select_bases(
            iteration=iteration,
            rewards=rewards,
            batch=batch,
        )

        proposal_limit = int(max_proposals) if max_proposals is not None else None
        if proposal_limit is not None and proposal_limit <= 0:
            return ProposalBatch(
                base_indices=base_indices,
                base_samples=base_samples,
                base_rewards=base_rewards,
                top_reward=top_reward,
                proposals=[],
            )

        proposals: list[Proposal] = []
        tr_target_edits: Optional[float] = None
        if self.candidate.strategy == "trust_region":
            if self.time_policy.tr_strategy is None:
                raise RuntimeError("trust_region strategy is not initialised")
            tr_domain = str(getattr(self.time_policy.tr_strategy.config, "domain", "edits") or "edits").lower()
            if tr_domain == "edits":
                tr_mean = getattr(self.time_policy.tr_strategy.config, "mean_edits", None)
                if tr_mean is None:
                    raise RuntimeError("trust_region mean_edits is missing")
                tr_target_edits = float(tr_mean)
        for base_position, (base_idx, base_sample) in enumerate(
            zip(base_indices, base_samples, strict=True)
        ):
            base_sample = base_sample.view(1, -1)
            parent_metadata = self._parent_trace_metadata(
                base_position=base_position,
                parent=base_sample,
                score=float(base_rewards[base_position].item()),
            )
            candidate_times = self.time_policy.candidate_times_for_iteration(
                iteration,
                base_sample=base_sample,
            )
            editable_seq_len = self.time_policy.editable_seq_len_for_sample(base_sample)
            for t_value in candidate_times:
                t_value = min(max(float(t_value), 0.0), 1.0)
                t_tensor = torch.full((1,), t_value, device=self.runtime.device)
                edits_value = (
                    float(min(float(tr_target_edits), float(editable_seq_len)))
                    if tr_target_edits is not None
                    else time_to_edits(t_value, seq_len=editable_seq_len)
                )
                mix_repeats = self.candidate.mixes_per_time
                if self.candidate.source == "topk" and self.topk_use_archive:
                    # In archive mode, parent count already equals mixes_per_time.
                    # Use one mix per selected parent to keep proposal budget stable.
                    mix_repeats = 1
                for _ in range(mix_repeats):
                    xt, corrupt_mask = self._sample_generator_interpolant(
                        base_sample,
                        t_tensor,
                        elite_probs,
                    )
                    if self.generator_exact_corruptions is not None:
                        edits_value = float(self.generator_exact_corruptions)
                    proposals.append(
                        Proposal(
                            base_index=int(base_idx),
                            t_value=t_value,
                            edits_value=float(edits_value),
                            xt=xt,
                            corrupt_mask=corrupt_mask,
                            **parent_metadata,
                        )
                    )
                    if proposal_limit is not None and len(proposals) >= proposal_limit:
                        return ProposalBatch(
                            base_indices=base_indices,
                            base_samples=base_samples,
                            base_rewards=base_rewards,
                            top_reward=top_reward,
                            proposals=proposals,
                        )

        return ProposalBatch(
            base_indices=base_indices,
            base_samples=base_samples,
            base_rewards=base_rewards,
            top_reward=top_reward,
            proposals=proposals,
        )

    def update_from_evaluations(
        self,
        *,
        proposals: list[Proposal],
        proposal_rewards: torch.Tensor,
        base_rewards: torch.Tensor,
    ) -> None:
        if self._ehrlich_adalead and proposals:
            self._remember_baseline_archive(
                torch.cat([p.xt.view(1, -1) for p in proposals]), proposal_rewards
            )
        if self.proposal_operator == "bo_evo":
            if len(proposals) != int(proposal_rewards.numel()):
                raise ValueError("bo_evo proposals and rewards must have the same length")
            rows = self._scored_rows(
                torch.cat([proposal.xt.view(1, -1) for proposal in proposals], dim=0),
                proposal_rewards,
            )
            if not rows:
                raise ValueError("bo_evo received no finite paid evaluations")
            self._bo_evo_parents = rows
            self._remember_baseline_archive(
                torch.cat([row[0] for row in rows], dim=0),
                torch.as_tensor([row[1] for row in rows]),
            )
        _ = base_rewards

    @staticmethod
    def _sequence_key(sequence: torch.Tensor) -> bytes:
        return sequence.detach().to("cpu", dtype=torch.long).contiguous().numpy().tobytes()

    def _remember_baseline_archive(self, batch: torch.Tensor, rewards: torch.Tensor) -> None:
        preserve_initial_multiplicity = (
            self.proposal_operator in {"adalead_official", "adalead_evodiff"}
            and not self._adalead_archive_initialized
        )
        for sequence, reward in zip(batch, rewards.view(-1), strict=True):
            score = float(reward.item())
            row = sequence.detach().to("cpu", dtype=torch.long).contiguous().view(1, -1)
            key = self._sequence_key(row)
            if self._ehrlich_adalead:
                # Every paid identity stays unavailable, including infeasible
                # observations. The shared Ehrlich GP uses a zero target for
                # paid negative-infinity outcomes.
                self._adalead_paid_keys.add(key)
                if score == -math.inf:
                    score = 0.0
            if not math.isfinite(score):
                continue
            if self.proposal_operator in {"adalead_official", "adalead_evodiff"} and (
                preserve_initial_multiplicity or key not in self._adalead_measurement_keys
            ):
                # The initial protein archive can contain duplicate rows. FLEXS
                # thresholds the measured DataFrame (with that multiplicity)
                # even though novelty checks use a set. On later rounds the
                # engine repeats the incumbent in current_batch without paying
                # for it again, so only genuinely unseen measurements append.
                self._adalead_measured_rows.append((row, score))
                self._adalead_measurement_keys.add(key)
            current = self._baseline_archive.get(key)
            if current is None or score > current[1]:
                self._baseline_archive[key] = (row, score)
        if self.proposal_operator in {"adalead_official", "adalead_evodiff"}:
            self._adalead_archive_initialized = True

    @staticmethod
    def _scored_rows(
        batch: torch.Tensor,
        rewards: torch.Tensor,
    ) -> list[tuple[torch.Tensor, float]]:
        rows: list[tuple[torch.Tensor, float]] = []
        for sequence, reward in zip(batch, rewards.view(-1), strict=True):
            score = float(reward.item())
            if math.isinf(score) and score < 0.0:
                score = 0.0
            if math.isfinite(score):
                rows.append(
                    (
                        sequence.detach().to("cpu", dtype=torch.long).contiguous().view(1, -1),
                        score,
                    )
                )
        return rows

    def _baseline_rows(self) -> list[tuple[torch.Tensor, float]]:
        return sorted(self._baseline_archive.values(), key=lambda row: row[1], reverse=True)

    def _proposal_target(self, max_proposals: Optional[int]) -> int:
        target = int(self.candidate.num_candidates) * int(self.candidate.mixes_per_time)
        return target if max_proposals is None else min(target, max(0, int(max_proposals)))

    def _mutate_exact(self, parent: torch.Tensor, edits: int) -> torch.Tensor:
        child = parent.clone()
        vocab_size = int(self.runtime.vocab_size)
        length = int(child.shape[1])
        target = max(0, int(edits))
        if target > length:
            raise ValueError(f"cannot make {target} distinct edits to a length-{length} sequence")
        if target == 0:
            return child
        if vocab_size < 2:
            raise ValueError("exact mutation requires an alphabet containing at least two tokens")

        # Baselines know only the public sequence canvas and alphabet. In
        # particular, they must not consult the generator's mutable-position
        # hooks or the oracle's hidden transition graph.
        positions = torch.randperm(length, device=child.device)[:target]
        old = child[0, positions]
        offsets = torch.randint(1, vocab_size, (target,), device=child.device)
        child[0, positions] = (old + offsets) % vocab_size
        return child

    def _mutate_bo_evo(self, parent: torch.Tensor) -> torch.Tensor:
        """Apply independent 1/L mutations, conditioned on changing at least one site."""
        child = parent.clone()
        length = int(child.shape[1])
        vocab_size = int(self.runtime.vocab_size)
        if length <= 0 or vocab_size < 2:
            raise ValueError("bo_evo mutation requires a non-empty sequence and at least two tokens")
        while True:
            positions = torch.rand(length, device=child.device) < (1.0 / float(length))
            if bool(positions.any().item()):
                break
        old = child[0, positions]
        offsets = torch.randint(1, vocab_size, old.shape, device=child.device)
        child[0, positions] = (old + offsets) % vocab_size
        return child

    def _mutate_adalead_official(self, parent: torch.Tensor) -> torch.Tensor:
        """Match FLEXS ``generate_random_mutant`` for tensor sequences.

        FLEXS interprets ``mu`` as an expected sequence-level edit count and
        mutates each site independently with probability ``mu / L``.  A drawn
        replacement comes from the complete alphabet, so it may equal the
        existing token; novelty is enforced by the caller exactly as in the
        released rollout loop.
        """

        child = parent.clone()
        length = int(child.shape[1])
        vocab_size = int(self.runtime.vocab_size)
        if length <= 0 or vocab_size <= 0:
            raise ValueError("official AdaLead mutation requires a non-empty sequence and alphabet")
        mutation_probability = min(1.0, self.baseline_adalead_mu / float(length))
        positions = torch.rand(length, device=child.device) < mutation_probability
        count = int(positions.sum().item())
        if count > 0:
            child[0, positions] = torch.randint(
                0,
                vocab_size,
                (count,),
                device=child.device,
            )
        return child

    @staticmethod
    def _adalead_means(
        predictor: Any,
        batch: torch.Tensor,
        *,
        context: str,
    ) -> torch.Tensor:
        means, _ = predictor(batch)
        values = torch.as_tensor(means, dtype=torch.double).view(-1)
        if values.numel() != int(batch.shape[0]):
            raise ValueError(
                f"official AdaLead surrogate returned {values.numel()} {context} predictions "
                f"for {int(batch.shape[0])} sequences"
            )
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"official AdaLead surrogate returned a non-finite {context} prediction")
        return values

    def _propose_adalead_official(self, target: int) -> ProposalBatch:
        """Run the released root-anchored AdaLead rollout with a strict query cap.

        The upstream implementation counts both repeated root predictions and
        mutant predictions against ``model_queries_per_batch``.  Its loop can
        overshoot when fewer than one root chunk remains; this adapter truncates
        only that final root chunk so the locked protein budget is exactly 128.
        """

        paid_batch_size = self.baseline_adalead_paid_batch_size
        if target != paid_batch_size:
            raise ValueError(
                "official AdaLead proposal target must equal its paid batch size "
                f"({paid_batch_size}), got {target}"
            )
        query_budget = self.baseline_adalead_surrogate_queries
        predictor = self.active_learning_selector.fit_surrogate_predictor()

        # FLEXS preserves measured-data order, filters by the signed relative
        # threshold, then uses numpy.resize to repeat that list to the paid
        # batch size.  Dict insertion order gives the same archive ordering.
        measured = list(self._adalead_measured_rows)
        if not measured:
            raise RuntimeError("official AdaLead measured archive is empty")
        best = max(score for _, score in measured)
        sign = 1.0 if best > 0.0 else (-1.0 if best < 0.0 else 0.0)
        cutoff = best * (1.0 - sign * self.baseline_fitness_threshold)
        retained = [row for row in measured if row[1] >= cutoff]
        if not retained:
            raise RuntimeError("official AdaLead threshold retained no parents")
        parents = [retained[idx % len(retained)] for idx in range(paid_batch_size)]

        measured_keys = self._adalead_paid_keys if self._ehrlich_adalead else set(self._baseline_archive)
        # key -> (predicted fitness, rollout root, measured root fitness)
        generated: dict[bytes, tuple[float, torch.Tensor, float, torch.Tensor]] = {}
        queries = 0
        root_queries = 0
        mutant_queries = 0
        mutation_attempts = 0
        maximum_mutation_attempts = max(10_000, query_budget * 1_000)

        while queries < query_budget:
            for start in range(0, len(parents), self.baseline_adalead_eval_batch_size):
                remaining = query_budget - queries
                if remaining <= 0:
                    break
                root_rows = parents[start : start + self.baseline_adalead_eval_batch_size]
                # Strict-budget adaptation: the released code evaluates the
                # whole chunk here and can overshoot on its final outer pass.
                root_rows = root_rows[:remaining]
                root_batch = torch.cat([row[0] for row in root_rows], dim=0).to(
                    device=self.runtime.device,
                    dtype=torch.long,
                )
                root_predictions = self._adalead_means(
                    predictor,
                    root_batch,
                    context="root",
                )
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

                # This is the upstream stopping rule: reserve one configured
                # eval chunk, compare every descendant to its fixed root score,
                # and restart from the retained roots when all walks stop.
                while (
                    nodes
                    and queries + self.baseline_adalead_eval_batch_size < query_budget
                ):
                    child_indices: list[int] = []
                    child_rows: list[torch.Tensor] = []
                    child_roots: list[torch.Tensor] = []
                    child_root_scores: list[float] = []
                    while len(child_rows) < len(nodes):
                        # Preserve the released FLEXS indexing exactly: the
                        # first draw uses nodes[-1], followed by nodes[0..n-2].
                        node_index, node, root, root_score = nodes[len(child_rows) - 1]
                        child = self._mutate_adalead_official(node)
                        mutation_attempts += 1
                        if mutation_attempts > maximum_mutation_attempts:
                            raise RuntimeError(
                                "official AdaLead could not draw a novel mutant within its safety limit"
                            )
                        key = self._sequence_key(child)
                        # Match FLEXS: candidates already measured or added by a
                        # previous surrogate batch are rejected.  Same-batch
                        # duplicates are allowed and collapse on dict update.
                        if key in measured_keys or key in generated:
                            continue
                        child_indices.append(node_index)
                        child_rows.append(child)
                        child_roots.append(root)
                        child_root_scores.append(root_score)

                    child_batch = torch.cat(child_rows, dim=0)
                    child_predictions = self._adalead_means(
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
                        key = self._sequence_key(child)
                        generated[key] = (float(prediction), root, root_score, child)
                        if float(prediction) >= float(root_predictions[idx]):
                            nodes.append((idx, child, root, root_score))

        if queries != query_budget:
            raise RuntimeError(
                f"official AdaLead used {queries} surrogate queries; expected exactly {query_budget}"
            )
        if len(generated) < paid_batch_size:
            raise RuntimeError(
                "official AdaLead produced fewer unique mutants than the paid batch size: "
                f"{len(generated)} < {paid_batch_size}"
            )

        ranked = sorted(generated.values(), key=lambda row: row[0], reverse=True)
        rows = [
            (root, root_score, child)
            for _prediction, root, root_score, child in ranked[:paid_batch_size]
        ]
        result = self._direct_proposal_batch(rows)
        result.preselected = self._ehrlich_adalead
        result.generation_diagnostics = {
            "operator": "adalead_official",
            "surrogate_query_count": queries,
            "surrogate_root_query_count": root_queries,
            "surrogate_mutant_query_count": mutant_queries,
            "paid_target_count": paid_batch_size,
            "unique_generated_count": len(generated),
        }
        if self._ehrlich_adalead:
            result.generation_diagnostics.update(
                selection="top_surrogate_mean",
                mutation_attempts=mutation_attempts,
                selected_predicted_means=[row[0] for row in ranked[:paid_batch_size]],
            )
        return result

    def _sample_bo_evo_parent(self) -> tuple[torch.Tensor, float]:
        parents = self._bo_evo_parents
        if not parents:
            raise RuntimeError("bo_evo parent batch is empty")
        scores = torch.as_tensor([row[1] for row in parents], dtype=torch.double)
        if self.baseline_parent_temperature == 0.0:
            weights = torch.full_like(scores, 1.0 / float(len(parents)))
        else:
            logits = self.baseline_parent_temperature * (scores - torch.max(scores))
            weights = torch.softmax(logits, dim=0)
        index = int(torch.multinomial(weights, 1).item())
        return parents[index]

    def _propose_bo_evo(self, target: int) -> ProposalBatch:
        batch_size = self.baseline_oracle_batch_size
        expected = batch_size * self.baseline_children_per_selection
        if target != expected:
            raise ValueError(
                "bo_evo proposal pool must equal oracle batch size times "
                f"baseline_children_per_selection ({expected}), got {target}"
            )
        predictor = self.active_learning_selector.fit_surrogate_moments_predictor()
        screen_counts = [self.baseline_children_per_selection] * batch_size

        virtual_parent, virtual_parent_score = self._sample_bo_evo_parent()
        virtual_parent = virtual_parent.to(device=self.runtime.device, dtype=torch.long)
        anchor_uncertainty: float | None = None
        archive_keys = set(self._baseline_archive)
        pending_keys: set[bytes] = set()
        rows: list[tuple[torch.Tensor, float, torch.Tensor]] = []
        for slot, screen_count in enumerate(screen_counts):
            children: list[torch.Tensor] = []
            child_keys: set[bytes] = set()
            attempts = 0
            while len(children) < screen_count:
                attempts += 1
                if attempts > 1000 * screen_count:
                    raise RuntimeError("bo_evo could not draw enough unseen mutation children")
                child = self._mutate_bo_evo(virtual_parent)
                key = self._sequence_key(child)
                if key in archive_keys or key in pending_keys or key in child_keys:
                    continue
                child_keys.add(key)
                children.append(child)
            child_batch = torch.cat(children, dim=0)
            means, stds = predictor(child_batch)
            means_t = torch.as_tensor(means, dtype=torch.double).view(-1)
            stds_t = torch.as_tensor(stds, dtype=torch.double).view(-1).clamp_min(0.0)
            if means_t.numel() != screen_count or stds_t.numel() != screen_count:
                raise ValueError("bo_evo surrogate returned the wrong number of moments")
            chosen_idx = int(torch.argmax(means_t + self.baseline_ucb_beta * stds_t).item())
            chosen = children[chosen_idx]
            pending_keys.add(self._sequence_key(chosen))
            rows.append((virtual_parent, virtual_parent_score, chosen))

            if slot + 1 == batch_size:
                continue
            chosen_uncertainty = max(0.0, float(stds_t[chosen_idx]))
            if anchor_uncertainty is None:
                anchor_uncertainty = chosen_uncertainty
                virtual_parent = chosen
                virtual_parent_score = float(means_t[chosen_idx])
            elif chosen_uncertainty > 2.0 * anchor_uncertainty:
                virtual_parent, virtual_parent_score = self._sample_bo_evo_parent()
                virtual_parent = virtual_parent.to(device=self.runtime.device, dtype=torch.long)
                anchor_uncertainty = None
            else:
                virtual_parent = chosen
                virtual_parent_score = float(means_t[chosen_idx])
        result = self._direct_proposal_batch(rows)
        result.generation_diagnostics = {
            "operator": "bo_evo",
            "surrogate_query_count": sum(screen_counts),
            "paid_target_count": batch_size,
            "children_per_selection": self.baseline_children_per_selection,
            "ucb_beta": self.baseline_ucb_beta,
            "parent_temperature": self.baseline_parent_temperature,
            "crossover_rate": 0.0,
            "mutation_operator": "independent_1_over_l_conditioned_nonempty",
        }
        return result

    def _direct_proposal_batch(
        self,
        rows: list[tuple[torch.Tensor, float, torch.Tensor]],
        *,
        parent_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> ProposalBatch:
        if parent_metadata is not None and len(parent_metadata) != len(rows):
            raise ValueError("direct proposal metadata must align with proposal rows")
        bases: list[torch.Tensor] = []
        base_rewards: list[float] = []
        base_by_key: dict[bytes, int] = {}
        proposals: list[Proposal] = []
        for row_index, (anchor, score, child) in enumerate(rows):
            anchor = anchor.to(device=self.runtime.device, dtype=torch.long).view(1, -1)
            child = child.to(device=self.runtime.device, dtype=torch.long).view(1, -1)
            key = self._sequence_key(anchor)
            base_index = base_by_key.get(key)
            if base_index is None:
                base_index = len(bases)
                base_by_key[key] = base_index
                bases.append(anchor)
                base_rewards.append(float(score))
            edits = self._sequence_distance(anchor[0], child[0])
            editable_length = int(anchor.shape[1])
            metadata = dict(parent_metadata[row_index]) if parent_metadata is not None else {}
            metadata.setdefault("parent_seq_hash", self._seq_hash(anchor))
            metadata.setdefault("parent_score", float(score))
            if self.proposal_operator == "hamming":
                hamming_edits = int((anchor[0] != child[0]).sum().item())
                conditional_log_prob = self._exact_hamming_conditional_log_prob(
                    length=int(anchor.shape[1]),
                    edits=hamming_edits,
                )
                metadata["hamming_conditional_log_prob"] = conditional_log_prob
                selection_log_prob = metadata.get("parent_selection_log_prob")
                if selection_log_prob is not None:
                    metadata["proposal_log_prob"] = float(selection_log_prob) + conditional_log_prob
                    metadata["proposal_probability_method"] = (
                        "archive_softmax_x_exact_hamming"
                    )
                else:
                    metadata["proposal_probability_method"] = "exact_hamming_given_parent"
            proposals.append(
                Proposal(
                    base_index=base_index,
                    t_value=edits_to_time(edits, seq_len=editable_length),
                    edits_value=float(edits),
                    xt=child,
                    corrupt_mask=(child[0] != anchor[0]).detach().to("cpu"),
                    direct_leaf=True,
                    **metadata,
                )
            )
        if not bases:
            raise RuntimeError(f"{self.proposal_operator} produced no proposals")
        return ProposalBatch(
            base_indices=list(range(len(bases))),
            base_samples=torch.cat(bases, dim=0),
            base_rewards=torch.as_tensor(base_rewards, dtype=torch.float, device=self.runtime.device),
            top_reward=max(score for _, score in self._baseline_archive.values()),
            proposals=proposals,
        )

    def _propose_sequence_baseline(
        self,
        *,
        iteration: int,
        batch: torch.Tensor,
        rewards: torch.Tensor,
        max_proposals: Optional[int],
    ) -> ProposalBatch:
        target = self._proposal_target(max_proposals)
        if target <= 0:
            top_idx = int(torch.argmax(rewards).item())
            return ProposalBatch(
                [top_idx],
                batch[top_idx : top_idx + 1],
                rewards[top_idx : top_idx + 1],
                float(rewards[top_idx].item()),
                [],
            )
        if self.proposal_operator == "hamming":
            _base_indices, bases, base_rewards = self._select_bases(
                iteration=iteration,
                rewards=rewards,
                batch=batch,
            )
            edits = self.baseline_mutation_edits
            if self.candidate.strategy == "trust_region":
                strategy = self.time_policy.tr_strategy
                if strategy is None or strategy.config.mean_edits is None:
                    raise RuntimeError("hamming trust region is missing its absolute edit state")
                edits = int(round(float(strategy.config.mean_edits)))
            rows: list[tuple[torch.Tensor, float, torch.Tensor]] = []
            parent_metadata: list[dict[str, Any]] = []
            for idx in range(target):
                base_position = idx % len(bases)
                parent = bases[base_position : base_position + 1]
                score = float(base_rewards[base_position].item())
                rows.append((parent, score, self._mutate_exact(parent, edits)))
                parent_metadata.append(
                    self._parent_trace_metadata(
                        base_position=base_position,
                        parent=parent,
                        score=score,
                    )
                )
            return self._direct_proposal_batch(rows, parent_metadata=parent_metadata)

        if self.proposal_operator == "bo_evo":
            return self._propose_bo_evo(target)

        if self.proposal_operator == "adalead_official":
            return self._propose_adalead_official(target)

        if self.proposal_operator == "adalead_evodiff":
            return propose_adalead_evodiff(self, target)

        archive = self._baseline_rows()
        predictor = self.active_learning_selector.fit_surrogate_predictor()
        if self.proposal_operator == "adalead":
            # Matched AdaLead-style arm: threshold measured parents and greedily
            # roll out by GP mean; the shared evaluator still ranks the M-pool by EI.
            best = archive[0][1]
            sign = 1.0 if best > 0.0 else (-1.0 if best < 0.0 else 0.0)
            cutoff = best * (1.0 - sign * self.baseline_fitness_threshold)
            parents = [row for row in archive if row[1] >= cutoff]
            population = [
                parents[int(torch.randint(len(parents), (1,)).item())]
                for _ in range(min(target, self.baseline_population_size))
            ]
            parent_batch = torch.cat([row[0] for row in population], dim=0).to(self.runtime.device)
            parent_mean, _ = predictor(parent_batch)
            rows: list[tuple[torch.Tensor, float, torch.Tensor]] = []
            for _ in range(self.baseline_generations):
                children = [self._mutate_exact(parent_batch[idx : idx + 1], self.baseline_mutation_edits) for idx in range(len(population))]
                child_batch = torch.cat(children, dim=0)
                child_mean, _ = predictor(child_batch)
                for idx, child in enumerate(children):
                    rows.append((population[idx][0], population[idx][1], child))
                    if child_mean[idx] >= parent_mean[idx]:
                        parent_batch[idx] = child[0]
                        parent_mean[idx] = child_mean[idx]
                    if len(rows) >= target:
                        return self._direct_proposal_batch(rows)
            raise ValueError("baseline_population_size * baseline_generations must fill M")

        raise RuntimeError(f"unhandled proposal operator {self.proposal_operator}")

    def update_trust_region(
        self,
        *,
        best_time: Optional[float],
        best_edits: Optional[float] = None,
        improved: bool,
        mean_edit_error: Optional[float] = None,
    ) -> None:
        self.time_policy.update_trust_region(
            best_time=best_time,
            best_edits=best_edits,
            improved=improved,
            mean_edit_error=mean_edit_error,
        )

    def expand_trust_region_once(self) -> tuple[Optional[float], Optional[float]]:
        return self.time_policy.expand_trust_region_once()

    def _select_bases(
        self,
        *,
        iteration: int,
        rewards: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        rewards_rank = torch.nan_to_num(rewards, nan=float("-inf"))
        source = self.candidate.source

        if source == "random":
            idx = torch.randint(0, batch.shape[0], (1,), device=self.runtime.device).tolist()
            return idx, batch[idx], rewards[idx]

        if source == "topk":
            if self.topk_use_archive:
                self.last_archive_snapshot = None
                self.last_archive_draws = []
                sampled = self._sample_from_topk_archive(iteration=iteration)
                if sampled is not None:
                    return sampled
            k = int(self.args.candidate_k or min(1, batch.shape[0]))
            k = max(1, min(k, batch.shape[0]))
            topk_unique = bool(self.args.topk_unique) if self.args.topk_unique is not None else False
            if topk_unique:
                selected = select_topk_unique(rewards_rank, batch, k, prefer_recent=True)
                if selected:
                    return selected, batch[selected], rewards[selected]
            scores = rewards_rank.detach().cpu()
            order = sorted(
                range(int(scores.numel())),
                key=lambda idx: (float(scores[idx].item()), int(idx)),
                reverse=True,
            )
            selected = order[:k]
            return selected, batch[selected], rewards[selected]
        max_reward = torch.max(rewards_rank)
        max_indices = torch.nonzero(rewards_rank == max_reward, as_tuple=False).view(-1)
        if max_indices.numel() == 0:
            selected = [int(torch.argmax(rewards_rank).item())]
            return selected, batch[selected], rewards[selected]
        # Prefer the newest entry when several states share the same best score.
        selected = [int(max_indices[-1].item())]
        return selected, batch[selected], rewards[selected]

    def _sample_from_topk_archive(
        self,
        *,
        iteration: int,
    ) -> Optional[tuple[list[int], torch.Tensor, torch.Tensor]]:
        if not self.topk_archive:
            return None
        # Archive draw count tracks mixes_per_time by design.
        k = max(1, int(self.candidate.mixes_per_time))
        # Default to sampling with replacement for weighted archive draws.
        replacement = True
        weights = self._archive_weights()
        if weights.numel() == 0:
            return None
        self.last_archive_snapshot = WeightedArchiveSnapshot(
            iteration=int(iteration),
            seq_hashes=[entry.seq_hash for entry in self.topk_archive],
            rewards=torch.tensor([float(entry.reward) for entry in self.topk_archive], dtype=torch.float),
            weights=weights.clone(),
            temperature=float(self.topk_weight_n),
        )
        try:
            draw_idx = torch.multinomial(weights, num_samples=k, replacement=replacement)
        except RuntimeError:
            if replacement:
                fallback = torch.arange(len(self.topk_archive), dtype=torch.long)
                reps = (k + len(fallback) - 1) // max(1, len(fallback))
                draw_idx = fallback.repeat(reps)[:k]
            else:
                draw_idx = torch.arange(len(self.topk_archive), dtype=torch.long)[:k]

        samples: list[torch.Tensor] = []
        reward_vals: list[float] = []
        self.last_archive_draws = []
        for idx in draw_idx.tolist():
            entry = self.topk_archive[int(idx)]
            samples.append(entry.seq.view(1, -1).to(self.runtime.device))
            reward_vals.append(float(entry.reward))
            self.last_archive_draws.append(
                WeightedArchiveDraw(
                    iteration=int(iteration),
                    archive_index=int(idx),
                    seq_hash=entry.seq_hash,
                    reward=float(entry.reward),
                    weight=float(weights[int(idx)].item()),
                )
            )
        base_samples = torch.cat(samples, dim=0)
        base_rewards = torch.tensor(reward_vals, dtype=torch.float, device=self.runtime.device)
        base_indices = list(range(base_samples.shape[0]))
        return base_indices, base_samples, base_rewards

    def _update_topk_archive_from_batch(
        self,
        *,
        batch: torch.Tensor,
        rewards: torch.Tensor,
        iteration: int,
    ) -> None:
        if self.candidate.source != "topk" or not self.topk_use_archive:
            return
        batch_cpu = batch.detach().to("cpu", dtype=torch.long)
        rewards_cpu = rewards.detach().to("cpu", dtype=torch.float)
        if batch_cpu.ndim == 1:
            batch_cpu = batch_cpu.view(1, -1)
        num_rows = min(int(batch_cpu.shape[0]), int(rewards_cpu.numel()))
        for row in range(num_rows):
            reward = float(rewards_cpu[row].item())
            if not math.isfinite(reward):
                continue
            seq = batch_cpu[row].clone().view(-1)
            self._archive_admit(seq=seq, reward=reward, iteration=iteration)
        self._prune_archive()

    def _archive_admit(
        self,
        *,
        seq: torch.Tensor,
        reward: float,
        iteration: int,
    ) -> None:
        seq = seq.detach().to("cpu", dtype=torch.long).contiguous().view(-1)
        seq_hash = self._seq_hash(seq)
        for idx, entry in enumerate(self.topk_archive):
            if entry.seq_hash != seq_hash:
                continue
            if reward >= float(entry.reward):
                self.topk_archive[idx] = WeightedArchiveEntry(
                    seq=seq.clone(),
                    reward=float(reward),
                    seq_hash=seq_hash,
                    added_iter=int(entry.added_iter),
                    last_updated_iter=int(iteration),
                )
            else:
                entry.last_updated_iter = int(iteration)
            return

        nearest_idx, nearest_dist = self._nearest_archive(seq)
        if nearest_idx is not None and self.topk_archive_d_min > 0 and nearest_dist < self.topk_archive_d_min:
            nearest = self.topk_archive[int(nearest_idx)]
            # Score takes priority when a new candidate is too close to an existing one.
            if reward > float(nearest.reward):
                self.topk_archive[int(nearest_idx)] = WeightedArchiveEntry(
                    seq=seq.clone(),
                    reward=float(reward),
                    seq_hash=seq_hash,
                    added_iter=int(iteration),
                    last_updated_iter=int(iteration),
                )
            return

        self.topk_archive.append(
            WeightedArchiveEntry(
                seq=seq.clone(),
                reward=float(reward),
                seq_hash=seq_hash,
                added_iter=int(iteration),
                last_updated_iter=int(iteration),
            )
        )

    def _prune_archive(self) -> None:
        if len(self.topk_archive) <= self.topk_archive_size:
            return
        weights = self._archive_weights()
        order = sorted(
            range(len(self.topk_archive)),
            key=lambda idx: (
                float(weights[idx].item()) if idx < int(weights.numel()) else float("-inf"),
                float(self.topk_archive[idx].reward),
                int(self.topk_archive[idx].last_updated_iter),
            ),
            reverse=True,
        )
        chosen: list[int] = []
        for idx in order:
            if len(chosen) >= self.topk_archive_size:
                break
            if self.topk_archive_d_min > 0:
                is_far_enough = True
                for prev_idx in chosen:
                    dist = self._sequence_distance(self.topk_archive[idx].seq, self.topk_archive[prev_idx].seq)
                    if dist < self.topk_archive_d_min:
                        is_far_enough = False
                        break
                if not is_far_enough:
                    continue
            chosen.append(idx)

        if len(chosen) < self.topk_archive_size:
            chosen_set = set(chosen)
            for idx in order:
                if idx in chosen_set:
                    continue
                chosen.append(idx)
                chosen_set.add(idx)
                if len(chosen) >= self.topk_archive_size:
                    break
        self.topk_archive = [self.topk_archive[idx] for idx in chosen[: self.topk_archive_size]]

    def _archive_weights(self) -> torch.Tensor:
        if not self.topk_archive:
            return torch.empty((0,), dtype=torch.float)
        rewards = torch.tensor([float(entry.reward) for entry in self.topk_archive], dtype=torch.float)
        logits = self.topk_weight_n * rewards
        finite = torch.isfinite(logits)
        if not bool(finite.any().item()):
            count = int(logits.numel())
            return torch.full((count,), 1.0 / float(max(1, count)), dtype=torch.float)

        masked = torch.where(finite, logits, torch.full_like(logits, float("-inf")))
        max_logit = torch.max(masked[finite])
        unnorm = torch.zeros_like(masked)
        unnorm[finite] = torch.exp(masked[finite] - max_logit)
        z = float(unnorm.sum().item())
        if not math.isfinite(z) or z <= 0.0:
            count = int(logits.numel())
            return torch.full((count,), 1.0 / float(max(1, count)), dtype=torch.float)
        return unnorm / z

    @staticmethod
    def _archive_snapshot_id(snapshot: WeightedArchiveSnapshot) -> str:
        """Return a stable, compact identifier for one weighted archive state."""

        digest = hashlib.sha256()
        digest.update(f"{int(snapshot.iteration)}|{float(snapshot.temperature).hex()}".encode())
        rewards = snapshot.rewards.detach().to("cpu", dtype=torch.double).view(-1).tolist()
        weights = snapshot.weights.detach().to("cpu", dtype=torch.double).view(-1).tolist()
        for seq_hash, reward, weight in zip(
            snapshot.seq_hashes,
            rewards,
            weights,
            strict=True,
        ):
            digest.update(
                f"|{seq_hash}|{float(reward).hex()}|{float(weight).hex()}".encode()
            )
        return f"archive-{int(snapshot.iteration)}-{digest.hexdigest()[:16]}"

    def _parent_trace_metadata(
        self,
        *,
        base_position: int,
        parent: torch.Tensor,
        score: float,
    ) -> dict[str, Any]:
        """Describe the sampled parent without changing proposal-selection interfaces."""

        metadata: dict[str, Any] = {
            "parent_seq_hash": self._seq_hash(parent),
            "parent_score": float(score),
        }
        snapshot = self.last_archive_snapshot
        if (
            self.candidate.source != "topk"
            or not self.topk_use_archive
            or snapshot is None
            or base_position >= len(self.last_archive_draws)
        ):
            return metadata

        draw = self.last_archive_draws[base_position]
        # A hash check protects the trace from silently attaching a stale draw
        # if a future base-selection path changes the returned row ordering.
        if draw.seq_hash != metadata["parent_seq_hash"]:
            return metadata
        probability = float(draw.weight)
        metadata.update(
            {
                "parent_archive_index": int(draw.archive_index),
                "parent_archive_probability": probability,
                "parent_selection_log_prob": (
                    math.log(probability) if probability > 0.0 else float("-inf")
                ),
                "archive_size": len(snapshot.seq_hashes),
                "archive_snapshot_id": self._archive_snapshot_id(snapshot),
                "proposal_probability_method": "archive_softmax_parent_only",
            }
        )
        return metadata

    def _exact_hamming_conditional_log_prob(self, *, length: int, edits: int) -> float:
        """Log probability of a child under ``_mutate_exact`` given its parent."""

        if length < 0 or edits < 0 or edits > length:
            raise ValueError("invalid exact-Hamming mutation dimensions")
        vocab_alternatives = int(self.runtime.vocab_size) - 1
        if edits > 0 and vocab_alternatives <= 0:
            raise ValueError("exact-Hamming mutation requires at least two tokens")
        if edits == 0:
            return 0.0
        return -(
            math.log(math.comb(length, edits))
            + float(edits) * math.log(float(vocab_alternatives))
        )

    def _nearest_archive(self, seq: torch.Tensor) -> tuple[Optional[int], int]:
        if not self.topk_archive:
            return None, 0
        best_idx = 0
        best_dist = self._sequence_distance(seq, self.topk_archive[0].seq)
        for idx in range(1, len(self.topk_archive)):
            dist = self._sequence_distance(seq, self.topk_archive[idx].seq)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx, best_dist

    def _sequence_distance(self, a: torch.Tensor, b: torch.Tensor) -> int:
        model = getattr(getattr(self.runtime, "process", None), "model", None)
        custom_distance = getattr(model, "sequence_edit_distance", None)
        if callable(custom_distance):
            return int(custom_distance(a, b))
        a1 = a.view(-1)
        b1 = b.view(-1)
        if a1.shape != b1.shape:
            raise ValueError("Cannot compute sequence distance for tensors with different shapes")
        return int((a1 != b1).sum().item())

    @staticmethod
    def _seq_hash(seq: torch.Tensor) -> str:
        seq_cpu = seq.detach().to("cpu", dtype=torch.long).contiguous().view(-1)
        return hashlib.sha1(seq_cpu.numpy().tobytes()).hexdigest()

    def _sample_interpolant(
        self,
        base_sample: torch.Tensor,
        t_tensor: torch.Tensor,
        elite_probs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        process = self.runtime.process
        custom_sampler = getattr(getattr(process, "model", None), "sample_interpolant", None)
        if callable(custom_sampler):
            return custom_sampler(
                base_sample=base_sample,
                t_tensor=t_tensor,
                elite_probs=elite_probs,
                transition_mask=self.runtime.transition_mask,
                mixing_mode=self.candidate.mixing_mode,
            )
        interpolant = getattr(process, "interpolant", "mask")
        if interpolant == "mask":
            return mask_interpolant_sample(base_sample, t_tensor, process.cfg.mask_id)

        if self.candidate.mixing_mode == "invalid":
            if self.runtime.transition_mask is None:
                raise ValueError("mixing_mode=invalid requires a transition_mask or transition_mask_path")
            return uniform_interpolant_sample_invalid(
                base_sample,
                t_tensor,
                self.runtime.transition_mask,
                mode=getattr(process, "corrupt_invalid_mode", "any"),
            )

        if elite_probs is not None:
            return categorical_interpolant_sample(base_sample, t_tensor, elite_probs)
        return uniform_interpolant_sample(base_sample, t_tensor, process.cfg.vocab_size)

    def _corrupt_mask(
        self,
        base_sample: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        corrupt_mask_fn = getattr(
            getattr(self.runtime.process, "model", None),
            "corrupt_mask",
            None,
        )
        if callable(corrupt_mask_fn):
            mask = corrupt_mask_fn(base_sample, xt)
            if not isinstance(mask, torch.Tensor) or mask.shape != base_sample.shape:
                raise ValueError(
                    "generator corrupt_mask must return a tensor matching base_sample"
                )
            return mask[0].detach().to(device="cpu", dtype=torch.bool)
        return (xt[0] != base_sample[0]).detach().to(device="cpu", dtype=torch.bool)

    def _sample_generator_interpolant(
        self,
        base_sample: torch.Tensor,
        t_tensor: torch.Tensor,
        elite_probs: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw an interpolant, optionally conditioned on an exact corruption count.

        Rejection sampling preserves the configured interpolant distribution
        conditional on the requested count.  The subsequent generator rollout
        still chooses the endpoint token; with masked rollout, the endpoint can
        either retain the parent token or differ at no more than these sites.
        """

        target = self.generator_exact_corruptions
        if target is not None:
            editable_length = self.time_policy.editable_seq_len_for_sample(base_sample)
            if target > editable_length:
                raise ValueError(
                    "generator_exact_corruptions cannot exceed the editable sequence length "
                    f"({target} > {editable_length})"
                )

        max_attempts = 4096
        for _ in range(max_attempts):
            xt = self._sample_interpolant(base_sample, t_tensor, elite_probs)
            corrupt_mask = self._corrupt_mask(base_sample, xt)
            if target is None or int(corrupt_mask.sum().item()) == target:
                return xt, corrupt_mask

        raise RuntimeError(
            "could not draw a generator interpolant with exactly "
            f"{target} corrupted position(s) after {max_attempts} attempts; "
            "choose a timestep whose expected edit count is close to the target"
        )


__all__ = ["InterpolantProposalPolicy"]
