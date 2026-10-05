"""Budgeted, prepaid-archive adaptation of Holo ``DiscreteEvolution``."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import Baseline, BaselineResult
from .utils import elapsed_since, evaluate_oracle, now_seconds, set_deterministic_seed, top_k_samples


def _tensor_rows(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        return [tensor.clone()]
    if tensor.ndim == 2:
        return [row.clone() for row in tensor]
    raise ValueError("initial_population must contain one-dimensional discrete sequences")


class HoloGeneticAlgorithmBaseline(Baseline):
    """Faithful Holo survivor/recombine/mutate mechanics under a finite budget.

    The released optimizer evaluates a fixed population, retains every member
    at or below a loss quantile, refills the rejected rows by sitewise
    recombination of survivors, and finally applies independent site mutation
    to the entire population. Duplicate and infeasible candidates are scored;
    neither is filtered before the black-box call.
    """

    name = "holo_genetic_algorithm"

    def __init__(
        self,
        *,
        population_size: int,
        mutation_prob: float,
        recombine_prob: float,
        survival_quantile: float,
        vocab_size: int,
        top_k: int = 100,
    ) -> None:
        self.population_size = int(population_size)
        self.mutation_prob = float(mutation_prob)
        self.recombine_prob = float(recombine_prob)
        self.survival_quantile = float(survival_quantile)
        self.vocab_size = int(vocab_size)
        self.top_k = int(top_k)
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        if not 0.0 < self.mutation_prob < 1.0:
            raise ValueError("mutation_prob must be in (0, 1)")
        if not 0.0 <= self.recombine_prob < 1.0:
            raise ValueError("recombine_prob must be in [0, 1)")
        if not 0.0 < self.survival_quantile < 1.0:
            raise ValueError("survival_quantile must be in (0, 1)")
        if self.survival_quantile * self.population_size < 2.0:
            raise ValueError("survival_quantile must retain at least two particles")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least two")

    def _initial_rows(self, initial_population: Any, *, device: torch.device) -> torch.Tensor:
        rows: list[torch.Tensor] = []
        if isinstance(initial_population, (list, tuple)):
            for value in initial_population:
                rows.extend(_tensor_rows(value))
        else:
            rows = _tensor_rows(initial_population)
        if not rows:
            raise ValueError("initial_population must be non-empty")
        shapes = {tuple(row.shape) for row in rows}
        if len(shapes) != 1:
            raise ValueError("all initial sequences must have the same shape")
        particles = torch.stack(rows).to(device=device, dtype=torch.float32)
        if torch.any(particles < 0) or torch.any(particles >= self.vocab_size):
            raise ValueError("initial_population contains tokens outside the vocabulary")
        return particles

    def _tiled_particles(self, rows: torch.Tensor) -> torch.Tensor:
        if int(rows.shape[0]) > self.population_size:
            raise ValueError("initial_population cannot exceed population_size")
        indices = torch.arange(self.population_size, device=rows.device) % int(rows.shape[0])
        return rows.index_select(0, indices).clone()

    @staticmethod
    def _selection_losses(scores: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        selection_scores = np.where(np.isfinite(scores), scores, -np.inf)
        return torch.as_tensor(-selection_scores, dtype=dtype, device=device)

    def _evolve(self, particles: torch.Tensor, losses: torch.Tensor) -> tuple[torch.Tensor, int]:
        threshold = torch.quantile(losses, self.survival_quantile)
        surviving = losses <= threshold if threshold < float("inf") else torch.ones_like(losses, dtype=torch.bool)
        survivor_rows = particles[surviving]
        survivor_count = int(survivor_rows.shape[0])
        missing = self.population_size - survivor_count
        if missing > 0:
            parent_1 = torch.randint(0, survivor_count, (missing,), device=particles.device)
            parent_2 = torch.randint(0, survivor_count, (missing,), device=particles.device)
            recombine = torch.rand_like(survivor_rows[parent_1]) < self.recombine_prob
            particles[~surviving] = torch.where(
                recombine,
                survivor_rows[parent_1],
                survivor_rows[parent_2],
            )
        mutate = torch.rand_like(particles) < self.mutation_prob
        replacements = torch.randint(
            self.vocab_size,
            particles.shape,
            device=particles.device,
            dtype=torch.long,
        ).to(dtype=particles.dtype)
        particles.copy_(torch.where(mutate, replacements, particles))
        return particles, survivor_count

    def run(
        self,
        model: Any,
        oracle: Any,
        budget: int,
        batch_size: int,
        seed: int,
        **kwargs: Any,
    ) -> BaselineResult:
        del model
        started = now_seconds()
        set_deterministic_seed(seed)
        budget_int = int(budget)
        if budget_int <= 0:
            raise ValueError("budget must be positive")
        if int(batch_size) != self.population_size:
            raise ValueError("batch_size must equal population_size for complete Holo generations")
        initial_population = kwargs.pop("initial_population", None)
        if initial_population is None:
            raise ValueError("Holo GA requires an explicit initial_population")
        raw_initial_scores = kwargs.pop("initial_scores", None)
        device = torch.device(str(kwargs.pop("device", "cpu")))
        if kwargs:
            raise TypeError(f"unexpected Holo GA options: {', '.join(sorted(kwargs))}")
        initial_rows = self._initial_rows(initial_population, device=device)

        initial_scores: np.ndarray | None = None
        initial_count = 0
        initial_survivor_count: int | None = None
        if raw_initial_scores is None:
            paid_budget = budget_int
            particles = self._tiled_particles(initial_rows)
        else:
            initial_scores = np.asarray(raw_initial_scores, dtype=float).reshape(-1)
            initial_count = int(initial_rows.shape[0])
            if initial_scores.size != initial_count:
                raise ValueError(
                    f"initial_scores has {initial_scores.size} values, expected {initial_count}"
                )
            paid_budget = budget_int - initial_count
            particles = self._tiled_particles(initial_rows)
            tiled_scores = initial_scores[np.arange(self.population_size) % initial_count]
            particles, initial_survivor_count = self._evolve(
                particles,
                self._selection_losses(tiled_scores, device=device, dtype=particles.dtype),
            )
        if paid_budget <= 0 or paid_budget % self.population_size != 0:
            raise ValueError(
                "budget after prepaid initial scores must be a positive multiple of population_size"
            )

        all_samples: list[torch.Tensor] = []
        all_scores: list[float] = []
        trace: list[dict[str, Any]] = []
        finite_initial = (
            initial_scores[np.isfinite(initial_scores)]
            if initial_scores is not None
            else np.zeros((0,), dtype=float)
        )
        best_score = float(np.max(finite_initial)) if finite_initial.size else -float("inf")
        oracle_calls = 0
        generation_count = paid_budget // self.population_size
        for generation in range(generation_count):
            evaluated = particles.round().clamp(0, self.vocab_size - 1).to(dtype=torch.long)
            samples = [row.detach().cpu().clone() for row in evaluated]
            scores_np, calls = evaluate_oracle(oracle, samples)
            oracle_calls += calls
            all_samples.extend(samples)
            all_scores.extend(float(value) for value in scores_np)
            finite = np.isfinite(scores_np)
            if finite.any():
                best_score = max(best_score, float(np.max(scores_np[finite])))
            losses = self._selection_losses(scores_np, device=device, dtype=particles.dtype)
            particles, survivors = self._evolve(particles, losses)
            trace.append(
                {
                    "generation": generation,
                    "oracle_calls": oracle_calls,
                    "charged_oracle_calls": initial_count + oracle_calls,
                    "best_score": best_score,
                    "feasible_fraction": float(finite.mean()),
                    "survivor_count": survivors,
                }
            )

        scores = np.asarray(all_scores, dtype=float)
        ranking_samples = all_samples
        ranking_scores = scores
        if initial_scores is not None:
            ranking_samples = [row.detach().cpu().clone() for row in initial_rows] + ranking_samples
            ranking_scores = np.concatenate((initial_scores, ranking_scores))
        best_samples, best_scores = top_k_samples(
            ranking_samples,
            ranking_scores,
            k=min(self.top_k, len(ranking_samples)),
        )
        return BaselineResult(
            samples=all_samples,
            scores=scores,
            best_samples=best_samples,
            best_scores=best_scores,
            oracle_calls=oracle_calls,
            generated_candidates=oracle_calls,
            trace=trace,
            metadata={
                "baseline": self.name,
                "seed": int(seed),
                "budget": budget_int,
                "batch_size": int(batch_size),
                "population_size": self.population_size,
                "generation_count": generation_count,
                "initial_oracle_evaluations": initial_count,
                "new_oracle_calls": oracle_calls,
                "total_charged_oracle_calls": initial_count + oracle_calls,
                "prepaid_initial_scores": initial_scores is not None,
                "initial_survivor_count": initial_survivor_count,
                "warm_start_policy": (
                    "tile_prepaid_archive_then_released_survive_recombine_mutate"
                    if initial_scores is not None
                    else "released_repeat_initial_rows"
                ),
                "mutation_prob": self.mutation_prob,
                "recombine_prob": self.recombine_prob,
                "survival_quantile": self.survival_quantile,
                "vocab_size": self.vocab_size,
                "device": str(device),
                "duplicate_policy": "score_all_like_released_holo",
                "feasibility_policy": "black_box_score_all",
                "runtime_seconds": elapsed_since(started),
            },
        )


__all__ = ["HoloGeneticAlgorithmBaseline"]
