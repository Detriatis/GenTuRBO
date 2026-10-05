"""Direct-grid adaptation of the repository's matched AdaLead-style GP/EI arm.

This follows the thresholded-parent, fixed-generation GP-mean rollout in
search.interpolant.proposal_policy. It is not an unmodified AdaLead reproduction.
The caller ranks the resulting pool with the shared EI selector; this module
has no oracle or generator access.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from sokoban.evolution import mutate_grid


class GridAdaLead:
    def __init__(self, cfg, *, seed):
        self.cfg = cfg
        self.rng = random.Random(seed)

    def propose(self, archive, predictor):
        cfg = self.cfg
        # Only measured, exact solutions can seed a rollout. Censored results
        # never acquire a made-up fitness or enter the parent threshold.
        ranked = sorted((r for r in archive.values()
                         if r['status'] == 'solved' and r.get('objective_status', 'exact') == 'exact'
                         and r['score'] is not None and np.isfinite(r['score'])),
                        key=lambda r: (-r['score'], r['key']))
        if not ranked:
            raise ValueError('AdaLead requires an exactly scored solvable parent')
        best = ranked[0]['score']
        cutoff = best - abs(best) * cfg.adalead_fitness_threshold
        eligible = [r for r in ranked if r['score'] >= cutoff]
        size = min(cfg.candidate_pool, cfg.adalead_population)
        population = torch.tensor([self.rng.choice(eligible)['canonical_board'] for _ in range(size)],
                                  dtype=torch.long)
        means = np.asarray(predictor(population)[0], dtype=float).copy()
        rows = []
        for _ in range(cfg.adalead_generations):
            children = torch.as_tensor(np.stack([
                mutate_grid(row.numpy(), self.rng, edits=cfg.adalead_mutation_edits)
                for row in population]), dtype=torch.long)
            child_means = np.asarray(predictor(children)[0], dtype=float)
            for idx, child in enumerate(children):
                rows.append(child.clone())
                if child_means[idx] >= means[idx]:
                    population[idx] = child
                    means[idx] = child_means[idx]
                if len(rows) == cfg.candidate_pool:
                    return torch.stack(rows)
        raise ValueError('AdaLead population * generations must fill the candidate pool')

    def metadata(self):
        return {
            'variant': 'matched_adalead_style_gp_ei',
            'reference_implementation': 'search.interpolant.proposal_policy:adalead',
            'population_size': self.cfg.adalead_population,
            'generations': self.cfg.adalead_generations,
            'fitness_threshold': self.cfg.adalead_fitness_threshold,
            'mutation_edits': self.cfg.adalead_mutation_edits,
            'mutation': 'equal_probability_interior_tile_swap_or_wall_floor_flip',
            'rollout_acceptance': 'nondecreasing_gp_mean',
            'pool_selection': 'shared_gp_expected_improvement_with_local_penalization',
            'recombination': False, 'solvability_repair': False,
            'proposal_oracle_calls': 0,
        }
