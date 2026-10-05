"""Sokoban evolutionary proposals using only paid fitness and public tile rules.

The direct-grid operators preserve entity counts, not solvability. They never
call a solver, a generator or a reward function.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from sokoban.boards import structural_errors

INTERIOR = tuple(y * 10 + x for y in range(1, 9) for x in range(1, 9))
WHOLE_GRID = tuple(range(100))


def editable_cells(region):
    if region not in {'interior', 'all'}:
        raise ValueError('editable region must be interior or all')
    return WHOLE_GRID if region == 'all' else INTERIOR


def crossover_grid(left, right, rng, *, editable_region='interior'):
    """Copy a random rectangle, then normalise inventory over the editable cells.

    Excess entities are removed uniformly and missing ones placed uniformly.
    Goals may overlap boxes/player. Occupied/goal cells are made floor. This
    does not check connectivity, deadlocks, solutions or objective values.
    """
    child = np.asarray(left, dtype=np.int64).copy().reshape(10, 10)
    other = np.asarray(right).reshape(10, 10)
    cells = editable_cells(editable_region)
    edges = range(0, 11) if editable_region == 'all' else range(1, 10)
    y0, y1 = sorted(rng.sample(edges, 2))
    x0, x1 = sorted(rng.sample(edges, 2))
    child[y0:y1, x0:x1] = other[y0:y1, x0:x1]
    child = child.flatten()
    walls = child == 0
    players = [i for i in cells if child[i] in (2, 6)]
    player = rng.choice(players or cells)

    def inventory(existing, count, allowed):
        existing = sorted(set(existing) & set(allowed))
        if len(existing) > count:
            return set(rng.sample(existing, count))
        return set(existing + rng.sample(sorted(set(allowed) - set(existing)), count - len(existing)))

    boxes = inventory([i for i in cells if child[i] in (3, 5)], 4,
                      [i for i in cells if i != player])
    goals = inventory([i for i in cells if child[i] in (4, 5, 6)], 4, cells)
    result = np.zeros(100, dtype=np.int64)
    for i in cells:
        result[i] = 0 if walls[i] else 1
    for i in goals:
        result[i] = 4
    for i in boxes:
        result[i] = 5 if i in goals else 3
    result[player] = 6 if player in goals else 2
    return result


def mutate_grid(board, rng, edits=2, *, editable_region='interior'):
    """Apply tile swaps or wall/plain-floor flips with equal probability."""
    child = np.asarray(board, dtype=np.int64).copy().flatten()
    cells = editable_cells(editable_region)
    for _ in range(edits):
        wall_candidates = [i for i in cells if child[i] in (0, 1)]
        if rng.random() < .5 and wall_candidates:
            i = rng.choice(wall_candidates)
            child[i] = 1 - child[i]
        else:
            i = rng.choice(cells)
            different = [j for j in cells if child[j] != child[i]]
            if different:
                j = rng.choice(different)
                child[i], child[j] = child[j], child[i]
    return child


class EvolutionPopulation:
    """Elitist (mu + lambda) survival, canonical deduplication, tournament parents.

    Unresolved fitness is excluded, exactly as for GP labels/incumbents. A
    structurally valid proven-unsolvable individual has finite fitness -1 and
    can survive if too few higher-scoring individuals exist. Offspring receive
    no free rejection/resampling before their paid oracle evaluation.
    """

    def __init__(self, *, size=32, tournament=3, crossover_rate=.5, mutation_edits=2, seed=0,
                 editable_region='interior'):
        if size < 1 or tournament < 1 or mutation_edits < 1 or not 0 <= crossover_rate <= 1:
            raise ValueError('invalid evolutionary settings')
        self.size, self.tournament = size, tournament
        self.crossover_rate, self.mutation_edits = crossover_rate, mutation_edits
        editable_cells(editable_region)
        self.editable_region = editable_region
        self.rng = random.Random(seed)
        self.members = []

    def observe(self, records):
        members = {r['key']: r for r in self.members}
        for record in records:
            if record.get('status') == 'unknown' or record.get('objective_status') in {'unknown', 'bounded'}:
                continue
            score = record.get('score')
            if score is None or not math.isfinite(score) or structural_errors(record['canonical_board']):
                continue
            members[record['key']] = record
        self.members = sorted(members.values(), key=lambda r: (-r['score'], r['key']))[:self.size]

    def _parent(self):
        if not self.members:
            raise RuntimeError('no finite, structurally valid evolutionary parent')
        contestants = self.rng.sample(self.members, min(self.tournament, len(self.members)))
        return min(contestants, key=lambda r: (-r['score'], r['key']))

    def propose(self, count, method, *, adapter=None, mask_count=20):
        if method == 'denoise_ea':
            parents = torch.tensor([self._parent()['canonical_board'] for _ in range(count)])
            return adapter.propose(parents, mask_count).cpu()
        if method not in {'grid_ga', 'grid_mutation'}:
            raise ValueError('unknown evolutionary method')
        children = []
        for _ in range(count):
            parent = self._parent()['canonical_board']
            if method == 'grid_ga' and self.rng.random() < self.crossover_rate:
                parent = crossover_grid(parent, self._parent()['canonical_board'], self.rng,
                                        editable_region=self.editable_region)
            children.append(mutate_grid(parent, self.rng, self.mutation_edits,
                                        editable_region=self.editable_region))
        return torch.tensor(np.stack(children), dtype=torch.long)

    def metadata(self, method):
        return {'selection': 'tournament', 'survival': 'elitist_mu_plus_lambda_unique_canonical',
                'population_size': self.size, 'tournament_size': self.tournament,
                'crossover_rate': self.crossover_rate if method == 'grid_ga' else 0,
                'mutation': 'frozen_generator' if method == 'denoise_ea' else 'tile_swap_or_wall_floor_flip',
                'mutation_edits': self.mutation_edits if method != 'denoise_ea' else None,
                'editable_cells': len(editable_cells(self.editable_region)) if method != 'denoise_ea' else 100,
                'editable_region': self.editable_region if method != 'denoise_ea' else 'all',
                'border_repair': 'none' if self.editable_region == 'all' or method == 'denoise_ea' else 'fixed_walls',
                'structural_normalisation': 'rectangle_crossover_inventory_only' if method == 'grid_ga' else 'none',
                'unresolved_fitness': 'excluded', 'offspring_prescreening': 'none'}
