"""FLEXS AdaLead with the locked protein-study mechanics and query accounting.

Port of the protein study's ``adalead_official`` operator, itself based on
FLEXS dd409167b5575b51967d325b4e48aef3577a505f. This is the no-recombination
configuration: threshold parents, root-anchored walks, and top predicted mean.
The seven public tiles replace the amino-acid alphabet. No inventory repair,
feasibility filtering, generator, or oracle is available to this operator.
"""
from __future__ import annotations

import numpy as np
import torch


def sequence_key(row):
    return row.detach().to('cpu', dtype=torch.long).contiguous().numpy().tobytes()


class OfficialAdaLead:
    def __init__(self, cfg):
        self.cfg = cfg
        self.measured = []
        self.measured_keys = set()
        self.paid_keys = set()
        self.initialized = False
        self.diagnostics = {}
        self.selected_metadata = []

    def observe(self, batch, scores):
        # Match protein/FLEXS initial multiplicity and later unique measured
        # rows. All paid identities, including censored results, are remembered.
        for row, score in zip(batch, scores, strict=True):
            row = row.detach().to('cpu', dtype=torch.long).view(1, -1).clone()
            key = sequence_key(row)
            self.paid_keys.add(key)
            score = float(score)
            if np.isfinite(score) and (not self.initialized or key not in self.measured_keys):
                self.measured.append((row, score))
                self.measured_keys.add(key)
        self.initialized = True

    def mutate(self, parent):
        child = parent.clone()
        length = child.shape[1]
        positions = torch.rand(length, device=child.device) < min(1.0, self.cfg.adalead_mu / length)
        count = int(positions.sum())
        if count:
            # Full alphabet replacement deliberately allows a no-op, just as
            # FLEXS and the protein operator do. The caller rejects known rows.
            child[0, positions] = torch.randint(0, 7, (count,), device=child.device)
        return child

    @staticmethod
    def means(predictor, batch):
        means = torch.as_tensor(predictor(batch)[0], dtype=torch.double).view(-1)
        if means.numel() != len(batch) or not torch.isfinite(means).all():
            raise ValueError('AdaLead requires one finite surrogate mean per sequence')
        return means

    def propose(self, predictor, *, count):
        cfg = self.cfg
        if not 1 <= count <= cfg.oracle_batch:
            raise ValueError('invalid AdaLead paid batch size')
        if not self.measured:
            raise RuntimeError('AdaLead has no finite measured parents')
        best = max(score for _, score in self.measured)
        cutoff = best - abs(best) * cfg.adalead_fitness_threshold
        retained = [row for row in self.measured if row[1] >= cutoff]
        # FLEXS numpy.resize semantics, in measured order, including duplicates.
        parents = [retained[i % len(retained)] for i in range(cfg.oracle_batch)]
        generated = {}
        queries = root_queries = mutant_queries = attempts = 0
        query_budget = cfg.adalead_surrogate_queries
        maximum_attempts = max(10_000, query_budget * 1_000)
        while queries < query_budget:
            for start in range(0, len(parents), cfg.adalead_eval_batch_size):
                remaining = query_budget - queries
                if remaining <= 0:
                    break
                roots = parents[start:start + cfg.adalead_eval_batch_size][:remaining]
                batch = torch.cat([row[0] for row in roots])
                root_means = self.means(predictor, batch)
                queries += len(roots)
                root_queries += len(roots)
                nodes = [(i, row[0], row[0], row[1]) for i, row in enumerate(roots)]
                while nodes and queries + cfg.adalead_eval_batch_size < query_budget:
                    children = []
                    while len(children) < len(nodes):
                        # Preserve the released code's node ordering and its
                        # allowance for duplicates within a single model batch.
                        idx, node, root, score = nodes[len(children) - 1]
                        child = self.mutate(node)
                        attempts += 1
                        if attempts > maximum_attempts:
                            raise RuntimeError('AdaLead could not draw novel mutants within its safety limit')
                        key = sequence_key(child)
                        if key in self.paid_keys or key in generated:
                            continue
                        children.append((idx, child, root, score))
                    predictions = self.means(predictor, torch.cat([r[1] for r in children]))
                    queries += len(children)
                    mutant_queries += len(children)
                    nodes = []
                    for (idx, child, root, score), mean in zip(children, predictions.tolist(), strict=True):
                        generated[sequence_key(child)] = (mean, child, root, score, float(root_means[idx]))
                        if mean >= float(root_means[idx]):
                            nodes.append((idx, child, root, score))
        if queries != query_budget or len(generated) < count:
            raise RuntimeError('AdaLead exhausted its model budget without enough unique candidates')
        ranked = sorted(generated.values(), key=lambda r: r[0], reverse=True)[:count]
        self.selected_metadata = [dict(predicted_mean=r[0], raw_parent_board=r[2][0].tolist(),
                                       parent_score=r[3], root_predicted_mean=r[4]) for r in ranked]
        self.diagnostics = dict(operator='adalead_official', surrogate_query_count=queries,
                                surrogate_root_query_count=root_queries,
                                surrogate_mutant_query_count=mutant_queries,
                                paid_target_count=count, unique_generated_count=len(generated),
                                mutation_attempts=attempts, threshold_cutoff=cutoff,
                                retained_measured_rows=len(retained))
        return torch.cat([r[1] for r in ranked])

    def metadata(self):
        return dict(variant='protein_matched_flexs_adalead',
                    upstream_commit='dd409167b5575b51967d325b4e48aef3577a505f',
                    surrogate_queries_per_batch=self.cfg.adalead_surrogate_queries,
                    sequences_batch_size=self.cfg.oracle_batch,
                    eval_batch_size=self.cfg.adalead_eval_batch_size,
                    mu=self.cfg.adalead_mu, threshold=self.cfg.adalead_fitness_threshold,
                    rho=0, recombination_rate=0.0, rng_backend='torch_global_seeded',
                    mutation='independent_mu_over_L_full_alphabet_replacement',
                    public_tile_values=list(range(7)), editable_cells=100,
                    rollout_acceptance='predicted_mean_at_least_fixed_root_mean',
                    pool_selection='top_surrogate_mean', local_penalization=False,
                    novelty='literal_sequence_identity', repair='none',
                    solvability_prescreening=False, proposal_oracle_calls=0,
                    budget_adaptation='truncate_final_root_chunk_and_return_exact_top_k')
