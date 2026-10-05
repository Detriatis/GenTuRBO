"""Separate faithful AdaLead arm, inheriting the paired Ehrlich GenTuRBO protocol.

Plan: python -m genturbo.ehrlich_adalead
Smoke: python -m genturbo.ehrlich_adalead --run --seeds 0 --smoke-budget 96
Run: python -m genturbo.ehrlich_adalead --run --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import replace
from pathlib import Path

from genturbo.control_study.runner import _TR_KEYS, bound_smoke_cells
from genturbo.kernel_study.runner import run_cells
from genturbo.sequence_baseline_study.config import load_sequence_baseline_study
from genturbo.sequence_baseline_study.runner import build_sequence_baseline_cells
from genturbo.serialization import atomic_write_json

CONFIG = "configs/experiments/ehrlich_adalead_official_v1/online.json"
STUDY = "ehrlich_adalead_official_v1"


def build_cells(repo_root: Path, *, config: str = CONFIG, run_root: Path | None = None,
                seeds: list[int] | None = None, smoke_budget: int | None = None,
                verify_inputs: bool = True):
    """Inherit starts, oracle, GP and budget; override only algorithm/execution."""
    root = repo_root.resolve()
    spec = json.loads((root / config).read_text())
    if spec["study"] != STUDY or spec["base_arm"] != "tr_gp":
        raise ValueError("expected the faithful AdaLead transfer from Ehrlich tr_gp")
    selected = spec["seeds"] if seeds is None else seeds
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(spec["seeds"]):
        raise ValueError("seeds must be a nonempty unique subset of the paired cohort")
    if smoke_budget is not None and run_root is None:
        raise ValueError("smoke runs require a separate --output directory")
    study = load_sequence_baseline_study(root / spec["base_study"], repo_root=root, verify_inputs=verify_inputs)
    originals = build_sequence_baseline_cells(study, repo_root=root)
    base = (run_root or root / spec["run_dir"]).resolve()
    canonical = (root / spec["run_dir"]).resolve()
    if smoke_budget is not None and (base == canonical or canonical in base.parents):
        raise ValueError("smoke output must be outside the canonical cohort")
    cells = []
    for old in originals:
        if old.arm != spec["base_arm"] or old.seed not in selected:
            continue
        dest = base / old.condition / "adalead_official" / f"seed{old.seed}"
        cfg = dict(old.resolved_config)
        for key in _TR_KEYS:
            cfg.pop(key, None)
        cfg.update(spec["overrides"])
        # Explicit token ordering and RNG adapter are shared with the retained
        # protein implementation. They do not consume RNG or alter proposals.
        cfg["baseline_adalead_alphabet"] = "ACDEFGHIKLMNPQRSTVWY"
        cfg["baseline_adalead_rng_backend"] = "torch_global_seeded"
        cfg.pop("baseline_crossover_rate", None)
        cfg["baseline_adalead_rho"] = 0
        cfg["baseline_adalead_recombination_rate"] = 0.0
        for key, value in tuple(cfg.items()):
            if key.endswith("_out"):
                cfg[key] = str(dest / Path(value).name)
        status = "smoke" if smoke_budget is not None else "canonical"
        cfg["plot_label"] = "adalead_official"
        cfg["metadata"] = {"experiment_status": status, "plot_label": "adalead_official"}
        if run_root is not None:
            # Packed runs may write outside the frozen source snapshot. Keep
            # the shared input archive rooted in that snapshot explicitly.
            cfg["initial_batch_path"] = str(root / cfg["initial_batch_path"])
        cells.append(replace(old, arm="adalead_official", study=STUDY,
                             run_dir=dest, resolved_config=cfg, status=status))
    if smoke_budget is not None:
        cells = bound_smoke_cells(cells, selected_archive_evaluations=smoke_budget, device="cpu")
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-budget", type=int)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cells = build_cells(root, config=args.config, seeds=args.seeds, run_root=args.output,
                        smoke_budget=args.smoke_budget)
    if args.run:
        print(json.dumps(run_cells(cells, repo_root=root), indent=2))
    else:
        print(json.dumps([{"seed": c.seed, "run_dir": str(c.run_dir), "config": c.resolved_config}
                          for c in cells], indent=2))


def verify_cell(cell, *, repo_root: Path):
    """Check algorithm accounting and literal novelty without new oracle calls."""
    cfg = cell.resolved_config
    def rows(value):
        path = repo_root / value
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt') as stream:
            return [json.loads(line) for line in stream]
    diagnostics = rows(cfg['proposal_diagnostics_out'])
    initial = rows(cfg['initial_batch_out'])
    saved = rows(cfg['initial_batch_path'])
    paid = rows(cfg['samples_out'])
    if [r['seq'] for r in initial] != [r['seq'] for r in saved]:
        raise ValueError('AdaLead did not preserve its paired saved initial sequences')
    batch_size = cfg['al_oracle_batch_size']
    budget = cfg['max_oracle_evaluations']
    rounds = (budget - cfg['initial_oracle_evaluations']) // batch_size
    if len(diagnostics) != rounds or len(initial) + len(paid) != budget:
        raise ValueError('AdaLead did not complete its exact query budget')
    seen = {tuple(r['seq']) for r in initial}
    for row in paid:
        key = tuple(row['seq'])
        if key in seen or row['oracle_evaluated'] is not True:
            raise ValueError('AdaLead selected a previously paid sequence or an unscored row')
        seen.add(key)
    for row in diagnostics:
        d = row['proposal_generation']
        means = d['selected_predicted_means']
        if (d['operator'] != 'adalead_official' or d['selection'] != 'top_surrogate_mean'
                or d['surrogate_query_count'] != cfg['baseline_adalead_surrogate_queries']
                or d['surrogate_root_query_count'] + d['surrogate_mutant_query_count'] != d['surrogate_query_count']
                or d['paid_target_count'] != batch_size or len(means) != batch_size
                or means != sorted(means, reverse=True)
                or row['oracle_scored_count'] != batch_size
                or row['oracle_budget_debit_count'] != batch_size
                or row['oracle_privileged_scored_count'] != 0):
            raise ValueError('AdaLead batch violates model-query, paid-query or selection protocol')
    result = {'status': 'passed', 'seed': cell.seed, 'paid_queries': budget,
              'batches': rounds, 'surrogate_queries': rounds * cfg['baseline_adalead_surrogate_queries'],
              'initial_sequences_match': True, 'paid_sequences_novel': True}
    atomic_write_json(cell.run_dir / 'adalead_validation.json', result)
    return result


if __name__ == "__main__":
    main()
