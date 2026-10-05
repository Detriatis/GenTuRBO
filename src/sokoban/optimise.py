"""Matched-budget Sokoban experiments using shared GenTuRBO acquisition and control."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from core.utils import seed_everything
from runners.common import restore_process_from_checkpoint
from search.interpolant.active_learning import ActiveLearningConfig, OneHotTanimotoExpectedImprovementSelector
from search.interpolant.trust_region import ScalarTrustRegion
from search.interpolant.types import TrustRegionConfig
from sokoban.adalead import GridAdaLead
from sokoban.adalead_official import OfficialAdaLead
from sokoban.boards import decode_board
from sokoban.evolution import EvolutionPopulation
from sokoban.hamming import mutate_exact
from sokoban.initial_archive import load_initial_sources
from sokoban.prepare import sha256
from sokoban.search import DISPLAY_ALPHABET, SokobanReward, SokobanRolloutAdapter, SokobanSwitchReward, board_key


@dataclass(frozen=True)
class PilotConfig:
    checkpoint: str = "runs/train/sokoban_mask_transformer/checkpoints/model.pt"
    output: str = "runs/sokoban"
    device: str = "cuda"
    precision: str = "float16"
    seeds: tuple = (0, 1, 2)
    methods: tuple = ("generator", "fixed", "turdbo")
    budget: int = 2000
    initial_size: int = 64
    candidate_pool: int = 128
    oracle_batch: int = 32
    archive_parents: int = 16
    parent_policy: str = "topk_uniform"
    local_penalization: bool = True
    fixed_masks: int = 20
    min_masks: int = 5
    max_masks: int = 40
    success_tolerance: int = 3
    failure_tolerance: int = 3
    region_scale: float = 1.5
    region_shrink_scalar: float | None = None
    gp_buffer: int = 256
    gp_refit_interval: int = 5
    solver_workers: int = 4
    solver_max_states: int = 200000
    solver_max_seconds: float = 30.0
    solver_max_discovered_states: int = 1000000
    objective: str = "solver_effort"
    evolution_population: int = 32
    evolution_tournament: int = 3
    ga_crossover_rate: float = 0.5
    ga_mutation_edits: int = 2
    ga_editable_region: str = "interior"
    adalead_population: int = 32
    adalead_generations: int = 4
    adalead_fitness_threshold: float = 0.05
    adalead_mutation_edits: int = 1
    adalead_surrogate_queries: int = 128
    adalead_eval_batch_size: int = 20
    adalead_mu: float = 1.0
    locality_parents: int = 16
    locality_repeats: int = 8
    locality_masks: tuple = (5, 10, 20, 40)

    def __post_init__(self):
        for field in ("initial_size", "candidate_pool", "oracle_batch", "archive_parents", "gp_buffer",
                      "gp_refit_interval", "solver_workers", "locality_parents", "locality_repeats",
                      "evolution_population", "evolution_tournament", "ga_mutation_edits",
                      "adalead_population", "adalead_generations", "adalead_mutation_edits"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.budget < self.initial_size or self.oracle_batch > self.candidate_pool:
            raise ValueError("budget must cover initialisation and pool must cover oracle batch")
        if not 1 <= self.min_masks <= self.fixed_masks <= self.max_masks <= 100:
            raise ValueError("invalid mask bounds")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if not self.methods or len(set(self.methods)) != len(self.methods) or set(self.methods) - {"generator", "fixed", "turdbo", "turdbo_sequence", "grid_ga", "grid_mutation", "denoise_ea", "adalead_gp_ei", "adalead_official", "hamming_gp_ei"}:
            raise ValueError("invalid methods")
        if self.parent_policy not in {"topk_uniform", "best"}:
            raise ValueError("invalid parent policy")
        if not isinstance(self.local_penalization, bool):
            raise ValueError("local_penalization must be a boolean")
        if self.region_shrink_scalar is not None and not 0 < self.region_shrink_scalar < 1:
            raise ValueError("region_shrink_scalar must be in (0,1)")
        if any(not 1 <= k <= 100 for k in self.locality_masks):
            raise ValueError("invalid locality masks")
        if self.success_tolerance < 1 or self.failure_tolerance < 1 or self.region_scale <= 1:
            raise ValueError("invalid trust-region settings")
        if not 0 <= self.ga_crossover_rate <= 1:
            raise ValueError("invalid GA crossover rate")
        if self.ga_editable_region not in {"interior", "all"}:
            raise ValueError("invalid GA editable region")
        if not 0 <= self.adalead_fitness_threshold <= 1:
            raise ValueError("invalid AdaLead fitness threshold")
        if "adalead_gp_ei" in self.methods and self.adalead_population * self.adalead_generations < self.candidate_pool:
            raise ValueError("AdaLead population * generations must fill the candidate pool")
        if "adalead_official" in self.methods:
            if (not np.isfinite(self.adalead_mu) or self.adalead_mu <= 0
                    or self.adalead_eval_batch_size < 1
                    or self.adalead_surrogate_queries <= self.oracle_batch + self.adalead_eval_batch_size):
                raise ValueError("invalid official AdaLead mutation or surrogate budget")
            if self.local_penalization:
                raise ValueError("official AdaLead uses top predicted mean without local penalization")
        if self.objective not in {"solver_effort", "box_switches"}:
            raise ValueError("invalid Sokoban objective")


def load_config(path, seen=None):
    """Resolve a named base relative to the child configuration."""
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("cyclic base configuration")
    values = json.loads(path.read_text())
    base = values.pop("base_config", None)
    inherited = load_config(path.parent / base, seen | {path}) if base else {}
    return {**inherited, **values}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_oracle(cfg):
    cls = SokobanSwitchReward if cfg.objective == "box_switches" else SokobanReward
    return cls(max_states=cfg.solver_max_states, max_seconds=cfg.solver_max_seconds,
               max_discovered_states=cfg.solver_max_discovered_states, workers=cfg.solver_workers)


def make_selector(cfg, oracle):
    return OneHotTanimotoExpectedImprovementSelector(reward_fn=oracle, config=ActiveLearningConfig(
        enabled=True, oracle_batch_size=cfg.oracle_batch, buffer_size=cfg.gp_buffer,
        onehot_max_len=100, onehot_alphabet=DISPLAY_ALPHABET, gp_device="cpu", gp_dtype="double",
        gp_hyperparameter_refit_interval=cfg.gp_refit_interval,
        acquisition="expected_improvement", local_penalization=cfg.local_penalization,
    ))


def make_adalead_selector(cfg, oracle):
    """The current protein AdaLead surrogate: fixed-scale one-hot Tanimoto GP."""
    return OneHotTanimotoExpectedImprovementSelector(reward_fn=oracle, config=ActiveLearningConfig(
        enabled=True, oracle_batch_size=cfg.oracle_batch, buffer_size=cfg.gp_buffer,
        onehot_max_len=100, onehot_alphabet=DISPLAY_ALPHABET[:7], gp_device="cpu", gp_dtype="double",
        gp_hyperparameter_refit_interval=cfg.gp_refit_interval, gp_jitter=1e-4,
        gp_outputscale=1.0, gp_learn_outputscale=False, gp_fit_mean_and_noise_only=True,
        acquisition="random", local_penalization=False,
    ))


def make_controller(cfg):
    return ScalarTrustRegion(config=TrustRegionConfig(
        kind="ScalarTrustRegion", eps=1e-6, log_every=0, mean_edits=cfg.fixed_masks,
        edits_init=cfg.fixed_masks, edits_min=cfg.min_masks, edits_max=cfg.max_masks,
        tau_success=cfg.success_tolerance, tau_failure=cfg.failure_tolerance, region_scalar=cfg.region_scale,
        region_shrink_scalar=cfg.region_shrink_scalar,
    ), seq_len=100, device=torch.device("cpu"))


def is_exact_solution(record):
    return record["status"] == "solved" and record.get("objective_status", "exact") == "exact"


def proposal_parents(cfg, archive, rng):
    """Choose exact archive parents; the sequence policy prefers the newest tie."""
    if cfg.parent_policy == "best":
        parents = [max(archive.values(), key=lambda r: (r["score"], r["query"]))]
        parent_ids = [0] * cfg.candidate_pool
    else:
        parents = sorted(archive.values(), key=lambda r: (-r["score"], r["key"]))[:cfg.archive_parents]
        parent_ids = torch.randint(len(parents), (cfg.candidate_pool,), generator=rng).tolist()
    boards = torch.tensor([parents[i]["canonical_board"] for i in parent_ids])
    return parents, parent_ids, boards


def record_scores(records):
    return torch.tensor([float("nan") if r["score"] is None else r["score"] for r in records])


def summarise(records):
    statuses = Counter(r["status"] for r in records)
    unique = {r["key"]: r for r in records}
    solved = sorted((r for r in unique.values() if is_exact_solution(r)),
                    key=lambda r: r["score"], reverse=True)
    top = solved[:10]
    distances = [sum(a != b for a, b in zip(x["canonical_board"], y["canonical_board"], strict=False))
                 for i, x in enumerate(top) for y in top[i + 1:]]
    def mean(field):
        values = [r[field] for r in top if r.get(field) is not None]
        return float(np.mean(values)) if values else None

    return {
        "queries": len(records), "unique_queried": len(unique), "status_counts": dict(statuses),
        "solvability": statuses["solved"] / len(records) if records else None,
        "unique_solved": sum(r["status"] == "solved" for r in unique.values()),
        "unique_exact_solved": len(solved),
        "objective_status_counts": dict(Counter(r.get("objective_status", r["status"]) for r in records)),
        "unresolved_objective": sum(r.get("objective_status") in {"bounded", "unknown"} for r in records),
        "cache_hits": sum(r["cache_hit"] for r in records),
        "best_effort": solved[0]["expanded_states"] if solved else None,
        "best_pushes": solved[0]["minimum_pushes"] if solved else None,
        "top10_count": len(top),
        "best_box_switches": solved[0].get("box_switches") if solved else None,
        "top10_mean_box_switches": mean("box_switches"),
        "top10_mean_effort": mean("expanded_states"),
        "top10_mean_pushes": mean("minimum_pushes"),
        "top10_mean_solution_pushes": mean("solution_pushes"),
        "top10_mean_pairwise_hamming": float(np.mean(distances)) if distances else None,
    }


def run_method(cfg, adapter, initial, initial_records, method, seed, output, *, selector_factory=make_selector):
    """Protocol runner; model, acquisition and controller are reused components.

    Initial queries are charged to every arm. Each selected proposal costs one
    query, including invalids, unknowns, and cached duplicates. All generated
    proposals are counted separately. The Hamming arm shares the adaptive
    controller, parents, acquisition and oracle with GenTuRBO, changing only the
    proposal operator to exact-distance public-tile mutations.
    """
    if method not in cfg.methods:
        raise ValueError("method not configured")
    if len(initial_records) != cfg.initial_size or len(initial) != cfg.initial_size:
        raise ValueError("initial boards and labels must match the charged initial budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    parent_rng = torch.Generator().manual_seed(seed + 20000)
    mutation_rng = torch.Generator().manual_seed(seed + 40000)
    adaptive = method in {"turdbo", "turdbo_sequence", "hamming_gp_ei"}
    archive = {}
    controller = make_controller(cfg)
    generated = cfg.initial_size
    adalead = GridAdaLead(cfg, seed=seed + 60000) if method == "adalead_gp_ei" else None
    official_adalead = OfficialAdaLead(cfg) if method == "adalead_official" else None
    adalead_batches = []
    evolution = (EvolutionPopulation(size=cfg.evolution_population, tournament=cfg.evolution_tournament,
                 crossover_rate=cfg.ga_crossover_rate, mutation_edits=cfg.ga_mutation_edits, seed=seed + 50000,
                 editable_region=cfg.ga_editable_region)
                 if method in {"grid_ga", "grid_mutation", "denoise_ea"} else None)
    with make_oracle(cfg) as oracle:
        # Labels are physically reused, but initialisation is charged in each arm.
        oracle.records = [{**r, "query": i + 1, "phase": "initial"} for i, r in enumerate(initial_records)]
        oracle.cache = {r["key"]: dict(r) for r in initial_records}
        if official_adalead is not None and selector_factory is make_selector:
            selector_factory = make_adalead_selector
        selector = selector_factory(cfg, oracle) if method in {"fixed", "turdbo", "turdbo_sequence", "adalead_gp_ei", "adalead_official", "hamming_gp_ei"} else None
        if selector is not None:
            selector.observe_oracle_batch(initial, record_scores(initial_records))
        if official_adalead is not None:
            official_adalead.observe(initial, record_scores(initial_records))
        if evolution is not None:
            evolution.observe(initial_records)
        for r in initial_records:
            if is_exact_solution(r):
                archive[r["key"]] = r
        if not archive:
            raise RuntimeError("no exactly scored solvable initial puzzle; retain seed rather than silently resampling")
        started = time.perf_counter()
        history = [{**summarise(oracle.records), "iteration": 0, "elapsed_seconds": 0.0,
                    "generated": generated, "mask_count": None}]
        write_json(output / "history.json", history)
        with (output / "queries.jsonl").open("w") as handle:
            for r in oracle.records:
                handle.write(json.dumps(r) + "\n")
            iteration = 0
            while len(oracle.records) < cfg.budget:
                iteration += 1
                count = min(cfg.oracle_batch, cfg.budget - len(oracle.records))
                best_before = max(r["score"] for r in archive.values())
                if method == "generator":
                    pool = adapter.complete(torch.full((count, 100), 7, dtype=torch.long, device=adapter.device)).cpu()
                    selected = list(range(count))
                    mask_count = 100
                    surrogate_scored = 0
                    new_candidates = len(pool)
                elif evolution is not None:
                    mask_count = cfg.fixed_masks if method == "denoise_ea" else None
                    pool = evolution.propose(count, method, adapter=adapter, mask_count=cfg.fixed_masks)
                    selected = list(range(count))
                    surrogate_scored = 0
                    new_candidates = len({board_key(row.numpy()) for row in pool} - set(oracle.cache))
                elif official_adalead is not None:
                    mask_count = None
                    pool = official_adalead.propose(selector.fit_surrogate_predictor(), count=count)
                    selected = list(range(len(pool)))
                    surrogate_scored = official_adalead.diagnostics['surrogate_query_count']
                    new_candidates = official_adalead.diagnostics['unique_generated_count']
                    adalead_batches.append(dict(iteration=iteration, **official_adalead.diagnostics))
                    write_json(output / 'adalead_batches.json', adalead_batches)
                else:
                    if adalead is not None:
                        mask_count = None
                        pool = adalead.propose(archive, selector.fit_surrogate_predictor())
                    else:
                        mask_count = cfg.fixed_masks if method == "fixed" else int(round(controller.config.mean_edits))
                        parents, parent_ids, parent_boards = proposal_parents(cfg, archive, parent_rng)
                        pool = (mutate_exact(parent_boards, mask_count, generator=mutation_rng)
                                if method == "hamming_gp_ei" else adapter.propose(parent_boards, mask_count).cpu())
                    # Remove known/equivalent candidates using identity only; no oracle pre-screening.
                    seen = set(oracle.cache)
                    eligible = []
                    for i, row in enumerate(pool):
                        key = board_key(row.numpy())
                        if key not in seen:
                            eligible.append(i)
                            seen.add(key)
                    selection = selector.select(pool[eligible], budget=count) if eligible else None
                    selected = [eligible[i] for i in selection.selected_indices] if selection else []
                    new_candidates = len(eligible)
                    surrogate_scored = sum(v is not None for v in getattr(selection, "surrogate_mean", []))
                    # A collapsed pool still consumes budget, and is visible in the trace.
                    selected += [i for i in range(len(pool)) if i not in selected][:count - len(selected)]
                generated += (official_adalead.diagnostics['surrogate_mutant_query_count']
                              if official_adalead is not None else len(pool))
                before = len(oracle.records)
                chosen = pool[selected]
                scores = oracle.score_batch(chosen)
                if selector is not None:
                    selector.observe_oracle_batch(chosen, scores)
                if official_adalead is not None:
                    official_adalead.observe(chosen, scores)
                for pool_index, record in zip(selected, oracle.records[before:], strict=True):
                    record.update(phase="search", iteration=iteration, mask_count=mask_count)
                    if official_adalead is not None:
                        record.update(official_adalead.selected_metadata[pool_index])
                        record['raw_hamming_distance'] = int((pool[pool_index] != torch.tensor(record['raw_parent_board'])).sum())
                    if method in {"hamming_gp_ei", "turdbo_sequence"}:
                        record.update(parent_key=parents[parent_ids[pool_index]]["key"],
                                      parent_score=parents[parent_ids[pool_index]]["score"],
                                      parent_query=parents[parent_ids[pool_index]]["query"],
                                      raw_parent_board=parent_boards[pool_index].tolist(),
                                      raw_hamming_distance=int((pool[pool_index] != parent_boards[pool_index]).sum()))
                    handle.write(json.dumps(record) + "\n")
                    if is_exact_solution(record):
                        archive[record["key"]] = record
                handle.flush()
                if evolution is not None:
                    evolution.observe(oracle.records[before:])
                best_after = max(r["score"] for r in archive.values())
                if adaptive:
                    controller.update(best_time=None, improved=best_after > best_before + 1e-6)
                progress = {**summarise(oracle.records), "iteration": iteration,
                            "elapsed_seconds": time.perf_counter() - started, "generated": generated,
                            "mask_count": mask_count, "new_candidates": new_candidates,
                            "surrogate_scored": surrogate_scored,
                            "next_mask_count": controller.config.mean_edits if adaptive else mask_count}
                history.append(progress)
                write_json(output / "history.json", history)
                print(json.dumps({"seed": seed, "method": method, **progress}), flush=True)
        result = {**history[-1], "method": method, "seed": seed, "status": "completed", "objective": cfg.objective,
                  "oracle": oracle.metadata(), "initial_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                  "timing_note": "search elapsed excludes shared initialisation; every arm is charged its initial queries",
                  "repair": "structural_inventory_only" if method == "grid_ga" else "none", "generator_frozen": True,
                  "search_policy": {"parent_policy": cfg.parent_policy, "archive_parents": cfg.archive_parents,
                                    "candidate_pool": cfg.candidate_pool, "oracle_batch": cfg.oracle_batch,
                                    "local_penalization": cfg.local_penalization, "gp_buffer": cfg.gp_buffer,
                                    "gp_refit_interval": cfg.gp_refit_interval},
                  "proposal_uses_generator": method not in {"grid_ga", "grid_mutation", "adalead_gp_ei", "adalead_official", "hamming_gp_ei"},
                  "hamming": ({"distance": "exact, raw child versus parent before canonicalisation",
                               "public_tile_values": list(range(7)), "editable_cells": 100,
                               "position_sampling": "uniform without replacement",
                               "replacement_sampling": "uniform among non-current public tiles",
                               "prescreening": "identity only; no feasibility or solver filtering",
                               "kernel": "one-hot Tanimoto", "acquisition": "expected_improvement",
                               "local_penalization": cfg.local_penalization, "adaptive_radius": True}
                              if method == "hamming_gp_ei" else None),
                  "adalead": (official_adalead.metadata() if official_adalead is not None
                              else adalead.metadata() if adalead is not None else None),
                  "evolution": evolution.metadata(method) if evolution is not None else None}
        if evolution is not None:
            result['search_policy'] = dict(parent_policy='tournament',
                                          population_size=cfg.evolution_population,
                                          tournament_size=cfg.evolution_tournament,
                                          oracle_batch=cfg.oracle_batch,
                                          offspring_per_update=cfg.oracle_batch,
                                          acquisition='none', surrogate_used=False)
        if official_adalead is not None:
            result['search_policy'] = dict(parent_policy='signed_relative_threshold_resize',
                                          acquisition='top_surrogate_mean', local_penalization=False,
                                          surrogate_queries_per_batch=cfg.adalead_surrogate_queries,
                                          oracle_batch=cfg.oracle_batch, gp_buffer=cfg.gp_buffer,
                                          gp_refit_interval=cfg.gp_refit_interval, gp_jitter=1e-4,
                                          gp_outputscale=1.0, gp_fit_mean_and_noise_only=True)
            result['surrogate_queries_total'] = sum(r['surrogate_query_count'] for r in adalead_batches)
        write_json(output / "report.json", result)
        top = sorted(archive.values(), key=lambda r: (-r["score"], r["key"]))[:10]
        write_json(output / "top10.json", top)
        (output / "top10.txt").write_text("\n\n".join(
            f"; {i} objective={cfg.objective} score={r['score']} minimum_pushes={r['minimum_pushes']} "
            f"witness_pushes={r.get('solution_pushes')}\n{decode_board(np.array(r['canonical_board']))}"
            for i, r in enumerate(top)) + "\n")
        return result


def run_locality(cfg, adapter, initial_records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    parents = [r for r in initial_records if is_exact_solution(r)][:cfg.locality_parents]
    if not parents:
        raise RuntimeError("no solved locality parents")
    parent_boards = torch.tensor([r["canonical_board"] for r in parents]).repeat_interleave(cfg.locality_repeats, dim=0)
    parent_keys = [r["key"] for r in parents for _ in range(cfg.locality_repeats)]
    baseline = np.array([r["score"] for r in parents for _ in range(cfg.locality_repeats)])
    reports = []
    with make_oracle(cfg) as oracle:
        for k in cfg.locality_masks:
            started = time.perf_counter()
            boards = adapter.propose(parent_boards, k).cpu()
            before = len(oracle.records)
            scores = oracle.score_batch(boards).numpy()
            records = oracle.records[before:]
            for r, parent in zip(records, parent_keys, strict=False):
                r.update(mask_count=k, parent_key=parent)
            changed = np.array([r["key"] != key for r, key in zip(records, parent_keys, strict=False)])
            solved = np.array([r["status"] == "solved" for r in records])
            report = {**summarise(records), "mask_count": k, "parents": len(parents),
                      "repeats": cfg.locality_repeats, "distinct_from_parent_fraction": float(changed.mean()),
                      "distinct_solvable_fraction": float((changed & solved).mean()),
                      "improvement_fraction": float((scores > baseline + 1e-6).mean()),
                      "mean_actual_cell_edits": float((boards != parent_boards).sum(1).float().mean()),
                      "elapsed_seconds": time.perf_counter() - started}
            reports.append(report)
            write_json(output / "report.json", reports)
            print(json.dumps({"locality": report}), flush=True)
        (output / "queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in oracle.records))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/search/sokoban/genturbo.json", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("initial", "locality", "search", "all"), default="all")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    parser.add_argument("--initial-from", type=Path, nargs="+", help="Reuse saved starts, with exactly one source per seed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    values = load_config(args.config)
    for key in ("checkpoint", "output"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.seed is not None:
        values["seeds"] = [args.seed]
    cfg = PilotConfig(**values)
    checkpoint_sha = sha256(Path(cfg.checkpoint)) if args.initial_from else None
    saved = load_initial_sources(cfg, args.initial_from, checkpoint_sha) if args.initial_from else {}
    if args.dry_run:
        print(json.dumps(asdict(cfg), indent=2))
        return
    root = Path(cfg.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    restored = restore_process_from_checkpoint(cfg.checkpoint, device_override=cfg.device)
    if restored.config.get("process") != "sokoban_masked_diffusion":
        raise ValueError("expected the paper-style Sokoban checkpoint")
    process = restored.process
    write_json(root / "config.json", asdict(cfg))
    write_json(root / "provenance.json", {
        "checkpoint_sha256": sha256(Path(cfg.checkpoint)), "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name() if cfg.device == "cuda" else None,
        "execution_device": cfg.device, "sampling_precision": cfg.precision, "cpu_threads": args.threads,
        "initial_sampling": "One unfiltered batch per seed; random generator seed = search seed + 10000. "
                            "CPU and CUDA draws are not bitwise interchangeable; reuse saved starts across methods.",
        "initial_sources": [str(path.resolve()) for path in args.initial_from] if args.initial_from else [],
        "initialisation_reused": bool(saved),
        "source_sha256": {str(p): sha256(p) for p in [Path(__file__), Path(__file__).with_name("search.py"),
                                                     Path(__file__).with_name("planning.py"),
                                                     Path(__file__).with_name("evolution.py")]},
        "implementation": "shared ScalarTrustRegion and OneHotTanimotoExpectedImprovementSelector with Sokoban protocol runner",
    })
    results = []
    for seed in cfg.seeds:
        seed_root = root / f"seed{seed}"
        seed_root.mkdir()
        started = time.perf_counter()
        if saved:
            initial, initial_records, hashes, source = saved[seed]
            for name in hashes:
                shutil.copyfile(source / f"seed{seed}" / name, seed_root / name)
            write_json(seed_root / "initial_reuse.json", {
                "source": str(source / f"seed{seed}"), "sha256": hashes, "physical_initial_solves": 0,
                "timing_note": "Copied initialisation timing is historical; no new initial generation or scoring.",
            })
        else:
            adapter = SokobanRolloutAdapter(process, precision=cfg.precision, seed=seed + 10000)
            initial = adapter.complete(torch.full((cfg.initial_size, 100), 7, dtype=torch.long, device=adapter.device)).cpu()
            with make_oracle(cfg) as oracle:
                oracle.score_batch(initial)
                initial_records = oracle.records
            np.save(seed_root / "initial.npy", initial.numpy())
            write_json(seed_root / "initial_records.json", initial_records)
            write_json(seed_root / "initial_report.json", {**summarise(initial_records), "elapsed_seconds": time.perf_counter() - started})
        print(json.dumps({"seed": seed, "phase": "initial_complete", "queries": len(initial_records),
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
        if args.mode in ("locality", "all"):
            run_locality(cfg, SokobanRolloutAdapter(process, precision=cfg.precision, seed=seed + 30000),
                         initial_records, seed_root / "locality")
        if args.mode in ("search", "all"):
            for method in cfg.methods:
                adapter = SokobanRolloutAdapter(process, precision=cfg.precision, seed=seed + 40000)
                results.append(run_method(cfg, adapter, initial, initial_records, method, seed, seed_root / method))
                write_json(root / "results.json", results)
    write_json(root / "complete.json", {"status": "completed", "seeds": cfg.seeds, "mode": args.mode, "runs": len(results)})


if __name__ == "__main__":
    main()
