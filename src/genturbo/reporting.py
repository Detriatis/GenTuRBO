"""Local, read-only analysis of completed experiments and reference tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from genturbo.io import iter_jsonl
from genturbo.protein_reporting import incumbent_values, row_score, write_records

PROTEIN_METHODS = ("kermut", "tanimoto", "hamming_gp", "holo_ga", "adalead", "bo_evo", "lambo2")
PROTEIN_TASKS = ("gfp_medium", "gfp_hard", "aav_medium", "aav_hard")
SOKOBAN_METHODS = ("turdbo_sequence", "hamming_gp_ei", "generator", "grid_ga", "adalead_official")


def _configure_ehrlich(root: Path):
    from genturbo.figures import ehrlich
    ehrlich.ROOT = root
    ehrlich.DATA_PREFIX = root / "figures/data/ehrlich_sequence_baselines_n20"
    ehrlich.FROZEN_ENDPOINTS = ehrlich.DATA_PREFIX.with_name(ehrlich.DATA_PREFIX.name + "_per_seed.csv")
    ehrlich.TRAJECTORY_DATA = ehrlich.DATA_PREFIX.with_name(ehrlich.DATA_PREFIX.name + "_trajectories.csv")
    ehrlich.OUTPUT = root / "figures/ehrlich_sequence_baselines_best_so_far_main"
    return ehrlich


def protein_tables(runs: Path, output: Path) -> None:
    """Read exactly charged runs; preserve external-method batch boundaries."""
    curves = {}
    endpoints = []
    for task in PROTEIN_TASKS:
        for method in PROTEIN_METHODS:
            for seed in range(20):
                folder = runs / task / method / f"seed{seed}"
                manifest = json.loads((folder / "run_manifest.json").read_text())
                if manifest.get("status") != "completed":
                    raise ValueError(f"Incomplete run: {folder}")
                initial = list(iter_jsonl(folder / "initial_batch.jsonl.gz"))
                selected = list(iter_jsonl(folder / "samples.jsonl.gz"))
                if len(initial) != 128 or len(selected) != 1872:
                    raise ValueError(f"Expected 128 initial + 1,872 selected labels: {folder}")
                if any(r.get("oracle_evaluated") is not True for r in selected):
                    raise ValueError(f"Unpaid selected row: {folder}")
                rows = initial + selected
                if method in ("kermut", "tanimoto", "hamming_gp"):
                    scores = [float(r["score"]) for r in rows]
                    calls = np.arange(128, 2001, 8)
                else:
                    scores = [row_score(r) for r in rows]
                    size = 468 if method == "holo_ga" else 8
                    calls = np.arange(128, 2001, size)
                    # A recorded batch identifier must not expose a partial generation.
                    if all("batch_id" in r for r in selected):
                        from itertools import groupby
                        sizes = [len(list(g)) for _, g in groupby(selected, lambda r: r["batch_id"])]
                        if any(n != size for n in sizes):
                            raise ValueError(f"Unexpected {method} batch boundary: {folder}")
                values = incumbent_values(scores, calls)
                curves[task, method, seed] = calls, values
                endpoints.append(dict(task=task, method=method, seed=seed, oracle_total=2000,
                                      initial_best=float(values[0]), final_best=float(values[-1])))
    trajectories = []
    for task in PROTEIN_TASKS:
        for method in PROTEIN_METHODS:
            calls = curves[task, method, 0][0]
            if any(not np.array_equal(calls, curves[task, method, seed][0]) for seed in range(20)):
                raise ValueError(f"Unequal grids: {task}/{method}")
            matrix = np.asarray([curves[task, method, seed][1] for seed in range(20)])
            for call, avg, sd in zip(calls, matrix.mean(0), matrix.std(0, ddof=1), strict=True):
                trajectories.append(dict(task=task, method=method, eval_idx=int(call), n_seeds=20,
                                         mean=float(avg), sample_std=float(sd)))
    write_records(output / "protein_comparison_per_seed.csv", endpoints)
    write_records(output / "protein_comparison_trajectories.csv", trajectories)


def sokoban_tables(runs: Path, output: Path) -> None:
    """Aggregate exact best scores directly from fresh histories and witnesses."""
    output.mkdir(parents=True, exist_ok=True)
    records, histories, winners = {}, {}, {}
    for path in sorted(runs.rglob("report.json")):
        row = json.loads(path.read_text())
        method, seed = row.get("method"), row.get("seed")
        if method not in SOKOBAN_METHODS:
            continue
        key = method, int(seed)
        if key in records:
            raise ValueError(f"Duplicate method/seed: {key}")
        if row.get("status") != "completed" or row.get("queries") != 2000:
            raise ValueError(f"Incomplete 2,000-query run: {path}")
        history = json.loads(path.with_name("history.json").read_text())
        top = json.loads(path.with_name("top10.json").read_text())
        if not top or any(t.get("objective_status") != "exact" for t in top):
            raise ValueError(f"Non-exact or missing winner archive: {path}")
        if history[0]["queries"] != 64 or history[-1]["queries"] != 2000:
            raise ValueError(f"Unexpected history endpoints: {path}")
        if any(b["queries"] <= a["queries"] or b["best_box_switches"] < a["best_box_switches"]
               for a, b in zip(history, history[1:])):
            raise ValueError(f"Nonmonotonic history: {path}")
        best = max(top, key=lambda t: t["score"])
        if best["score"] != history[-1]["best_box_switches"] or best["score"] != row["best_box_switches"]:
            raise ValueError(f"Winner/history mismatch: {path}")
        records[key], histories[key] = row, history
        if method not in winners or best["score"] > winners[method]["score"]:
            winners[method] = best
    required = {(m, s) for m in SOKOBAN_METHODS for s in range(20)}
    if set(records) != required:
        raise ValueError(f"Expected 100 current Sokoban cells; missing {sorted(required - set(records))}")
    for seed in range(20):
        if len({records[m, seed]["initial_sha256"] for m in SOKOBAN_METHODS}) != 1:
            raise ValueError(f"Shared initial archive differs at seed {seed}")
        if len({histories[m, seed][0]["best_box_switches"] for m in SOKOBAN_METHODS}) != 1:
            raise ValueError(f"Shared initial score differs at seed {seed}")
    queries = sorted({h["queries"] for history in histories.values() for h in history})
    curves = {m: [[next(h["best_box_switches"] for h in reversed(histories[m, s]) if h["queries"] <= q)
                   for q in queries] for s in range(20)] for m in SOKOBAN_METHODS}
    data = dict(metric="best_box_switches", seeds=list(range(20)), queries=queries, curves=curves,
                endpoints={m: [histories[m, s][-1]["best_box_switches"] for s in range(20)]
                           for m in SOKOBAN_METHODS})
    for name, value in (("figure_data.json", data), ("winner.json", winners["turdbo_sequence"]),
                        ("generator_winner.json", winners["generator"])):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    per_seed = [dict(method=m, seed=s, queries=2000, initial_best=histories[m, s][0]["best_box_switches"],
                     final_best=records[m, s]["best_box_switches"],
                     secondary_top10_mean=records[m, s]["top10_mean_box_switches"])
                for m in SOKOBAN_METHODS for s in range(20)]
    write_records(output / "best_so_far_per_seed.csv", per_seed)


def _reference_inputs(source: Path, root: Path) -> None:
    """Copy only numerical inputs; never copy manuscript or rendered graphics."""
    for relative in ("figures/data", "figures/sokoban"):
        directory = source / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing reference directory: {directory}")
        target = root / relative
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in (".csv", ".json"):
                shutil.copy2(path, target / path.name)


def render(suite: str, root: Path) -> None:
    from genturbo.figures import comparison, style
    style.apply_style()
    comparison.ROOT, comparison.DATA, comparison.FIGURES = root, root / "figures/data", root / "figures"
    if suite in ("ehrlich", "all"):
        _configure_ehrlich(root).render()
    if suite in ("protein", "all"):
        comparison.protein_comparison()
    if suite in ("sokoban", "all"):
        from genturbo.figures import sokoban
        sokoban.render(root)
    if suite in ("supporting", "all"):
        from genturbo.figures import neighbourhood, cone
        comparison.ehrlich_locality()
        comparison.ehrlich_kernels()
        neighbourhood.ROOT = root
        neighbourhood.main([])
        cone.main(["--input-dir", str(root / "figures/data/visitation"),
                   "--output-prefix", str(root / "figures/ehrlich_infeasible_cone_half_uniform"),
                   "--clean", "--no-colourbars", "--no-infeasible-bars",
                   "--cutoff-multiple", "0.5", "--paper-width", "6.075"])
    if suite == "all":
        from genturbo.figures import tables
        tables.render(root)


def handle_report(args: argparse.Namespace) -> int:
    root = args.output.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Choose a fresh report output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "figures/data").mkdir(parents=True)
    reference = args.reference
    if reference is None and args.artifacts is not None and args.runs is None:
        reference = args.artifacts / "reference/paper"
    if reference is not None:
        if args.runs is not None:
            raise ValueError("Choose fresh runs or reference replay, not both")
        _reference_inputs(reference, root)
    elif args.runs is not None:
        runs = args.runs.resolve()
        if args.suite in ("ehrlich", "all"):
            source = runs / "ehrlich/l256_q4_k8" if (runs / "ehrlich").exists() else runs
            _configure_ehrlich(root).rebuild_trajectory_data(source, source)
        if args.suite in ("protein", "all"):
            protein_tables(runs / "protein" if (runs / "protein").exists() else runs, root / "figures/data")
        if args.suite in ("sokoban", "all"):
            sokoban_tables(runs / "sokoban" if (runs / "sokoban").exists() else runs, root / "figures/sokoban")
        if args.suite in ("supporting", "all"):
            supporting = runs / "supporting/figure_inputs"
            for name in ("ehrlich_locality_calibration.csv", "ehrlich_kernel_trajectories.csv", "ehrlich_neighbourhood_source.json"):
                shutil.copy2(supporting / name, root / "figures/data" / name)
    else:
        raise ValueError("Supply --runs, --reference, or --artifacts")
    if args.suite in ("supporting", "all"):
        visitation = (args.artifacts / "runs/ehrlich_radial_search_v1/hamming_ball_visitation"
                      if reference is not None and args.artifacts is not None
                      else args.runs / "supporting/visitation" if args.runs is not None
                      else reference / "figures/data/visitation")
        target = root / "figures/data/visitation"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("hamming_ball_visitation.csv", "hamming_ball_visitation.json"):
            if not (visitation / name).is_file():
                raise FileNotFoundError("The visitation figure needs the frozen numerical inputs; supply --artifacts")
            shutil.copy2(visitation / name, target / name)
    render(args.suite, root)
    (root / "report.json").write_text(json.dumps({"suite": args.suite,
        "input_kind": "reference_replay" if reference is not None else "fresh_runs",
        "no_new_oracle_calls": True}, indent=2) + "\n")
    print(f"Generated report: {root}")
    return 0


def add_parser(subparsers):
    parser = subparsers.add_parser("report", help="Recreate tables and figures from runs or reference inputs")
    parser.add_argument("suite", choices=("ehrlich", "protein", "sokoban", "supporting", "all"))
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--reference", type=Path, help="Reference numerical root containing figures/data and figures/sokoban")
    parser.add_argument("--artifacts", type=Path, help="Verified artifact root; use its reference/paper tables")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=handle_report)
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(required=True)
    add_parser(subs)
    return handle_report(parser.parse_args(["report", *(argv or [])]))
