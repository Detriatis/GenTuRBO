#!/usr/bin/env python3
"""Render the twenty-seed Ehrlich baseline trajectories used in the manuscript."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import fmean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from genturbo.figures.style import (COLOURS, LINE_STYLES, COLUMN_WIDTH, PAGE_WIDTH,
                          apply_style, style_axis, legend_above, save_figure)


ROOT = Path(__file__).resolve().parents[1]
DATA_PREFIX = ROOT / "figures/data/ehrlich_sequence_baselines_n20"
FROZEN_ENDPOINTS = DATA_PREFIX.with_name(DATA_PREFIX.name + "_per_seed.csv")
TRAJECTORY_DATA = DATA_PREFIX.with_name(DATA_PREFIX.name + "_trajectories.csv")
OUTPUT = ROOT / "figures/ehrlich_sequence_baselines_best_so_far_main"
SEED_COUNT = 20
T_CRITICAL_95 = 2.093024054408263  # Student t(19), 97.5th percentile.

ALL_ARMS = (
    "unconditional",
    "no_tr_no_gp",
    "tr_no_gp",
    "no_tr_gp",
    "tr_gp",
    "oracle_upper_bound",
    "unconditional_gp",
    "hamming_random",
    "hamming_gp",
    "adalead_official",
    "bo_evo",
    "lambo2",
    "holo_ga",
)

PLOT_ARMS = (
    "tr_gp",
    "hamming_gp",
    "unconditional_gp",
    "adalead_official",
    "bo_evo",
    "lambo2",
    "holo_ga",
)

LABELS = {
    "tr_gp": "GenTuRBO",
    "unconditional_gp": "Uncond. + GP/EI",
    "hamming_gp": "Hamming + GP/EI",
    "adalead_official": "AdaLead",
    "bo_evo": "BO-EVO",
    "lambo2": "LaMBO-2 (online)",
    "holo_ga": "Holo GA",
    "unconditional": "Unconditional",
    "no_tr_no_gp": "RT + Random",
    "tr_no_gp": "TR + Random",
    "no_tr_gp": "RT + GP/EI",
    "oracle_upper_bound": "Privileged oracle",
    "hamming_random": "Hamming + Random",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="Optional l256_q4_k8 run directory used to rebuild the frozen trajectory CSV.",
    )
    parser.add_argument("--adalead-root", type=Path,
                        help="Separate l256_q4_k8 directory containing the faithful adalead_official cohort.")
    return parser.parse_args()


def score_for_analysis(row: dict[str, object]) -> float:
    for key in ("score_for_analysis", "diagnostic_score", "score"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def read_jsonl_gz(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_trajectory(runs_root: Path, arm: str, seed: int) -> list[float]:
    run_dir = runs_root / arm / f"seed{seed}"
    for name in ("run_manifest.json", "scientific_completion.json"):
        if json.loads((run_dir / name).read_text())["status"] != "completed":
            raise ValueError(f"{arm} seed {seed} is not complete")
    sample_rows = read_jsonl_gz(run_dir / "samples.jsonl.gz")
    if arm == "unconditional":
        rows = sample_rows
    else:
        initial_rows = read_jsonl_gz(run_dir / "initial_batch.jsonl.gz")
        if len(initial_rows) != 64:
            raise ValueError(f"{arm} seed {seed} has {len(initial_rows)} initial rows")
        rows = [*initial_rows, *sample_rows]

    if len(rows) != 2000:
        raise ValueError(f"{arm} seed {seed} has {len(rows)} paid evaluation rows")
    if any(row.get("oracle_evaluated") is not True for row in rows):
        raise ValueError(f"{arm} seed {seed} contains an unpaid evaluation")

    incumbent = 0.0
    trajectory: list[float] = []
    for row in rows:
        incumbent = max(incumbent, score_for_analysis(row))
        trajectory.append(incumbent)
    if arm == "holo_ga":
        # Holo evaluates a whole generation together. Reveal improvements only
        # when all 484 evaluations in that generation have been charged.
        for start in range(64, 2000, 484):
            trajectory[start:start + 483] = [trajectory[start - 1]] * 483
    return trajectory


def load_frozen_endpoints(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    frozen: dict[tuple[str, int], tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frozen[(row["arm"], int(row["seed"]))] = (
                float(row["final_best"]),
                float(row["full_budget_auc"]),
            )
    return frozen


def rebuild_trajectory_data(runs_root: Path, adalead_root: Path, *, reference: Path | None = None) -> None:
    frozen = load_frozen_endpoints(reference) if reference is not None else {}
    trajectories: dict[tuple[str, int], list[float]] = {}
    records = []
    for arm in ALL_ARMS:
        for seed in range(SEED_COUNT):
            trajectory = load_trajectory(adalead_root if arm == "adalead_official" else runs_root, arm, seed)
            observed_final = trajectory[-1]
            observed_auc = fmean(trajectory)
            if (arm, seed) in frozen:
                expected_final, expected_auc = frozen[(arm, seed)]
                if not math.isclose(observed_final, expected_final, abs_tol=1e-12):
                    raise ValueError(f"{arm} seed {seed}: changed historical endpoint")
                if not math.isclose(observed_auc, expected_auc, abs_tol=1e-12):
                    raise ValueError(f"{arm} seed {seed}: changed historical AUC")
            trajectories[(arm, seed)] = trajectory
            records.append(dict(condition="l256_q4_k8", arm=arm, seed=seed,
                                final_best=observed_final, full_budget_auc=observed_auc))
    write_csv(FROZEN_ENDPOINTS, records)
    write_statistics(records)

    with TRAJECTORY_DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "arm",
                "oracle_evaluations",
                "n",
                "best_so_far_mean",
                "best_so_far_sample_std",
            ),
        )
        writer.writeheader()
        for arm in ALL_ARMS:
            for index in range(2000):
                values = [trajectories[(arm, seed)][index] for seed in range(SEED_COUNT)]
                writer.writerow(
                    {
                        "arm": arm,
                        "oracle_evaluations": index + 1,
                        "n": SEED_COUNT,
                        "best_so_far_mean": fmean(values),
                        "best_so_far_sample_std": stdev(values),
                    }
                )


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_statistics(records: list[dict]) -> None:
    metrics = ("final_best", "full_budget_auc")
    lookup = {(row["arm"], row["seed"]): row for row in records}
    summary = []
    for arm in ALL_ARMS:
        row = dict(arm=arm, seed_count=SEED_COUNT, status="complete")
        for metric in metrics:
            values = [lookup[arm, seed][metric] for seed in range(SEED_COUNT)]
            row[metric + "_mean"] = fmean(values)
            row[metric + "_sample_std"] = stdev(values)
        summary.append(row)
    write_csv(DATA_PREFIX.with_name(DATA_PREFIX.name + "_summary.csv"), summary)

    pairs = {'tr_without_gp': ('tr_no_gp', 'no_tr_no_gp'), 'gp_without_tr': ('no_tr_gp', 'no_tr_no_gp'), 'tr_with_gp': ('tr_gp', 'no_tr_gp'), 'gp_with_tr': ('tr_gp', 'tr_no_gp'), 'combined_vs_neither': ('tr_gp', 'no_tr_no_gp'), 'oracle_headroom': ('oracle_upper_bound', 'tr_gp'), 'conditional_vs_unconditional_gp': ('no_tr_gp', 'unconditional_gp'), 'generator_tr_vs_hamming_random': ('tr_no_gp', 'hamming_random'), 'generator_tr_gp_vs_hamming_gp': ('tr_gp', 'hamming_gp'), 'hamming_gp_effect': ('hamming_gp', 'hamming_random'), 'generator_tr_gp_vs_adalead_official': ('tr_gp', 'adalead_official'), 'generator_tr_gp_vs_bo_evo': ('tr_gp', 'bo_evo'), 'generator_tr_gp_vs_lambo2': ('tr_gp', 'lambo2'), 'generator_tr_gp_vs_holo_ga': ('tr_gp', 'holo_ga')}
    contrasts = []
    for name, (treatment, reference) in pairs.items():
        for metric in metrics:
            deltas = [lookup[treatment, seed][metric] - lookup[reference, seed][metric]
                      for seed in range(SEED_COUNT)]
            average, sd = fmean(deltas), stdev(deltas)
            radius = T_CRITICAL_95 * sd / math.sqrt(SEED_COUNT)
            contrasts.append(dict(seed_count=SEED_COUNT, status="complete", contrast=name,
                                  treatment_arm=treatment, reference_arm=reference, metric=metric,
                                  mean_delta=average, std_delta=sd,
                                  ci95_low=average-radius, ci95_high=average+radius,
                                  wins=sum(x > 0 for x in deltas), ties=sum(x == 0 for x in deltas),
                                  losses=sum(x < 0 for x in deltas)))
    write_csv(DATA_PREFIX.with_name(DATA_PREFIX.name + "_contrasts.csv"), contrasts)


def load_plot_data() -> dict[str, tuple[list[int], list[float], list[float]]]:
    grouped: dict[str, tuple[list[int], list[float], list[float]]] = {
        arm: ([], [], []) for arm in ALL_ARMS
    }
    with TRAJECTORY_DATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm = row["arm"]
            if int(row["n"]) != SEED_COUNT:
                raise ValueError(f"Unexpected seed count for {arm}")
            calls, means, stds = grouped[arm]
            calls.append(int(row["oracle_evaluations"]))
            means.append(float(row["best_so_far_mean"]))
            stds.append(float(row["best_so_far_sample_std"]))
    for arm, (calls, _, _) in grouped.items():
        if calls != list(range(1, 2001)):
            raise ValueError(f"trajectory CSV is incomplete for {arm}")
    return grouped


def render() -> None:
    apply_style()
    grouped = load_plot_data()
    figure, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.8), layout="constrained")

    for arm in PLOT_ARMS:
        calls, means, stds = grouped[arm]
        colour = COLOURS[arm]
        axis.fill_between(
            calls,
            [mean - std for mean, std in zip(means, stds, strict=True)],
            [mean + std for mean, std in zip(means, stds, strict=True)],
            step="post",
            color=colour,
            alpha=0.14 if arm == "tr_gp" else 0.08,
            linewidth=0,
        )
        axis.plot(
            calls,
            means,
            drawstyle="steps-post",
            color=colour,
            linestyle=LINE_STYLES[arm],
            linewidth=1.7 if arm == "tr_gp" else 1.15,
            label=LABELS[arm],
            zorder=4 if arm == "tr_gp" else 3,
        )

    axis.axvline(64, color="#555555", linestyle=":", linewidth=1.0, zorder=2)
    axis.set_xlim(0, 2000)
    upper = max(mean + sd for arm in PLOT_ARMS
                for mean, sd in zip(grouped[arm][1], grouped[arm][2]))
    axis.set_ylim(0, max(0.24, math.ceil(upper / 0.05) * 0.05))
    axis.set_xticks((0, 500, 1000, 1500, 2000))
    axis.set_xlabel("Charged oracle evaluations", fontsize=8)
    axis.set_ylabel("Best reward so far", fontsize=8)
    axis.tick_params(axis="both", labelsize=7)
    style_axis(axis)
    legend_above(figure, [axis], ncol=2)
    save_figure(figure, OUTPUT)

    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.35))
    figure.subplots_adjust(left=0.075, right=0.96, bottom=0.145, top=0.72, wspace=0.28)
    panels = (
        ("Generator and component controls", ("tr_gp", "tr_no_gp", "no_tr_gp",
                                              "no_tr_no_gp", "unconditional", "oracle_upper_bound")),
        ("Sequence-search baselines",
         ("tr_gp", *ALL_ARMS[6:])),
    )
    for index, (axis, (title, arms)) in enumerate(zip(axes, panels, strict=True)):
        for arm in arms:
            calls, means, stds = grouped[arm]
            axis.fill_between(calls, [m-s for m, s in zip(means, stds)],
                              [m+s for m, s in zip(means, stds)], step="post",
                              color=COLOURS[arm], alpha=0.14 if arm == "tr_gp" else 0.08, linewidth=0)
            axis.plot(calls, means, drawstyle="steps-post", color=COLOURS[arm],
                      linestyle=LINE_STYLES.get(arm, "--" if arm == "oracle_upper_bound" else "-"),
                      linewidth=1.7 if arm == "tr_gp" else 1.15, label=LABELS[arm])
        upper = max(m+s for arm in arms for m, s in zip(grouped[arm][1], grouped[arm][2]))
        axis.set(xlim=(0, 2000), ylim=(0, math.ceil(upper / 0.05) * 0.05),
                 title=f"({chr(97 + index)}) {title}", xlabel="Charged oracle evaluations", ylabel="Best reward so far")
        axis.set_xticks((0, 500, 1000, 1500, 2000))
        axis.axvline(64, color="#555555", linestyle=":", linewidth=1)
        style_axis(axis)
        position = axis.get_position()
        legend_above(figure, [axis], ncol=2,
                     anchor=(position.x0 + position.width / 2, 1.0))
    save_figure(figure, ROOT / "figures/ehrlich_sequence_baselines_best_so_far")


def main() -> None:
    args = parse_args()
    if args.runs_root is not None:
        if args.adalead_root is None:
            raise ValueError("Rebuilding requires --adalead-root for the current faithful cohort")
        rebuild_trajectory_data(args.runs_root, args.adalead_root)
    if not TRAJECTORY_DATA.exists():
        raise FileNotFoundError(
            f"{TRAJECTORY_DATA} does not exist; supply --runs-root to build it"
        )
    render()


if __name__ == "__main__":
    main()
