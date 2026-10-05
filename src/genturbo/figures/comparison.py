#!/usr/bin/env python3
"""Style-only redraws from saved publication tables; no models or experiments."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from genturbo.figures.style import (COLOURS, LINE_STYLES, PAGE_WIDTH, apply_style,
                          style_axis, legend_above, save_figure)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures/data"
FIGURES = ROOT / "figures"
TASKS = ("gfp_medium", "gfp_hard", "aav_medium", "aav_hard")
TASK_LABELS = {task: task.split("_")[0].upper() + "-" + task.split("_")[1].title()
               for task in TASKS}


def read_rows(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_band(axis, calls, centre, low, high, method, label):
    colour = COLOURS[method]
    axis.fill_between(calls, low, high, step="post", color=colour,
                      alpha=0.14 if method in ("tr_gp", "kermut") else 0.08, linewidth=0)
    axis.step(calls, centre, where="post", color=colour, linestyle=LINE_STYLES[method],
              linewidth=1.7 if method in ("tr_gp", "kermut", "tanimoto") else 1.15,
              label=label, zorder=4 if method in ("tr_gp", "kermut") else 3)


def protein_comparison():
    rows = read_rows("protein_comparison_trajectories.csv")
    methods = ("kermut", "tanimoto", "hamming_gp", "holo_ga", "adalead", "bo_evo", "lambo2")
    labels = ("GenTuRBO (Kermut)", "GenTuRBO (Tanimoto)", "Hamming + GP/EI", "Holo GA", "AdaLead", "BO-EVO", "LaMBO-2 (online)")
    # Match the existing 0.83\textwidth placement, not an oversized source canvas.
    figure, axes = plt.subplots(2, 2, figsize=(0.83 * PAGE_WIDTH, 4.25), layout="constrained")
    for index, (axis, task) in enumerate(zip(axes.flat, TASKS, strict=True)):
        for method, label in zip(methods, labels, strict=True):
            group = [row for row in rows if row["task"] == task and row["method"] == method]
            if not group:
                raise ValueError(f"Missing reported curve: {task}/{method}")
            calls = [int(row["eval_idx"]) for row in group]
            mean = np.array([float(row["mean"]) for row in group])
            sd = np.array([float(row["sample_std"]) for row in group])
            plot_band(axis, calls, mean, mean - sd, mean + sd, method, label)
        axis.set(title=f"({chr(97 + index)}) {TASK_LABELS[task]}", xlim=(128, 2000),
                 xticks=(128, 500, 1000, 1500, 2000), ylim=(0, None))
        if index >= 2:
            axis.set_xlabel("Charged oracle evaluations")
        if index % 2 == 0:
            axis.set_ylabel("Best observed proxy fitness")
        style_axis(axis)
    legend_above(figure, axes.flat, ncol=3)
    save_figure(figure, FIGURES / "protein_kermut_external20_best_so_far")


def ehrlich_locality():
    rows = read_rows("ehrlich_locality_calibration.csv")
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.9), layout="constrained")
    for index, (axis, condition) in enumerate(zip(axes, ("l128_q4_k4", "l256_q4_k8"), strict=True)):
        group = [row for row in rows if row["condition"] == condition]
        maximum = max(max(float(row["expected_edits"]), float(row["realised_edits_max"])) for row in group)
        axis.plot([0, maximum], [0, maximum], color="#777777", ls="--", lw=1, label="Ideal")
        for population, label, colour in (("all", "All proposals", "#0072B2"),
                                           ("eligible", "Eligible", "#D55E00"),
                                           ("selected", "Selected", "#009E73")):
            points = [row for row in group if row["population"] == population]
            x = [float(row["expected_edits"]) for row in points]
            y = np.array([float(row["realised_edits_mean"]) for row in points])
            low = np.array([float(row["realised_edits_min"]) for row in points])
            high = np.array([float(row["realised_edits_max"]) for row in points])
            axis.errorbar(x, y, yerr=[y - low, high - y], marker="o", markersize=2.5,
                          linewidth=0.85, capsize=1.5, color=colour, label=label, alpha=0.9)
        title = ("L=128, q=4, K=4", "L=256, q=4, K=8")[index]
        axis.set(title=f"({chr(97 + index)}) {title}", xlabel="Desired absolute edits", ylabel="Realised absolute edits")
        style_axis(axis)
    legend_above(figure, axes, ncol=4)
    save_figure(figure, FIGURES / "ehrlich_expected_vs_realised_edits")


def ehrlich_kernels():
    rows = read_rows("ehrlich_kernel_trajectories.csv")
    arms = ("onehot_ei", "spaced_pair_w32_ei", "spaced_pair_w48_ei", "random")
    labels = ("One-hot EI", "Spaced pair (W=32) EI", "Spaced pair (W=48) EI", "Random selection")
    colours = ("#0072B2", "#E69F00", "#009E73", "#777777")
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.9), layout="constrained")
    for index, (axis, condition) in enumerate(zip(axes, ("l128_q4_k4", "l256_q4_k8"), strict=True)):
        for arm, label, colour, ls in zip(arms, labels, colours, ("-", "--", "-.", ":"), strict=True):
            by_call = defaultdict(list)
            for row in rows:
                if row["condition"] == condition and row["arm"] == arm:
                    by_call[int(row["oracle_evaluations"])].append(float(row["best_so_far"]))
            calls = sorted(by_call)
            mean = np.array([np.mean(by_call[call]) for call in calls])
            sd = np.array([np.std(by_call[call], ddof=1) for call in calls])
            # Original kernel diagnostic joins recorded points linearly.
            axis.fill_between(calls, mean - sd, mean + sd, color=colour, alpha=0.12, linewidth=0)
            axis.plot(calls, mean, color=colour, lw=1.3, ls=ls, label=label)
        axis.axvline(64, color="#777777", ls=":", lw=0.8)
        title = ("L=128, q=4, K=4", "L=256, q=4, K=8")[index]
        axis.set(title=f"({chr(97 + index)}) {title}", xlabel="Oracle evaluations", ylabel="Best reward so far")
        style_axis(axis)
    legend_above(figure, axes, ncol=4)
    save_figure(figure, FIGURES / "ehrlich_kernel_best_so_far")
