#!/usr/bin/env python3
"""Exact Hamming-ball counts with and without the saved Ehrlich constraints."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
from genturbo.figures.style import COLUMN_WIDTH, apply_style, style_axis, legend_above, save_figure

ROOT = Path(__file__).resolve().parents[1]


def feasible_shell_counts(reference: list[int], support: list[list[int]]) -> list[int]:
    """Count strings by distance, using Python integers (no overflow/rounding).

    dp[a][j] counts permitted prefixes ending in token a with j substitutions.
    All initial tokens are allowed, as in the Ehrlich task implementation.
    """
    q, length = len(support), len(reference)
    if not length or q < 2 or any(len(row) != q for row in support):
        raise ValueError("Need a nonempty reference and a square support matrix")
    if any(token not in range(q) for token in reference):
        raise ValueError("Reference token outside the alphabet")
    predecessors = [[a for a in range(q) if support[a][b]] for b in range(q)]
    dp = [[0] * (length + 1) for _ in range(q)]
    for a in range(q):
        dp[a][int(a != reference[0])] = 1
    for position in range(1, length):
        updated = [[0] * (length + 1) for _ in range(q)]
        for b in range(q):
            delta = int(b != reference[position])
            incoming = [dp[a] for a in predecessors[b]]
            for edits in range(position + 1):
                updated[b][edits + delta] = sum(row[edits] for row in incoming)
        dp = updated
    return [sum(row[edits] for row in dp) for edits in range(length + 1)]


def cumulative(values: list[int]) -> list[int]:
    return list(itertools.accumulate(values))


def check_counts() -> None:
    """Compare against exhaustive enumeration, including adjacent/end edits."""
    rng = random.Random(123)
    reference = [0, 1, 2, 0, 1]
    matrices = [
        [[1] * 3 for _ in range(3)],
        [[int(a == b) for b in range(3)] for a in range(3)],
        [[int(b == (a + 1) % 3) for b in range(3)] for a in range(3)],
        *[[[rng.randrange(2) for _ in range(3)] for _ in range(3)] for _ in range(20)],
    ]
    for support in matrices:
        expected = [0] * 6
        for candidate in itertools.product(range(3), repeat=5):
            if all(support[a][b] for a, b in zip(candidate, candidate[1:])):
                expected[sum(a != b for a, b in zip(candidate, reference))] += 1
        assert feasible_shell_counts(reference, support) == expected
    for length in (1, 2, 8):
        actual = feasible_shell_counts([0] * length, [[1] * 4 for _ in range(4)])
        assert actual == [math.comb(length, k) * 3**k for k in range(length + 1)]
    print("Checks passed: 23 exhaustive support matrices and 3 unrestricted cases")


def render(ambient: list[int], feasible: list[int], max_edits: int, scale: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator, MaxNLocator

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 2.55), layout="constrained")
    x = list(range(max_edits + 1))
    if scale == "linear":
        exponent = len(str(ambient[max_edits])) - 1
        unit = 10**exponent
        all_values = [n / unit for n in ambient[:max_edits + 1]]
        feasible_values = [n / unit for n in feasible[:max_edits + 1]]
        upper = 1.08 * max(all_values)
        ylabel = "Number of sequences"
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.text(0, 1.015, rf"$\times 10^{{{exponent}}}$", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7)
    else:
        all_values = [math.log10(n) for n in ambient[:max_edits + 1]]
        feasible_values = [math.log10(n) for n in feasible[:max_edits + 1]]
        step = max(1, math.ceil(all_values[-1] / 7 / 5) * 5)
        upper = math.ceil(all_values[-1] / step) * step
        ylabel = "Number of sequences (log scale)"
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"$10^{{{int(value)}}}$"))
    ax.plot(x, all_values, color="#555555", linestyle="--", linewidth=1.5,
            marker="o" if max_edits <= 12 else None, markersize=3, clip_on=False, label="All sequences")
    ax.plot(x, feasible_values, color="#0072B2", linewidth=1.6,
            marker="o" if max_edits <= 12 else None, markersize=3, clip_on=False, label="Ehrlich-feasible sequences")
    ax.set(xlim=(0, max_edits), ylim=(0, upper),
           xlabel=r"Edit distance budget, $k$", ylabel=ylabel)
    ax.set_xticks(list(range(max_edits + 1)) if max_edits <= 12 else
                 ([0, 64, 128, 192, 256] if max_edits == 256 else
                  list(range(0, max_edits + 1, max(1, max_edits // 4)))))
    ax.grid(axis="y", alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"figures/ehrlich_neighbourhood_counts.{ext}",
                    dpi=400, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def render_full_comparison(ambient: list[int], feasible: list[int], *, retention_only: bool = False) -> None:
    """Full-domain feasible percentage, optionally paired with log counts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    assert len(ambient) == len(feasible) == 257
    x = list(range(257))
    all_counts, feasible_counts = ambient, feasible
    retained = [100 * f / a for a, f in zip(ambient, feasible)]
    apply_style()
    blue, grey = '#0072B2', '#555555'
    if retention_only:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4), layout='constrained')
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.9), sharex=True,
                                 gridspec_kw={'height_ratios': [1.1, 1]}, layout='constrained')
        ax = axes[0]
        ax.plot(x, [math.log10(a) for a in all_counts], '--', color=grey,
                lw=1.5, ms=3, label='All sequences', clip_on=False)
        ax.plot(x, [math.log10(f) for f in feasible_counts], '-', color=blue,
                lw=1.6, ms=3, label='Ehrlich-feasible', clip_on=False)
        ax.set(ylim=(0, 350), ylabel='Number of sequences', title='(a) Cumulative count (log scale)')
        ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf'$10^{{{int(value)}}}$'))
        ax.legend(loc='upper left', frameon=False, handlelength=1.8)

    ax = axes[-1]
    ax.axhline(100, color=grey, ls='--', lw=1)
    ax.plot(x, retained, '-', color=blue, lw=1.8, ms=3.5, clip_on=False)
    ax.fill_between(x, 0, retained, color=blue, alpha=.08, lw=0)
    ax.set(ylim=(0, 110), ylabel='Sequences remaining feasible',
           xlabel=r'Edit distance budget, $k$')
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.plot(12, retained[12], 'o', color=blue, ms=3)
    ax.annotate(f'By 12 edits:\n{retained[12]:.1f}% remain',
                xy=(12, retained[12]), xytext=(70, 39),
                ha='center', va='center', fontsize=8, color=blue,
                arrowprops={'arrowstyle': '-', 'color': blue, 'lw': .8})
    exponent = math.floor(math.log10(retained[-1]))
    mantissa = retained[-1] / 10**exponent
    ax.text(.96, .78, 'At 256 edits:\n' + rf'${mantissa:.2f}\times10^{{{exponent}}}\%$ remain',
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color=blue)
    for ax in axes:
        ax.set_xlim(0, 256)
        ax.set_xticks([0, 64, 128, 192, 256])
        style_axis(ax)
    save_figure(fig, ROOT / 'figures/ehrlich_neighbourhood_counts')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edits", type=int, default=256)
    parser.add_argument("--scale", choices=("linear", "log"), default="log")
    parser.add_argument("--view", choices=("retention", "comparison", "counts"), default="retention",
                        help="Paper feasible-percentage figure, two-panel comparison, or single count panel")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.view == "retention" and args.max_edits != 256:
        parser.error("The retention view uses the full 0-256 edit range")
    if args.view == "comparison" and (args.max_edits != 256 or args.scale != "log"):
        parser.error("The comparison view uses max-edits=256 and scale=log; use --view counts for alternatives")
    if args.check:
        check_counts()
    source = json.loads((ROOT / "figures/data/ehrlich_neighbourhood_source.json").read_text())
    reference, support = source["reference"], source["transition_support"]
    length, q = len(reference), len(support)
    assert length == source["instance_config"]["seq_len"] == 256
    assert q == source["instance_config"]["vocab_size"] == 20
    assert all(support[a][b] for a, b in zip(reference, reference[1:]))
    if not 1 <= args.max_edits <= length:
        parser.error("max-edits must be between one and the sequence length")
    ambient_shells = [math.comb(length, k) * (q - 1)**k for k in range(length + 1)]
    feasible_shells = feasible_shell_counts(reference, support)
    ambient, feasible = cumulative(ambient_shells), cumulative(feasible_shells)
    assert ambient[0] == feasible[0] == 1 and ambient[-1] == q**length
    assert all(0 <= f <= a for a, f in zip(ambient_shells, feasible_shells))
    # Independent one-substitution enumeration checks the real 256-token case.
    one_edit = 0
    for position, original in enumerate(reference):
        for token in range(q):
            if token == original:
                continue
            candidate = reference.copy()
            candidate[position] = token
            one_edit += all(support[a][b] for a, b in zip(candidate, candidate[1:]))
    assert feasible_shells[1] == one_edit
    # Summing distance shells must equal unrestricted path counting on G.
    paths = [1] * q
    for _ in range(length - 1):
        paths = [sum(paths[a] for a in range(q) if support[a][b]) for b in range(q)]
    assert feasible[-1] == sum(paths)
    with (ROOT / "figures/data/ehrlich_neighbourhood_counts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["edit_budget", "all_within", "ehrlich_feasible_within", "all_exact", "ehrlich_feasible_exact"])
        writer.writerows(zip(range(length + 1), ambient, feasible, ambient_shells, feasible_shells))
    if args.view in ("retention", "comparison"):
        render_full_comparison(ambient, feasible, retention_only=args.view == "retention")
    else:
        render(ambient, feasible, args.max_edits, args.scale)
    for k in (0, 1, 4, 12, 32, 64, 256):
        print(json.dumps({"k": k, "all_log10": math.log10(ambient[k]),
                          "feasible_log10": math.log10(feasible[k]),
                          "feasible_fraction": feasible[k] / ambient[k]}))


if __name__ == "__main__":
    main()
