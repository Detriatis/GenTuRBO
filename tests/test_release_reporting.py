"""Scientific report contracts, independent of model weights and manuscript files."""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
from pathlib import Path

import pytest

from genturbo.figures.neighbourhood import feasible_shell_counts
from genturbo.protein_reporting import incumbent_values, row_score
from genturbo.reporting import main, sokoban_tables


def test_exact_neighbourhood_counts_match_exhaustive_sequences():
    reference = [0, 1, 0, 1]
    support = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    expected = [0] * 5
    for sequence in itertools.product(range(3), repeat=4):
        if all(support[a][b] for a, b in zip(sequence, sequence[1:])):
            expected[sum(a != b for a, b in zip(sequence, reference))] += 1
    assert feasible_shell_counts(reference, support) == expected


def test_analysis_score_precedence_does_not_leak_different_oracle():
    assert row_score({"score_for_analysis": float("nan"), "score": 100.0}) is None
    assert row_score({"score_for_analysis": 0.4, "score": 100.0}) == 0.4
    assert incumbent_values([0.1, 0.9, None, 0.3], [1, 4]).tolist() == [0.1, 0.9]


@pytest.mark.parametrize("status,queries", [("running", 2000), ("completed", 1999)])
def test_sokoban_rejects_incomplete_cohort(tmp_path, status, queries):
    run = tmp_path / "runs/seed0/generator"
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps(dict(method="generator", seed=0, status=status, queries=queries)))
    with pytest.raises(ValueError, match="Incomplete"):
        sokoban_tables(tmp_path / "runs", tmp_path / "report")


def test_reporting_refuses_to_overwrite_existing_outputs(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("retain")
    with pytest.raises(ValueError, match="fresh"):
        main(["all", "--reference", str(tmp_path / "missing"), "--output", str(output)])
    assert (output / "keep.txt").read_text() == "retain"


@pytest.mark.integration
def test_reference_statistics_replay(tmp_path):
    root = os.environ.get("GENTURBO_ARTIFACTS")
    if not root:
        pytest.skip("Set GENTURBO_ARTIFACTS to the verified companion bundle directory")
    source = Path(root) / "reference/paper/figures/data"
    main(["all", "--artifacts", root, "--output", str(tmp_path / "report")])
    destination = tmp_path / "report/figures/data"
    for name, count in (("protein_all_pairwise_comparisons.csv", 84),
                        ("table2_all_external_comparisons.csv", 32),
                        ("ehrlich_neighbourhood_counts.csv", 257)):
        with (source / name).open() as a, (destination / name).open() as b:
            expected, observed = list(csv.DictReader(a)), list(csv.DictReader(b))
        assert len(observed) == len(expected) == count
        for left, right in zip(expected, observed, strict=True):
            assert left.keys() == right.keys()
            for key in left:
                if left[key] != right[key]:
                    assert math.isclose(float(left[key]), float(right[key]), rel_tol=1e-12, abs_tol=1e-12)
