"""Compact public-recipe contracts; full transition checks precede release."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pytest

from genturbo.reproduce import build_recipes, resolve_recipe, missing_inputs, _seeds, _check_device

ROOT = Path(__file__).resolve().parents[1]


def test_current_main_matrix_and_faithful_adalead():
    cells = build_recipes(ROOT, "ehrlich")
    assert len(cells) == 260
    assert set(Counter(c.method for c in cells).values()) == {20}
    assert "adalead_gp_ei" not in {c.method for c in cells}
    official = next(c for c in cells if c.method == "adalead_official")
    assert official.config["proposal_operator"] == "adalead_official"
    assert official.config["baseline_adalead_surrogate_queries"] == 128
    assert official.config["baseline_adalead_rho"] == 0
    assert all(c.config["max_oracle_evaluations"] == 2000 for c in cells)
    mapped = resolve_recipe(official, root=ROOT, artifacts=ROOT, output=ROOT / "unused", device="cpu")
    assert "device" not in mapped.config["reward"]
    assert not ({"reproduction_suite", "device_override"} & set(mapped.config["metadata"]))


def test_protein_current_full_history_protocol_and_fixed_noise():
    cells = build_recipes(ROOT, "protein")
    assert len(cells) == 560
    assert len({(c.task, c.method, c.seed) for c in cells}) == 560
    for cell in cells:
        cfg = cell.config
        assert cfg["max_oracle_evaluations"] == 2000
        if cell.method in {"kermut", "tanimoto", "hamming_gp", "adalead", "bo_evo"}:
            assert cfg["al_buffer_size"] == 2000
            assert cfg["al_gp_hyperparameter_refit_interval"] == 8
        if cell.method in {"tanimoto", "hamming_gp"}:
            assert cfg["al_acquisition"] == "expected_improvement"
            assert cfg["al_sequential_selection"] is False
            if cell.task == "gfp_medium":
                assert cfg["al_gp_noise_variance"] == 0.001
        if cell.method == "lambo2":
            assert cfg["lambo2"]["initialization"] == "online_archive_random"
            assert "checkpoint" not in cfg
            assert cfg["lambo2"]["training_history"] == "complete_paid_history"


def test_supporting_components_use_full_cohorts():
    cells = build_recipes(ROOT, "supporting", analyses=["protein-components"])
    assert len(cells) == 320
    for cell in cells:
        if cell.method == "tanimoto_rt_gp":
            assert cell.config["al_acquisition"] == "expected_improvement"
            assert cell.config["al_buffer_size"] == 2000
            assert cell.config["candidate_strategy"] == "random"
            if cell.task == "gfp_medium":
                assert cell.config["al_gp_noise_variance"] == 0.001


def test_relocation_preserves_original_recipe_and_requires_inputs(tmp_path):
    original = next(c for c in build_recipes(ROOT, "protein") if c.method == "kermut")
    cfg_before = dict(original.config)
    mapped = resolve_recipe(original, root=ROOT, artifacts=tmp_path / "assets", output=tmp_path / "outputs", device="cpu")
    assert original.config == cfg_before
    assert mapped.config["al_kermut_resources"].startswith(str(tmp_path / "assets/runs/"))
    assert mapped.config["initial_batch_path"].startswith(str(tmp_path / "assets/benchmarks/"))
    assert mapped.config["samples_out"].endswith("protein/gfp_medium/kermut/seed0/samples.jsonl.gz")
    assert mapped.config["reward"]["device"] == "cpu"
    assert mapped.config["al_gp_device"] == original.config["al_gp_device"]
    assert missing_inputs(mapped)


def test_plan_writes_nothing(tmp_path, capsys):
    from genturbo.cli import main
    output = tmp_path / "output"
    assert main(["reproduce", "ehrlich", "--seeds", "0", "--methods", "adalead_official", "--output", str(output), "--dry-run"]) == 0
    assert not output.exists()
    assert '"cell_count": 1' in capsys.readouterr().out


def test_seed_range_and_unavailable_device_fail_closed(monkeypatch):
    import torch
    assert _seeds(["0-2", "5"]) == [0, 1, 2, 5]
    with pytest.raises(ValueError):
        _seeds(["1", "1"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        _check_device("cuda")


def test_training_plans_preserve_budgets_and_keep_outputs_separate(tmp_path):
    from genturbo.training import training_plan
    output = tmp_path / "new-training"
    ehrlich = training_plan("ehrlich", root=ROOT, artifacts=tmp_path / "assets", output=output, device="cpu")
    cfg = ehrlich["resolved_config"]
    assert cfg["steps"] == 18000 and cfg["dataset_cfg"]["batch_size"] == 128
    assert cfg["run_root"] == str(output) and cfg["run_name"] == "model"
    assert cfg["checkpoint_path"] is None and cfg["resume_if_checkpoint_exists"] is False
    sokoban = training_plan("sokoban", root=ROOT, artifacts=tmp_path / "assets", output=output, device="cpu")
    assert sokoban["resolved_config"]["steps"] == 292000
    assert sokoban["resolved_config"]["dataset_cfg"]["batch_size"] == 1536
    assert sokoban["resolved_config"]["precision"] == "float32"
    lambo = training_plan("lambo2-ehrlich", root=ROOT, artifacts=tmp_path / "assets", output=output, device="cpu")
    spec = lambo["resolved_config"]["lambo2"]["pretraining"]
    assert spec["epochs"] == 9 and spec["matched_token_exposures"] == 294912000
    assert spec["checkpoint"] == str(output / "generator.ckpt")
    assert spec["reference_checkpoint"].startswith(str(tmp_path / "assets"))
    assert not output.exists()


def test_training_cli_default_plan_is_read_only(tmp_path, capsys):
    from genturbo.cli import main
    output = tmp_path / "training"
    assert main(["train", "ehrlich", "--device", "cpu", "--output", str(output)]) == 0
    assert not output.exists()
    assert '"execute": false' in capsys.readouterr().out
