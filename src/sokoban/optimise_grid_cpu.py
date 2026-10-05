"""Run direct-grid GA, AdaLead or adaptive Hamming GP/EI from saved starts."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from sokoban.initial_archive import load_initial_archives as validate_initial_archives
from sokoban.initial_archive import load_initial_sources
from sokoban.optimise import PilotConfig, load_config, run_method, write_json
from sokoban.prepare import sha256


def load_initial_archives(cfg, source, checkpoint_sha256):
    validate_cpu_config(cfg)
    return validate_initial_archives(cfg, source, checkpoint_sha256)


def validate_cpu_config(cfg):
    if cfg.device != "cpu" or set(cfg.methods) - {"grid_ga", "adalead_gp_ei", "adalead_official", "hamming_gp_ei"} or cfg.objective != "box_switches":
        raise ValueError("CPU archive runner requires device=cpu, methods drawn from grid_ga/adalead_gp_ei/adalead_official/hamming_gp_ei, objective=box_switches")


def run_cpu(cfg, source, checkpoint_sha256, *, threads=1):
    validate_cpu_config(cfg)
    if threads < 1:
        raise ValueError("threads must be positive")
    sources = [Path(s).resolve() for s in ([source] if isinstance(source, (str, Path)) else source)]
    archives = load_initial_sources(cfg, sources, checkpoint_sha256)
    root = Path(cfg.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    write_json(root / "config.json", asdict(cfg))
    write_json(root / "provenance.json", {
        "checkpoint_sha256": checkpoint_sha256, "checkpoint_loaded": False,
        "generator_calls": 0, "gpu": None, "execution_device": "cpu",
        "torch": str(torch.__version__), "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "initial_source": str(sources[0]) if len(sources) == 1 else None,
        "initial_sources": [str(s) for s in sources],
        "initial_source_metadata_sha256": {str(s): {name: sha256(s / name) for name in ("config.json", "provenance.json")}
                                           for s in sources},
        "source_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                          ("optimise_grid_cpu.py", "optimise.py", "search.py", "planning.py", "evolution.py", "adalead.py", "adalead_official.py", "hamming.py")},
        "timing_note": "Initial puzzles and labels reused and charged; copied initial timing is historical. "
                       "Search runs on the recorded CPU platform, so compare query efficiency, not GPU wall time.",
    })
    results = []
    for seed, (initial, records, hashes, source) in archives.items():
        seed_root = root / f"seed{seed}"
        seed_root.mkdir()
        for name in hashes:
            shutil.copyfile(source / f"seed{seed}" / name, seed_root / name)
        write_json(seed_root / "initial_reuse.json", {"source": str(source / f"seed{seed}"),
                                                      "sha256": hashes, "physical_initial_solves": 0})
        # No adapter is constructed: grid operators and the oracle run on CPU.
        for method in cfg.methods:
            results.append(run_method(cfg, None, initial, records, method, seed, seed_root / method))
            write_json(root / "results.json", results)
    write_json(root / "complete.json", {"status": "completed", "seeds": cfg.seeds,
                                        "mode": "grid_ga_cpu" if list(cfg.methods) == ["grid_ga"] else "direct_grid_cpu",
                                        "methods": cfg.methods, "runs": len(results)})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/search/sokoban/box_switches_grid_ga_cpu.json"))
    parser.add_argument("--initial-from", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint-sha256", required=True, help="Expected generator identity for the saved initial puzzles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    values = load_config(args.config)
    values["output"] = args.output
    if args.seeds is not None:
        values["seeds"] = args.seeds
    cfg = PilotConfig(**values)
    validate_cpu_config(cfg)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.dry_run:
        load_initial_sources(cfg, args.initial_from, args.checkpoint_sha256)
        print(json.dumps({"config": asdict(cfg), "initial_archives_verified": len(cfg.seeds),
                          "generator_calls": 0, "checkpoint_loaded": False}, indent=2))
    else:
        run_cpu(cfg, args.initial_from, args.checkpoint_sha256, threads=args.threads)


if __name__ == "__main__":
    main()
