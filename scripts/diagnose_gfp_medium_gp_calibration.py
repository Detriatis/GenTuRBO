#!/usr/bin/env python3
"""Small offline Tanimoto comparison; diagnostic labels never enter search archives.

prepare: fixed validation/test mutants plus explicitly selection-biased logged
generator samples. run: independent history/setting replays in a CPU process pool.
Only the requested split is evaluated; freeze a setting before running test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import time
import warnings

import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_CONFIG = Path("configs/experiments/gfp_medium_calibration_v1.json")
DEFAULT_OUTPUT = Path("results/protein_gfp_medium_calibration_v1")


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sequence(row):
    return "".join(ALPHABET[int(token)] for token in row["seq"])


def mutate_exact(parent, distance, rng):
    tokens = np.asarray([ALPHABET.index(char) for char in parent])
    positions = rng.choice(len(tokens), distance, replace=False)
    # A nonzero cyclic offset gives a uniform draw from the other 19 residues.
    tokens[positions] = (tokens[positions] + rng.integers(1, 20, distance)) % 20
    return "".join(ALPHABET[token] for token in tokens)


def diverse_parents(initial, count):
    seqs = [row["sequence"] for row in initial]
    tokens = np.asarray([list(seq) for seq in seqs])
    chosen = [int(np.argmax([row["score"] for row in initial]))]
    nearest = np.full(len(seqs), len(seqs[0]) + 1)
    while len(chosen) < count:
        nearest = np.minimum(nearest, (tokens != tokens[chosen[-1]]).sum(axis=1))
        nearest[chosen] = -1
        chosen.append(int(np.argmax(nearest)))
    return [seqs[index] for index in chosen]


def unique_archive(rows):
    archive = {}
    for row in rows:
        seq = row["sequence"]
        archive[seq] = max(archive.get(seq, -np.inf), row["score"])
    return list(archive), np.asarray(list(archive.values()))


def prepare(config, output):
    import torch
    from adapters.protein.reward import ProteinGFPAVCNNOracleReward

    torch.set_num_threads(int(config.get("threads", 1)))
    histories, future = {}, []
    initial = None
    excluded = set()
    endpoints = {}
    for cohort in ("learned", "fixed"):
        for seed in config["seeds"]:
            root = Path(config[f"{cohort}_history_root"]) / f"seed{seed}"
            starts = [{"sequence": sequence(row), "score": row["score"]}
                      for row in read_rows(root / "initial_batch.jsonl.gz")]
            if initial is None:
                initial = starts
            if starts != initial:
                raise ValueError(f"Starting archive mismatch: {root}")
            paid = [row for row in read_rows(root / "samples.jsonl.gz")
                    if row.get("oracle_evaluated", True)]
            batches = []
            for batch in range(16):
                rows = [{"sequence": sequence(row), "score": row["score"]}
                        for row in paid if row["batch_id"] == batch]
                if len(rows) != 8:
                    raise ValueError(f"Expected eight paid rows: {root}, batch {batch}")
                batches.append(rows)
            key = f"{cohort}_seed{seed}"
            histories[key] = batches
            excluded.update(row["sequence"] for row in starts)
            excluded.update(row["sequence"] for batch in batches for row in batch)
            endpoints[key] = {"labels": len(starts) + len(paid),
                              "best": max(row["score"] for row in starts + paid)}
            future.extend({"sequence": sequence(row), "score": row["score"],
                           "source_history": key, "source_batch": row["batch_id"]}
                          for row in paid if 16 <= row["batch_id"] < 64)

    parents = diverse_parents(initial, config["parents"])
    seen = set(excluded)
    probes = []
    seeds = np.random.SeedSequence(config["probe_seed"]).spawn(3)
    for split, child_seed in zip(("validation", "test"), seeds[:2]):
        rng = np.random.default_rng(child_seed)
        for parent_index, parent in enumerate(parents):
            for distance in config["distances"]:
                accepted = 0
                while accepted < config["draws_per_parent_distance_split"]:
                    seq = mutate_exact(parent, distance, rng)
                    if seq in seen:
                        continue
                    seen.add(seq)
                    probes.append({"sequence": seq, "split": split, "operator": "uniform",
                                   "distance": distance, "parent_index": parent_index})
                    accepted += 1
    oracle = ProteinGFPAVCNNOracleReward(
        protein="GFP", benchmark_root=config.get("benchmark_root", "benchmarks/gfp_aav"), sequence_length=237, device="cpu")
    for start in range(0, len(probes), 128):
        chunk = probes[start:start + 128]
        for row, score in zip(chunk, oracle.score_sequences([r["sequence"] for r in chunk])):
            row["score"] = float(score)
    cnn_initial = oracle.score_sequences([row["sequence"] for row in initial])
    rng = np.random.default_rng(seeds[2])
    rng.shuffle(future)
    cursor = iter(future)
    for split in ("validation", "test"):
        for _ in range(config["generator_acquired_per_split"]):
            row = next(cursor, None)
            while row is not None and row["sequence"] in seen:
                row = next(cursor, None)
            if row is None:
                raise ValueError("Not enough distinct post-replay generator acquisitions")
            seen.add(row["sequence"])
            probes.append({**row, "operator": "generator_acquired", "split": split})
    data = {"config": config, "initial": initial, "initial_cnn_scores": cnn_initial,
            "parents": parents, "histories": histories, "probes": probes,
            "completed_endpoints": endpoints,
            "extra_oracle_evaluations": len([r for r in probes if r["operator"] == "uniform"]) + len(initial),
            "generator_note": "Previously acquired (not unselected) samples from batches 16-63; secondary selection-biased diagnostic."}
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "prepared.json.gz"
    if destination.exists():
        raise FileExistsError(destination)
    with gzip.open(destination, "wt") as stream:
        json.dump(data, stream)
    labels = np.asarray([r["score"] for r in initial])
    audit = {"n": len(initial), "rmse_stored_vs_cnn": float(np.sqrt(np.mean((labels-cnn_initial)**2))),
             "correlation_stored_vs_cnn": float(np.corrcoef(labels, cnn_initial)[0, 1]),
             "best_stored": float(labels.max()), "cnn_at_stored_best": float(cnn_initial[int(labels.argmax())]),
             "extra_oracle_evaluations": data["extra_oracle_evaluations"],
             "probes": len(probes), "history_training_overlap": len({r["sequence"] for r in probes} & excluded)}
    (output / "initial_label_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


def features(seqs):
    from search.interpolant.active_learning import make_sequence_onehot_features
    matrix, valid = make_sequence_onehot_features(seqs, max_len=237, alphabet=ALPHABET)
    if not np.all(valid):
        raise ValueError("Invalid protein sequence")
    return matrix


def metrics(y, mean, sd, noise_raw, best):
    import torch
    from scipy.stats import spearmanr
    from botorch.acquisition.analytic import _log_ei_helper
    from search.interpolant.active_learning import _expected_improvement

    variance = np.maximum(sd**2, 1e-15)
    obs_variance = variance + noise_raw
    error = y - mean
    ei = _expected_improvement(mean, sd, best=best, xi=0.0)
    safe_sd = torch.as_tensor(np.maximum(sd, 1e-15), dtype=torch.double)
    z = (torch.as_tensor(mean, dtype=torch.double) - best) / safe_sd
    log_ei = (_log_ei_helper(z) + safe_sd.log()).numpy()
    top = np.argsort(-ei, kind="stable")[:8]
    stable_top = np.argsort(-log_ei, kind="stable")[:8]
    result = {
        "n": len(y), "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": float(spearmanr(y, mean).statistic) if np.ptp(mean) and np.ptp(y) else None,
        "latent_nll": float(np.mean(0.5*(np.log(2*np.pi*variance) + error**2/variance))),
        "observation_nll": float(np.mean(0.5*(np.log(2*np.pi*obs_variance) + error**2/obs_variance))),
        "latent_95_coverage": float(np.mean(np.abs(error) <= 1.95996398454*np.sqrt(variance))),
        "observation_95_coverage": float(np.mean(np.abs(error) <= 1.95996398454*np.sqrt(obs_variance))),
        "latent_mean_sd": float(np.mean(sd)), "improving_candidates": int(np.sum(y > best)),
        "ei_top8_improvements": int(np.sum(y[top] > best)),
        "ei_top8_regret": float(y.max() - y[top].max()),
        "log_ei_top8_regret": float(y.max() - y[stable_top].max()),
        "ei_logei_top8_overlap": len(set(top) & set(stable_top)),
        "ei_nonpositive_fraction": float(np.mean(ei <= 0)),
        "ei_negative_fraction": float(np.mean(ei < 0)),
        "variance_floor_count": int(np.sum(sd**2 <= 1e-15)),
    }
    return result


def replay(job):
    prepared, output, history, variant, split, threads, smoke = job
    import torch
    from surrogate.features import TanimotoGPRegressor

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    start = time.monotonic()
    with gzip.open(prepared, "rt") as stream:
        data = json.load(stream)
    config = data["config"]
    probes = [r for r in data["probes"] if r["split"] == split]
    if smoke:
        probes = probes[:64]
    x_probe = features([r["sequence"] for r in probes])
    y_probe = np.asarray([r["score"] for r in probes])
    masks = {"all": np.ones(len(probes), dtype=bool),
             "uniform": np.asarray([r["operator"] == "uniform" for r in probes]),
             "generator_acquired": np.asarray([r["operator"] == "generator_acquired" for r in probes])}
    masks.update({f"distance_{d}": np.asarray([r.get("distance") == d for r in probes])
                  for d in config["distances"]})
    rows = list(data["initial"])
    batches = data["histories"][history]
    reports = []
    # Skipped intermediate conditioning calls do not change hyperparameters or
    # outcome statistics. Fit at 0,8,16; inspect just before both later refits too.
    last_batch = 0
    for batch in ([0] if smoke else config["snapshot_batches"]):
        for new_rows in batches[last_batch:batch]:
            rows.extend(new_rows)
        last_batch = batch
        seqs, y = unique_archive(rows)
        x = features(seqs)
        refit = batch % config["refit_interval"] == 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if refit:
                gp = TanimotoGPRegressor(device="cpu", dtype="double", jitter=1e-4,
                                         **config["variants"][variant]).fit(x, y)
            else:
                gp.refresh_training_data(x, y)
            predictions = [gp.predict_mean_std(x_probe[i:i+128]) for i in range(0, len(probes), 128)]
        mean = np.concatenate([p[0] for p in predictions]).reshape(-1)
        sd = np.concatenate([p[1] for p in predictions]).reshape(-1)
        model = gp.model_
        noise = model.likelihood.noise.item()
        target_sd = model.outcome_transform.stdvs.item()
        params = {"noise_standardized": noise, "noise_raw": noise*target_sd**2,
                  "outputscale": model.covar_module.outputscale.item(),
                  "mean_standardized": model.mean_module.constant.item(),
                  "outcome_mean": model.outcome_transform.means.item(), "outcome_sd": target_sd}
        report = {"batch": batch, "paid_labels": len(rows), "unique_labels": len(seqs),
                  "refit": refit, "parameters": params,
                  "warnings": sorted({str(w.message) for w in caught}),
                  "metrics": {name: metrics(y_probe[mask], mean[mask], sd[mask], params["noise_raw"], y.max())
                              for name, mask in masks.items() if np.any(mask)}}
        reports.append(report)
    result = {"history": history, "variant": variant, "split": split,
              "seconds": time.monotonic()-start, "snapshots": reports}
    path = Path(output) / f"{history}__{variant}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {"history": history, "variant": variant, "seconds": round(result["seconds"], 2)}


def run(args):
    with gzip.open(args.output / "prepared.json.gz", "rt") as stream:
        data = json.load(stream)
    variants = args.variants or list(data["config"]["variants"])
    if args.split == "test":
        decision = json.loads((args.output / "selection.json").read_text())
        if set(variants) - {"learned_unit", decision["selected_variant"]}:
            raise ValueError("Test only the frozen selection and original baseline")
    histories = list(data["histories"])
    if args.smoke:
        histories, variants = histories[:1], variants[:1]
    dest = args.output / ("smoke" if args.smoke else args.split)
    dest.mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.output / "prepared.json.gz"), str(dest), h, v, args.split, args.threads, args.smoke)
            for h in histories for v in variants if not (dest / f"{h}__{v}.json").exists()]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(replay, job) for job in jobs]):
            print(json.dumps(future.result()), flush=True)
    summarize(dest)


def summarize(directory):
    grouped = {}
    for path in sorted(directory.glob("*__*.json")):
        result = json.loads(path.read_text())
        for snapshot in result["snapshots"]:
            for group, values in snapshot["metrics"].items():
                grouped.setdefault((result["variant"], group), []).append(values)
    summary = []
    for (variant, group), values in sorted(grouped.items()):
        summary.append({"variant": variant, "probe_group": group, "replay_snapshots": len(values),
                        **{key: float(np.mean([v[key] for v in values if v[key] is not None]))
                           for key in values[0] if any(v[key] is not None for v in values)}})
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for row in sorted((r for r in summary if r["probe_group"] == "uniform"), key=lambda r:r["observation_nll"]):
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "summarize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(json.loads(args.config.read_text()), args.output)
    elif args.phase == "run":
        run(args)
    else:
        summarize(args.output / args.split)


if __name__ == "__main__":
    main()
