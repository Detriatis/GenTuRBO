"""CLI argument parser for interpolant CTMC search."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Optional


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolant search using a trained CTMC flow.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--search-config", type=str, default=None, help="JSON file with search params (overridden by CLI)"
    )
    parser.add_argument(
        "--benchmark-task",
        type=str,
        default=None,
        help="Operational benchmark task identifier (for example gfp_hard or aav_medium).",
    )
    parser.add_argument(
        "--trace-level",
        choices=("none", "summary", "full"),
        default=None,
        help="Diagnostic trace volume. Full traces require --trace-record-limit.",
    )
    parser.add_argument(
        "--trace-record-limit",
        type=int,
        default=None,
        help="Maximum full-trace records written; required when --trace-level=full.",
    )
    parser.add_argument(
        "--compress-jsonl",
        action="store_true",
        dest="compress_jsonl",
        default=None,
        help="Write automatically derived JSONL outputs with gzip compression.",
    )
    parser.add_argument(
        "--no-compress-jsonl",
        action="store_false",
        dest="compress_jsonl",
    )
    parser.add_argument(
        "--run-manifest-out",
        type=str,
        default=None,
        help="Optional run-manifest JSON path. Defaults beside samples_out.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument(
        "--config", type=str, default=None, help="Optional override config JSON (defaults to checkpoint metadata)"
    )
    parser.add_argument("--flow-steps", type=int, default=None, help="Number of CTMC steps for rollouts")
    parser.add_argument("--max-time", type=float, default=None, help="Max time horizon for CTMC rollouts")
    parser.add_argument(
        "--rollout-denoise-steps",
        type=int,
        default=None,
        help="Optional exact number of CTMC integration substeps over [sampled t, max_time].",
    )
    parser.add_argument("--device", type=str, default=None, help="Device string (e.g., cpu, cuda:0, mps)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for initial sampling")
    parser.add_argument(
        "--initial-oracle-evaluations",
        type=int,
        default=None,
        help="Initial oracle-evaluated samples used to seed BO. Defaults to batch-size.",
    )
    parser.add_argument(
        "--initial-batch-path",
        type=str,
        default=None,
        help="Optional JSONL path containing an initial evaluated archive to start BO from.",
    )
    parser.add_argument(
        "--initial-batch-out",
        type=str,
        default=None,
        help="Optional JSONL path for the initial evaluated archive. Defaults beside samples_out.",
    )
    parser.add_argument(
        "--initial-batch-advance-rng",
        action="store_true",
        dest="initial_batch_advance_rng",
        default=None,
        help="When loading an initial archive, discard an equivalent model sample first so later RNG matches a normal run.",
    )
    parser.add_argument(
        "--no-initial-batch-advance-rng",
        action="store_false",
        dest="initial_batch_advance_rng",
    )
    parser.add_argument(
        "--initial-batch-rescore-with-reward",
        action="store_true",
        dest="initial_batch_rescore_with_reward",
        default=None,
        help=(
            "When loading an initial archive, ignore stored scores and rescore "
            "the archive with the active reward/oracle before seeding BO."
        ),
    )
    parser.add_argument(
        "--no-initial-batch-rescore-with-reward",
        action="store_false",
        dest="initial_batch_rescore_with_reward",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        dest="sample_only",
        default=None,
        help="Sample and score an initial batch from the model, then exit without running optimisation.",
    )
    parser.add_argument(
        "--no-sample-only",
        action="store_false",
        dest="sample_only",
    )
    parser.add_argument(
        "--sample-only-count",
        type=int,
        default=None,
        help="Number of model samples to draw in sample-only mode. Defaults to batch-size.",
    )
    parser.add_argument(
        "--sample-only-batch-size",
        type=int,
        default=None,
        help=(
            "Optional generation chunk size in sample-only mode. This bounds model "
            "memory without changing the requested sample or oracle count."
        ),
    )
    parser.add_argument("--num-iterations", type=int, default=None, help="Number of search iterations")
    parser.add_argument(
        "--num-trust-regions",
        type=int,
        default=None,
        help="Number of independent trust regions for multi-region Thompson-over-union search. Values > 1 enable the multi-region path.",
    )
    parser.add_argument(
        "--tr-restart-patience",
        type=int,
        default=None,
        help="Restart patience for multi-region trust regions. A region is reinitialised from pure noise after this many consecutive collapsed/inactive iterations.",
    )
    parser.add_argument(
        "--multi-region-score-mode",
        type=str,
        default=None,
        help="How multi-region slotwise Thompson compares regions: absolute uses raw Thompson value, improvement uses (Thompson - local incumbent).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Optional early-stop patience in outer search iterations. If the best score does not improve for this many iterations, terminate the run.",
    )
    parser.add_argument(
        "--max-oracle-evaluations",
        type=int,
        default=None,
        help=(
            "Optional hard stop on the cumulative oracle budget counter. "
            "By default every physically scored row is debited; oracle_best can "
            "explicitly use selected/archive accounting instead."
        ),
    )
    parser.add_argument(
        "--diagnostic-inject-full-dataset-best",
        action="store_true",
        dest="diagnostic_inject_full_dataset_best",
        default=None,
        help=(
            "Diagnostic only: inject the highest-scoring GFP/AAV full-DMS sequence "
            "into each terminal acquisition pool without giving its score to the GP."
        ),
    )
    parser.add_argument(
        "--no-diagnostic-inject-full-dataset-best",
        action="store_false",
        dest="diagnostic_inject_full_dataset_best",
    )
    parser.add_argument(
        "--diagnostic-full-dataset-best-source",
        type=str,
        default=None,
        help="Source for diagnostic full-dataset-best injection. Currently only 'dms' is supported.",
    )
    parser.add_argument(
        "--diagnostic-full-dataset-best-csv",
        type=str,
        default=None,
        help="Optional explicit ground_truth.csv path for diagnostic full-dataset-best injection.",
    )
    parser.add_argument(
        "--diagnostic-stop-on-injected-selection",
        action="store_true",
        dest="diagnostic_stop_on_injected_selection",
        default=None,
        help="Stop the run after the injected full-DMS best candidate is acquired and written out.",
    )
    parser.add_argument(
        "--no-diagnostic-stop-on-injected-selection",
        action="store_false",
        dest="diagnostic_stop_on_injected_selection",
    )
    parser.add_argument(
        "--diagnostic-injection-affects-collapse",
        action="store_true",
        dest="diagnostic_injection_affects_collapse",
        default=None,
        help=(
            "Let the injected diagnostic candidate count toward proposal-collapse acquisition "
            "availability. Disabled by default so collapse diagnostics describe the generated pool."
        ),
    )
    parser.add_argument(
        "--no-diagnostic-injection-affects-collapse",
        action="store_false",
        dest="diagnostic_injection_affects_collapse",
    )
    parser.add_argument(
        "--num-candidates", type=int, default=None, help="Number of candidate interpolant timesteps to test"
    )
    parser.add_argument(
        "--num-times",
        type=int,
        default=None,
        help="Alias for num-candidates (number of candidate times to sample).",
    )
    parser.add_argument(
        "--candidate-strategy",
        type=str,
        default=None,
        help="Candidate strategy: uniform, random, or trust_region",
    )
    parser.add_argument(
        "--proposal-operator",
        choices=(
            "generator",
            "hamming",
            "adalead",
            "adalead_official",
            "adalead_evodiff",
            "bo_evo",
        ),
        default=None,
        help="Fixed-sequence candidate-pool operator; defaults to the generator.",
    )
    parser.add_argument(
        "--generator-exact-corruptions",
        type=int,
        default=None,
        help=(
            "Condition each generator interpolant draw on exactly this many "
            "proposal-corrupted positions. Intended for small, fixed-edit "
            "genealogy experiments; combine with --rollout-use-corrupt-mask "
            "to bound the realised child edits by the same count."
        ),
    )
    parser.add_argument("--baseline-population-size", type=int, default=None)
    parser.add_argument(
        "--baseline-generations",
        type=int,
        default=None,
        help="Maximum virtual generations; population times generations must cover the proposal pool.",
    )
    parser.add_argument("--baseline-fitness-threshold", type=float, default=None)
    parser.add_argument("--baseline-mutation-edits", type=int, default=None)
    parser.add_argument(
        "--baseline-adalead-surrogate-queries",
        type=int,
        default=None,
        help="Strict FLEXS AdaLead model-query budget, including root predictions.",
    )
    parser.add_argument("--baseline-adalead-paid-batch-size", type=int, default=None)
    parser.add_argument(
        "--baseline-adalead-mu",
        type=float,
        default=None,
        help="FLEXS AdaLead expected mutations per sequence (per-site probability is mu/L).",
    )
    parser.add_argument(
        "--baseline-adalead-eval-batch-size",
        type=int,
        default=None,
        help="FLEXS AdaLead rollout evaluation chunk size (upstream default: 20).",
    )
    parser.add_argument("--baseline-adalead-rho", type=int, default=None)
    parser.add_argument("--baseline-adalead-recombination-rate", type=float, default=None)
    parser.add_argument("--baseline-adalead-alphabet", type=str, default=None)
    parser.add_argument("--baseline-adalead-rng-backend", type=str, default=None)
    parser.add_argument(
        "--baseline-parent-temperature",
        type=float,
        default=None,
        help="BO-EVO fitness-proportional parent softmax temperature (default: 10).",
    )
    parser.add_argument(
        "--baseline-ucb-beta",
        type=float,
        default=None,
        help="BO-EVO upper-confidence-bound uncertainty weight (default: 0.2).",
    )
    parser.add_argument(
        "--baseline-children-per-selection",
        type=int,
        default=None,
        help="BO-EVO mutation children screened for each paid selection (default: 8).",
    )
    parser.add_argument(
        "--candidate-source",
        type=str,
        default=None,
        help="Which sequence to mix with the prior: best or random",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Number of top sequences to use when candidate_source=topk",
    )
    parser.add_argument(
        "--topk-archive-size",
        type=int,
        default=None,
        help="Max number of entries kept in the weighted top-k archive.",
    )
    parser.add_argument(
        "--topk-weight-n",
        type=float,
        default=None,
        help="Weight scale n for archive sampling weights exp(n * reward).",
    )
    parser.add_argument(
        "--topk-archive-d-min",
        type=int,
        default=None,
        help="Minimum Hamming distance for diversity-aware archive admission/pruning.",
    )
    parser.add_argument(
        "--topk-use-archive",
        action="store_true",
        dest="topk_use_archive",
        default=None,
        help="Use weighted archive when candidate_source=topk.",
    )
    parser.add_argument(
        "--no-topk-use-archive",
        action="store_false",
        dest="topk_use_archive",
    )
    parser.add_argument(
        "--topk-unique",
        action="store_true",
        dest="topk_unique",
        default=None,
        help="Ensure topk candidate_source selects unique sequences.",
    )
    parser.add_argument(
        "--no-topk-unique",
        action="store_false",
        dest="topk_unique",
    )
    parser.add_argument(
        "--mixing-mode",
        type=str,
        default=None,
        help="How to mix with the prior: random or invalid (uniform interpolant only)",
    )
    parser.add_argument(
        "--mixes-per-time",
        type=int,
        default=None,
        help="How many independent mixes to draw per time value.",
    )
    parser.add_argument(
        "--transition-mask-path",
        type=str,
        default=None,
        help="Optional transition mask path for invalid mixing.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=None,
        help="Pool size of candidate timesteps before selection (>= num-candidates)",
    )
    parser.add_argument(
        "--tr-mean", type=float, default=None, help="Trust-region center as t in [0,1] (compatibility alias)."
    )
    parser.add_argument(
        "--tr-domain",
        type=str,
        default=None,
        help="Trust-region state domain: edits (default) or time.",
    )
    parser.add_argument("--tr-sigma", type=float, default=None, help="Deprecated (truncnorm removed).")
    parser.add_argument("--tr-sigma-min", type=float, default=None, help="Deprecated (truncnorm removed).")
    parser.add_argument("--tr-sigma-max", type=float, default=0.5, help="Deprecated (truncnorm removed).")
    parser.add_argument("--tr-expand", type=float, default=1.05, help="Trust-region expansion factor.")
    parser.add_argument("--tr-shrink", type=float, default=0.8, help="Trust-region shrink factor.")
    parser.add_argument("--tr-eps", type=float, default=1e-6, help="Improvement threshold for expansion.")
    parser.add_argument("--tr-log-every", type=int, default=None, help="Log trust-region stats every N iterations.")
    parser.add_argument(
        "--tr-edits-delta",
        type=float,
        default=None,
        help="Fixed shift in trust-region center per update, in expected edit units.",
    )
    parser.add_argument(
        "--time-min",
        type=float,
        default=None,
        help="Minimum trust-region center in time space when tr-domain=time.",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=None,
        help="Maximum trust-region center in time space when tr-domain=time.",
    )
    parser.add_argument(
        "--tr-edits-min",
        type=float,
        default=None,
        help="Minimum expected edit budget for trust-region center/sampling.",
    )
    parser.add_argument(
        "--edits-min",
        type=float,
        default=None,
        help="Alias for tr-edits-min (used by ScalarTrustRegion).",
    )
    parser.add_argument(
        "--edits-max",
        type=float,
        default=None,
        help="Maximum expected edit budget (used by ScalarTrustRegion).",
    )
    parser.add_argument(
        "--edits-init",
        type=float,
        default=None,
        help="Initial edit budget for ScalarTrustRegion (defaults to tr-edits/tr-mean mapping).",
    )
    parser.add_argument(
        "--tau-success",
        type=int,
        default=None,
        help="Consecutive successes required before expanding ScalarTrustRegion.",
    )
    parser.add_argument(
        "--tau-failure",
        type=int,
        default=None,
        help="Consecutive failures required before shrinking ScalarTrustRegion.",
    )
    parser.add_argument(
        "--region-scalar",
        type=float,
        default=None,
        help="Backward-compatible scale factor for ScalarTrustRegion expansion and contraction.",
    )
    parser.add_argument(
        "--region-expand-scalar",
        type=float,
        default=None,
        help="Scale factor for ScalarTrustRegion expansion updates; in time domain this expands noise radius 1-t.",
    )
    parser.add_argument(
        "--region-contract-scalar",
        type=float,
        default=None,
        help="Reciprocal contraction scale for ScalarTrustRegion shrink updates.",
    )
    parser.add_argument(
        "--region-shrink-scalar",
        type=float,
        default=None,
        help="Multiplicative shrink factor for ScalarTrustRegion contraction updates; in time domain this shrinks noise radius 1-t.",
    )
    parser.add_argument(
        "--tr-alpha",
        type=float,
        default=None,
        help="Multiplicative time scaling factor applied as t_used = clamp(alpha * t_base).",
    )
    parser.add_argument(
        "--tr-alpha-k",
        type=float,
        default=None,
        help=(
            "Sensitivity for calibration from normalized mean edit error "
            "(expected - actual, scaled by editable tokens). In time-domain TR this updates alpha; "
            "in edit-domain TR this updates mean_edits directly."
        ),
    )
    parser.add_argument(
        "--legacy-hamming-tr-error",
        action="store_true",
        dest="legacy_hamming_tr_error",
        default=None,
        help="Use plain Hamming distance for ancestor_to_child_edits in trust-region error updates and traces.",
    )
    parser.add_argument(
        "--no-legacy-hamming-tr-error",
        action="store_false",
        dest="legacy_hamming_tr_error",
    )
    parser.add_argument(
        "--tr-alpha-min",
        type=float,
        default=None,
        help="Lower clamp for trust-region alpha.",
    )
    parser.add_argument(
        "--tr-alpha-max",
        type=float,
        default=None,
        help="Upper clamp for trust-region alpha.",
    )
    parser.add_argument(
        "--tr-sampling",
        type=str,
        default=None,
        help="Trust-region sampling mode (only fixed_budget is supported).",
    )
    parser.add_argument(
        "--tr-type",
        type=str,
        default=None,
        help="Trust-region behaviour type: AdditiveTrustRegion or ScalarTrustRegion.",
    )
    parser.add_argument(
        "--tr-edits",
        type=int,
        default=None,
        help="Fixed trust-region center as expected token edits (alternative to tr-mean).",
    )
    parser.add_argument(
        "--elite-prior-k", type=int, default=None, help="Use top-k elites to build a per-position prior."
    )
    parser.add_argument("--elite-prior-alpha", type=float, default=0.1, help="Additive smoothing for elite prior.")
    parser.add_argument("--elite-prior-min", type=int, default=4, help="Minimum elite pool size to use prior.")
    parser.add_argument(
        "--elite-prior-weighted",
        action="store_true",
        dest="elite_prior_weighted",
        default=None,
        help="Weight elite contributions by their scores.",
    )
    parser.add_argument(
        "--no-elite-prior-weighted",
        action="store_false",
        dest="elite_prior_weighted",
    )
    parser.add_argument("--elite-prior-mix", type=float, default=1.0, help="Mix weight for elite prior vs uniform.")
    parser.add_argument(
        "--fixed-time",
        type=float,
        default=None,
        help="Use a single fixed timestep in [0,1] (overrides candidate sampling).",
    )
    parser.add_argument(
        "--fixed-times",
        type=float,
        nargs="*",
        default=None,
        help="Use fixed timesteps in [0,1] (overrides candidate sampling).",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override CTMC temperature for sampling")
    parser.add_argument(
        "--stop-when-filled",
        action="store_true",
        dest="stop_when_filled",
        help="Stop CTMC rollout early once masks are filled",
    )
    parser.add_argument("--no-stop-when-filled", action="store_false", dest="stop_when_filled")
    parser.set_defaults(stop_when_filled=None)
    parser.add_argument(
        "--rollout-use-corrupt-mask",
        action="store_true",
        dest="rollout_use_corrupt_mask",
        default=None,
        help="During rollout, only allow token updates at proposal-corrupted positions.",
    )
    parser.add_argument(
        "--no-rollout-use-corrupt-mask",
        action="store_false",
        dest="rollout_use_corrupt_mask",
    )
    parser.add_argument(
        "--target-sequence", type=str, default=None, help="Optional path to target tensor (.pt) for reward"
    )
    parser.add_argument(
        "--target-max", action="store_true", dest="target_max", help="Use a target sequence of all (vocab_size-1)"
    )
    parser.add_argument("--no-target-max", action="store_false", dest="target_max")
    parser.set_defaults(target_max=None)
    parser.add_argument(
        "--oracle", default=None, help="JSON object/string oracle config; equivalent to reward for lookup oracles."
    )
    parser.add_argument(
        "--generator", type=str, default=None, help="Generator backend for task configs, for example evodiff."
    )
    parser.add_argument(
        "--generation-mode", type=str, default=None, help="Generation mode for task configs, for example masked_infill."
    )
    parser.add_argument(
        "--candidate-key",
        type=str,
        default=None,
        help="Candidate key used for deduplication/scoring, for example mutant4.",
    )
    parser.add_argument(
        "--fixed-context", type=str, default=None, help="Fixed context sequence for context-conditioned protein tasks."
    )
    parser.add_argument(
        "--mutable-positions-0-indexed",
        type=int,
        nargs="*",
        default=None,
        help="Zero-indexed mutable positions for context-conditioned tasks.",
    )
    parser.add_argument(
        "--oracle-batch-size",
        type=int,
        default=None,
        help="Alias for al_oracle_batch_size in task configs.",
    )
    parser.add_argument("--reward", type=str, default=None, help="JSON string for reward config")
    parser.add_argument("--reward-config", type=str, default=None, help="Path to JSON reward config")
    parser.add_argument(
        "--objective-direction",
        type=str,
        default=None,
        help="Objective direction. The interpolant BO loop currently supports maximize.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Optional expected sequence length; checked against the restored process.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--leaves-out", type=str, default=None, help="Optional path to save per-iteration leaves (pt)")
    parser.add_argument("--print-leaves", action="store_true", help="Print leaf token sequences each iteration")
    parser.add_argument("--samples-out", type=str, default=None, help="Optional JSONL path to save sequences + scores")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory (optional)")
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        dest="profile_timings",
        default=None,
        help="Write per-stage wall-clock timing diagnostics as JSONL.",
    )
    parser.add_argument(
        "--no-profile-timings",
        action="store_false",
        dest="profile_timings",
    )
    parser.add_argument(
        "--timings-out",
        type=str,
        default=None,
        help="Optional JSONL path for timing diagnostics. Defaults beside samples_out when profiling is enabled.",
    )
    parser.add_argument(
        "--profile-sync-cuda",
        action="store_true",
        dest="profile_sync_cuda",
        default=None,
        help="Synchronize CUDA before/after each timed block for more accurate GPU timings.",
    )
    parser.add_argument(
        "--no-profile-sync-cuda",
        action="store_false",
        dest="profile_sync_cuda",
    )
    parser.add_argument(
        "--active-learning",
        action="store_true",
        dest="active_learning",
        default=None,
        help="Use a GP acquisition model to choose which rolled-out candidates receive oracle evaluation.",
    )
    parser.add_argument(
        "--no-active-learning",
        action="store_false",
        dest="active_learning",
    )
    parser.add_argument(
        "--al-oracle-batch-size",
        type=int,
        default=None,
        help="Maximum number of rolled-out candidates to send to the true oracle each iteration when active_learning is enabled.",
    )
    parser.add_argument(
        "--al-oracle-best-budget-mode",
        choices=("physical", "selected_archive"),
        default=None,
        help=(
            "How max_oracle_evaluations is debited for oracle_best acquisition: "
            "physical counts every oracle-scored proposal (default), while "
            "selected_archive counts only proposals selected into the archive."
        ),
    )
    parser.add_argument(
        "--al-initial-batch-matches-buffer-size",
        action="store_true",
        dest="al_initial_batch_matches_buffer_size",
        default=None,
        help="When active_learning is enabled, use al_buffer_size as the initial sampled batch size used to seed the GP buffer.",
    )
    parser.add_argument(
        "--no-al-initial-batch-matches-buffer-size",
        action="store_false",
        dest="al_initial_batch_matches_buffer_size",
    )
    parser.add_argument(
        "--al-sequential-selection",
        action="store_true",
        dest="al_sequential_selection",
        default=None,
        help="Select oracle points one at a time, retraining the GP buffer after each evaluation.",
    )
    parser.add_argument(
        "--no-al-sequential-selection",
        action="store_false",
        dest="al_sequential_selection",
    )
    parser.add_argument(
        "--al-local-penalization",
        action="store_true",
        dest="al_local_penalization",
        default=None,
        help="Use batch Bayesian optimization style local penalization when choosing multiple oracle points from one GP fit.",
    )
    parser.add_argument(
        "--no-al-local-penalization",
        action="store_false",
        dest="al_local_penalization",
    )
    parser.add_argument(
        "--al-local-penalization-strength",
        type=float,
        default=None,
        help="Strength of the local penalty applied after each chosen batch point; larger values push the selector further away from already chosen fingerprints.",
    )
    parser.add_argument(
        "--al-local-penalization-power",
        type=float,
        default=None,
        help="Similarity power used by local penalization; larger values make the penalty concentrate more on very similar fingerprints.",
    )
    parser.add_argument(
        "--al-acquisition",
        type=str,
        default=None,
        help=(
            "Active-learning acquisition rule: expected_improvement, weighted_expected_improvement, "
            "batched_expected_improvement, weighted_batched_expected_improvement, "
            "explore_then_batched_expected_improvement, random, thompson, or slotwise_thompson."
        ),
    )
    parser.add_argument(
        "--al-qei-mc-samples",
        type=int,
        default=None,
        help="Monte Carlo sample count used by batched_expected_improvement (qEI).",
    )
    parser.add_argument(
        "--al-explore-batch-size",
        type=int,
        default=None,
        help="Number of high-uncertainty candidates to select before qEI in explore_then_batched_expected_improvement.",
    )
    parser.add_argument(
        "--al-exploit-batch-size",
        type=int,
        default=None,
        help="Optional number of qEI candidates after exploration; when omitted, qEI fills the remaining oracle budget.",
    )
    parser.add_argument(
        "--al-wei-lambda",
        type=float,
        default=None,
        help=(
            "Kernel penalty strength for weighted EI acquisitions. The selector uses "
            "log(EI + eps) + lambda * logK, or the analogous qEI form for weighted batched EI."
        ),
    )
    parser.add_argument(
        "--al-wei-standardize",
        action="store_true",
        dest="al_wei_standardize",
        default=None,
        help="Use z-scored log EI/qEI and z-scored local-kernel log probability for weighted EI.",
    )
    parser.add_argument(
        "--no-al-wei-standardize",
        action="store_false",
        dest="al_wei_standardize",
    )
    parser.add_argument(
        "--al-wei-eps",
        type=float,
        default=None,
        help="Numerical epsilon used inside log(EI + eps) for weighted EI acquisitions.",
    )
    parser.add_argument(
        "--al-thompson-above-incumbent-only",
        action="store_true",
        dest="al_thompson_above_incumbent_only",
        default=None,
        help="In Thompson mode, only allow candidates whose sampled posterior value exceeds the best observed oracle score.",
    )
    parser.add_argument(
        "--no-al-thompson-above-incumbent-only",
        action="store_false",
        dest="al_thompson_above_incumbent_only",
    )
    parser.add_argument(
        "--al-ei-threshold",
        type=float,
        default=None,
        help="Optional EI threshold; if no candidate exceeds it, the proposal batch can be resampled.",
    )
    parser.add_argument(
        "--al-debug-random-sample-count",
        type=int,
        default=None,
        help="When active_learning is enabled, print this many random candidate predictions per batch for debugging.",
    )
    parser.add_argument(
        "--al-score-full-proposal-batch",
        action="store_true",
        dest="al_score_full_proposal_batch",
        default=None,
        help=(
            "When active_learning is enabled, score every acquisition-eligible proposal for diagnostics "
            "after selecting the oracle batch. These extra scores are written out but are not fed back into the GP."
        ),
    )
    parser.add_argument(
        "--no-al-score-full-proposal-batch",
        action="store_false",
        dest="al_score_full_proposal_batch",
    )
    parser.add_argument(
        "--al-max-resample-batches",
        type=int,
        default=None,
        help="Maximum number of extra proposal batches to try when a batch yields no candidate above the EI threshold.",
    )
    parser.add_argument(
        "--proposal-collapse-min-acq-frac",
        type=float,
        default=None,
        help=(
            "For active-learning random-time or trust-region runs, treat a proposal batch as collapsed when fewer "
            "than this fraction of generated proposals remain acquisition-eligible after duplicate/invalid/cache "
            "filtering."
        ),
    )
    parser.add_argument(
        "--proposal-collapse-min-acq-count",
        type=int,
        default=None,
        help=(
            "For active-learning random-time or trust-region runs, treat a proposal batch as collapsed when fewer "
            "than this many proposals remain acquisition-eligible after duplicate/invalid/cache filtering."
        ),
    )
    parser.add_argument(
        "--proposal-collapse-max-resample-batches",
        type=int,
        default=None,
        help=(
            "Maximum collapsed proposal batches to discard per search iteration by default. Trust-region runs "
            "expand one step toward the higher-noise state before resampling; random-time runs redraw candidate "
            "times. Use --proposal-collapse-top-up to keep eligible candidates and generate only the missing count."
        ),
    )
    parser.add_argument(
        "--proposal-collapse-top-up",
        action="store_true",
        dest="proposal_collapse_top_up",
        default=None,
        help=(
            "Use proposal-collapse top-up mode: keep acquisition-eligible candidates from a collapsed pool, "
            "expand/redraw time, and generate only the missing proposal count until the threshold or cutoff."
        ),
    )
    parser.add_argument(
        "--no-proposal-collapse-top-up",
        action="store_false",
        dest="proposal_collapse_top_up",
    )
    parser.add_argument(
        "--al-buffer-size",
        type=int,
        default=None,
        help="Maximum number of oracle-evaluated molecules retained in the GP training buffer.",
    )
    parser.add_argument(
        "--al-buffer-mode",
        type=str,
        default=None,
        help="Rolling buffer policy for active learning: recent or top.",
    )
    parser.add_argument(
        "--al-buffer-diversity-max-similarity",
        type=float,
        default=None,
        help="Optional greedy diversity filter on the GP buffer: reject entries with Tanimoto similarity >= this threshold to a higher-priority retained entry.",
    )
    parser.add_argument(
        "--al-feature-space",
        type=str,
        default=None,
        help=(
            "Active-learning feature space: onehot_tanimoto, "
            "ehrlich_onehot, ehrlich_diffusion, ehrlich_spaced_ngram, ehrlich_spaced_pair, "
            "ehrlich_spaced_trigram, ehrlich_local_spaced_ngram, "
            "ehrlich_local_spaced_ngram_additive, ehrlich_trigram, "
            "esm, or kermut."
        ),
    )
    parser.add_argument(
        "--feature-space",
        type=str,
        default=None,
        help="Alias for al_feature_space in JSON search configs.",
    )
    parser.add_argument(
        "--al-kermut-resources",
        type=str,
        default=None,
        help="Kermut reference NPZ: wild_type, alphabet, conditional_probs [L,20], coords [L,3].",
    )
    parser.add_argument(
        "--al-onehot-max-len",
        type=int,
        default=None,
        help="Fixed sequence length for onehot_tanimoto active-learning features.",
    )
    parser.add_argument(
        "--al-onehot-alphabet",
        type=str,
        default=None,
        help="Alphabet used by onehot_tanimoto active-learning features.",
    )
    parser.add_argument(
        "--al-ehrlich-diffusion-tau",
        type=float,
        default=None,
        help="Heat-kernel diffusion time for ehrlich_diffusion active learning, default 1.0.",
    )
    parser.add_argument(
        "--al-ehrlich-black-box-feasibility",
        action="store_true",
        dest="al_ehrlich_black_box_feasibility",
        default=False,
        help=(
            "Do not use the Ehrlich transition mask before acquisition. Paid invalid "
            "oracle outcomes remain non-finite in traces but train the GP with target 0."
        ),
    )
    parser.add_argument(
        "--no-al-ehrlich-black-box-feasibility",
        action="store_false",
        dest="al_ehrlich_black_box_feasibility",
    )
    parser.add_argument(
        "--al-ehrlich-diffusion-combine",
        choices=("mean", "sum", "product"),
        default=None,
        help="How ehrlich_diffusion combines per-position similarities, default mean.",
    )
    parser.add_argument(
        "--al-ehrlich-diffusion-use-transition-weights",
        "--al-ehrlich-use-transition-weights",
        action="store_true",
        dest="al_ehrlich_diffusion_use_transition_weights",
        default=None,
        help="Use Ehrlich transition probabilities as graph weights (default).",
    )
    parser.add_argument(
        "--no-al-ehrlich-diffusion-use-transition-weights",
        "--no-al-ehrlich-use-transition-weights",
        action="store_false",
        dest="al_ehrlich_diffusion_use_transition_weights",
        help="Use only the Ehrlich transition support graph, without probability weights.",
    )
    parser.add_argument(
        "--al-ehrlich-spaced-ngram-patterns",
        choices=("instance", "regular", "local_pair", "local_trigram", "local_all"),
        default=None,
        help=(
            "Offset patterns for ehrlich_spaced_ngram: instance uses the benchmark's motif gaps "
            "without revealing motif tokens; regular uses a fixed gap; local_pair uses every "
            "pair separation within a generic local window; local_trigram uses every three-token "
            "offset pattern within that window; local_all uses every n-token offset pattern and "
            "requires --al-ehrlich-spaced-ngram-n. Default: instance."
        ),
    )
    parser.add_argument(
        "--al-ehrlich-spaced-ngram-n",
        type=int,
        default=None,
        help="Number of tokens per spaced n-gram; default: the Ehrlich motif length.",
    )
    parser.add_argument(
        "--al-ehrlich-spaced-ngram-orders",
        type=int,
        nargs="+",
        default=None,
        help="Orders combined by an additive local spaced n-gram kernel, for example 3 4.",
    )
    parser.add_argument(
        "--al-ehrlich-spaced-ngram-weights",
        type=float,
        nargs="+",
        default=None,
        help="Fixed positive mixture weights for the additive local spaced n-gram orders.",
    )
    parser.add_argument(
        "--al-ehrlich-spaced-ngram-windows",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional per-component local windows aligned with "
            "--al-ehrlich-spaced-ngram-orders. When omitted, every additive component "
            "uses --al-ehrlich-spaced-ngram-window (default 8)."
        ),
    )
    parser.add_argument(
        "--al-ehrlich-spaced-ngram-gap",
        type=int,
        default=None,
        help="Gap between tokens when spaced n-gram patterns=regular; default: 1 (contiguous).",
    )
    parser.add_argument(
        "--al-ehrlich-spaced-ngram-window",
        type=int,
        default=None,
        help=(
            "Window size for local patterns and the shared additive fallback; "
            "default: 8 residues (separations 1 through 7)."
        ),
    )
    parser.add_argument(
        "--al-esm-model",
        type=str,
        default=None,
        help="ESM model name: defaults to esm2_t33_650M_UR50D for Kermut, esm2_t30_150M_UR50D otherwise.",
    )
    parser.add_argument(
        "--al-esm-batch-size",
        type=int,
        default=None,
        help="Batch size for ESM protein featurization.",
    )
    parser.add_argument(
        "--al-gp-device",
        type=str,
        default=None,
        help="Device for the active-learning GP, for example cpu or cuda.",
    )
    parser.add_argument(
        "--al-gp-dtype",
        type=str,
        default=None,
        help="Torch dtype string for the active-learning GP, for example double or float.",
    )
    parser.add_argument(
        "--al-gp-jitter",
        type=float,
        default=None,
        help="Optional Cholesky jitter used during active-learning GP fit and posterior evaluation.",
    )
    parser.add_argument(
        "--al-gp-noise-variance",
        type=float,
        default=None,
        help="Optional fixed observation-noise variance for the active-learning GP.",
    )
    parser.add_argument(
        "--al-gp-outputscale",
        type=float,
        default=None,
        help="Optional fixed outputscale for the active-learning GP kernel.",
    )
    parser.add_argument(
        "--al-gp-learn-outputscale",
        action="store_true",
        dest="al_gp_learn_outputscale",
        default=None,
        help="Learn the active-learning GP kernel outputscale during fitting.",
    )
    parser.add_argument(
        "--no-al-gp-learn-outputscale",
        action="store_false",
        dest="al_gp_learn_outputscale",
    )
    parser.add_argument(
        "--al-gp-fit-mean-and-noise-only",
        action="store_true",
        dest="al_gp_fit_mean_and_noise_only",
        default=False,
        help=("Fit only the constant mean and likelihood noise, with the GP outputscale fixed at 1.0."),
    )
    parser.add_argument(
        "--no-al-gp-fit-mean-and-noise-only",
        action="store_false",
        dest="al_gp_fit_mean_and_noise_only",
    )
    parser.add_argument(
        "--al-gp-hyperparameter-refit-interval",
        type=int,
        default=1,
        help=(
            "Refit trainable GP hyperparameters every N archive updates while "
            "conditioning the posterior on all observations every update."
        ),
    )
    parser.add_argument(
        "--al-gp-lengthscale",
        type=float,
        default=None,
        help=(
            "RBF lengthscale: learned initial value for Kermut (default 1.0); "
            "fixed for other protein embedding GPs unless learning is enabled (default sqrt(feature_dim))."
        ),
    )
    parser.add_argument(
        "--al-gp-learn-lengthscale",
        action="store_true",
        dest="al_gp_learn_lengthscale",
        default=None,
        help="Learn the active-learning GP RBF lengthscale during fitting.",
    )
    parser.add_argument(
        "--no-al-gp-learn-lengthscale",
        action="store_false",
        dest="al_gp_learn_lengthscale",
    )
    parser.add_argument(
        "--write-proposal-diagnostics",
        action="store_true",
        dest="write_proposal_diagnostics",
        default=None,
        help="Write per-iteration proposal filtering diagnostics as JSONL.",
    )
    parser.add_argument(
        "--no-write-proposal-diagnostics",
        action="store_false",
        dest="write_proposal_diagnostics",
    )
    parser.add_argument(
        "--proposal-diagnostics-out",
        type=str,
        default=None,
        help="Optional JSONL path for per-iteration proposal filtering diagnostics.",
    )
    parser.add_argument(
        "--proposal-traces-out",
        type=str,
        default=None,
        help="Optional JSONL path for full proposal traces. Defaults beside samples_out when full-batch scoring is enabled.",
    )
    parser.add_argument(
        "--protein-mode",
        action="store_true",
        default=False,
        help="Use protein reward/ESM defaults inside the normal CTMC interpolant search path.",
    )
    parser.add_argument(
        "--protein-model",
        type=str,
        default=None,
        help="Deprecated for CTMC protein search; use --checkpoint to select the CTMC model.",
    )
    parser.add_argument(
        "--protein-objective",
        type=str,
        default=None,
        help="Protein target objective for protein-mode. Currently: toy_amp.",
    )
    parser.add_argument(
        "--protein-seq-len",
        type=int,
        default=None,
        help="Deprecated standalone protein BO option; normal CTMC search uses the checkpoint sequence length.",
    )
    parser.add_argument("--protein-min-len", type=int, default=None, help="Minimum valid protein sequence length.")
    parser.add_argument("--protein-max-len", type=int, default=None, help="Maximum valid protein sequence length.")
    parser.add_argument(
        "--protein-initial-size",
        type=int,
        default=None,
        help="Deprecated standalone protein BO option; normal CTMC search uses --batch-size or --al-buffer-size.",
    )
    parser.add_argument(
        "--protein-proposal-pool-size",
        type=int,
        default=None,
        help="Deprecated standalone protein BO option; normal CTMC search uses --num-candidates and --mixes-per-time.",
    )
    parser.add_argument(
        "--protein-repetition-penalty",
        type=float,
        default=None,
        help="Deprecated standalone EvoDiff option; ignored by normal CTMC protein search.",
    )
    parser.add_argument(
        "--protein-remix-mask-fraction",
        type=float,
        default=None,
        help="Deprecated standalone EvoDiff option; ignored by normal CTMC protein search.",
    )
    parser.add_argument(
        "--protein-allow-random-fallback",
        action="store_true",
        default=False,
        help="Deprecated standalone EvoDiff option; ignored by normal CTMC protein search.",
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    defaults = parser.parse_args([])
    args = parser.parse_args(raw_argv)
    args._defaults = vars(defaults)
    provided: set[str] = set()
    for token in raw_argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            provided.add(str(action.dest))
    args._provided = frozenset(provided)
    return args


__all__ = ["parse_args"]
