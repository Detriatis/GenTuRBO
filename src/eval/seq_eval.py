"""Utilities for scoring and summarising discrete sequences."""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Iterable, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

RewardFn = Callable[[Tensor], float]


def _coerce_batch_scores(raw: object, *, expected: int, device: torch.device) -> Tensor:
    """Normalize custom reward score output to a flat float tensor of length expected."""
    if isinstance(raw, Tensor):
        scores = raw.to(device=device, dtype=torch.float)
    else:
        scores = torch.as_tensor(raw, dtype=torch.float, device=device)
    if scores.ndim == 0:
        scores = scores.view(1)
    scores = scores.flatten()
    if scores.numel() != expected:
        raise ValueError(f"reward score_batch returned {scores.numel()} scores, expected {expected}")
    return scores


def unique_fraction(x: Tensor) -> float:
    """Compute the fraction of unique sequences in a batch."""
    if x.ndim <= 1:
        return 1.0
    if x.ndim != 2:
        raise ValueError(f"unique_fraction expects [B, L] or [L], got shape {tuple(x.shape)}")
    batch = x.shape[0]
    if batch == 0:
        return 0.0
    if x.device.type == "mps":
        unique_rows = torch.unique(x.detach().cpu(), dim=0).shape[0]
    else:
        unique_rows = torch.unique(x, dim=0).shape[0]
    return float(unique_rows) / float(batch)


def mean_pairwise_hamming(x: Tensor) -> float:
    """Average pairwise Hamming distance fraction across a batch."""
    if x.ndim == 1:
        return 0.0
    if x.ndim != 2:
        raise ValueError(f"mean_pairwise_hamming expects [B, L] or [L], got shape {tuple(x.shape)}")
    batch, length = x.shape
    if batch < 2 or length == 0:
        return 0.0
    x0 = x.unsqueeze(0)
    x1 = x.unsqueeze(1)
    neq = (x0 != x1).float()
    total_pairs = batch * (batch - 1)
    mean = neq.sum() / (float(total_pairs) * float(length))
    return float(mean.item())


def sequence_edit_distance(
    a: Tensor | Iterable[int],
    b: Tensor | Iterable[int],
    *,
    ignore_token_id: Optional[int] = None,
) -> int:
    """Levenshtein edit distance between two token sequences.

    When ``ignore_token_id`` is provided, all occurrences of that token are
    removed from both sequences before alignment. This is useful for treating
    padding as structural absence rather than a literal token identity.
    """

    def _to_list(seq: Tensor | Iterable[int]) -> list[int]:
        if isinstance(seq, Tensor):
            values = [int(v) for v in seq.detach().view(-1).cpu().tolist()]
        else:
            values = [int(v) for v in seq]
        if ignore_token_id is None:
            return values
        ignore = int(ignore_token_id)
        return [v for v in values if v != ignore]

    left = _to_list(a)
    right = _to_list(b)
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) < len(right):
        left, right = right, left

    prev = list(range(len(right) + 1))
    for i, left_tok in enumerate(left, start=1):
        curr = [i]
        for j, right_tok in enumerate(right, start=1):
            substitution = prev[j - 1] + (0 if left_tok == right_tok else 1)
            insertion = curr[j - 1] + 1
            deletion = prev[j] + 1
            curr.append(min(substitution, insertion, deletion))
        prev = curr
    return int(prev[-1])


def _normalized_token_list(
    seq: Tensor | Iterable[int],
    *,
    ignore_token_id: Optional[int] = None,
) -> list[int]:
    if isinstance(seq, Tensor):
        values = [int(v) for v in seq.detach().view(-1).cpu().tolist()]
    else:
        values = [int(v) for v in seq]
    if ignore_token_id is None:
        return values
    ignore = int(ignore_token_id)
    return [v for v in values if v != ignore]


def _edit_script(
    source: list[int],
    target: list[int],
) -> list[tuple[str, Optional[int], Optional[int], Optional[int], Optional[int]]]:
    n = len(source)
    m = len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        s_tok = source[i - 1]
        for j in range(1, m + 1):
            t_tok = target[j - 1]
            sub_cost = 0 if s_tok == t_tok else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    ops_rev: list[tuple[str, Optional[int], Optional[int], Optional[int], Optional[int]]] = []
    i = n
    j = m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source[i - 1] == target[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops_rev.append(("equal", i - 1, j - 1, source[i - 1], target[j - 1]))
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops_rev.append(("replace", i - 1, j - 1, source[i - 1], target[j - 1]))
            i -= 1
            j -= 1
            continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops_rev.append(("delete", i - 1, None, source[i - 1], None))
            i -= 1
            continue
        ops_rev.append(("insert", None, j - 1, None, target[j - 1]))
        j -= 1

    ops_rev.reverse()
    return ops_rev


def _anchored_edit_counter(
    base: Tensor | Iterable[int],
    target: Tensor | Iterable[int],
    *,
    ignore_token_id: Optional[int] = None,
) -> Counter[tuple]:
    source = _normalized_token_list(base, ignore_token_id=ignore_token_id)
    dest = _normalized_token_list(target, ignore_token_id=ignore_token_id)
    ops = _edit_script(source, dest)

    counter: Counter[tuple] = Counter()
    src_cursor = 0
    insert_rank_by_slot: dict[int, int] = {}
    for kind, _, _, _, dst_tok in ops:
        if kind == "equal":
            src_cursor += 1
            continue
        if kind == "replace":
            counter[("replace", src_cursor, int(dst_tok))] += 1
            src_cursor += 1
            continue
        if kind == "delete":
            counter[("delete", src_cursor)] += 1
            src_cursor += 1
            continue
        if kind == "insert":
            rank = insert_rank_by_slot.get(src_cursor, 0)
            insert_rank_by_slot[src_cursor] = rank + 1
            counter[("insert", src_cursor, rank, int(dst_tok))] += 1
            continue
        raise RuntimeError(f"Unexpected edit op '{kind}'")
    return counter


def _logical_edit_result(
    proposal_edits: int,
    leaf_edits: int,
    model_edits: int,
    retained: int,
) -> Dict[str, Optional[float]]:
    retained_frac = float(retained) / float(proposal_edits) if proposal_edits > 0 else None
    return {
        "proposal_edits": float(proposal_edits),
        "leaf_edits": float(leaf_edits),
        "model_edits": float(model_edits),
        "retained_proposal_edits": float(retained),
        "dropped_proposal_edits": float(proposal_edits - retained),
        "introduced_leaf_edits": float(leaf_edits - retained),
        "retained_proposal_frac": retained_frac,
    }


def _positional_edit_diagnostics(
    base: Tensor | Iterable[int],
    proposal: Tensor | Iterable[int],
    leaf: Tensor | Iterable[int],
    *,
    ignore_token_id: Optional[int],
) -> Optional[Dict[str, Optional[float]]]:
    """Return O(L) diagnostics when all three sequences are fixed length.

    Fixed-length scientific sequences use positional substitutions as their
    logical edits. Sequences containing a configured padding token are not
    fixed length logically, so those (and physically unequal inputs) retain
    the alignment-based path below.
    """

    if isinstance(base, Tensor) and isinstance(proposal, Tensor) and isinstance(leaf, Tensor):
        base_flat = base.detach().reshape(-1)
        proposal_flat = proposal.detach().reshape(-1)
        leaf_flat = leaf.detach().reshape(-1)
        if not (
            base_flat.numel() == proposal_flat.numel() == leaf_flat.numel()
            and base_flat.device == proposal_flat.device == leaf_flat.device
        ):
            return None

        proposal_changed = proposal_flat != base_flat
        leaf_changed = leaf_flat != base_flat
        retained_mask = proposal_changed & (leaf_flat == proposal_flat)
        counts = [
            proposal_changed.sum(),
            leaf_changed.sum(),
            (proposal_flat != leaf_flat).sum(),
            retained_mask.sum(),
        ]
        if ignore_token_id is not None:
            ignore = int(ignore_token_id)
            counts.append((base_flat == ignore).sum() + (proposal_flat == ignore).sum() + (leaf_flat == ignore).sum())
        values = torch.stack(counts).detach().cpu().tolist()
        if ignore_token_id is not None and int(values[-1]) > 0:
            return None
        proposal_edits, leaf_edits, model_edits, retained = (int(value) for value in values[:4])
    else:
        base_values = _normalized_token_list(base)
        proposal_values = _normalized_token_list(proposal)
        leaf_values = _normalized_token_list(leaf)
        if not (len(base_values) == len(proposal_values) == len(leaf_values)):
            return None
        if ignore_token_id is not None:
            ignore = int(ignore_token_id)
            if any(ignore in values for values in (base_values, proposal_values, leaf_values)):
                return None

        proposal_edits = sum(left != right for left, right in zip(base_values, proposal_values, strict=True))
        leaf_edits = sum(left != right for left, right in zip(base_values, leaf_values, strict=True))
        model_edits = sum(left != right for left, right in zip(proposal_values, leaf_values, strict=True))
        retained = sum(
            base_token != proposal_token and proposal_token == leaf_token
            for base_token, proposal_token, leaf_token in zip(
                base_values,
                proposal_values,
                leaf_values,
                strict=True,
            )
        )

    return _logical_edit_result(proposal_edits, leaf_edits, model_edits, retained)


def logical_edit_diagnostics(
    base: Tensor | Iterable[int],
    proposal: Tensor | Iterable[int],
    leaf: Tensor | Iterable[int],
    *,
    ignore_token_id: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """Logical edit diagnostics relative to a shared ancestor sequence.

    Equal-length, unpadded sequences use positional substitution diagnostics.
    Variable-length and padded sequences use Levenshtein alignment.
    """

    positional = _positional_edit_diagnostics(
        base,
        proposal,
        leaf,
        ignore_token_id=ignore_token_id,
    )
    if positional is not None:
        return positional

    proposal_counter = _anchored_edit_counter(base, proposal, ignore_token_id=ignore_token_id)
    leaf_counter = _anchored_edit_counter(base, leaf, ignore_token_id=ignore_token_id)
    proposal_edits = int(sum(proposal_counter.values()))
    leaf_edits = int(sum(leaf_counter.values()))
    retained = int(sum(min(count, leaf_counter.get(key, 0)) for key, count in proposal_counter.items()))
    model_edits = int(sequence_edit_distance(proposal, leaf, ignore_token_id=ignore_token_id))
    return _logical_edit_result(proposal_edits, leaf_edits, model_edits, retained)


def score_batch(x: Tensor, reward_fn: RewardFn) -> Tensor:
    """
    Compute rewards for a batch of sequences.

    Args:
        x: Tensor of shape [B, L] or [L]
        reward_fn: Callable returning a float reward for a 1D sequence
    Returns:
        Tensor of shape [B] (or scalar if input is [L])
    """
    if x.ndim == 1:
        scorer = getattr(reward_fn, "score_batch", None)
        if callable(scorer):
            scores = _coerce_batch_scores(scorer(x), expected=1, device=x.device)
            return scores[0]
        return torch.tensor(float(reward_fn(x)), dtype=torch.float, device=x.device)
    if x.ndim != 2:
        raise ValueError(f"score_batch expects [B, L] or [L], got shape {tuple(x.shape)}")
    scorer = getattr(reward_fn, "score_batch", None)
    if callable(scorer):
        return _coerce_batch_scores(scorer(x), expected=x.shape[0], device=x.device)
    rewards = [float(reward_fn(x_i)) for x_i in x]
    return torch.tensor(rewards, dtype=torch.float, device=x.device)


def summarise_rewards(x: Tensor, rewards: Tensor) -> Dict[str, float]:
    """
    Compute simple reward statistics.

    Metrics: mean, max, median, p90, unique_frac (row-unique fraction of x).
    """
    rew = rewards.detach().flatten().float()
    if rew.numel() == 0:
        raise ValueError("summarise_rewards requires at least one reward value")
    metrics: Dict[str, float] = {
        "mean_reward": rew.mean().item(),
        "best_reward": rew.max().item(),
        "p50_reward": rew.median().item(),
        "p90_reward": torch.quantile(rew, 0.90).item(),
    }

    metrics["unique_frac"] = unique_fraction(x)
    return metrics


def incorrect_transition_counts(x: Tensor, transition_mask: Tensor) -> Tensor:
    """Count invalid transitions per sequence."""
    if x.ndim == 1:
        x = x.view(1, -1)
    if x.ndim != 2:
        raise ValueError(f"incorrect_transition_counts expects [B, L] or [L], got {tuple(x.shape)}")
    if transition_mask.ndim != 2 or transition_mask.shape[0] != transition_mask.shape[1]:
        raise ValueError("transition_mask must have shape [V, V]")
    prev = x[:, :-1]
    nxt = x[:, 1:]
    allowed = transition_mask.to(x.device)[prev, nxt]
    incorrect = (~allowed).sum(dim=1)
    return incorrect


def summarise_transition_errors(x: Tensor, transition_mask: Tensor) -> Dict[str, float]:
    """Summarise transition violations across a batch."""
    counts = incorrect_transition_counts(x, transition_mask).float()
    if counts.numel() == 0:
        raise ValueError("summarise_transition_errors requires at least one sequence")
    return {
        "avg_incorrect_transitions": counts.mean().item(),
        "max_incorrect_transitions": counts.max().item(),
    }


def evaluate_sequences(seqs: Tensor, reward_fn: RewardFn) -> tuple[Tensor, Dict[str, float]]:
    """Score sequences and return both per-sequence rewards and summary metrics."""
    rewards = score_batch(seqs, reward_fn)
    metrics = summarise_rewards(seqs, rewards if rewards.ndim > 0 else rewards.view(1))
    return rewards, metrics


def sample_and_score(
    dataset: IterableDataset,
    reward_fn: RewardFn,
    num_batches: int,
    *,
    dataloader: Optional[DataLoader] = None,
    dataloader_factory: Optional[Callable[[IterableDataset], DataLoader]] = None,
    writer: Optional[Callable[[dict[str, Tensor]], None]] = None,
    dataloader_kwargs: Optional[dict[str, object]] = None,
) -> list[dict[str, Tensor]]:
    """
    Sample ``num_batches`` from an iterable dataset and attach rewards.

    Records are returned and optionally streamed to ``writer`` for integration
    with existing storage/logging utilities.
    """
    if dataloader is None:
        if dataloader_factory is not None:
            dataloader = dataloader_factory(dataset)
        else:
            kwargs = {"batch_size": None, "num_workers": 0}
            if dataloader_kwargs:
                kwargs.update(dataloader_kwargs)
            dataloader = DataLoader(dataset, **kwargs)

    it = iter(dataloader)
    records: list[dict[str, Tensor]] = []
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)
        x = batch["x"]
        c = batch.get("c")
        rewards = score_batch(x, reward_fn)
        record: dict[str, Tensor] = {"x": x, "reward": rewards}
        if c is not None:
            record["c"] = c
        records.append(record)
        if writer is not None:
            writer(record)
    return records


__all__ = [
    "RewardFn",
    "logical_edit_diagnostics",
    "sequence_edit_distance",
    "mean_pairwise_hamming",
    "score_batch",
    "summarise_rewards",
    "unique_fraction",
    "evaluate_sequences",
    "sample_and_score",
    "incorrect_transition_counts",
    "summarise_transition_errors",
]
