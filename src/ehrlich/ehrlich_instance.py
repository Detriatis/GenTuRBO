"""Ehrlich benchmark instance generation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import gcd, isfinite
from numbers import Integral

import torch
from torch import Tensor

from .ehrlich_generator import sample_markov_sequences


@dataclass
class EhrlichConfig:
    vocab_size: int
    seq_len: int
    motif_count: int
    motif_len: int
    quantization: int
    bandwidth: int | None = None
    temperature: float = 0.5
    motif_from_sequence: bool = False

    def validate(self) -> None:
        v = int(self.vocab_size)
        L = int(self.seq_len)
        c = int(self.motif_count)
        k = int(self.motif_len)
        q = int(self.quantization)
        if v < 2:
            raise ValueError("vocab_size must be >= 2")
        if L < 2:
            raise ValueError("seq_len must be >= 2")
        if c < 1:
            raise ValueError("motif_count must be >= 1")
        if k < 1:
            raise ValueError("motif_len must be >= 1")
        if q < 1:
            raise ValueError("quantization must be >= 1")
        if self.motif_from_sequence:
            if L < k:
                raise ValueError("seq_len must be >= motif_len when motif_from_sequence is true")
        else:
            if L < c * k:
                raise ValueError("seq_len must be >= motif_count * motif_len")
        if k % q != 0:
            raise ValueError("quantization must evenly divide motif_len")
        if q > k:
            raise ValueError("quantization must not exceed motif_len")
        if not isfinite(float(self.temperature)) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.bandwidth is not None:
            if isinstance(self.bandwidth, bool) or not isinstance(self.bandwidth, Integral):
                raise ValueError("bandwidth must be an integer")
            bandwidth = int(self.bandwidth)
            if bandwidth < 1:
                raise ValueError("bandwidth must be positive")
            max_bandwidth = v // 2
            if bandwidth > max_bandwidth:
                raise ValueError(
                    f"bandwidth must be <= vocab_size // 2 ({max_bandwidth}) for a circular band"
                )


@dataclass
class EhrlichInstance:
    cfg: EhrlichConfig
    transition_probs: Tensor
    transition_mask: Tensor
    motifs: Tensor
    spacings: Tensor
    spacing_max: Tensor
    base_sequence: Tensor | None = None
    _indices_cache: dict[torch.device, list[Tensor]] = field(default_factory=dict, init=False, repr=False)

    def initial_solution(self, *, generator: torch.Generator | None = None) -> Tensor:
        if self.base_sequence is not None:
            return self.base_sequence.clone()
        seqs = sample_markov_sequences(
            self.transition_probs,
            self.cfg.seq_len,
            batch_size=1,
            generator=generator,
        )
        return seqs[0]

    def indices_for_device(self, device: torch.device) -> list[Tensor]:
        cached = self._indices_cache.get(device)
        if cached is not None:
            return cached
        indices: list[Tensor] = []
        L = int(self.cfg.seq_len)
        for s in self.spacings:
            s_dev = s.to(device)
            max_offset = int(s_dev.max().item())
            num_starts = L - max_offset
            if num_starts <= 0:
                raise ValueError("Spacing exceeds sequence length")
            starts = torch.arange(num_starts, device=device).view(-1, 1)
            idx = starts + s_dev.view(1, -1)
            indices.append(idx)
        self._indices_cache[device] = indices
        return indices


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def _banded_mask(vocab_size: int, bandwidth: int) -> Tensor:
    v = int(vocab_size)
    w = int(bandwidth)
    if w < 1 or w > v // 2:
        raise ValueError("bandwidth must be between 1 and vocab_size // 2")
    mask = torch.zeros((v, v), dtype=torch.bool)
    for i in range(v):
        for offset in range(-w, w + 1):
            j = (i + offset) % v
            mask[i, j] = True
    return mask


def _transition_matrix(cfg: EhrlichConfig, g: torch.Generator) -> tuple[Tensor, Tensor]:
    v = int(cfg.vocab_size)
    bandwidth = cfg.bandwidth if cfg.bandwidth is not None else max(1, (v * 2) // 5)
    mask = _banded_mask(v, bandwidth)
    perm = torch.randperm(v, generator=g)
    mask = mask[perm]
    mask[torch.arange(v), torch.arange(v)] = True

    logits = torch.randn((v, v), generator=g)
    probs = torch.softmax(logits / float(cfg.temperature), dim=1)
    probs = probs * mask
    row_sums = probs.sum(dim=1, keepdim=True)
    if (row_sums <= 0).any():
        raise ValueError("Transition matrix row had zero mass after masking")
    probs = probs / row_sums
    return probs, mask


def _sample_motifs(cfg: EhrlichConfig, transition_probs: Tensor, g: torch.Generator) -> Tensor:
    total_len = int(cfg.motif_count) * int(cfg.motif_len)
    seq = sample_markov_sequences(
        transition_probs,
        total_len,
        batch_size=1,
        generator=g,
    )[0]
    motifs = seq.view(int(cfg.motif_count), int(cfg.motif_len)).clone()
    return motifs


def _sample_motifs_from_sequence(
    cfg: EhrlichConfig, transition_probs: Tensor, g: torch.Generator
) -> tuple[Tensor, Tensor, Tensor]:
    L = int(cfg.seq_len)
    c = int(cfg.motif_count)
    k = int(cfg.motif_len)
    base_seq = sample_markov_sequences(
        transition_probs,
        L,
        batch_size=1,
        generator=g,
    )[0]
    motifs = torch.empty((c, k), dtype=torch.long)
    spacings = torch.empty((c, k), dtype=torch.long)
    for i in range(c):
        positions = torch.randperm(L, generator=g)[:k].sort().values
        motifs[i] = base_seq[positions]
        spacings[i] = positions - positions[0]
    return base_seq, motifs, spacings


def _sample_spacings(cfg: EhrlichConfig, g: torch.Generator) -> Tensor:
    c = int(cfg.motif_count)
    k = int(cfg.motif_len)
    slack = (int(cfg.seq_len) - c * k) // c
    spacings = torch.zeros((c, k), dtype=torch.long)
    if k == 1:
        return spacings
    for i in range(c):
        # Uniform weights normalized to sum to 1, using the provided generator.
        u = torch.rand(k - 1, generator=g)
        weights = u / u.sum()
        increments = 1 + torch.floor(weights * float(slack)).to(torch.long)
        s = torch.zeros(k, dtype=torch.long)
        s[0] = 0
        for j in range(1, k):
            s[j] = s[j - 1] + increments[j - 1]
        spacings[i] = s
    return spacings


def make_ehrlich_instance(cfg: EhrlichConfig, seed: int) -> EhrlichInstance:
    cfg.validate()
    g = _make_generator(seed)
    probs, mask = _transition_matrix(cfg, g)
    _validate_external_transition(cfg, probs, mask)
    base_sequence: Tensor | None = None
    if cfg.motif_from_sequence:
        base_sequence, motifs, spacings = _sample_motifs_from_sequence(cfg, probs, g)
    else:
        motifs = _sample_motifs(cfg, probs, g)
        spacings = _sample_spacings(cfg, g)
    spacing_max = spacings.max(dim=1).values
    return EhrlichInstance(
        cfg=cfg,
        transition_probs=probs,
        transition_mask=mask,
        motifs=motifs,
        spacings=spacings,
        spacing_max=spacing_max,
        base_sequence=base_sequence,
    )


def make_ehrlich_instance_from_transition(
    cfg: EhrlichConfig,
    seed: int,
    transition_probs: Tensor,
    transition_mask: Tensor,
) -> EhrlichInstance:
    cfg.validate()
    _validate_external_transition(cfg, transition_probs, transition_mask)
    g = _make_generator(seed)
    base_sequence: Tensor | None = None
    if cfg.motif_from_sequence:
        base_sequence, motifs, spacings = _sample_motifs_from_sequence(cfg, transition_probs, g)
    else:
        motifs = _sample_motifs(cfg, transition_probs, g)
        spacings = _sample_spacings(cfg, g)
    spacing_max = spacings.max(dim=1).values
    return EhrlichInstance(
        cfg=cfg,
        transition_probs=transition_probs,
        transition_mask=transition_mask,
        motifs=motifs,
        spacings=spacings,
        spacing_max=spacing_max,
        base_sequence=base_sequence,
    )


def _validate_external_transition(
    cfg: EhrlichConfig,
    transition_probs: Tensor,
    transition_mask: Tensor,
) -> None:
    """Validate a caller-supplied finite-state Markov transition kernel."""

    v = int(cfg.vocab_size)
    expected_shape = (v, v)
    if transition_probs.shape != expected_shape:
        raise ValueError(f"transition_probs must have shape ({v}, {v})")
    if transition_mask.shape != expected_shape:
        raise ValueError(f"transition_mask must have shape ({v}, {v})")
    if not torch.is_floating_point(transition_probs):
        raise TypeError("transition_probs must have a floating-point dtype")
    if transition_mask.dtype != torch.bool:
        raise TypeError("transition_mask must have dtype torch.bool")
    if not bool(torch.isfinite(transition_probs).all().item()):
        raise ValueError("transition_probs must contain only finite values")
    if bool((transition_probs < 0).any().item()):
        raise ValueError("transition_probs must be non-negative")

    row_sums = transition_probs.sum(dim=1)
    expected_sums = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, expected_sums, rtol=1e-5, atol=1e-6):
        raise ValueError("transition_probs must be row-stochastic")

    support = transition_probs > 0
    mask_on_probs_device = transition_mask.to(device=transition_probs.device)
    if not torch.equal(support, mask_on_probs_device):
        raise ValueError("transition_mask must exactly match the positive support of transition_probs")

    adjacency = support.detach().to(device="cpu", dtype=torch.bool)
    if not _is_strongly_connected(adjacency):
        raise ValueError("transition_probs must define an irreducible Markov chain")
    period = _markov_chain_period(adjacency)
    if period != 1:
        raise ValueError(
            f"transition_probs must define an aperiodic Markov chain (period={period})"
        )


def _is_strongly_connected(adjacency: Tensor) -> bool:
    """Return whether every state is reachable from every other state."""

    graph = adjacency.tolist()
    size = len(graph)

    def _reachable(*, transpose: bool) -> bool:
        seen = {0}
        queue: deque[int] = deque([0])
        while queue:
            node = queue.popleft()
            for neighbour in range(size):
                linked = graph[neighbour][node] if transpose else graph[node][neighbour]
                if linked and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return len(seen) == size

    return _reachable(transpose=False) and _reachable(transpose=True)


def _markov_chain_period(adjacency: Tensor) -> int:
    """Compute the common period of a strongly connected directed graph."""

    graph = adjacency.tolist()
    size = len(graph)
    distances = [-1] * size
    distances[0] = 0
    queue: deque[int] = deque([0])
    while queue:
        node = queue.popleft()
        for neighbour, linked in enumerate(graph[node]):
            if linked and distances[neighbour] < 0:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)

    period = 0
    for source, row in enumerate(graph):
        for destination, linked in enumerate(row):
            if linked:
                cycle_difference = distances[source] + 1 - distances[destination]
                period = gcd(period, abs(cycle_difference))
    return period


def construct_optimal_sequence(instance: EhrlichInstance) -> Tensor:
    if instance.base_sequence is not None:
        return instance.base_sequence.clone()
    cfg = instance.cfg
    L = int(cfg.seq_len)
    c = int(cfg.motif_count)
    k = int(cfg.motif_len)
    x = torch.empty((L,), dtype=torch.long)
    cursor = 0
    last_token: int | None = None

    for i in range(c):
        motif = instance.motifs[i]
        spacing = instance.spacings[i]
        max_offset = int(spacing.max().item())
        if cursor + max_offset >= L:
            raise ValueError("Motif placement exceeds sequence length")
        prev_pos = cursor + int(spacing[0].item())
        x[prev_pos] = int(motif[0].item())
        last_token = int(motif[0].item())

        for j in range(1, k):
            pos = cursor + int(spacing[j].item())
            if pos < prev_pos:
                raise ValueError("Non-monotone spacing detected in optimal construction")
            if last_token is None:
                raise RuntimeError("Unexpected missing token during construction")
            if pos > prev_pos + 1:
                x[prev_pos + 1 : pos] = last_token
            token = int(motif[j].item())
            x[pos] = token
            last_token = token
            prev_pos = pos

        cursor = cursor + max_offset + 1

    if last_token is None:
        raise RuntimeError("Failed to construct optimal sequence")
    if cursor < L:
        x[cursor:] = last_token
    return x


__all__ = [
    "EhrlichConfig",
    "EhrlichInstance",
    "construct_optimal_sequence",
    "make_ehrlich_instance_from_transition",
    "make_ehrlich_instance",
]
