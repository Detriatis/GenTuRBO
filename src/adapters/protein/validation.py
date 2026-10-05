"""Protein sequence validation helpers."""

from __future__ import annotations

from typing import Iterable

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
CANONICAL_AA_SET = frozenset(CANONICAL_AA)


def normalize_sequence(seq: str) -> str:
    """Uppercase a sequence and remove whitespace separators."""
    return "".join(str(seq).upper().split())


def validate_sequence(
    seq: str,
    *,
    min_len: int = 1,
    max_len: int | None = None,
    alphabet: Iterable[str] = CANONICAL_AA,
) -> str:
    """Return a normalized canonical amino-acid sequence or raise ``ValueError``."""
    if not isinstance(seq, str):
        raise TypeError("protein sequence must be a string")
    clean = normalize_sequence(seq)
    allowed = frozenset(str(aa).upper() for aa in alphabet)
    if len(clean) < int(min_len):
        raise ValueError(f"protein sequence length {len(clean)} is below min_len={int(min_len)}")
    if max_len is not None and len(clean) > int(max_len):
        raise ValueError(f"protein sequence length {len(clean)} exceeds max_len={int(max_len)}")
    invalid = sorted({aa for aa in clean if aa not in allowed})
    if invalid:
        raise ValueError(f"protein sequence contains non-canonical residues: {''.join(invalid)}")
    return clean


def is_valid_sequence(
    seq: str,
    *,
    min_len: int = 1,
    max_len: int | None = None,
    alphabet: Iterable[str] = CANONICAL_AA,
) -> bool:
    """Return whether ``seq`` passes ``validate_sequence``."""
    try:
        validate_sequence(seq, min_len=min_len, max_len=max_len, alphabet=alphabet)
    except (TypeError, ValueError):
        return False
    return True


def unique_valid_sequences(
    seqs: Iterable[str],
    *,
    min_len: int = 1,
    max_len: int | None = None,
    alphabet: Iterable[str] = CANONICAL_AA,
) -> list[str]:
    """Normalize, validate, and deduplicate sequences while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for seq in seqs:
        try:
            clean = validate_sequence(seq, min_len=min_len, max_len=max_len, alphabet=alphabet)
        except (TypeError, ValueError):
            continue
        if clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


__all__ = [
    "CANONICAL_AA",
    "CANONICAL_AA_SET",
    "is_valid_sequence",
    "normalize_sequence",
    "unique_valid_sequences",
    "validate_sequence",
]
