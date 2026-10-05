"""The 10x10 Boxoban token contract; no solver or reward information."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

TILES = ("#", " ", "@", "$", ".", "*", "+")
TILE_TO_ID = {tile: index for index, tile in enumerate(TILES)}
MASK_ID = len(TILES)
HEIGHT = WIDTH = 10
SEQ_LEN = HEIGHT * WIDTH


def structural_errors(tokens: Sequence[int] | np.ndarray) -> list[str]:
    """Check geometry and entity counts, without claiming solvability.

    Geometry means a 10x10 rectangle enclosed by walls. Disconnected floor
    regions are allowed. Goals under boxes/players count as goals.
    """
    board = np.asarray(tokens)
    if board.shape not in {(SEQ_LEN,), (HEIGHT, WIDTH)}:
        return ["expected 100 tokens or a 10x10 grid"]
    if not np.issubdtype(board.dtype, np.integer) or np.any((board < 0) | (board >= len(TILES))):
        return ["unrecognised tile token (masks are not data tiles)"]
    board = board.reshape(HEIGHT, WIDTH)
    errors = []
    if np.any(board[[0, -1], :] != 0) or np.any(board[:, [0, -1]] != 0):
        errors.append("border must consist of walls")
    for name, ids, expected in (("player", (2, 6), 1), ("boxes", (3, 5), 4), ("goals", (4, 5, 6), 4)):
        count = int(np.isin(board, ids).sum())
        if count != expected:
            errors.append(f"expected {expected} {name}, found {count}")
    return errors


def encode_board(rows: Sequence[str]) -> np.ndarray:
    """Encode one strict board, preserving spaces rather than stripping rows."""
    if len(rows) != HEIGHT or any(len(row) != WIDTH for row in rows):
        raise ValueError("expected exactly ten rows of ten characters")
    try:
        tokens = np.array([TILE_TO_ID[tile] for row in rows for tile in row], dtype=np.uint8)
    except KeyError as exc:
        raise ValueError(f"unrecognised tile {exc.args[0]!r}") from exc
    errors = structural_errors(tokens)
    if errors:
        raise ValueError("; ".join(errors))
    return tokens


def decode_board(tokens: Sequence[int] | np.ndarray) -> str:
    """Render even structurally invalid samples; '?' denotes a remaining mask."""
    board = np.asarray(tokens)
    if board.shape != (SEQ_LEN,) or not np.issubdtype(board.dtype, np.integer):
        raise ValueError("expected a flat integer array of 100 tokens")
    if np.any((board < 0) | (board > MASK_ID)):
        raise ValueError("unrecognised tile token")
    alphabet = (*TILES, "?")
    return "\n".join("".join(alphabet[int(i)] for i in row) for row in board.reshape(HEIGHT, WIDTH))


def parse_levels(text: str, *, source: str = "<text>") -> Iterator[tuple[str, np.ndarray]]:
    """Read Boxoban '; id' records, failing loudly on malformed input."""
    level_id: str | None = None
    rows: list[str] = []
    seen: set[str] = set()

    def finish() -> tuple[str, np.ndarray]:
        try:
            return str(level_id), encode_board(rows)
        except ValueError as exc:
            raise ValueError(f"{source}, level {level_id}: {exc}") from exc

    for line in text.splitlines():
        if line.startswith(";"):
            if level_id is not None:
                yield finish()
            level_id = line[1:].strip()
            if not level_id or level_id in seen:
                raise ValueError(f"{source}: missing or duplicate level id {level_id!r}")
            seen.add(level_id)
            rows = []
        elif line == "":
            continue
        elif level_id is None:
            raise ValueError(f"{source}: board before first level id")
        else:
            rows.append(line)
    if level_id is not None:
        yield finish()
