"""CLI entrypoint for interpolant-based CTMC search.

Implementation lives in `search.interpolant`.
"""

from __future__ import annotations

from search.interpolant.args import parse_args
from search.interpolant.engine import run_interpolant_search


def main() -> None:
    run_interpolant_search(parse_args())


if __name__ == "__main__":
    main()
