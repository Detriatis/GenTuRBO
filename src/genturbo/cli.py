"""Public local experiment and reporting commands."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genturbo", description="Reproduce GenTuRBO experiments locally")
    commands = parser.add_subparsers(dest="command", required=True)
    from genturbo.reproduce import add_parser
    add_parser(commands)
    from genturbo.training import add_parser as add_training
    add_training(commands)
    from genturbo.reporting import add_parser as add_report
    add_report(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        return int(result or 0)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
