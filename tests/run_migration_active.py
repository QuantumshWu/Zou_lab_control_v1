"""Run only tests explicitly admitted by the active-migration whitelist."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WHITELIST = ROOT / "tests" / "migration_active_tests.txt"


def _entries() -> tuple[str, ...]:
    entries = tuple(
        line.strip()
        for line in WHITELIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if entries != tuple(sorted(set(entries))):
        raise RuntimeError("migration test whitelist must be sorted and unique")
    for entry in entries:
        path = ROOT / entry
        if not entry.startswith("tests/test_") or path.suffix != ".py":
            raise RuntimeError(f"invalid migration test entry: {entry}")
        if not path.is_file():
            raise RuntimeError(f"missing migration test: {entry}")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        metavar="TEST[::NODE]",
        help="run a whitelisted file or node; repeat to select several",
    )
    parser.add_argument("-k", "--keyword")
    parser.add_argument("-x", "--exitfirst", action="store_true")
    parser.add_argument("--maxfail", type=int)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    entries = _entries()
    allowed = frozenset(entries)
    selected = tuple(args.select) if args.select else entries
    for node in selected:
        if node.split("::", 1)[0].replace("\\", "/") not in allowed:
            raise SystemExit(f"test is outside migration whitelist: {node}")

    pytest_args = list(selected)
    if args.keyword is not None:
        pytest_args.extend(("-k", args.keyword))
    if args.exitfirst:
        pytest_args.append("-x")
    if args.maxfail is not None:
        pytest_args.append(f"--maxfail={args.maxfail}")
    if args.quiet:
        pytest_args.append("-q")
    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
