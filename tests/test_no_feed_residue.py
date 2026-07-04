"""MECHANICAL guard for hard-rule #1: the task-console redesign dropped the old
"feed"/"Feed" LAYER name -- the word must appear NOWHERE in the Python sources or
tests (the LogicNode / node vocabulary replaced it).

A naming rule that lives only in a .md gets silently violated (it already was: a
blind rename left "feed" residue scattered through the codebase).  So this pins it
as a test that FAILS the moment the banned word creeps back -- the same reason every
other mechanically-enforceable design rule in this repo is a test, not a doc line.

NOTE: "producer" is NOT banned.  It was once purged alongside "feed", but the
declarative *producer model* (signals grouped BY their producing node in the signal
picker -- the authoritative task-console design) re-adopted it as first-class design
vocabulary.  Only "feed" stays gone.

Allowed exception: the substring "feedback" (a different word).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# "feed" as a WHOLE word: a boundary that also rejects an underscore neighbour, since
# identifiers glue words with underscores (e.g. a stray ``camera_feed`` must be caught).
# "feedback" is a different word -- scrubbed out before the search below.
_BANNED = re.compile(r"(?<![A-Za-z0-9_])feed(?![A-Za-z0-9_])", re.IGNORECASE)


def _offenders():
    roots = [REPO_ROOT / "Zou_lab_control", REPO_ROOT / "tests", REPO_ROOT / "task_console.py"]
    hits: list[str] = []
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.py"))
    for path in files:
        # THIS guard file necessarily names the banned word (docstring + regex)
        if path.name == "test_no_feed_residue.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            scrubbed = line.replace("feedback", "").replace("Feedback", "")
            if _BANNED.search(scrubbed):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()[:120]}")
    return hits


def test_no_feed_vocabulary_anywhere():
    offenders = _offenders()
    assert not offenders, (
        "banned 'feed' vocabulary re-appeared (hard-rule #1 -- use the LogicNode / "
        "node vocabulary instead):\n  " + "\n  ".join(offenders))
