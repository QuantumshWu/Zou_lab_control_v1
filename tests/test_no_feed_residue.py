"""MECHANICAL guard for hard-rule #1: the task-console redesign uses the LogicNode
vocabulary ONLY -- the words "feed"/"Feed" and the old "producer"/"Producer" layer
name must appear NOWHERE in the Python sources or tests.

A naming rule that lives only in a .md gets silently violated (it already was: a
blind rename left "feed"/"producer" residue scattered through the codebase).  So
this pins it as a test that FAILS the moment the banned vocabulary creeps back --
the same reason every other mechanically-enforceable design rule in this repo is a
test, not a doc paragraph.

Allowed exceptions (NOT the banned concept):
  * the substring "feedback" (a different word);
  * the vendor DCAM driver enums ``TIMESTAMP_PRODUCER`` / ``FRAMESTAMP_PRODUCER``
    (third-party API constants, not our architecture's "Producer").
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The banned tokens as WHOLE words (so "feedback" / "feed_forward"-free): a word
# boundary that also rejects an underscore neighbour, since identifiers glue words
# with underscores (e.g. a stray ``camera_feed`` must be caught).
_BANNED = re.compile(r"(?<![A-Za-z0-9_])(feed|producer)(?![A-Za-z0-9_])", re.IGNORECASE)
_ALLOWED_VENDOR = ("TIMESTAMP_PRODUCER", "FRAMESTAMP_PRODUCER")


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
        # the vendor DCAM driver legitimately ships *_PRODUCER enum constants
        if "dcam" in path.parts or path.name.startswith("dcamapi"):
            continue
        # THIS guard file necessarily names the banned words (docstring + regex)
        if path.name == "test_no_feed_residue.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            scrubbed = line
            for token in _ALLOWED_VENDOR:
                scrubbed = scrubbed.replace(token, "")
            scrubbed = scrubbed.replace("feedback", "").replace("Feedback", "")
            if _BANNED.search(scrubbed):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()[:120]}")
    return hits


def test_no_feed_or_producer_vocabulary_anywhere():
    offenders = _offenders()
    assert not offenders, (
        "banned 'feed'/'producer' vocabulary re-appeared (hard-rule #1 -- use the "
        "LogicNode / node vocabulary instead):\n  " + "\n  ".join(offenders))
