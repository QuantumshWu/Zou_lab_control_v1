"""Contract: each panel shows a SIGNAL LEGEND in its footer (FRONTEND).

So the SignalHub namespace is never a mystery, every panel's footer states what it
READS from the hub, the node it reads FROM and what that node PROVIDES, and
flags any read whose name is published by MORE THAN ONE node (ambiguous).

Offscreen Qt; no demo GUI fixtures.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


class _LogicNode:
    """A node that only declares which signals it publishes."""

    def __init__(self, prefix, signals):
        self.prefix = prefix
        self.running = True
        self._signals = frozenset(signals)

    def published_signals(self):
        return self._signals

    def start(self, rate_hz=5.0):
        self.running = True
        return self

    def stop(self):
        self.running = False


def _console(nodes):
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    console = TaskConsole(hub=SignalHub(), state=default_console_state(), running_nodes=nodes)
    console._timer.stop()
    return console


def _add(console, kind, source):
    from Zou_lab_control.frontend.task_console import PanelConfig, PanelCard
    cfg = PanelConfig(kind=kind, title=source, source=source)
    card = PanelCard(cfg, names_provider=console.hub.names)
    console.cards.append(card)
    return card


def test_footer_legend_shows_reads_provides_and_duplicates():
    # "rate" is published by BOTH nodes -> ambiguous; "occupied" only by B.
    a = _LogicNode("", {"rate", "frame"})
    b = _LogicNode("b_", {"rate", "occupied"})
    console = _console([a, b])
    try:
        rate_card = _add(console, "monitor", "value = rate")
        occ_card = _add(console, "sites", "value = occupied")
        console._refresh_signal_info()

        info = rate_card._signal_info
        assert "reads: rate" in info
        assert "from" in info and "rate" in info          # the node it reads from + its signals
        assert "duplicated: rate" in info                 # rate is published by 2 nodes

        info2 = occ_card._signal_info
        assert "reads: occupied" in info2
        assert "duplicated" not in info2                  # occupied is unambiguous
        # the legend reaches the visible footer text, not just the attribute
        assert "reads: occupied" in occ_card.footer.text()
    finally:
        console.shutdown()
