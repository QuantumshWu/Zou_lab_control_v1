"""An immediate panel re-render must use the board's current coherent namespace."""

from __future__ import annotations

import numpy as np
import pytest


def test_rerender_uses_current_shot_not_stale_cache(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from Zou_lab_control.frontend import devtools

    console = devtools.demo_console(shots=2)
    try:
        card = next(item for item in console.cards if item.config.kind == "2d")
        stale = np.full(
            np.asarray(console.hub.latest("frame_0")).shape, -999.0, dtype=float)
        card._last_namespace = {"frame_0": stale}

        for _ in range(3):
            for node in console.running_nodes:
                node.step()
        current = np.asarray(console.hub.latest("frame_0"))

        captured = {}
        card.refresh = lambda namespace: captured.setdefault("namespace", namespace)
        card._rerender_last()

        namespace = captured.get("namespace")
        assert namespace is not None
        assert np.array_equal(np.asarray(namespace["frame_0"]), current)
        assert not np.array_equal(np.asarray(namespace["frame_0"]), stale)
    finally:
        session = console.session
        console.shutdown()
        session.close()
