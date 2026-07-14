"""A per-site data axis must never inherit a producer's scan-grid geometry."""

from __future__ import annotations

import numpy as np
import pytest


def test_site_vector_structure_has_no_grid_shape(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import LogicNode, SignalSpec

    class _SiteVectorProducer(LogicNode):
        layer = "processor"
        node_label = "site vector"
        grid_shape = (5, 7)

        def shot(self):
            return {"sites": np.zeros(35, dtype=float)}

        def _bare_published_signals(self):
            return frozenset({"sites"})

        def _bare_output_specs(self):
            return (SignalSpec(
                "sites", "per-site value", points_shape=(1,), data_shape=(35,),
                dtype=np.float64, repeat_capacity=1),)

    hub = SignalHub()
    producer = _SiteVectorProducer(hub)
    producer.step()
    console = TaskConsole(
        hub=hub, state=default_console_state(), running_nodes=[producer], window_px=(900, 600))
    console._timer.stop()
    try:
        structure = console._signal_structure("sites")
        assert structure is not None
        assert tuple(structure["points_shape"]) == (1,)
        assert tuple(structure["data_shape"]) == (35,)
        assert tuple(structure["grid_shape"]) == ()
    finally:
        console.shutdown()


def test_site_underlay_uses_the_same_repeat_projection_as_the_site_values(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from Zou_lab_control.frontend.live import reduce_repeat
    from Zou_lab_control.frontend.task_console import PanelCard, PanelConfig

    sites = np.array([[[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]]])
    centers = np.array([[[[4.0, 4.0], [8.0, 8.0]]]])
    frames = np.stack([
        np.full((1, 4, 4), 10.0),
        np.full((1, 4, 4), 20.0),
        np.full((1, 4, 4), 40.0),
    ])
    namespace = {"sites": sites, "centers": centers, "frame": frames}

    def structure_provider(name):
        if name == "sites":
            return {"points_shape": (1,), "data_shape": (2,), "grid_shape": ()}
        return None

    def sites_inputs_provider(name):
        return ("centers", "frame") if name == "sites" else (None, None)

    def underlay(mode):
        card = PanelCard(
            PanelConfig(
                kind="sites", inputs=["sites"], source="value = signal",
                params={"repeat_mode": mode}),
            structure_provider=structure_provider,
            sites_inputs_provider=sites_inputs_provider,
        )
        try:
            _, image = card._sites_aux(namespace)
            return np.asarray(image, dtype=float)
        finally:
            card.shutdown()

    average = underlay("average")
    latest = underlay("replace")
    np.testing.assert_allclose(average, np.squeeze(reduce_repeat(frames, "average")))
    np.testing.assert_allclose(latest, np.squeeze(reduce_repeat(frames, "replace")))
    assert not np.array_equal(average, latest)
