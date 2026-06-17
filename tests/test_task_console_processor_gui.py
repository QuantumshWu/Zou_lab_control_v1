"""Contract: a Data-processing action is a LOGIC NODE that runs end-to-end.

A processor is a logic node (Logic tab), added STOPPED, with an auto-generated Edit
form from the spec's ParamDecls (incl. the 'text' folder-path kind).  Start runs the
action ONCE on its own thread: a ProcessorRun publishes the result to the hub (display
suppressed -- it makes no plot).  You then add a Plot panel on the Monitor board
pointed at a published signal to view it.

Offscreen Qt + the saved-frames virtual==real path (na.write_virtual_run); no demo
GUI fixtures.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
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


def _add_processor_node(console, name):
    kc = console.kind_combo
    idx = next(j for j in range(kc.count()) if kc.itemData(j) == ("processor", name))
    assert kc.itemText(idx).startswith("Processor:")
    kc.setCurrentIndex(idx)
    console._add_panel()
    row = console.logic_nodes[-1]
    return row, console._logic_editors[id(row)]


def test_processor_node_runs_and_publishes(tmp_path):
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    data_dir = tmp_path / "run"
    na.write_virtual_run(str(data_dir), prefix="img", groups=40, shots_per_group=4,
                         short_shot=3, ref_shots=(1, 2, 4), grid_shape=(4, 5),
                         loading_probability=0.5, seed=4)
    exp = na.connect("virtual", sitemap={"grid_shape": (4, 5)})
    exp.readout.sitemap_from_dir(str(data_dir), prefix="img", method="psf")

    console = TaskConsole(hub=SignalHub(), state=default_console_state(),
                          processors=exp.readout.processor_specs(), session=exp)
    console._timer.stop()
    try:
        row, editor = _add_processor_node(console, "Readout fidelity")
        assert row.node.kind == "processor"
        # the Edit form opened, auto-generated incl. the 'text' folder-path field
        assert editor.form is not None
        assert editor.form._widgets["data_dir"][0] == "text"
        # default STOPPED: no node built, nothing on the hub
        assert console._logic_nodes[id(row)] is None

        # fill the folder + Start: the action runs ONCE on its own thread, publishes
        editor.form._widgets["data_dir"][1].setText(str(data_dir))
        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        deadline = time.monotonic() + 8.0
        while not getattr(node, "finished", False) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert node.finished
        assert float(console.hub.latest("processor_done")) == 1.0
        assert np.asarray(console.hub.latest("fidelity_site")).shape == (20,)
        # display suppressed: running the processor created NO plot panel
        assert console.cards == []
    finally:
        console.shutdown()
        exp.close()


def test_detect_sites_node_publishes_detected_centers(tmp_path):
    """The 'Detect sites' data-processing logic node detects centers from saved
    frames and publishes its OWN centers + underlay signals (no live node needed);
    a Plot panel can then read them."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state, PanelConfig, PanelCard
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    data_dir = tmp_path / "run"
    na.write_virtual_run(str(data_dir), prefix="img", groups=40, shots_per_group=4,
                         short_shot=3, ref_shots=(1, 2, 4), grid_shape=(4, 5),
                         loading_probability=0.5, seed=5)
    exp = na.connect("virtual", sitemap={"grid_shape": (4, 5)})

    console = TaskConsole(hub=SignalHub(), state=default_console_state(),
                          processors=exp.readout.processor_specs(), session=exp)
    console._timer.stop()
    try:
        row, editor = _add_processor_node(console, "Detect sites")
        editor.form._widgets["data_dir"][1].setText(str(data_dir))
        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        deadline = time.monotonic() + 8.0
        while not getattr(node, "finished", False) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert node.finished
        assert np.asarray(console.hub.latest("site_centers")).shape == (20, 2)   # detected centers
        assert np.asarray(console.hub.latest("sitemap_frame")).ndim == 2          # underlay image

        # a Plot panel pointed at the published centers + underlay builds the site map
        card = PanelCard(PanelConfig(kind="sites", source="value = np.ones(len(site_centers))",
                                     params={"centers": "site_centers", "image": "sitemap_frame"}),
                         parent=console.board, names_provider=console.hub.names)
        console._attach_card(card)
        console.refresh_once()
        assert card.plotter is not None
    finally:
        console.shutdown()
        exp.close()
