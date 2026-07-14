"""Contract: a processor is a LOGIC NODE that publishes to the hub end-to-end.

A processor is a logic node (Logic tab), added STOPPED, with an auto-generated Edit
form from the spec's ParamDecls.  Two execution styles share the model:

Start runs the ``Readout fidelity`` action once on its own thread (a
``ProcessorRun``) over a saved frames folder and self-stops.

Display is suppressed (no auto plot); you add a Plot panel on the Monitor
board pointed at a published signal to view it.

Offscreen Qt + the virtual==real path (na.simulation.write_virtual_run / a virtual session); no
demo GUI fixtures.
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
    na.simulation.write_virtual_run(str(data_dir), prefix="img", groups=40, shots_per_group=4,
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
        # the Edit form opened, auto-generated incl. the folder field -- a PATH picker (path_mode="dir",
        # Browse to a frames folder), as the fidelity processor declares it
        assert editor.form is not None
        assert editor.form._decls["data_dir"].kind == "path"
        # default STOPPED: no node built, nothing on the hub
        assert console._logic_nodes[id(row)] is None

        # fill the folder + Start: the action runs ONCE on its own thread, publishes
        editor.form._widgets["data_dir"].setText(str(data_dir))
        console._start_logic_node(row)
        node = console._logic_nodes[id(row)]
        deadline = time.monotonic() + 8.0
        while not getattr(node, "finished", False) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert node.finished
        assert np.asarray(console.hub.latest("fidelity_site")).shape == (1, 1, 20)
        assert console.hub.schema("fidelity_site").data_shape == (20,)
        # display suppressed: running the processor created NO plot panel
        assert console.cards == []
    finally:
        console.shutdown()
        exp.close()
