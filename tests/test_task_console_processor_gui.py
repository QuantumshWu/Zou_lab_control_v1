"""Contract: the Add-Panel "Data processing" category runs a processor end-to-end.

Add Panel offers three clear categories -- Plot / Measurement / Data processing.
Picking a data-processing action creates a result panel bound to its default view,
opens an Edit form (auto-generated from the spec's ParamDecls, including the new
'text' kind for a folder path), and Start runs the action ONCE: a ProcessorFeed
publishes the result to the hub and the result panel shows it.

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


def test_add_panel_data_processing_runs_and_publishes(tmp_path):
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
                          processors=exp.readout.processor_specs())
    console._timer.stop()
    try:
        # the combo lists the action under the "Data processing" category
        kc = console.kind_combo
        idx = next(j for j in range(kc.count())
                   if kc.itemData(j) == ("processor", "Readout fidelity"))
        assert kc.itemText(idx).startswith("Data processing:")
        kc.setCurrentIndex(idx)
        console._add_panel()

        # a result panel bound to the processor, on the EXISTING atom 'sites' kind
        card = next(c for c in console.cards if c.config.params.get("processor") == "Readout fidelity")
        assert card.config.kind == "sites"
        assert card.config.source == "value = fidelity_site"

        # the Edit form opened, auto-generated incl. the 'text' folder-path field
        editor = console._panel_editors[id(card)]
        form = editor.meas_panel
        assert form is not None
        assert form._widgets["data_dir"][0] == "text"

        # fill the folder + Start: the action runs ONCE on its own thread, publishes
        form._widgets["data_dir"][1].setText(str(data_dir))
        console._start_processor(form)
        feed = console._active_proc_feed
        deadline = time.monotonic() + 8.0
        while not getattr(feed, "finished", False) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert feed.finished
        assert float(console.hub.latest("processor_done")) == 1.0
        assert np.asarray(console.hub.latest("fidelity_site")).shape == (20,)
        # the result panel reads its bound signal without error
        console.refresh_once()
        assert card.plotter is not None
    finally:
        console.shutdown()
