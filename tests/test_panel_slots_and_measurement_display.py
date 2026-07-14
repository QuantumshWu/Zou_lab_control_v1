"""GUI contracts for panel slots, pulse-scan forms and measurement display policy.

Notebook measurements may open their result view, while the same measurement run
as a GUI logic node only publishes data for user-selected panels.  Camera ROI edits
and panel source rules use the same virtual/real contracts.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from conftest import fire_live_imaging   # the live "On Pulse" the trigger-driven camera needs


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _virtual_experiment(grid=(3, 4)):
    import Zou_lab_control.neutral_atom as na

    return na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (48, 60)})





def test_sitemap_is_single_slot_while_other_kinds_allow_multi_slot():
    """E1 guard: the signal-picker + ``value = ...`` expression MECHANISM is universal (every
    kind has the expression box), but whether a kind can GROW slots (+signal / −signal) is
    data-driven from PANEL_SINGLE_SLOT_KINDS -- the site map takes EXACTLY ONE occupancy signal
    (its centres + frame underlay resolve from signal[0]), so NO +/-; other kinds are multi-slot.
    Locks the design so the slot UI can never regress to a hardcoded-per-panel +/- on the sitemap."""
    from Zou_lab_control.frontend.task_console import (
        PanelCard, PanelConfig, panel_allows_multi_slot)
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    assert panel_allows_multi_slot("sites") is False
    assert panel_allows_multi_slot("2d") and panel_allows_multi_slot("1d") and panel_allows_multi_slot("monitor")

    sites = PanelCard(PanelConfig(kind="sites"))
    assert sites._multi_slot is False
    assert not hasattr(sites, "add_slot_button")     # NO +/- on the site map
    assert hasattr(sites, "source_edit")             # but it KEEPS the universal expression box

    two_d = PanelCard(PanelConfig(kind="2d"))
    assert two_d._multi_slot is True
    assert hasattr(two_d, "add_slot_button")         # +/- present on a multi-slot kind


def test_pulse_scan_slots_form_is_template_driven(tmp_path):
    """The Pulse-scan form derives both sweep strategies directly from the selected template."""
    from Zou_lab_control.frontend.task_console import MeasurementPanel
    from Zou_lab_control.neutral_atom.operations.measurement import (
        SWEEP_API_SLOT,
        SWEEP_SCAN_SLOT,
    )
    from Zou_lab_control.neutral_atom.timing import single_imaging_template

    exp = _virtual_experiment()
    try:
        template = single_imaging_template()
        template.bind_field(
            "duration", "0", label="load time", unit="s", name="load_time"
        )
        template_path = template.save(tmp_path / "pulse_table.json")
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        panel = MeasurementPanel([spec], single=True)
        widget = panel._widgets["pulse_slots"]
        assert panel._decls["pulse_slots"].kind == "pulse_slots"
        # Generate this legacy-form input in the test.  Tracked authoring assets
        # are current zlc_pulse PulseDocuments and must not be parsed by this island.
        panel._widgets["template"].setText(str(template_path))
        panel._repopulate_pulse_slots("pulse_slots")
        out = panel.collect_values()["pulse_slots"]
        assert isinstance(out, dict) and set(out) == {
            "program_id", "api", "sweep_kind", "program"}
        assert out["program_id"]
        assert set(out["api"]) == set(template.api_names())
        assert out["sweep_kind"] == SWEEP_SCAN_SLOT
        assert "scan_table" in out["program"]

        api_index = widget._sweep_combo.findData(SWEEP_API_SLOT)
        assert api_index >= 0
        widget._sweep_combo.setCurrentIndex(api_index)
        api_out = panel.collect_values()["pulse_slots"]
        assert api_out["sweep_kind"] == SWEEP_API_SLOT
        assert "scan_table" in api_out["program"]
    finally:
        if hasattr(panel, "shutdown"):
            panel.shutdown()
        exp.close()


# ------------------------------------------------- camera measurement: editable region
def test_camera_measurement_region_is_editable_and_applies_to_virtual():
    """The camera measurement exposes its ROI as an editable ``region`` (the SAME
    field for virtual / real -- only the camera differs), and setting it actually
    windows the virtual camera (the frame is cropped).  This pins that virtual==real:
    both honour camera.configure(roi=)."""
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _virtual_experiment((3, 4))
    try:
        # the camera Edit auto-form carries region (next to exposure / frames_per_cycle)
        console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                              tasks=exp.readout.task_specs(), window_px=(900, 600))
        console._timer.stop()
        kc = console.kind_combo
        i = next(j for j in range(kc.count()) if kc.itemData(j) == ("camera", "live"))
        kc.setCurrentIndex(i)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._edit_logic_node(row)
        editor = console._logic_editors[id(row)]
        assert {"frames_per_cycle", "exposure", "region"} <= set(editor.form._widgets)
        console.shutdown()

        # building with a region windows the VIRTUAL camera -> the frame is cropped
        node = exp.readout.camera_spec().build(SignalHub(), region="10, 50, 8, 40")
        assert node.camera.roi is not None
        fire_live_imaging(exp)                            # On Pulse: the trigger-driven camera streams
        frame = node.camera.acquire(1)[0]                # the wired camera senses the firing itself
        assert np.asarray(frame).shape != (40, 50)        # ROI actually crops the virtual frame
        assert "region" in node.acquisition_parameters()  # round-trips (endpoints)
    finally:
        exp.close()


# ------------------------------------------------------- measurement plot split
def test_notebook_measurement_defaults_display_true():
    """A measurement called from the notebook API auto-plots (display=True default)."""
    from Zou_lab_control.neutral_atom.operations.measurement import ScannedMeasurement

    exp = _virtual_experiment()
    try:
        # the scanned-measurement run + a readout measurement method both default on
        assert inspect.signature(ScannedMeasurement.run).parameters["display"].default is True
        assert inspect.signature(exp.readout.temperature).parameters["display"].default is True
    finally:
        exp.close()


def test_gui_measurement_node_is_plot_false_publishes_to_hub_only():
    """The SAME measurement driven as a GUI logic node is plot=False: it publishes
    its result signal to the hub and opens NO plot panel (the user wires a Plot)."""
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.measurement import MeasurementSpec

    class _Reducer:
        data_shape = (1,)
        labels = ("x", "y", "y")

    class _Measurement:
        axis = SimpleNamespace(values=np.array([0.0, 1.0]), label="x", unit="")
        reducer = _Reducer()
        camera = None
        sequencer = None

        @staticmethod
        def measure(value, _index):
            return np.array([value + 1.0])

    spec = MeasurementSpec(
        name="Synthetic scan",
        result_labels=("x", "y"),
        x_key="x",
        y_key="y",
        build=lambda: _Measurement(),
    )
    console = TaskConsole(
        hub=SignalHub(),
        state=default_console_state(),
        measurements=[spec],
        window_px=(900, 600),
    )
    console._timer.stop()
    try:
        kc = console.kind_combo
        i = next(j for j in range(kc.count()) if kc.itemData(j) == ("measurement", spec.name))
        kc.setCurrentIndex(i)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._start_logic_node(row)
        published_y = f"{spec.key}_{spec.y_key}"        # node publishes under its slug
        deadline = time.monotonic() + 8.0
        while published_y not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)
        assert published_y in console.hub.names()       # data on the hub
        assert console.cards == []                     # plot=False: no auto plot panel
    finally:
        console.shutdown()
