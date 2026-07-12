"""Pulse editor consumes the sequencer PortCatalog; raw DAC lanes never become UI ports."""

from __future__ import annotations

import pytest
from conftest import pulse_editor_for_test

from Zou_lab_control.neutral_atom.ports import PortCatalog
from Zou_lab_control.neutral_atom.timing import PulseTableState


def _dac_catalog() -> PortCatalog:
    return PortCatalog.from_channels(
        ["b0", "b1", "ttl", "da_clk0"],
        channel_labels={"b0": "da_x[0]", "b1": "da_x[1]",
                        "ttl": "probe", "da_clk0": "da_clk0"},
    )


def test_editor_rejects_raw_lane_equality_without_semantic_topology(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.pulse_gui import ensure_qt_app

    ensure_qt_app()
    catalog = _dac_catalog()
    # Same raw lane order but deliberately no DAC grouping: raw equality is not
    # enough to establish semantic topology.
    stale = PulseTableState(port_catalog=PortCatalog.from_channels(list(catalog.raw_lanes)),
                            visible_ports=["b0", "b1", "ttl"])

    class Sequencer:
        clock_hz = 50e6
        port_catalog = catalog
        channels = list(catalog.raw_lanes)

    with pytest.raises(ValueError, match="has no matching hardware port"):
        pulse_editor_for_test(stale, sequencer=Sequencer())


def test_new_dac_scan_axis_uses_semantic_port_name(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, ensure_qt_app

    ensure_qt_app()
    state = PulseTableState(port_catalog=_dac_catalog(),
                            visible_ports=["da_x", "ttl"])
    editor = PulseSequenceEditor(state)
    try:
        editor._toggle_dac_scan(editor.drag_container.pulse_cards()[0], "da_x")
        assert editor.state.scan_names == ["da_x"]
        assert editor.state.compiler_scan_vars == ["s0"]  # compiler token stays private
    finally:
        editor.close()


def test_connection_is_installation_owned_and_frontend_only_adopts_descriptor(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, ensure_qt_app
    from zlc_workbench.pulse_control import PulseTargetDescriptor

    ensure_qt_app()
    server_catalog = _dac_catalog()
    state = PulseTableState(
        port_catalog=server_catalog,
        visible_ports=["da_x", "ttl"],
        time_step_ns=20,
    )
    target = PulseTargetDescriptor(
        "installation-a", 1, server_catalog, 100_000_000, "FPGA installation"
    )

    class Port:
        def __init__(self):
            self.target = target

    editor = PulseSequenceEditor(
        state, target_descriptor=target, command_port=Port()
    )
    try:
        editor._apply_connection()

        assert editor.state.port_catalog == server_catalog
        assert editor.state.time_step_ns == 10
        assert editor._clock_hz == 100_000_000
        assert not hasattr(editor, "sequencer")
        assert not editor.conn_target_combo.isEnabled()
        assert not editor.conn_connect_button.isEnabled()
        assert editor.conn_status.text() == "FPGA installation"
    finally:
        editor.close()
