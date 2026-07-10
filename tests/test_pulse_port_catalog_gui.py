"""Pulse editor consumes the sequencer PortCatalog; raw DAC lanes never become UI ports."""

from __future__ import annotations

import pytest

from Zou_lab_control.neutral_atom.ports import PortCatalog
from Zou_lab_control.neutral_atom.timing import PulseTableState


def _dac_catalog() -> PortCatalog:
    return PortCatalog.from_channels(
        ["b0", "b1", "ttl", "da_clk0"],
        channel_labels={"b0": "da_x[0]", "b1": "da_x[1]",
                        "ttl": "probe", "da_clk0": "da_clk0"},
    )


def test_editor_rebinds_same_raw_lanes_to_device_topology(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, _display_rows, ensure_qt_app

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

    editor = PulseSequenceEditor(stale, sequencer=Sequencer())
    try:
        assert editor.state.port_catalog.fingerprint == catalog.fingerprint
        rows = _display_rows(editor.state)
        assert [(row["kind"], row["name"]) for row in rows] == [
            ("bus", "da_x"), ("channel", "ttl")]
        assert not ({"b0", "b1", "da_clk0"} & {str(row["name"]) for row in rows})
        assert editor.read_state().port_catalog.fingerprint == catalog.fingerprint
    finally:
        editor.close()


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


def test_remote_connection_binds_server_catalog_and_clock_before_adoption(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.pulse_gui import PulseSequenceEditor, ensure_qt_app
    import Zou_lab_control.neutral_atom.devices.sequencer as sequencer_module

    ensure_qt_app()
    server_catalog = _dac_catalog()
    local_catalog = PortCatalog.from_channels(list(server_catalog.raw_lanes))
    state = PulseTableState(
        port_catalog=local_catalog,
        visible_ports=["b0", "b1", "ttl"],
        time_step_ns=20,
    )
    constructed = []

    class Remote:
        def __init__(self, *, host, port):
            constructed.append((host, port))
            self.host, self.port = host, port
            self.port_catalog = None
            self.channels = []
            self.clock_hz = None
            self.closed = False

        def open(self):
            self.port_catalog = server_catalog
            self.channels = list(server_catalog.raw_lanes)
            self.clock_hz = 100_000_000
            return self

        def close(self):
            self.closed = True
            self.port_catalog = None
            self.channels = []
            self.clock_hz = None

    monkeypatch.setattr(sequencer_module, "RemoteSequencer", Remote)
    editor = PulseSequenceEditor(state)
    try:
        editor.conn_target_combo.setCurrentIndex(
            editor.conn_target_combo.findData("remote"))
        editor.conn_addr_edit.setText("fpga-box:18861")
        editor._apply_connection()

        assert constructed == [("fpga-box", 18861)]
        assert isinstance(editor.sequencer, Remote)
        assert editor.state.port_catalog == server_catalog
        assert editor.state.time_step_ns == 10
        assert editor._clock_hz == 100_000_000
    finally:
        editor.close()
