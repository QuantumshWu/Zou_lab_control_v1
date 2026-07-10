from __future__ import annotations

import sys
import types
import json

import pytest

from Zou_lab_control.neutral_atom import PortCatalog, PortSpec
from Zou_lab_control.neutral_atom.devices.sequencer import (
    RemoteSequencer,
    RuntimeBusDelay,
    RuntimeBusSegment,
    RuntimeSequenceProgram,
    SequencerService,
    compile_pulse_table_runtime_program,
)
from Zou_lab_control.neutral_atom.devices.virtual import VirtualSequencer
from Zou_lab_control.neutral_atom.timing import Pulse, PulsePeriod, PulseSequence, PulseTableState
from Zou_lab_control.neutral_atom.timing.pulse_table import ApiSlot, ScanSlot


def _dac_catalog(*, reverse_bus_indices: bool = False) -> PortCatalog:
    raw = ("x0", "x1", "y0", "y1", "clk0", "clk1", "ttl")
    x_index, y_index = ((1, 0) if reverse_bus_indices else (0, 1))
    return PortCatalog(raw, (
        PortSpec("da_x", "dac", ("x0", "x1"), bus_index=x_index, width=2,
                 encoding="offset_binary", safe_value=2, latch_clock="clk0"),
        PortSpec("da_y", "dac", ("y0", "y1"), bus_index=y_index, width=2,
                 encoding="offset_binary", safe_value=2, latch_clock="clk1"),
        PortSpec("clk0", "clock", ("clk0",)),
        PortSpec("clk1", "clock", ("clk1",)),
        PortSpec("ttl", "digital", ("ttl",)),
    ))


def _dac_state(catalog: PortCatalog) -> PulseTableState:
    return PulseTableState(
        port_catalog=catalog,
        periods=[PulsePeriod(100, (0,) * len(catalog.raw_lanes), unit="ns")],
        analog_bus_modes={
            "da_x": [{"mode": "edge", "value": 0}],
            "da_y": [{"mode": "edge", "value": 1}],
        },
        repeat_forever=False,
        time_step_ns=20,
    )


def test_pulse_document_persists_one_topology_contract():
    state = _dac_state(_dac_catalog())
    payload = state.to_dict()

    assert "port_catalog" in payload
    assert not ({
        "channels", "channel_labels", "analog_buses", "clk_channels", "visible_channels",
    } & payload.keys())
    assert "visible_ports" in payload
    restored = PulseTableState.from_dict(payload)
    assert restored.port_catalog == state.port_catalog
    assert restored.bus_channels() == {"da_x": ["x0", "x1"], "da_y": ["y0", "y1"]}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", None, r"missing=.*schema"),
        ("schema", "old.PortCatalog", "port-catalog schema"),
        ("version", None, r"missing=.*version"),
        ("version", 0, "port-catalog version"),
        ("version", 2, "port-catalog version"),
    ],
)
def test_port_catalog_rejects_missing_or_unknown_contract_identity(field, value, message):
    payload = _dac_catalog().to_dict()
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value

    with pytest.raises(ValueError, match=message):
        PortCatalog.from_dict(payload)


@pytest.mark.parametrize("field", ["raw_lanes", "ports", "fingerprint"])
def test_port_catalog_requires_every_current_schema_field(field):
    payload = _dac_catalog().to_dict()
    payload.pop(field)
    with pytest.raises(ValueError, match=rf"missing=.*{field}"):
        PortCatalog.from_dict(payload)


def test_port_catalog_rejects_unknown_top_level_fields():
    payload = _dac_catalog().to_dict()
    payload["channels"] = list(payload["raw_lanes"])
    with pytest.raises(ValueError, match=r"unexpected=.*channels"):
        PortCatalog.from_dict(payload)


def test_pulse_table_requires_exact_current_version_shape():
    payload = _dac_state(_dac_catalog()).to_dict()
    payload.pop("delay_units")
    with pytest.raises(ValueError, match=r"missing=.*delay_units"):
        PulseTableState.from_dict(payload)


def test_pulse_sequence_requires_exact_current_version_shape_and_types():
    payload = PulseSequence(
        [Pulse("ttl", 0.0, 1e-6)], name="shot", repeat_forever=False,
    ).to_dict()
    assert PulseSequence.from_dict(payload).to_dict() == payload

    for field in payload:
        broken = dict(payload)
        broken.pop(field)
        with pytest.raises(ValueError, match=rf"missing=.*{field}"):
            PulseSequence.from_dict(broken)

    with pytest.raises(ValueError, match=r"unexpected=.*legacy_repeat"):
        PulseSequence.from_dict({**payload, "legacy_repeat": 1})
    with pytest.raises(ValueError, match="PulseSequence version"):
        PulseSequence.from_dict({**payload, "version": payload["version"] + 1})
    with pytest.raises(ValueError, match="repeat_forever must be a boolean"):
        PulseSequence.from_dict({**payload, "repeat_forever": "false"})


def test_runtime_records_require_their_exact_current_shapes():
    delay = RuntimeBusDelay(1, 20).to_dict()
    segment = RuntimeBusSegment(
        bus_index=0, bus_name="da_x", start_tick=1, stop_tick=2,
        start_value=512, stop_value=513, mode="ramp", value_select=0,
        stop_value_select=0, start_tick_coeffs=[0], stop_tick_coeffs=[1],
    ).to_dict()

    for factory, payload in (
        (RuntimeBusDelay.from_dict, delay),
        (RuntimeBusSegment.from_dict, segment),
    ):
        for field in payload:
            broken = dict(payload)
            broken.pop(field)
            with pytest.raises(ValueError, match=rf"missing=.*{field}"):
                factory(broken)
        with pytest.raises(ValueError, match=r"unexpected=.*legacy"):
            factory({**payload, "legacy": 0})


def test_runtime_program_requires_every_current_field_and_rejects_unknowns():
    catalog = _dac_catalog()
    payload = compile_pulse_table_runtime_program(
        _dac_state(catalog), port_catalog=catalog,
        clock_hz=50_000_000, repeat_forever=False,
    ).to_dict()
    assert RuntimeSequenceProgram.from_dict(payload).to_dict() == payload

    required = set(payload) - {"source_sequence", "source_table"}
    for field in required:
        broken = dict(payload)
        broken.pop(field)
        with pytest.raises(ValueError, match=rf"missing=.*{field}"):
            RuntimeSequenceProgram.from_dict(broken)
    with pytest.raises(ValueError, match=r"unexpected=.*trigger_count"):
        RuntimeSequenceProgram.from_dict({**payload, "trigger_count": 1})

    payload = _dac_state(_dac_catalog()).to_dict()
    payload["legacy_channels"] = list(payload["port_catalog"]["raw_lanes"])
    with pytest.raises(ValueError, match=r"unexpected=.*legacy_channels"):
        PulseTableState.from_dict(payload)


@pytest.mark.parametrize(
    ("factory", "payload", "field"),
    [
        (ScanSlot.from_dict,
         {"name": "exposure", "kind": "duration", "target": "0", "label": "Exposure",
          "unit": "ns", "nominal": 100.0}, "label"),
        (ApiSlot.from_dict,
         {"name": "a1", "kind": "duration", "target": "0", "unit": "ns"}, "unit"),
        (PulsePeriod.from_dict,
         {"duration": 100.0, "unit": "ns", "name": "image", "states": [0] * 7}, "name"),
    ],
)
def test_pulse_nested_records_require_their_complete_shape(factory, payload, field):
    payload.pop(field)
    with pytest.raises(ValueError, match=rf"missing=.*{field}"):
        factory(payload)


@pytest.mark.parametrize(
    "field",
    [
        "key", "kind", "lanes", "label", "bus_index", "width",
        "encoding", "safe_value", "latch_clock",
    ],
)
def test_port_spec_requires_the_complete_versioned_catalog_shape(field):
    payload = _dac_catalog().to_dict()
    payload["ports"][0].pop(field)

    with pytest.raises(ValueError, match="invalid sequencer port spec fields"):
        PortCatalog.from_dict(payload)


def test_port_spec_rejects_unknown_fields_inside_current_catalog_version():
    payload = _dac_catalog().to_dict()
    payload["ports"][0]["old_channel_alias"] = "x0"

    with pytest.raises(ValueError, match="unexpected=.*old_channel_alias"):
        PortCatalog.from_dict(payload)


def test_same_raw_lanes_with_different_topology_is_rejected():
    state = _dac_state(_dac_catalog())
    all_digital = PortCatalog.from_channels(state.port_catalog.raw_lanes)

    with pytest.raises(ValueError, match="port topology does not match"):
        compile_pulse_table_runtime_program(
            state,
            port_catalog=all_digital,
            clock_hz=50_000_000,
            repeat_forever=False,
        )


def test_runtime_dac_order_comes_only_from_bus_index():
    catalog = _dac_catalog(reverse_bus_indices=True)
    program = compile_pulse_table_runtime_program(
        _dac_state(catalog), port_catalog=catalog,
        clock_hz=50_000_000, repeat_forever=False)

    assert program.bus_names == ["da_y", "da_x"]
    assert [segment.bus_index for segment in program.bus_segments] == [0, 1]
    assert program.port_catalog_fingerprint == catalog.fingerprint


def test_virtual_snapshot_separates_raw_lanes_from_logical_ports():
    sequencer = VirtualSequencer(sleep_scale=0)
    snapshot = sequencer.snapshot()

    assert "channels" not in snapshot
    assert snapshot["raw_channels"] == list(sequencer.port_catalog.raw_lanes)
    assert snapshot["port_catalog_fingerprint"] == sequencer.port_catalog.fingerprint
    assert [port["key"] for port in snapshot["port_catalog"]["ports"] if port["kind"] == "dac"] \
        == ["da_x", "da_y", "da_z"]


def test_remote_fetches_its_only_catalog_from_server(monkeypatch):
    server_catalog = _dac_catalog()

    class Root:
        def snapshot(self):
            return {"port_catalog": server_catalog.to_dict(), "clock_hz": 50_000_000}

    class Connection:
        closed = False
        root = Root()

        def close(self):
            self.closed = True

    fake_rpyc = types.SimpleNamespace(
        connect=lambda *args, **kwargs: Connection(),
        utils=types.SimpleNamespace(
            classic=types.SimpleNamespace(ssl_connect=lambda *args, **kwargs: Connection())))
    monkeypatch.setitem(sys.modules, "rpyc", fake_rpyc)

    remote = RemoteSequencer(host="127.0.0.1", port=18861)
    assert remote.port_catalog is None
    assert remote.channels == []
    assert remote.clock_hz is None
    remote.open()
    assert remote.port_catalog == server_catalog
    assert remote.channels == list(server_catalog.raw_lanes)
    assert remote.clock_hz == 50_000_000
    remote.close()
    assert remote.port_catalog is None
    assert remote.channels == []
    assert remote.clock_hz is None


def test_remote_constructor_has_connection_params_only():
    import inspect

    params = inspect.signature(RemoteSequencer).parameters
    assert not ({"channels", "port_catalog", "clock_hz"} & set(params))
    with pytest.raises(TypeError):
        RemoteSequencer(host="127.0.0.1", port=18861, clock_hz=50_000_000)


@pytest.mark.parametrize(
    "snapshot",
    [
        {"clock_hz": 50_000_000},
        {"port_catalog": _dac_catalog().to_dict()},
        {"port_catalog": _dac_catalog().to_dict(), "clock_hz": 0},
    ],
)
def test_remote_rejects_incomplete_server_hardware_snapshot(monkeypatch, snapshot):
    class Connection:
        closed = False
        root = types.SimpleNamespace(snapshot=lambda: snapshot)

        def close(self):
            self.closed = True

    fake_rpyc = types.SimpleNamespace(
        connect=lambda *args, **kwargs: Connection(),
        utils=types.SimpleNamespace(
            classic=types.SimpleNamespace(ssl_connect=lambda *args, **kwargs: Connection())))
    monkeypatch.setitem(sys.modules, "rpyc", fake_rpyc)

    remote = RemoteSequencer(host="127.0.0.1", port=18861)
    with pytest.raises(RuntimeError, match="complete PortCatalog.*positive clock_hz"):
        remote.open()
    assert remote.port_catalog is None
    assert remote.channels == []
    assert remote.clock_hz is None


def test_sequence_view_keeps_exact_editable_dac_and_scan_source():
    state = _dac_state(_dac_catalog())
    state.bind_field("dac", "da_x@0", name="field_x", label="field x", unit="value")
    state.set_scan_table([[-1.0], [1.0]])
    state.scan_code = "scan_table = [[-1], [1]]"
    service = SequencerService(port_catalog=state.port_catalog, clock_hz=50_000_000)

    service.prepare(state.to_sequence())
    recorded = PulseTableState.from_dict(json.loads(service.last_payload_json))
    assert recorded.port_catalog == state.port_catalog
    assert recorded.scan_names == ["field_x"]
    assert recorded.scan_table == state.scan_table
    assert recorded.scan_code == state.scan_code


def test_pulse_state_has_no_parallel_topology_views():
    state = _dac_state(_dac_catalog())
    for name in ("channels", "channel_labels", "analog_buses", "clk_channels"):
        assert name not in vars(state)
        assert not hasattr(state, name)
