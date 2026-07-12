from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from Zou_lab_control.neutral_atom.ports import PortCatalog
from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
from zlc_workbench.pulse_control import PulseCommandPort, PulseTargetDescriptor


class _Facade:
    def __init__(self):
        self.calls = []

    def prepare(self, payload):
        self.calls.append(("prepare", payload))
        return "prepared"

    def fire(self, payload=None):
        self.calls.append(("fire", payload))
        return "running"

    def set_safe_state(self):
        self.calls.append(("safe", None))

    def scan_progress(self):
        self.calls.append(("progress", None))
        return {"scanning": False}

    def snapshot(self):
        self.calls.append(("snapshot", None))
        return {"state": "safe"}


def _target(generation=1):
    return PulseTargetDescriptor(
        "installation-a",
        generation,
        PortCatalog.from_channels(["ch00", "ch01"]),
        50e6,
        "Virtual installation",
    )


def test_command_port_runs_semantic_pulse_commands_without_raw_escape():
    facade = _Facade()
    port = PulseCommandPort(facade, _target(), lambda: 1)
    state = PulseTableState(port_catalog=port.target.port_catalog)

    assert port.prepare(state) == "prepared"
    assert port.run(state) == "prepared"
    port.stop()
    assert port.scan_progress() == {"scanning": False}
    assert port.snapshot() == {"state": "safe"}
    assert [name for name, _ in facade.calls] == [
        "prepare",
        "safe",
        "prepare",
        "fire",
        "safe",
        "progress",
        "snapshot",
    ]
    assert not hasattr(port, "sequencer")
    assert not hasattr(port, "fire")
    assert not hasattr(port, "set_safe_state")


def test_stale_generation_fails_before_any_facade_call():
    facade = _Facade()
    port = PulseCommandPort(facade, _target(generation=4), lambda: 5)

    with pytest.raises(RuntimeError, match="stale installation generation"):
        port.stop()
    assert facade.calls == []


def test_frontend_pulse_api_cannot_accept_or_construct_raw_hardware():
    from Zou_lab_control.frontend.pulse_gui import (
        PulseSequenceEditor,
        show_pulse_gui,
    )

    for callable_ in (PulseSequenceEditor, show_pulse_gui):
        parameters = inspect.signature(callable_).parameters
        assert "sequencer" not in parameters
        assert "experiment" not in parameters
        assert "target_descriptor" in parameters
        assert "command_port" in parameters

    source = Path(inspect.getsourcefile(PulseSequenceEditor)).read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "RemoteSequencer",
        "VirtualSequencer",
        "self.sequencer",
        "experiment.devices",
    ):
        assert forbidden not in source


def test_standalone_launcher_composes_a_session_instead_of_raw_adapters():
    root = Path(__file__).resolve().parents[1]
    source = (root / "pulse_gui.py").read_text(encoding="utf-8")

    assert 'na.connect("virtual")' in source
    assert '"remote_template"' in source
    assert "managed_pulse_command_port" in source
    assert "na.RemoteSequencer" not in source
    assert "na.VirtualSequencer" not in source
    assert "sequencer" not in inspect.signature(
        __import__("Zou_lab_control.frontend", fromlist=["show_pulse_gui"]).show_pulse_gui
    ).parameters
