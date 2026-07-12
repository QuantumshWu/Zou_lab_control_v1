"""Pulse-owned TargetIR packs byte-for-byte like the installed host path."""

from __future__ import annotations

from pathlib import Path

import pytest

from fpga.pulse_streamer.host.image import StreamerParams, pack_program
from Zou_lab_control.neutral_atom.devices.sequencer import (
    compile_pulse_table_runtime_program,
    compile_pulse_table_scan_runtime_program,
)
from zlc_pulse import (
    PulseExecutionForm,
    compile_pulse_document,
    load_pulse_document,
    pack_target_ir,
)
from zlc_workbench.pulse_compile_bridge import (
    _legacy_compile_input,
)


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("name", "form"),
    [
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE),
        ("pulse_test.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE),
        ("T.json", PulseExecutionForm.STATIC_ONCE),
    ],
)
def test_target_ir_wire_image_equals_existing_proven_path(name, form):
    document = load_pulse_document(ROOT / "pulses" / name)
    params = StreamerParams()
    ir = compile_pulse_document(document, clock_hz=50e6, execution_form=form)
    current = pack_target_ir(ir, params)

    state, catalog = _legacy_compile_input(document)
    if form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        legacy = compile_pulse_table_scan_runtime_program(
            state,
            clock_hz=50e6,
            repeat_forever=False,
            port_catalog=catalog,
        )
    else:
        legacy = compile_pulse_table_runtime_program(
            state,
            clock_hz=50e6,
            repeat_forever=False,
            port_catalog=catalog,
        )
    assert current.as_dict() == pack_program(legacy, params)
    assert len(current.digest) == 64


def test_wire_image_is_immutable_and_geometry_bound():
    document = load_pulse_document(ROOT / "pulses" / "probe_template.json")
    ir = compile_pulse_document(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
    )
    first = pack_target_ir(ir, StreamerParams())
    second = pack_target_ir(ir, StreamerParams(max_edges=2048))
    assert first.geometry_fingerprint != second.geometry_fingerprint
    assert first.digest != second.digest
    with pytest.raises(TypeError):
        first.words[0] = (0, 0)
