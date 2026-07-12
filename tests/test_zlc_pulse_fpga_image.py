"""Pulse-owned TargetIR packs byte-for-byte like the installed host path."""

from __future__ import annotations

from pathlib import Path

import pytest

from fpga.pulse_streamer.host.image import StreamerParams
from zlc_pulse import (
    PulseExecutionForm,
    compile_pulse_document,
    load_pulse_document,
    pack_target_ir,
)


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("name", "form", "expected_digest"),
    [
        ("imaging_template.json", PulseExecutionForm.STATIC_ONCE, "3f3100880aee3be52141ffc1cdda477db38644835ecd6deb361c5e9fcde7c3cb"),
        ("mot_field_template.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "7e32834207c74fd646e4f9012ae5fc0aceec12dec9c5a5546882a43784b1420a"),
        ("pulse_test.json", PulseExecutionForm.AUTONOMOUS_SCAN_ONCE, "e5410fce0ff366390ea56feb74bd7f7fc8e640264d5b5906d95fefb083d3d355"),
        ("T.json", PulseExecutionForm.STATIC_ONCE, "a31cedfa143c00571fbb8ebd882ddc45ae6bfd36774aa44368f66b99df5eb270"),
    ],
)
def test_target_ir_wire_image_matches_the_frozen_wire_golden(name, form, expected_digest):
    document = load_pulse_document(ROOT / "pulses" / name)
    params = StreamerParams()
    ir = compile_pulse_document(document, clock_hz=50e6, execution_form=form)
    current = pack_target_ir(ir, params)

    assert current.digest == expected_digest


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
