"""A >4096-point scan is refused with typed facts, not a bare ValueError.

S15.4 of the design requires preflight to answer an over-capacity scan with
`FormalScanCapacityExceeded(resident_limit, capability_unavailable_reason)`.
The code raised a bare `ValueError` whose message named neither the limit nor
what would lift it, which is exactly the wrong answer for the 9999-point
experiments this rig is built for: the frozen bitstream *does* have ping-pong
refill hardware, so the refusal is about missing evidence, not missing silicon,
and it has to say so.

Two call sites check the same thing at different stages - the logical R*P domain
before row expansion, and the expanded artifact at the deployment boundary.  They
must raise one type carrying one reason, or the two answers drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zlc_pulse import (
    AUTONOMOUS_REFILLED_UNAVAILABLE_REASON,
    FormalScanCapacityExceeded,
    require_autonomous_scan_resident_capacity,
)
from zlc_pulse.deployment import resident_scan_point_capacity

from fpga.pulse_streamer.host.image import StreamerParams


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "REAL_HARDWARE_BRINGUP_zh.md"


def test_the_typed_refusal_carries_the_three_facts_an_operator_needs():
    error = FormalScanCapacityExceeded(
        9999, 4096, AUTONOMOUS_REFILLED_UNAVAILABLE_REASON
    )

    assert error.requested_points == 9999
    assert error.resident_limit == 4096
    assert error.capability_unavailable_reason == (
        AUTONOMOUS_REFILLED_UNAVAILABLE_REASON
    )
    # Callers that already treat an over-capacity table as a bad argument keep working.
    assert isinstance(error, ValueError)
    text = str(error)
    assert "9999 points > 4096" in text
    assert "fully resident capacity" in text


def test_the_reason_says_evidence_is_missing_not_hardware():
    reason = AUTONOMOUS_REFILLED_UNAVAILABLE_REASON
    # The one thing this message must never imply is that the board cannot do it.
    assert "does implement ping-pong bank refill" in reason
    # Every S15.4 clause is named, so the reader knows what would lift the limit.
    assert "single FiniteScanStreamer I/O owner" in reason
    assert "conservative hard upper bound" in reason
    assert "EVERY potential" in reason and "seam" in reason
    # The non-sticky UNDERFLOW is why a final DONE proves nothing.
    assert "clears UNDERFLOW" in reason
    assert "docs/REAL_HARDWARE_BRINGUP_zh.md" in reason


def test_the_logical_domain_gate_raises_the_same_type():
    """The pre-expansion gate must answer identically to the deployment gate."""

    from dataclasses import replace

    from zlc_pulse import FrozenScanTable, load_pulse_document

    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    assert document.scan_table is not None
    table = document.scan_table
    # Five logical points against a four-point window: the smallest table that
    # exercises the refusal without depending on the template's own row count.
    document = replace(
        document,
        scan_table=FrozenScanTable(table.columns, table.rows * 5),
        scan_recipe=None,
    )

    with pytest.raises(FormalScanCapacityExceeded) as caught:
        require_autonomous_scan_resident_capacity(document, 4)

    error = caught.value
    assert error.requested_points == 5
    assert error.resident_limit == 4
    assert error.capability_unavailable_reason == (
        AUTONOMOUS_REFILLED_UNAVAILABLE_REASON
    )

    # One point below the window is accepted, so the gate is a boundary and not
    # a blanket refusal of every multi-point scan.
    require_autonomous_scan_resident_capacity(document, 5)


def test_the_two_gates_do_not_carry_two_different_reasons():
    """Both call sites must quote the one constant, never a local rewording."""

    source = (ROOT / "zlc_pulse" / "deployment.py").read_text(encoding="utf-8")
    assert source.count("class FormalScanCapacityExceeded") == 1
    assert source.count("AUTONOMOUS_REFILLED_UNAVAILABLE_REASON = (") == 1
    # Two raise sites, both handing over the same constant.
    assert source.count("raise FormalScanCapacityExceeded(") == 2
    assert source.count("            AUTONOMOUS_REFILLED_UNAVAILABLE_REASON,\n") == 2
    # No second opinion left behind in a message.
    assert "exceeds the bound sequencer's fully resident" not in source


def test_the_resident_limit_is_derived_from_the_board_config_not_a_literal():
    # StreamerParams() IS the board config; nothing here retypes 2048 or 4096.
    params = StreamerParams()
    assert resident_scan_point_capacity(params) == 2 * params.bank_size


def test_the_runbook_documents_the_enabling_path_for_9999_points():
    """The goal allows deferring to real hardware ONLY with a written runbook."""

    text = RUNBOOK.read_text(encoding="utf-8")
    assert "AUTONOMOUS_REFILLED" in text
    assert "FormalScanCapacityExceeded" in text
    assert "9999" in text
    # The S22 checklist for this section: evidence, config/commands, criteria,
    # rollback, and the "host only supplies frozen chunks" invariant.
    for required in (
        "AUTONOMOUS_REFILLED_UNAVAILABLE_REASON",
        "单一 I/O owner",
        "保守硬上界",
        "residual",
        "回退",
        "只**供应预先冻结的 chunk**",
    ):
        assert required in text, required
    # It must NOT offer host stepping as the way out.
    assert "host 逐点驱动不是" in text
