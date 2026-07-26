"""Approved host-side topology manifest for the frozen laboratory streamer."""

from __future__ import annotations

from dataclasses import replace

from fpga.pulse_streamer.host.image import StreamerParams

from .artifact import CompiledPulseArtifact
from .document import FrozenScanTable, PulseDocument
from .fpga import pack_target_ir
from .target import (
    DAC_OFFSET_BINARY,
    PORT_CLOCK,
    PORT_DAC,
    PORT_DIGITAL,
    PulseTarget,
)
from .validation import validate_target_ir_for_target


# This digest proves which host topology was approved.  It does not attest the
# contents of the running .bit: the existing geometry handshake plus deployment
# SOP remain the bitstream identity boundary under the frozen-RTL constraint.
APPROVED_DEPLOYED_TARGET_ABI = (
    "d9ce9aea5da7380f0670ee89c5f936f5f32edfc00c98939b4dd06af32a8563c9"
)


def _autonomous_scan_repeat_domain(
    document: PulseDocument,
) -> tuple[FrozenScanTable, int, int]:
    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    table = document.scan_table
    if table is None:
        raise ValueError("autonomous scan requires a frozen scan table")
    repeat = document.repeat
    if repeat is not None and (
        repeat.start_period_id != document.periods[0].period_id
        or repeat.end_period_id != document.periods[-1].period_id
    ):
        raise ValueError(
            "formal scan repeat axis requires a whole-document RepeatRegion"
        )
    repeat_count = 1 if repeat is None else repeat.count
    return table, repeat_count, repeat_count * len(table.rows)


def expand_autonomous_scan_repeats(document: PulseDocument) -> PulseDocument:
    """Freeze a whole-document logical repeat into a repeat-major scan table.

    A partial-period RepeatRegion cannot represent a dataset repeat axis: its
    triggers need not occur once per loop.  Exact scan therefore removes the
    hardware loop and duplicates the frozen SCAN_SLOT rows in ``repeat, point``
    order.  The transport streams the resulting immutable table through the
    frozen RTL's ping-pong banks.
    """

    table, repeat_count, _ = _autonomous_scan_repeat_domain(document)
    repeat = document.repeat
    if repeat is None:
        return document
    return replace(
        document,
        scan_table=FrozenScanTable(
            table.columns,
            tuple(row for _repeat in range(repeat_count) for row in table.rows),
        ),
        scan_recipe=None,
        repeat=None,
    )


def require_approved_target_abi(target_abi_fingerprint: str) -> None:
    if target_abi_fingerprint != APPROVED_DEPLOYED_TARGET_ABI:
        raise ValueError(
            "compiled artifact target differs from the approved deployed topology"
        )


def validate_deployed_target(target: PulseTarget, params: StreamerParams) -> None:
    """Bind the approved semantic target to the frozen RTL's physical lane groups."""

    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    if not isinstance(params, StreamerParams):
        raise TypeError("params must be StreamerParams")
    if len(target.raw_lanes) != params.channel_count:
        raise ValueError(
            "deployed PulseTarget raw lane count differs from frozen streamer channel_count"
        )
    digital_ports = tuple(port for port in target.ports if port.kind == PORT_DIGITAL)
    dac_ports = tuple(
        sorted(
            (port for port in target.ports if port.kind == PORT_DAC),
            key=lambda port: int(port.bus_index),
        )
    )
    clock_ports = tuple(port for port in target.ports if port.kind == PORT_CLOCK)
    if len(digital_ports) != params.num_delay_ch:
        raise ValueError("deployed PulseTarget digital-port count differs from frozen RTL")
    expected_digital_lanes = set(target.raw_lanes[: params.num_delay_ch])
    if {port.lanes[0] for port in digital_ports} != expected_digital_lanes:
        raise ValueError("deployed PulseTarget digital lanes differ from frozen RTL positions")
    if len(dac_ports) != params.bus_count or len(clock_ports) != params.bus_count:
        raise ValueError("deployed PulseTarget must declare every frozen DAC bus and clock")
    clock_by_key = {port.key: port for port in clock_ports}
    midpoint = 1 << (params.bus_width - 1)
    for bus_index, port in enumerate(dac_ports):
        if port.bus_index != bus_index:
            raise ValueError("deployed PulseTarget DAC bus indices differ from frozen RTL")
        if (
            port.width != params.bus_width
            or port.encoding != DAC_OFFSET_BINARY
            or port.safe_value != midpoint
        ):
            raise ValueError(f"DAC port {port.key!r} differs from frozen bus geometry")
        base = params.num_delay_ch + bus_index * (params.bus_width + 1)
        expected_bus_lanes = set(target.raw_lanes[base : base + params.bus_width])
        expected_clock_lane = target.raw_lanes[base + params.bus_width]
        if set(port.lanes) != expected_bus_lanes:
            raise ValueError(f"DAC port {port.key!r} occupies the wrong frozen lane slice")
        clock = clock_by_key.get(port.latch_clock)
        if clock is None or clock.lanes != (expected_clock_lane,) or clock.safe_value != 0:
            raise ValueError(f"DAC port {port.key!r} has the wrong frozen latch clock")
    require_approved_target_abi(target.abi_fingerprint)


def validate_artifact_for_deployment(
    artifact: CompiledPulseArtifact,
    target: PulseTarget,
    params: StreamerParams,
    clock_hz: float,
) -> None:
    """Prove a target and compiled artifact at a standalone deployment boundary."""

    validate_deployed_target(target, params)
    _validate_artifact_against_bound_deployment(
        artifact,
        target,
        params,
        clock_hz,
    )


def _validate_artifact_against_bound_deployment(
    artifact: CompiledPulseArtifact,
    target: PulseTarget,
    params: StreamerParams,
    clock_hz: float,
) -> None:
    """Validate one artifact after the session constructor bound target geometry."""

    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    if artifact.target_ir.clock_hz != float(clock_hz):
        raise ValueError("compiled artifact clock differs from deployed hardware clock")
    validate_target_ir_for_target(artifact.target_ir, target)
    expected = pack_target_ir(artifact.target_ir, params)
    if artifact.wire_image != expected:
        raise ValueError(
            "compiled artifact wire image differs from deterministic deployed packing"
        )


__all__ = [
    "APPROVED_DEPLOYED_TARGET_ABI",
    "expand_autonomous_scan_repeats",
    "require_approved_target_abi",
    "validate_artifact_for_deployment",
    "validate_deployed_target",
]
