"""Approved host-side topology manifest for the frozen laboratory streamer."""

from __future__ import annotations

from fpga.pulse_streamer.host.image import StreamerParams

from .artifact import CompiledPulseArtifact
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
    """Prove a compiled artifact against the bound topology before any I/O."""

    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    validate_deployed_target(target, params)
    if artifact.target_ir.clock_hz != float(clock_hz):
        raise ValueError("compiled artifact clock differs from deployed hardware clock")
    validate_resident_scan_capacity(artifact, params)
    validate_target_ir_for_target(artifact.target_ir, target)
    expected = pack_target_ir(artifact.target_ir, params)
    if artifact.wire_image != expected:
        raise ValueError(
            "compiled artifact wire image differs from deterministic deployed packing"
        )


def validate_resident_scan_capacity(
    artifact: CompiledPulseArtifact,
    params: StreamerParams,
) -> None:
    """Reject scans whose timing would depend on unqualified host refill."""

    total_points = len(artifact.target_ir.scan_points)
    resident_capacity = 2 * params.bank_size
    if total_points > resident_capacity:
        raise ValueError(
            "formal autonomous scan exceeds the frozen bitstream's fully resident "
            f"capacity: {total_points} points > {resident_capacity}"
        )


__all__ = [
    "APPROVED_DEPLOYED_TARGET_ABI",
    "require_approved_target_abi",
    "validate_artifact_for_deployment",
    "validate_deployed_target",
    "validate_resident_scan_capacity",
]
