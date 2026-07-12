"""Frozen-bitstream wire image derived solely from pulse-owned TargetIR."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from fpga.pulse_streamer.host.image import (
    StreamerParams,
    build_fingerprint,
    pack_program,
)
from zlc_storage import canonical_digest

from .ir import TargetIR, target_ir_to_tree


@dataclass(frozen=True)
class PulseWireImage:
    geometry_fingerprint: int
    source_ir_digest: str
    words: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        fingerprint = self.geometry_fingerprint
        if isinstance(fingerprint, bool) or not isinstance(fingerprint, int):
            raise TypeError("geometry_fingerprint must be an integer")
        if fingerprint < 0 or fingerprint > 0xFFFFFFFF:
            raise ValueError("geometry_fingerprint must fit 32 bits")
        source = self.source_ir_digest
        if (
            not isinstance(source, str)
            or len(source) != 64
            or any(character not in "0123456789abcdef" for character in source)
        ):
            raise ValueError("source_ir_digest must be a lowercase SHA-256 digest")
        words = tuple(self.words)
        previous = -1
        for address, value in words:
            if (
                isinstance(address, bool)
                or not isinstance(address, int)
                or address < 0
                or address <= previous
            ):
                raise ValueError("wire word addresses must be strictly increasing non-negative ints")
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError("wire values must be unsigned 32-bit integers")
            previous = address
        object.__setattr__(self, "geometry_fingerprint", fingerprint)
        object.__setattr__(self, "words", words)

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "schema": "zlc_pulse.PulseWireImage/v1",
                "geometry_fingerprint": self.geometry_fingerprint,
                "source_ir_digest": self.source_ir_digest,
                "words": [list(item) for item in self.words],
            }
        )

    def as_dict(self) -> dict[int, int]:
        return dict(self.words)


def pack_target_ir(
    value: TargetIR,
    params: StreamerParams | None = None,
) -> PulseWireImage:
    if not isinstance(value, TargetIR):
        raise TypeError("pack_target_ir requires TargetIR")
    geometry = params or StreamerParams()
    carrier = SimpleNamespace(
        ticks=value.ticks,
        masks=value.masks,
        slot_count=len(value.slot_kinds),
        tick_slot_coeffs=value.tick_slot_coeffs,
        scan_points=value.scan_points,
        bus_segments=value.bus_segments,
        repeat_forever=value.repeat_forever,
        repeat_from_index=0,
        loop_start_index=value.loop_start_index,
        loop_count=value.loop_count,
        loop_end_tick=value.loop_end_tick,
        loop_end_slot_coeffs=value.loop_end_slot_coeffs,
        channel_delays=value.channel_delays,
        bus_delays=tuple(
            {"bus_index": item.bus_index, "delay": item.delay_ticks}
            for item in value.bus_delays
        ),
        clk_enable=value.clk_enable,
    )
    packed = pack_program(carrier, geometry)
    return PulseWireImage(
        geometry_fingerprint=build_fingerprint(geometry) & 0xFFFFFFFF,
        source_ir_digest=canonical_digest(target_ir_to_tree(value)),
        words=tuple(
            sorted((int(address), int(word) & 0xFFFFFFFF) for address, word in packed.items())
        ),
    )


def pulse_wire_image_to_tree(value: PulseWireImage) -> dict[str, object]:
    if not isinstance(value, PulseWireImage):
        raise TypeError("value must be PulseWireImage")
    return {
        "schema": "zlc_pulse.PulseWireImage/v1",
        "geometry_fingerprint": value.geometry_fingerprint,
        "source_ir_digest": value.source_ir_digest,
        "words": [list(item) for item in value.words],
        "digest": value.digest,
    }


def pulse_wire_image_from_tree(tree: object) -> PulseWireImage:
    if not isinstance(tree, dict) or set(tree) != {
        "schema",
        "geometry_fingerprint",
        "source_ir_digest",
        "words",
        "digest",
    }:
        raise ValueError("PulseWireImage has an unknown field set")
    if tree["schema"] != "zlc_pulse.PulseWireImage/v1":
        raise ValueError("PulseWireImage schema differs")
    if not isinstance(tree["words"], list):
        raise TypeError("PulseWireImage words must be a list")
    words = []
    for item in tree["words"]:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("PulseWireImage word must be [address, value]")
        words.append((item[0], item[1]))
    value = PulseWireImage(
        tree["geometry_fingerprint"],
        tree["source_ir_digest"],
        tuple(words),
    )
    if tree["digest"] != value.digest:
        raise ValueError("PulseWireImage digest differs")
    return value


__all__ = [
    "PulseWireImage",
    "pack_target_ir",
    "pulse_wire_image_from_tree",
    "pulse_wire_image_to_tree",
]
