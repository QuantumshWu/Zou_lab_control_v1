"""Frozen-bitstream wire image derived solely from pulse-owned TargetIR."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from fpga.pulse_streamer.host.image import (
    StreamerParams,
    build_fingerprint,
    pack_program,
    region_bases,
)
from zlc_storage import canonical_digest, sha256_text

from .ir import TargetIR, target_ir_to_tree
from .validation import validate_target_ir_for_geometry


@dataclass(frozen=True)
class PulseWireImage:
    geometry_fingerprint: int
    source_ir_digest: str
    words: tuple[tuple[int, int], ...]
    _digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        fingerprint = self.geometry_fingerprint
        if isinstance(fingerprint, bool) or not isinstance(fingerprint, int):
            raise TypeError("geometry_fingerprint must be an integer")
        if fingerprint < 0 or fingerprint > 0xFFFFFFFF:
            raise ValueError("geometry_fingerprint must fit 32 bits")
        sha256_text(self.source_ir_digest, "source_ir_digest")
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
            if address > 0xFFFFFFFF:
                raise ValueError("wire word addresses must be unsigned 32-bit integers")
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
                raise ValueError("wire values must be unsigned 32-bit integers")
            previous = address
        object.__setattr__(self, "geometry_fingerprint", fingerprint)
        object.__setattr__(self, "words", words)
        object.__setattr__(
            self,
            "_digest",
            canonical_digest(_pulse_wire_image_payload_tree(self)),
        )

    @property
    def digest(self) -> str:
        return self._digest

    def as_dict(self) -> dict[int, int]:
        return dict(self.words)


def pack_target_ir(
    value: TargetIR,
    params: StreamerParams | None = None,
) -> PulseWireImage:
    if not isinstance(value, TargetIR):
        raise TypeError("pack_target_ir requires TargetIR")
    geometry = params or StreamerParams()
    validate_target_ir_for_geometry(value, geometry)
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
        source_ir_digest=value.fingerprint,
        words=tuple(
            sorted((int(address), int(word) & 0xFFFFFFFF) for address, word in packed.items())
        ),
    )


def scan_chunk_wire_words(
    value: TargetIR,
    params: StreamerParams,
    chunk_index: int,
    *,
    target_bank: int,
) -> tuple[tuple[int, int], ...]:
    """Serialize one immutable TargetIR scan chunk for a freed hardware bank."""

    if not isinstance(value, TargetIR):
        raise TypeError("scan chunk source must be TargetIR")
    if not isinstance(params, StreamerParams):
        raise TypeError("params must be StreamerParams")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
        raise TypeError("chunk_index must be an integer")
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    if target_bank not in (0, 1):
        raise ValueError("target_bank must be 0 or 1")
    first = chunk_index * params.bank_size
    if first >= len(value.scan_points):
        raise ValueError("scan chunk is outside the compiled TargetIR")
    stop = min(first + params.bank_size, len(value.scan_points))
    base = (
        region_bases(params)["scan"]
        + target_bank * params.bank_size * params.scan_words
    )
    value_mask = (1 << params.tick_width) - 1
    slot_count = len(value.slot_kinds)
    return tuple(
        (
            base + local_index * params.scan_words + slot_index,
            (
                int(value.scan_points[point_index][slot_index]) & value_mask
                if slot_index < slot_count
                else 0
            ),
        )
        for local_index, point_index in enumerate(range(first, stop))
        for slot_index in range(params.num_slots)
    )


def _pulse_wire_image_payload_tree(value: PulseWireImage) -> dict[str, object]:
    if not isinstance(value, PulseWireImage):
        raise TypeError("value must be PulseWireImage")
    return {
        "schema": "zlc_pulse.PulseWireImage",
        "geometry_fingerprint": value.geometry_fingerprint,
        "source_ir_digest": value.source_ir_digest,
        "words": np.asarray(value.words, dtype="<u4").reshape(len(value.words), 2),
    }


def pulse_wire_image_to_tree(value: PulseWireImage) -> dict[str, object]:
    return {**_pulse_wire_image_payload_tree(value), "digest": value.digest}


def pulse_wire_image_from_tree(tree: object) -> PulseWireImage:
    if not isinstance(tree, dict) or set(tree) != {
        "schema",
        "geometry_fingerprint",
        "source_ir_digest",
        "words",
        "digest",
    }:
        raise ValueError("PulseWireImage has an unknown field set")
    if tree["schema"] != "zlc_pulse.PulseWireImage":
        raise ValueError("PulseWireImage schema differs")
    words_array = tree["words"]
    if (
        not isinstance(words_array, np.ndarray)
        or words_array.dtype != np.dtype("<u4")
        or words_array.ndim != 2
        or words_array.shape[1] != 2
    ):
        raise TypeError("PulseWireImage words must be a two-column <u4 ndarray")
    words = tuple((int(address), int(word)) for address, word in words_array)
    value = PulseWireImage(
        tree["geometry_fingerprint"],
        tree["source_ir_digest"],
        words,
    )
    if tree["digest"] != value.digest:
        raise ValueError("PulseWireImage digest differs")
    return value


__all__ = [
    "PulseWireImage",
    "pack_target_ir",
    "pulse_wire_image_from_tree",
    "pulse_wire_image_to_tree",
    "scan_chunk_wire_words",
]
