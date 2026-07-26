"""Source-neutral binding for PulseScan's external ``y`` signal.

PulseScan owns a sequencer program and consumes one selected output from an
already-running producer.  It does not own, start, stop, or interpret that
producer and it never branches on Camera, Occupancy, selector, or Fit types.
The process-local event source is supplied separately by the composition root
at Run preparation; this serializable binding contains only domain-neutral
selection facts.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import DataTransformSpec
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.node_input import BoundDatasetInput, BoundNodeInputs

from .authoring import PULSE_SCAN_SOURCE_INPUT_SPEC
from .contracts import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    PulseScanProgram,
)


@dataclass(frozen=True, slots=True)
class ScanSignalBinding:
    """Frozen identity of the external signal selected as scan ``y``.

    ``transform`` is authoritative user intent, never a display default.  The
    runtime signal-source owner applies it before PulseScan receives a Value;
    PulseScan itself does not guess axes or reduce trailing data.
    """

    producer_definition: DefinitionKey
    output: DatasetOutputDeclaration
    transform: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.producer_definition, DefinitionKey):
            raise TypeError("producer_definition must be DefinitionKey")
        if not isinstance(self.output, DatasetOutputDeclaration):
            raise TypeError("output must be DatasetOutputDeclaration")
        if self.transform is not None:
            if not isinstance(self.transform, DataTransformSpec):
                raise TypeError("transform must be DataTransformSpec or None")
            if not self.transform.operations:
                raise ValueError("an empty transform must be None")


@dataclass(frozen=True, slots=True)
class PulseScanBoundRequest:
    """One pulse program plus one external, source-neutral signal binding."""

    program: PulseScanProgram
    signal: ScanSignalBinding

    def __post_init__(self) -> None:
        if not isinstance(
            self.program,
            (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
        ):
            raise TypeError("program must be a current PulseScan program")
        if not isinstance(self.signal, ScanSignalBinding):
            raise TypeError("signal must be ScanSignalBinding")


def bind_scan_signal(binding: BoundDatasetInput) -> ScanSignalBinding:
    """Strip producer request/device details from one generic Dataset input."""

    if not isinstance(binding, BoundDatasetInput):
        raise TypeError("binding must be BoundDatasetInput")
    if binding.spec != PULSE_SCAN_SOURCE_INPUT_SPEC:
        raise ValueError("binding belongs to another Logic-node input")
    return ScanSignalBinding(
        binding.producer_definition,
        binding.output,
        binding.transform_spec,
    )


def bind_pulse_scan_request(
    program: PulseScanProgram,
    inputs: BoundNodeInputs,
) -> PulseScanBoundRequest:
    """Bind the declaration-owned signal without inspecting its producer."""

    if not isinstance(inputs, BoundNodeInputs):
        raise TypeError("inputs must be BoundNodeInputs")
    return PulseScanBoundRequest(
        program,
        bind_scan_signal(inputs.dataset(PULSE_SCAN_SOURCE_INPUT_SPEC)),
    )


__all__ = [
    "PulseScanBoundRequest",
    "ScanSignalBinding",
    "bind_pulse_scan_request",
    "bind_scan_signal",
]
