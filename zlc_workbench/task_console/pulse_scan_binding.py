"""TaskConsole binding intent for a source-neutral Pulse scan.

The catalog can freeze the pulse program and the selected console signal, but
it cannot resolve another row's runtime/domain request.  That resolution
belongs to the TaskConsole composition root.  Keeping this value deliberately
small prevents the form layer from smuggling a camera into Pulse scan again.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zlc_data import DataTransformSpec
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
from zlc_neutral_atom.camera_measurement import (
    CameraMeasurementRequest,
    camera_frame_output_index,
)
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.occupancy import (
    OCCUPANCY_EXACT_SOURCE_OUTPUT_NAMES,
    OCCUPANCY_STREAM_PROCESSOR_KEY,
)
from zlc_neutral_atom.scan import (
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    DirectCameraScanSource,
    OccupancyScanSource,
    ScanSourceBinding,
)
from zlc_storage import canonical_text

from .occupancy_binding import (
    ConsoleProducerBinding,
    OccupancyBindingIntent,
)


PULSE_SCAN_CAMERA_FRAME_SOURCE = "camera-frame"
PULSE_SCAN_OCCUPANCY_SOURCE = "occupancy"


def classify_pulse_scan_producer(
    definition_key: DefinitionKey,
    output_name: str,
) -> str | None:
    """Return the exact Pulse Scan source family, or ``None`` if rejected."""

    if definition_key == CAMERA_MEASUREMENT_KEY:
        try:
            camera_frame_output_index(output_name)
        except (TypeError, ValueError):
            pass
        else:
            return PULSE_SCAN_CAMERA_FRAME_SOURCE
    if (
        definition_key == OCCUPANCY_STREAM_PROCESSOR_KEY
        and output_name in OCCUPANCY_EXACT_SOURCE_OUTPUT_NAMES
    ):
        return PULSE_SCAN_OCCUPANCY_SOURCE
    return None


@dataclass(frozen=True, slots=True)
class PulseScanBindingIntent:
    """One frozen pulse program and one explicitly selected y signal."""

    program: AutonomousScanSlotProgram | ApiSlotSegmentedProgram
    y_signal: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.program,
            (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
        ):
            raise TypeError("program must be a current PulseScanProgram")
        canonical_text(self.y_signal, "Pulse scan y_signal")


@dataclass(frozen=True, slots=True)
class PulseScanSourceBinding:
    """The physical producer plus any Figure-owned authoritative selection."""

    source_kind: str
    producer: ConsoleProducerBinding
    transform_spec: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.producer, ConsoleProducerBinding):
            raise TypeError("producer must be a ConsoleProducerBinding")
        expected = classify_pulse_scan_producer(
            self.producer.definition_key,
            self.producer.output_name,
        )
        if expected is None or self.source_kind != expected:
            raise ValueError(
                "source_kind must match an exact Camera frame or Occupancy "
                "counts/occupied producer"
            )
        if self.transform_spec is not None:
            if not isinstance(self.transform_spec, DataTransformSpec):
                raise TypeError("transform_spec must be DataTransformSpec or None")
            if not self.transform_spec.operations:
                raise ValueError("an empty transform_spec must be None")


def _retirement_nodes(*nodes: object | None) -> tuple[object, ...]:
    """Return each concrete Workbench runtime node exactly once."""

    unique: dict[int, object] = {}
    for node in nodes:
        if node is not None:
            unique.setdefault(id(node), node)
    return tuple(unique.values())


def resolve_typed_scan_source(
    binding: PulseScanSourceBinding,
    *,
    resolve_producer: Callable[[str], ConsoleProducerBinding],
    resolve_calibration: Callable[
        [OccupancyBindingIntent],
        CalibrationArtifactRef,
    ],
) -> tuple[ScanSourceBinding, tuple[object, ...]]:
    """Resolve Workbench routing into one canonical physical scan source.

    This seam deliberately receives no pulse program and constructs no scan
    request.  The scan domain validates Camera cardinality, output vocabulary,
    calibration identity and authoritative transform when the concrete source
    value is built.  The Workbench contributes only producer/runtime routing
    and returns the monitor nodes which must terminate before exact ownership.
    """

    if not isinstance(binding, PulseScanSourceBinding):
        raise TypeError("binding must be PulseScanSourceBinding")
    if not callable(resolve_producer):
        raise TypeError("resolve_producer must be callable")
    if not callable(resolve_calibration):
        raise TypeError("resolve_calibration must be callable")

    producer = binding.producer
    if binding.source_kind == PULSE_SCAN_CAMERA_FRAME_SOURCE:
        if not isinstance(producer.request, CameraMeasurementRequest):
            raise TypeError(
                "the selected Camera producer has no CameraMeasurementRequest"
            )
        source = DirectCameraScanSource(
            producer.request,
            producer.output_name,
            binding.transform_spec,
        )
        return source, _retirement_nodes(producer.run_node)

    if binding.source_kind == PULSE_SCAN_OCCUPANCY_SOURCE:
        if not isinstance(producer.request, OccupancyBindingIntent):
            raise TypeError(
                "the selected Occupancy producer has no OccupancyBindingIntent"
            )
        camera = resolve_producer(producer.request.camera_frame_signal)
        if not isinstance(camera, ConsoleProducerBinding):
            raise TypeError(
                "resolve_producer must return ConsoleProducerBinding"
            )
        if not isinstance(camera.request, CameraMeasurementRequest):
            raise TypeError(
                "the Occupancy input producer has no CameraMeasurementRequest"
            )
        calibration_ref = resolve_calibration(producer.request)
        if not isinstance(calibration_ref, CalibrationArtifactRef):
            raise TypeError(
                "resolve_calibration must return CalibrationArtifactRef"
            )
        source = OccupancyScanSource(
            camera.request,
            producer.output_name,
            calibration_ref,
            producer.request.model_kind,
            binding.transform_spec,
        )
        if camera.output_name != source.camera_output_name:
            raise ValueError(
                "the Occupancy input is not the Camera request's sole output"
            )
        return source, _retirement_nodes(
            producer.run_node,
            camera.run_node,
        )

    raise RuntimeError(
        f"unknown Pulse scan source kind {binding.source_kind!r}"
    )


__all__ = [
    "PULSE_SCAN_CAMERA_FRAME_SOURCE",
    "PULSE_SCAN_OCCUPANCY_SOURCE",
    "PulseScanBindingIntent",
    "PulseScanSourceBinding",
    "classify_pulse_scan_producer",
    "resolve_typed_scan_source",
]
