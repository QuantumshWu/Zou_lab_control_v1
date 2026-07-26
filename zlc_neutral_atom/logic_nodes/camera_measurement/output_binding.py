"""Physical binding for one running Camera Measurement frame output.

The Camera domain owns this value because only it can join the selected public
``frame_i`` output to the endpoint-read working point and the exact live stream
generation that publishes it.  Downstream processors may require this typed
binding; a GUI or generic signal router must never reconstruct camera physics
from a signal name or ndarray shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import StreamGenerationId, ValueSchema
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.devices.camera.contract import (
    CameraCapabilityEvidence,
    ReadoutBindingKey,
    validate_camera_frame_schema_facts,
)
from zlc_neutral_atom.runtime.resources import DeviceBindingStamp
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_storage import canonical_digest

from .definition import (
    CAMERA_FRAME_OUTPUT_CONTRACT_ID,
    camera_frame_output_index,
)


@dataclass(frozen=True, slots=True)
class CameraFrameOutputBinding:
    """Immutable physical identity of one active ``frame_i`` signal.

    ``capability_evidence`` is captured after any requested camera
    reconfiguration has completed and been read back.  ``stream_generation``
    prevents the same schema from being silently rebound to a restarted Camera
    Measurement.
    """

    output: DatasetOutputDeclaration
    readout_event_index: int
    readout_binding: ReadoutBindingKey
    capability_evidence: CameraCapabilityEvidence
    binding_stamp: DeviceBindingStamp
    frame_schema: ValueSchema
    stream_id: StreamId
    stream_generation: StreamGenerationId

    def __post_init__(self) -> None:
        if not isinstance(self.output, DatasetOutputDeclaration):
            raise TypeError("output must be DatasetOutputDeclaration")
        if self.output.contract_id != CAMERA_FRAME_OUTPUT_CONTRACT_ID:
            raise ValueError("Camera frame binding has another output contract")
        if (
            isinstance(self.readout_event_index, bool)
            or not isinstance(self.readout_event_index, int)
            or self.readout_event_index < 0
        ):
            raise ValueError("readout_event_index must be a non-negative integer")
        if camera_frame_output_index(self.output.name) != self.readout_event_index:
            raise ValueError("Camera output name and readout event index differ")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        evidence = self.capability_evidence
        if not isinstance(evidence, CameraCapabilityEvidence):
            raise TypeError("capability_evidence must be CameraCapabilityEvidence")
        if self.readout_binding.value != evidence.source_id:
            raise ValueError("Camera source and readout binding differ")
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("binding_stamp must be DeviceBindingStamp")
        facts = evidence.physical_facts
        if (
            self.binding_stamp.physical_identity.stable_device_identity
            != facts.camera_identity
        ):
            raise ValueError("Camera working point belongs to another device binding")
        if not isinstance(self.frame_schema, ValueSchema):
            raise TypeError("frame_schema must be ValueSchema")
        validate_camera_frame_schema_facts(
            spatial_y_axis_id=facts.spatial_y_axis_id,
            spatial_x_axis_id=facts.spatial_x_axis_id,
            coordinate_frame=facts.coordinate_frame,
            output_shape_yx=facts.output_shape_yx,
            dtype=facts.dtype,
            count_unit=facts.count_unit,
            frame_schema=self.frame_schema,
        )
        if not isinstance(self.stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(self.stream_generation, StreamGenerationId):
            raise TypeError("stream_generation must be StreamGenerationId")

    @property
    def identity(self) -> str:
        """Canonical ephemeral binding identity used in processor provenance."""

        return canonical_digest(
            {
                "owner": "zlc_neutral_atom.camera-measurement.frame-output-binding",
                "output_name": self.output.name,
                "output_contract": self.output.contract_id,
                "readout_event_index": self.readout_event_index,
                "readout_binding": self.readout_binding.value,
                "capability_fingerprint": self.capability_evidence.fingerprint,
                "binding_instance_id": self.binding_stamp.binding_instance_id,
                "frame_schema": self.frame_schema.fingerprint,
                "stream_id": self.stream_id.value,
                "stream_generation": self.stream_generation.value,
            }
        )


__all__ = ["CameraFrameOutputBinding"]
