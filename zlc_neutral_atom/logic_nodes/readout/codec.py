"""Owner tree mappings embedded by durable neutral-atom artifacts."""

from __future__ import annotations

from typing import Any

from zlc_data import AxisId
from zlc_data.codec import value_schema_from_tree, value_schema_to_tree
from zlc_storage import exact_mapping as _exact_map

from zlc_neutral_atom.devices.camera.contract import (
    CAMERA_FRAME_FACT_FIELDS,
    ReadoutBindingKey,
    camera_frame_facts_from_tree,
    camera_frame_facts_to_tree,
    readout_binding_key_from_tree,
    readout_binding_key_to_tree,
)

from .contracts import (
    CalibrationCaptureLayout,
    FrameContract,
)


_FRAME_FIELDS = CAMERA_FRAME_FACT_FIELDS | {
    "binding",
    "exposure_seconds",
    "gain",
    "readout_mode",
    "opaque_frame_settings_fingerprint",
    "frame_schema",
}


def frame_contract_to_tree(value: FrameContract) -> dict[str, Any]:
    if not isinstance(value, FrameContract):
        raise TypeError("value must be FrameContract")
    return {
        "binding": readout_binding_key_to_tree(value.binding),
        **camera_frame_facts_to_tree(value),
        "exposure_seconds": value.exposure_seconds,
        "gain": value.gain,
        "readout_mode": value.readout_mode,
        "opaque_frame_settings_fingerprint": value.opaque_frame_settings_fingerprint,
        "frame_schema": value_schema_to_tree(value.frame_schema),
    }


def frame_contract_from_tree(tree: Any) -> FrameContract:
    data = _exact_map(
        tree,
        _FRAME_FIELDS,
        "frame contract",
        discriminator=None,
    )
    value = FrameContract(
        binding=readout_binding_key_from_tree(data["binding"]),
        **camera_frame_facts_from_tree(data),
        exposure_seconds=data["exposure_seconds"],
        gain=data["gain"],
        readout_mode=data["readout_mode"],
        opaque_frame_settings_fingerprint=data["opaque_frame_settings_fingerprint"],
        frame_schema=value_schema_from_tree(data["frame_schema"]),
    )
    return value


def calibration_capture_layout_to_tree(
    value: CalibrationCaptureLayout,
) -> dict[str, Any]:
    if not isinstance(value, CalibrationCaptureLayout):
        raise TypeError("value must be CalibrationCaptureLayout")
    return {
        "readout_event_axis_id": value.readout_event_axis_id.value,
        "reference_event_indices": list(value.reference_event_indices),
        "readout_event_index": value.readout_event_index,
    }


def calibration_capture_layout_from_tree(tree: Any) -> CalibrationCaptureLayout:
    data = _exact_map(
        tree,
        {"readout_event_axis_id", "reference_event_indices", "readout_event_index"},
        "calibration capture layout",
        discriminator=None,
    )
    return CalibrationCaptureLayout(
        readout_event_axis_id=AxisId(data["readout_event_axis_id"]),
        reference_event_indices=tuple(data["reference_event_indices"]),
        readout_event_index=data["readout_event_index"],
    )


__all__ = [
    "calibration_capture_layout_from_tree",
    "calibration_capture_layout_to_tree",
    "frame_contract_from_tree",
    "frame_contract_to_tree",
]
