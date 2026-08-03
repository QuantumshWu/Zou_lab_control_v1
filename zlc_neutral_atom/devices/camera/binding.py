"""Bind one camera endpoint to the broker under its stable instance identity."""

from __future__ import annotations

from zlc_neutral_atom.runtime.ports import BoundDevice, DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import PhysicalDeviceIdentity, ResourceKey
from zlc_storage import canonical_text

from .endpoint import CameraCaptureEndpoint, CameraMonitorEndpoint


def bind_camera_endpoint(
    broker: DeviceBroker,
    *,
    instance_id: str,
    identity: PhysicalDeviceIdentity,
    endpoint: CameraCaptureEndpoint | CameraMonitorEndpoint,
):
    """Bind the sole camera command, close, and DISARM surface."""

    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    instance_id = canonical_text(instance_id, "camera instance_id")
    if not isinstance(identity, PhysicalDeviceIdentity):
        raise TypeError("identity must be PhysicalDeviceIdentity")
    if not isinstance(endpoint, (CameraCaptureEndpoint, CameraMonitorEndpoint)):
        raise TypeError("endpoint must be a camera endpoint")
    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("camera endpoint binding is not installed")
        return binding

    proof = broker.verify_identity(lambda: identity)
    binding = broker.bind(
        key=ResourceKey(("device", instance_id)),
        identity=proof,
        execute_command=lambda command: endpoint.execute_command(
            current_binding(), command
        ),
        capability_probe=lambda: endpoint.capability_probe(current_binding()),
        close_session=lambda command: endpoint.close_session(
            current_binding(), command
        ),
        interrupt_operations={SafetyOperation.DISARM: endpoint.interrupt},
    )
    return broker.verify_capability(binding)


__all__ = ["bind_camera_endpoint"]
