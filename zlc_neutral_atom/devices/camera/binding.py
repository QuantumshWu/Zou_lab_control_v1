"""Composition-owned binding of one camera endpoint to the device broker."""

from __future__ import annotations

from zlc_neutral_atom.installation_assets import InstallationAsset
from zlc_neutral_atom.installation_runtime import _identity_for
from zlc_neutral_atom.runtime.ports import BoundDevice, DeviceBroker, SafetyOperation

from .endpoint import CameraCaptureEndpoint, CameraMonitorEndpoint


def bind_camera_endpoint(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    endpoint: CameraCaptureEndpoint | CameraMonitorEndpoint,
):
    """Bind the sole camera command, close, and DISARM surface."""

    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    if not isinstance(asset, InstallationAsset):
        raise TypeError("asset must be InstallationAsset")
    if not isinstance(endpoint, (CameraCaptureEndpoint, CameraMonitorEndpoint)):
        raise TypeError("endpoint must be a camera endpoint")
    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("camera endpoint binding is not installed")
        return binding

    identity = _identity_for(asset, asset_map_revision)
    proof = broker.verify_identity(lambda: identity)
    binding = broker.bind(
        key=asset.resource_key,
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
