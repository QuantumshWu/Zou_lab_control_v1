"""Composition of the current remote-pulse sequencer attachment."""

from __future__ import annotations

import math
import uuid

from fpga.pulse_streamer.host.image import StreamerParams, build_fingerprint
from zlc_neutral_atom.installation_assets import (
    InstallationAsset,
    InstallationAssetMap,
    adapter_kind,
)
from zlc_neutral_atom.installation_runtime import (
    _InstallationComposition,
    _InstallationRuntime,
    _catalog,
    _identity_for,
)
from zlc_neutral_atom.devices.sequencer.remote_pulse import RemotePulseExecutionEndpoint
from zlc_neutral_atom.installation_plan import installation_device_plan
from zlc_neutral_atom.runtime.ports import BoundDevice, DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import DeviceIdentityEvidenceKind, ResourceArbiter, ResourceKey
from zlc_neutral_atom.runtime.run import RunController
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_pulse import PulseDocument, RemotePulseExecutionClient, bind_pulse_document_target, validate_pulse_document_clock_grid
from zlc_storage import canonical_digest, normalized_text

def _bind_remote_sequencer(
    broker: DeviceBroker,
    asset: InstallationAsset,
    asset_map_revision: str,
    client: RemotePulseExecutionClient,
    *,
    endpoint_label: str,
    max_blocking_call_seconds: float | None,
) -> BoundPulsePort:
    """Bind the current remote protocol behind the same typed pulse Port.

    The GUI and notebook never receive ``client`` or the endpoint.  They submit a
    declarative PulseRunRequest through the ordinary RunController, so remote use
    cannot grow a second prepare/fire/safe authority.
    """

    endpoint = RemotePulseExecutionEndpoint(
        client,
        endpoint_label=endpoint_label,
        max_blocking_call_seconds=max_blocking_call_seconds,
    )
    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("remote sequencer endpoint binding is not installed")
        return binding

    identity = _identity_for(asset, asset_map_revision)
    proof = broker.verify_identity(lambda: identity)
    binding = broker.bind(
        key=asset.resource_key,
        identity=proof,
        execute_command=lambda command: endpoint.execute_command(
            current_binding(),
            command,
        ),
        capability_probe=lambda: endpoint.capability_probe(current_binding()),
        close_session=lambda command: endpoint.close_session(
            current_binding(),
            command,
        ),
        interrupt_operations={SafetyOperation.SAFE_STATE: endpoint.interrupt},
    )
    return BoundPulsePort(
        broker.verify_capability(binding),
        lambda session_id, run_id, artifact_digest, point_count: (
            endpoint.observe_scan_progress(
                current_binding(),
                session_id,
                run_id,
                artifact_digest,
                point_count,
            )
        ),
        lambda session_id, run_id, artifact_digest, timeout: (
            endpoint.wait_continuous_failure(
                current_binding(),
                session_id,
                run_id,
                artifact_digest,
                timeout,
            )
        ),
    )


def create_remote_pulse_installation(
    *,
    host: str,
    port: int,
    transport_timeout_seconds: float,
    max_blocking_call_seconds: float | None = None,
    required_pulse_document: PulseDocument | None = None,
) -> _InstallationComposition:
    """Compose one sequencer-only installation over the current pulse RPC.

    Network reachability is proven before publishing the owned installation.
    A typo or an unavailable server therefore remains retryable.  Once the remote
    generation has entered SAFE and is bound, all later failures are treated like
    any other partial hardware startup and fail closed.
    """

    endpoint_host = normalized_text(host, "remote pulse host")
    if isinstance(port, bool) or not isinstance(port, int):
        raise TypeError("remote pulse port must be an integer")
    if not 1 <= port <= 65535:
        raise ValueError("remote pulse port must be between 1 and 65535")
    if isinstance(transport_timeout_seconds, bool) or not isinstance(
        transport_timeout_seconds,
        (int, float),
    ):
        raise TypeError("transport_timeout_seconds must be a number")
    transport_timeout = float(transport_timeout_seconds)
    if not math.isfinite(transport_timeout) or transport_timeout <= 0:
        raise ValueError("transport_timeout_seconds must be finite and positive")
    if max_blocking_call_seconds is None:
        blocking_limit = None
    else:
        if isinstance(max_blocking_call_seconds, bool) or not isinstance(
            max_blocking_call_seconds,
            (int, float),
        ):
            raise TypeError("max_blocking_call_seconds must be a number or None")
        blocking_limit = float(max_blocking_call_seconds)
        if not math.isfinite(blocking_limit) or blocking_limit <= 0:
            raise ValueError("max_blocking_call_seconds must be finite and positive")
        if blocking_limit >= transport_timeout:
            raise ValueError(
                "max_blocking_call_seconds must be shorter than the transport timeout"
            )
    if required_pulse_document is not None and not isinstance(
        required_pulse_document,
        PulseDocument,
    ):
        raise TypeError("required_pulse_document must be PulseDocument or None")
    client = RemotePulseExecutionClient.connect(
        endpoint_host,
        port,
        transport_timeout_seconds=transport_timeout,
    )
    try:
        snapshot = client.snapshot()
        host_geometry = build_fingerprint(StreamerParams())
        if snapshot.geometry_fingerprint != host_geometry:
            raise ValueError(
                "remote pulse geometry differs from the current host compiler "
                f"geometry: remote=0x{snapshot.geometry_fingerprint:08x}, "
                f"host=0x{host_geometry:08x}"
            )
        if required_pulse_document is not None:
            bind_pulse_document_target(
                required_pulse_document,
                snapshot.target,
            )
            validate_pulse_document_clock_grid(
                required_pulse_document,
                snapshot.clock_hz,
            )
    except BaseException as primary:
        try:
            client.close()
        except BaseException as close_error:
            try:
                primary.add_note(
                    "close error: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            except BaseException:
                pass
        raise
    endpoint_label = f"{endpoint_host}:{port}"
    devices: dict[str, object] = {"sequencer": client}
    resources: ResourceArbiter | None = None
    broker: DeviceBroker | None = None
    try:
        safe_timeout = (
            client.transport_timeout_seconds * 0.9
            if max_blocking_call_seconds is None
            else blocking_limit
        )
        client.safe_state(timeout=safe_timeout)
        endpoint_identity = canonical_digest(
            {
                "protocol": "zlc.current-pulse-rpc",
                "host": endpoint_host,
                "port": port,
            }
        )
        assets = InstallationAssetMap(
            (
                InstallationAsset(
                    asset_id="remote-sequencer",
                    role="sequencer",
                    resource_key=ResourceKey.parse("device/sequencer"),
                    adapter_kind=adapter_kind(client),
                    evidence_kind=(
                        DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT
                    ),
                    expected_identity=f"remote-pulse-endpoint:{endpoint_identity}",
                ),
            )
        )
        installation_id = f"installation-{assets.revision[:20]}"
        runtime_instance_id = uuid.uuid4().hex
        broker = DeviceBroker()
        pulse_port = _bind_remote_sequencer(
            broker,
            assets.require("sequencer", client),
            assets.revision,
            client,
            endpoint_label=endpoint_label,
            max_blocking_call_seconds=blocking_limit,
        )
        catalog = _catalog(
            installation_id,
            runtime_instance_id,
            assets,
            devices,
            installation_device_plan("remote_pulse"),
        )
        resources = ResourceArbiter()
        controller = RunController(resources)
        runtime = _InstallationRuntime(
            installation_id=installation_id,
            runtime_instance_id=runtime_instance_id,
            catalog=catalog,
            resources=resources,
            broker=broker,
            controller=controller,
            camera_ports={},
            camera_monitor_ports={},
            pulse_ports={"sequencer": pulse_port},
            rf_ports={},
            raw_graph=devices,
            close_order=("sequencer",),
        )
        return _InstallationComposition(runtime=runtime)
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        for action in (
            None if broker is None else broker.shutdown,
            client.close,
        ):
            if action is None:
                continue
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(error)
        if not cleanup_errors:
            final_action = None if resources is None else resources.shutdown
            if final_action is not None:
                try:
                    final_action()
                except BaseException as error:
                    cleanup_errors.append(error)
        for error in cleanup_errors:
            try:
                primary.add_note(
                    "remote pulse installation startup cleanup also failed: "
                    f"{type(error).__name__}: {error}"
                )
            except BaseException:
                pass
        raise

__all__ = ["create_remote_pulse_installation"]
