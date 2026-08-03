"""Built-in remote sequencer leaf type and its sole connection factory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
from zlc_neutral_atom.device_types import (
    CAPABILITY_EXTERNAL_TRIGGER_CLIENT,
    CAPABILITY_PULSE_EXECUTE,
    DeviceTypeDescriptor,
)
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.devices.sequencer.remote_pulse import RemotePulseExecutionEndpoint
from zlc_neutral_atom.installation_config import DeviceInstanceConfig
from zlc_neutral_atom.runtime.ports import BoundDevice, DeviceBroker, SafetyOperation
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
)
from zlc_pulse import (
    PulseDocument,
    PulseServerSnapshot,
    RemotePulseExecutionClient,
    bind_pulse_document_target,
    require_deployed_geometry_facts,
    validate_pulse_document_clock_grid,
)
from zlc_storage import normalized_text


DEFAULT_REMOTE_PORT = 18861
DEFAULT_TRANSPORT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class _RemotePulseConnection:
    client: RemotePulseExecutionClient
    snapshot: PulseServerSnapshot
    endpoint_label: str
    max_blocking_call_seconds: float | None


def _bind_remote_sequencer(
    broker: DeviceBroker,
    instance_id: str,
    connection: _RemotePulseConnection,
) -> BoundPulsePort:
    endpoint = RemotePulseExecutionEndpoint(
        connection.client,
        endpoint_label=connection.endpoint_label,
        max_blocking_call_seconds=connection.max_blocking_call_seconds,
    )
    binding: BoundDevice | None = None

    def current_binding() -> BoundDevice:
        if binding is None:
            raise RuntimeError("remote sequencer binding is not installed")
        return binding

    identity = PhysicalDeviceIdentity(
        f"remote-pulse-endpoint:{connection.endpoint_label}",
        DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
    )
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
        interrupt_operations={SafetyOperation.SAFE_STATE: endpoint.interrupt},
    )
    return BoundPulsePort(
        capability_attestation=broker.verify_capability(binding),
        editor_connection_mode="remote",
        _scan_progress_reader=lambda session_id, run_id, digest, count: (
            endpoint.observe_scan_progress(
                current_binding(), session_id, run_id, digest, count
            )
        ),
        _continuous_failure_waiter=lambda session_id, run_id, digest, timeout: (
            endpoint.wait_continuous_failure(
                current_binding(), session_id, run_id, digest, timeout
            )
        ),
    )


def _connect_remote(
    instance: DeviceInstanceConfig,
    _dependencies: Mapping[str, object],
    broker: object,
    required_pulse_document: object | None,
):
    if not isinstance(broker, DeviceBroker):
        raise TypeError("broker must be DeviceBroker")
    values = _REMOTE_SCHEMA.freeze(instance.parameters)
    host = normalized_text(values["host"], "remote pulse host")
    port = values["port"]
    timeout = float(values["transport_timeout_seconds"])
    assert isinstance(port, int)
    if required_pulse_document is not None and not isinstance(
        required_pulse_document,
        PulseDocument,
    ):
        raise TypeError("required_pulse_document must be PulseDocument or None")
    client = RemotePulseExecutionClient.connect(
        host,
        port,
        transport_timeout_seconds=timeout,
    )
    try:
        snapshot = client.snapshot()
        require_deployed_geometry_facts(
            snapshot.geometry_fingerprint,
            snapshot.clock_hz,
        )
        if required_pulse_document is not None:
            bind_pulse_document_target(required_pulse_document, snapshot.target)
            validate_pulse_document_clock_grid(
                required_pulse_document,
                snapshot.clock_hz,
            )
        safe_snapshot = client.safe_state(timeout=timeout * 0.9)
        if safe_snapshot.connection_generation != snapshot.connection_generation:
            raise RuntimeError("remote pulse generation changed while entering SAFE")
        connection = _RemotePulseConnection(
            client,
            safe_snapshot,
            f"{host}:{port}",
            None,
        )
        port_capability = _bind_remote_sequencer(
            broker,
            instance.instance_id,
            connection,
        )
    except BaseException as primary:
        try:
            client.close()
        except BaseException as error:
            primary.add_note(f"remote client close also failed: {error}")
        raise
    return (
        {
            CAPABILITY_PULSE_EXECUTE: port_capability,
            CAPABILITY_EXTERNAL_TRIGGER_CLIENT: client,
        },
        client.close,
    )


_REMOTE_SCHEMA = AuthoringSchema(
    (
        AuthoringField("host", "text", "Host", "", True),
        AuthoringField(
            "port",
            "int",
            "Port",
            DEFAULT_REMOTE_PORT,
            True,
            minimum=1,
            maximum=65535,
        ),
        AuthoringField(
            "transport_timeout_seconds",
            "float",
            "Transport timeout",
            DEFAULT_TRANSPORT_TIMEOUT_SECONDS,
            True,
            unit="s",
            minimum=math.nextafter(0.0, math.inf),
        ),
    )
)


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        type_id="sequencer.remote_pulse",
        domain="sequencer",
        label="Remote pulse sequencer",
        authoring_schema=_REMOTE_SCHEMA,
        capabilities=(CAPABILITY_PULSE_EXECUTE, CAPABILITY_EXTERNAL_TRIGGER_CLIENT),
        requirements=(),
        factory=_connect_remote,
    ),
)


__all__ = ["DEVICE_TYPES"]
