from __future__ import annotations

import pytest

from zlc_neutral_atom.runtime import (
    BoundDevice,
    DeviceBroker,
    DeviceIdentityAck,
    DeviceIdentityEvidenceKind,
    ResourceKey,
    SafeStateAck,
)
from zlc_workbench._endpoint_binding import require_current_endpoint_binding


@pytest.fixture
def binding() -> BoundDevice:
    broker = DeviceBroker()
    identity = broker.verify_identity(
        lambda: DeviceIdentityAck(
            "test-device",
            DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
            "test-connection",
            "test-assets-v1",
        )
    )
    return broker.bind(
        key=ResourceKey.parse("device/test/endpoint-binding"),
        identity=identity,
        execute_command=lambda _command: None,
        cleanup_operations={},
        verify_safe_state=lambda: SafeStateAck("test-device-safe"),
    )


@pytest.mark.parametrize("endpoint", ("camera", "sequencer"))
def test_endpoint_binding_accepts_current_identity(
    binding: BoundDevice,
    endpoint: str,
) -> None:
    require_current_endpoint_binding(
        binding,
        endpoint=endpoint,
        binding_id=binding.binding_id,
        connection_generation=binding.connection_generation,
    )


@pytest.mark.parametrize(
    ("binding_id", "connection_generation"),
    (("wrong-binding", None), (None, "wrong-generation")),
)
@pytest.mark.parametrize("endpoint", ("camera", "sequencer"))
def test_endpoint_binding_rejects_each_stale_identity_component(
    binding: BoundDevice,
    endpoint: str,
    binding_id: str | None,
    connection_generation: str | None,
) -> None:
    expected_binding_id = binding.binding_id if binding_id is None else binding_id
    expected_generation = (
        binding.connection_generation
        if connection_generation is None
        else connection_generation
    )
    with pytest.raises(
        RuntimeError,
        match=rf"{endpoint} endpoint binding generation changed",
    ):
        require_current_endpoint_binding(
            binding,
            endpoint=endpoint,
            binding_id=expected_binding_id,
            connection_generation=expected_generation,
        )


@pytest.mark.parametrize("endpoint", ("camera", "sequencer"))
def test_endpoint_binding_allows_bound_device_before_first_probe(
    binding: BoundDevice,
    endpoint: str,
) -> None:
    require_current_endpoint_binding(
        binding,
        endpoint=endpoint,
        binding_id=None,
        connection_generation=None,
    )


@pytest.mark.parametrize("endpoint", ("camera", "sequencer"))
def test_endpoint_binding_requires_opaque_bound_device(endpoint: str) -> None:
    with pytest.raises(TypeError, match=rf"{endpoint} endpoint requires BoundDevice"):
        require_current_endpoint_binding(
            object(),
            endpoint=endpoint,
            binding_id=None,
            connection_generation=None,
        )
