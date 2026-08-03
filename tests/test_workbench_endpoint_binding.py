from __future__ import annotations

import pytest

from zlc_neutral_atom.runtime.ports import (
    require_current_endpoint_binding,
)
from zlc_neutral_atom.runtime.ports import BoundDevice, DeviceBroker
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
)


@pytest.fixture
def binding() -> BoundDevice:
    broker = DeviceBroker()
    identity = broker.verify_identity(
        lambda: PhysicalDeviceIdentity(
            stable_device_identity="test-device",
            evidence_kind=DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        )
    )
    value = broker.bind(
        key=ResourceKey.parse("device/test/endpoint-binding"),
        identity=identity,
        execute_command=lambda _command: None,
    )
    yield value
    broker.shutdown()


@pytest.mark.parametrize("endpoint", ("camera", "sequencer"))
def test_endpoint_binding_accepts_current_instance(
    binding: BoundDevice,
    endpoint: str,
) -> None:
    require_current_endpoint_binding(
        binding,
        endpoint,
        binding.binding_instance_id,
    )


@pytest.mark.parametrize("endpoint", ("camera", "sequencer"))
def test_endpoint_binding_rejects_another_instance(
    binding: BoundDevice,
    endpoint: str,
) -> None:
    with pytest.raises(RuntimeError, match=rf"{endpoint} endpoint binding instance changed"):
        require_current_endpoint_binding(
            binding,
            endpoint,
            "another-binding-instance",
        )


@pytest.mark.parametrize("endpoint", ("camera", "sequencer"))
def test_endpoint_binding_allows_initial_probe_before_instance_is_frozen(
    binding: BoundDevice,
    endpoint: str,
) -> None:
    require_current_endpoint_binding(binding, endpoint, None)


@pytest.mark.parametrize("endpoint", ("camera", "sequencer"))
def test_endpoint_binding_requires_an_opaque_bound_device(endpoint: str) -> None:
    with pytest.raises(TypeError, match=rf"{endpoint} endpoint requires BoundDevice"):
        require_current_endpoint_binding(object(), endpoint, None)
