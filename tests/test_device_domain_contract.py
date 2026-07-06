"""CONTRACT (#E4): ONE role->base table for device-contract validation.

``DEVICE_DOMAINS`` (``devices/registry.py``, extended by ``register_device_domain``) is the
SINGLE source of the role->base relation.  ``validate_device_contract`` reads it directly, so
a lab-registered domain (RF source, DAQ, ...) is enforced at load time exactly like the
built-ins -- previously ``devices/base.py`` hardcoded a second ``ROLE_BASES`` copy of the
same three roles and validation read only that, so user-registered domains were never
contract-checked.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.base import BaseDevice  # noqa: E402
from Zou_lab_control.neutral_atom.devices.registry import (  # noqa: E402
    DEVICE_DOMAINS,
    load_devices,
    register_device_domain,
    validate_device_contract,
)


def test_every_registered_domain_key_is_enforced_from_the_one_table():
    """Enumerate DEVICE_DOMAINS (no re-typed role literals): for EVERY registered key, a device
    that is a BaseDevice but not the domain's declared base must fail loud; an unregistered
    role name only requires BaseDevice."""

    class Impostor(BaseDevice):
        pass

    impostor = Impostor()
    assert DEVICE_DOMAINS, "the built-in domains (camera/sequencer/trap_array) must be registered"
    for key, domain in DEVICE_DOMAINS.items():
        assert issubclass(domain.base_type, BaseDevice)
        # a plain BaseDevice subclass is never the domain base -- otherwise the raise below is vacuous
        assert not isinstance(impostor, domain.base_type), (
            f"domain {key!r} declares BaseDevice-level base_type; the contract check would be vacuous")
        with pytest.raises(TypeError, match=domain.base_type.__name__):
            validate_device_contract(key, impostor)
    # a role with no registered domain needs only the common BaseDevice contract
    validate_device_contract("some_unregistered_role", impostor)
    with pytest.raises(TypeError, match="BaseDevice"):
        validate_device_contract("some_unregistered_role", object())


def test_user_registered_domain_extends_load_time_validation():
    """``register_device_domain`` must extend contract validation automatically: a wrong-typed
    device under the new role fails loud at ``load_devices``; a right-typed one builds."""

    class RFSourceDevice(BaseDevice):
        pass

    class NotAnRFSource(BaseDevice):
        pass

    register_device_domain("rf_source", RFSourceDevice, "RF source")
    try:
        with pytest.raises(TypeError, match="must inherit RFSourceDevice"):
            load_devices({"rf_source": {"type": "NotAnRFSource"}},
                         lookup={"NotAnRFSource": NotAnRFSource}, open_devices=False)
        ds = load_devices({"rf_source": {"type": "RFSourceDevice"}},
                          lookup={"RFSourceDevice": RFSourceDevice}, open_devices=False)
        assert isinstance(ds.devices["rf_source"], RFSourceDevice)
        validate_device_contract("rf_source", ds.devices["rf_source"])  # the direct call agrees
    finally:
        DEVICE_DOMAINS.pop("rf_source", None)                           # leave the table as found
