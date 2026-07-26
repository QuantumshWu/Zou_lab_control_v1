"""Single public topology plan for each supported installation backend.

Installation composition validates the devices it built against this immutable
plan, while DeviceManager renders the same values.  Public roles, their domains,
and expected adapter identities therefore cannot drift between the actual
installation and its editor.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_neutral_atom.installation_config import (
    SUPPORTED_INSTALLATION_BACKENDS,
    default_installation_authoring_schema,
)
from zlc_storage import canonical_text


@dataclass(frozen=True, slots=True)
class InstallationDevicePlan:
    role: str
    domain: str
    adapter_kind: str
    summary: str
    configuration_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("role", "domain", "adapter_kind", "summary"):
            object.__setattr__(
                self,
                field,
                canonical_text(getattr(self, field), field),
            )
        keys = tuple(
            canonical_text(key, "installation configuration key")
            for key in self.configuration_keys
        )
        if len(keys) != len(set(keys)):
            raise ValueError("installation configuration keys must be unique")
        object.__setattr__(self, "configuration_keys", keys)


_PLANS = {
    "virtual": (
        InstallationDevicePlan(
            "sequencer",
            "sequencer",
            "zlc_neutral_atom.devices.simulation.apparatus.VirtualSequencer",
            "In-process pulse target execution",
        ),
        InstallationDevicePlan(
            "rf",
            "rf",
            "zlc_neutral_atom.devices.simulation.apparatus.VirtualRfSource",
            "In-process RF-table source driven by the virtual sequencer",
        ),
        InstallationDevicePlan(
            "camera",
            "camera",
            "zlc_neutral_atom.devices.simulation.apparatus.VirtualCamera",
            "Externally triggered readout camera",
            ("seed",),
        ),
        InstallationDevicePlan(
            "mot_camera",
            "camera",
            "zlc_neutral_atom.devices.simulation.apparatus.VirtualCamera",
            "MOT camera with live and finite triggered acquisition",
            ("seed",),
        ),
    ),
    "remote_pulse": (
        InstallationDevicePlan(
            "sequencer",
            "sequencer",
            "zlc_pulse.client.RemotePulseExecutionClient",
            "Remote pulse execution endpoint",
            ("host", "port", "transport_timeout_seconds"),
        ),
    ),
}

if frozenset(_PLANS) != SUPPORTED_INSTALLATION_BACKENDS:
    raise RuntimeError(
        "installation device plans differ from the supported config backends"
    )
for _backend, _plan in _PLANS.items():
    _declared_keys = frozenset(
        default_installation_authoring_schema(_backend).keys
    )
    _referenced_keys = frozenset(
        key for device in _plan for key in device.configuration_keys
    )
    if not _referenced_keys <= _declared_keys:
        raise RuntimeError(
            f"installation plan {_backend!r} references undeclared config fields"
        )


def installation_device_plan(backend: str) -> tuple[InstallationDevicePlan, ...]:
    """Return the exact public device topology for ``backend``."""

    name = canonical_text(backend, "installation backend")
    try:
        return _PLANS[name]
    except KeyError as error:
        raise ValueError(f"unsupported installation backend {name!r}") from error


__all__ = ["InstallationDevicePlan", "installation_device_plan"]
