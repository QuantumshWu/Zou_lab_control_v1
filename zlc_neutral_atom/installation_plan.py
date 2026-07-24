"""Single public topology plan for each supported installation backend.

The runtime bootstrap validates the devices it composed against this immutable
plan, while DeviceManager renders the same values.  Public roles, their domains,
and expected adapter identities therefore cannot drift between the actual
installation and its editor.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import canonical_text


@dataclass(frozen=True, slots=True)
class InstallationDevicePlan:
    role: str
    domain: str
    adapter_kind: str

    def __post_init__(self) -> None:
        for field in ("role", "domain", "adapter_kind"):
            object.__setattr__(
                self,
                field,
                canonical_text(getattr(self, field), field),
            )


_PLANS = {
    "virtual": (
        InstallationDevicePlan(
            "sequencer",
            "sequencer",
            "zlc_neutral_atom.bootstrap._virtual_hardware.VirtualSequencer",
        ),
        InstallationDevicePlan(
            "rf",
            "rf",
            "zlc_neutral_atom.bootstrap._virtual_hardware.VirtualRfSource",
        ),
        InstallationDevicePlan(
            "camera",
            "camera",
            "zlc_neutral_atom.bootstrap._virtual_hardware.VirtualCamera",
        ),
        InstallationDevicePlan(
            "mot_camera",
            "camera",
            "zlc_neutral_atom.bootstrap._virtual_hardware.VirtualCamera",
        ),
    ),
    "remote_pulse": (
        InstallationDevicePlan(
            "sequencer",
            "sequencer",
            "zlc_pulse.client.RemotePulseExecutionClient",
        ),
    ),
}


def installation_device_plan(backend: str) -> tuple[InstallationDevicePlan, ...]:
    """Return the exact public device topology for ``backend``."""

    name = canonical_text(backend, "installation backend")
    try:
        return _PLANS[name]
    except KeyError as error:
        raise ValueError(f"unsupported installation backend {name!r}") from error


__all__ = ["InstallationDevicePlan", "installation_device_plan"]
