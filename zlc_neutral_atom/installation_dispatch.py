"""Private dispatch through the fixed built-in installation namespace."""

from __future__ import annotations

from zlc_neutral_atom.installation_package import installation_package_for_config
from zlc_neutral_atom.installation_config import InstallationConfigDocument
from zlc_neutral_atom.installation_runtime import _InstallationComposition
from zlc_pulse import PulseDocument


def create_installation(
    document: InstallationConfigDocument,
    *,
    required_pulse_document: PulseDocument | None = None,
) -> _InstallationComposition:
    """Compose the exact package owning the document's frozen config type."""

    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")
    config = document.config
    package = installation_package_for_config(config)
    return package.compose(config, required_pulse_document)


__all__ = ["create_installation"]
