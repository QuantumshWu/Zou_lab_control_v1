from __future__ import annotations

from pathlib import Path

from zlc_neutral_atom.installation_package import discover_installation_packages
from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    supported_installation_backends,
)
from zlc_neutral_atom.installation_dispatch import create_installation


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_device_namespace_is_the_only_executable_backend_dispatch() -> None:
    packages = discover_installation_packages()
    package_files = tuple(
        path
        for path in (ROOT / "zlc_neutral_atom" / "devices").rglob("package.py")
        if "__pycache__" not in path.parts
    )

    assert len(packages) == len(package_files)
    assert tuple(package.backend for package in packages) == tuple(
        sorted(supported_installation_backends())
    )
    assert len({package.config_type for package in packages}) == len(packages)

    dispatch_source = (
        ROOT / "zlc_neutral_atom" / "installation_dispatch.py"
    ).read_text(encoding="utf-8")
    assert "devices.simulation" not in dispatch_source
    assert "devices.sequencer" not in dispatch_source
    assert "isinstance(config" not in dispatch_source


def test_virtual_document_composes_through_its_discovered_owner() -> None:
    installation = create_installation(
        InstallationConfigDocument.from_parameters("virtual", {"seed": 17})
    )
    try:
        assert installation.runtime.device_catalog.roles("sequencer") == (
            "sequencer",
        )
        assert installation.runtime.device_catalog.roles("camera") == (
            "camera",
            "mot_camera",
        )
    finally:
        assert installation.runtime.shutdown(timeout=2.0)
