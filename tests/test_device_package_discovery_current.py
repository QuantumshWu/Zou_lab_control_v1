from __future__ import annotations

from pathlib import Path

import pytest

import zlc_neutral_atom.device_types as device_types
from zlc_neutral_atom.authoring import AuthoringSchema
from zlc_neutral_atom.device_types import (
    DeviceTypeDescriptor,
    discover_device_types,
)
from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    InstallationConfigDocument,
    installation_template,
)
from zlc_neutral_atom.installation_runtime import create_installation


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_leaf_namespace_is_the_only_device_type_discovery() -> None:
    descriptors = discover_device_types()
    leaf_files = tuple(
        path
        for path in (ROOT / "zlc_neutral_atom" / "devices").rglob(
            "device_types.py"
        )
        if "__pycache__" not in path.parts
    )

    assert len(leaf_files) == 3
    assert tuple(value.type_id for value in descriptors) == (
        "camera.dcam",
        "camera.pylon",
        "camera.virtual_mot",
        "camera.virtual_readout",
        "rf.virtual",
        "sequencer.remote_pulse",
        "sequencer.virtual",
    )
    assert len({value.type_id for value in descriptors}) == len(descriptors)
    assert all(value.capabilities for value in descriptors)

    generic_source = (ROOT / "zlc_neutral_atom" / "device_types.py").read_text(
        encoding="utf-8"
    )
    assert "zlc_neutral_atom.devices." not in generic_source
    for retired in (
        "installation_package",
        "installation_plan",
        "installation_dispatch",
        "installation_assets",
    ):
        assert not (ROOT / "zlc_neutral_atom" / f"{retired}.py").exists()


def test_virtual_template_composes_through_generic_capabilities() -> None:
    installation = create_installation(installation_template("virtual", seed=17))
    try:
        catalog = installation.runtime.device_catalog
        assert tuple(catalog) == ("sequencer", "rf", "camera", "mot-camera")
        assert catalog.roles("camera") == ("camera", "mot_camera")
        camera = catalog.require("camera")
        assert camera.role == "camera"
        assert installation.runtime.require_capability(
            camera.ref,
            "camera.capture",
        ) is not None
        with pytest.raises(ValueError, match="does not provide"):
            installation.runtime.require_capability(camera.ref, "rf.table")
    finally:
        assert installation.runtime.shutdown(timeout=2.0)


def test_synthetic_leaf_needs_no_manager_facade_or_runtime_type_branch(
    monkeypatch,
) -> None:
    capability = object()
    closed: list[str] = []

    def factory(instance, dependencies, _broker, required_pulse_document):
        assert instance.instance_id == "echo"
        assert dependencies == {}
        assert required_pulse_document is None
        return {"test.echo": capability}, lambda: closed.append("echo")

    descriptor = DeviceTypeDescriptor(
        type_id="test.echo",
        domain="test",
        label="Echo",
        authoring_schema=AuthoringSchema(()),
        capabilities=("test.echo",),
        requirements=(),
        factory=factory,
    )
    monkeypatch.setattr(
        device_types,
        "discover_device_types",
        lambda: (descriptor,),
    )
    document = InstallationConfigDocument(
        (DeviceInstanceConfig("echo", "echo", "test.echo", {}),)
    )

    installation = create_installation(document)
    reference = installation.runtime.device_catalog.require("echo").ref
    assert installation.runtime.require_capability(reference, "test.echo") is capability
    assert installation.runtime.shutdown(timeout=2.0)
    assert closed == ["echo"]
