from __future__ import annotations

from pathlib import Path

from Zou_lab_control.api import NodeApi, WorkspacePaths, connect
from zlc_neutral_atom.logic_node import discover_logic_nodes


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_namespace_descriptors_are_unique_and_complete() -> None:
    descriptors = discover_logic_nodes()
    descriptor_files = tuple(
        path
        for path in (ROOT / "zlc_neutral_atom" / "logic_nodes").rglob(
            "logic_node.py"
        )
        if "ui" not in path.parts and "__pycache__" not in path.parts
    )

    assert len(descriptors) == len(descriptor_files)
    assert tuple(value.api_name for value in descriptors) == tuple(
        sorted(value.api_name for value in descriptors)
    )
    assert len({value.api_name for value in descriptors}) == len(descriptors)
    assert len({value.definition.key for value in descriptors}) == len(descriptors)
    for descriptor in descriptors:
        assert descriptor.definition.key.owner_package.startswith(
            "zlc_neutral_atom.logic_nodes."
        )
        assert callable(descriptor.build_request)
        assert callable(descriptor.bind_execute)

    logic_root = ROOT / "zlc_neutral_atom" / "logic_nodes"
    assert not tuple(logic_root.rglob("package.py"))
    assert not tuple(logic_root.rglob("api.py"))
    assert not (ROOT / "zlc_neutral_atom" / "logic_node_package.py").exists()
    assert not (ROOT / "zlc_neutral_atom" / "logic_node_declaration.py").exists()

    application_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (
            ROOT / "Zou_lab_control" / "api",
            ROOT / "Zou_lab_control" / "workbench",
        )
        for path in root.glob("*.py")
    )
    assert "zlc_neutral_atom.logic_nodes." not in application_sources


def test_experiment_nodes_are_one_generic_projection_of_discovery(tmp_path) -> None:
    experiment = connect(
        "virtual",
        workspace=WorkspacePaths.for_workspace((tmp_path / "workspace").resolve()),
    )
    try:
        descriptors = discover_logic_nodes()
        assert experiment.nodes.names == tuple(
            descriptor.api_name for descriptor in descriptors
        )
        for descriptor in descriptors:
            api = getattr(experiment.nodes, descriptor.api_name)
            assert isinstance(api, NodeApi)
            assert api.descriptor is descriptor
        assert isinstance(experiment.nodes.calibration, NodeApi)
        assert isinstance(experiment.nodes.mot_field, NodeApi)
    finally:
        experiment.close()
