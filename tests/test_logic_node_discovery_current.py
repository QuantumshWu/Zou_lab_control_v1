from __future__ import annotations

from pathlib import Path

from Zou_lab_control.api import WorkspacePaths, connect
from zlc_neutral_atom.logic_node_package import discover_logic_node_packages


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_namespace_packages_are_unique_and_complete() -> None:
    packages = discover_logic_node_packages()
    package_files = tuple(
        path
        for path in (ROOT / "zlc_neutral_atom" / "logic_nodes").rglob("package.py")
        if "__pycache__" not in path.parts
    )

    assert len(packages) == len(package_files)
    names = tuple(package.api_name for package in packages)
    pending = {package.api_name: package for package in packages}
    resolved: set[str] = set()
    expected_order: list[str] = []
    while pending:
        ready = tuple(
            sorted(
                name
                for name, package in pending.items()
                if set(package.api_dependencies) <= resolved
            )
        )
        assert ready
        for name in ready:
            pending.pop(name)
            resolved.add(name)
            expected_order.append(name)
    assert names == tuple(expected_order)
    assert len({package.api_name for package in packages}) == len(packages)
    assert len({package.declaration.definition.key for package in packages}) == len(
        packages
    )
    assert all(
        (package.declaration.bind_request is None)
        != (package.bind_hosted_request is None)
        for package in packages
    )
    assert (ROOT / "Zou_lab_control" / "api").is_dir()
    assert "zlc_neutral_atom.logic_nodes." not in (
        ROOT / "Zou_lab_control" / "workbench" / "_composition.py"
    ).read_text(encoding="utf-8")
    assert "zlc_neutral_atom.logic_nodes." not in (
        ROOT / "Zou_lab_control" / "api" / "__init__.py"
    ).read_text(encoding="utf-8")

    application_api = ROOT / "Zou_lab_control" / "api"
    application_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in application_api.glob("*.py")
    )
    assert "zlc_neutral_atom.logic_nodes." not in application_sources
    assert ".pulse_scan" not in application_sources
    assert ".occupancy" not in application_sources

    leaf_api_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "zlc_neutral_atom" / "logic_nodes").rglob("api.py")
    )
    assert "_borrow_services" not in leaf_api_sources
    assert "services." not in leaf_api_sources


def test_experiment_nodes_are_projected_from_discovered_packages(tmp_path) -> None:
    experiment = connect(
        "virtual",
        workspace=WorkspacePaths.for_workspace(
            ROOT,
            repository_root=tmp_path,
        ),
    )
    try:
        packages = discover_logic_node_packages()
        assert experiment.nodes.names == tuple(
            package.api_name for package in packages
        )
        assert experiment.nodes.calibration is not None
        assert experiment.nodes.mot_field is not None
    finally:
        experiment.close()
