"""The checked-in neutral-atom tutorial is the executable user contract.

The old guard copied selected notebook calls into Python.  That let the copy
stay green when the actual notebook drifted.  This file instead validates and
executes the one shipped notebook itself; product-level physics and data
contracts remain covered by their owning tests.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
TUTORIAL = TUTORIALS / "neutral_atom_tutorial.ipynb"

_REQUIRED_MARKDOWN = (
    "连上装置",
    "拍一组图",
    "标定:站点、PSF 与读出模型",
    "阈值与保真度来自数据",
    "逐发判定",
    "自主 Pulse scan",
    "Release-recapture survival scan",
    "图形界面",
    "收尾",
)

_REQUIRED_CURRENT_CODE = (
    "Zou_lab_control.api",
    "exp.nodes.calibration",
    "exp.nodes.occupancy",
    "exp.nodes.camera_measurement",
    "exp.nodes.pulse_scan",
    "exp.nodes.temperature",
    "resolve_api_parameters",
    "FrozenScanTable",
    "temperature_release_recapture_request",
    "aggregate_fidelity",
    "global_fidelity",
    "schema.cell_schema.data_shape",
    "schema.point_table.columns",
    "WorkspacePaths.for_workspace",
    "workspace=WORKSPACE",
    "WORKSPACE.pulses_root",
)

_FORBIDDEN_OLD_OR_RAW_CODE = (
    "Zou_lab_control.notebook",
    "exp.readout.sitemap",
    "exp.readout.load_calibration",
    "exp.readout.detection_request",
    "exp.readout.detect",
    "exp.readout.load_occupancy",
    "exp.readout.camera_measurement_request",
    "exp.readout.prepare_camera_measurement",
    "exp.readout.prepare_scan_source",
    "exp.readout.materialize_scan",
    "exp.readout.temperature_release_recapture",
    "Zou_lab_control.frontend",
    "Zou_lab_control.neutral_atom",
    "load_sitemap_pulse",
    ".target.raw_lanes",
    'trigger_channel="ch11"',
    "exp.devices",
    "exp.camera",
    "triggered_frames(",
    "detection_time(",
    "fit_temperature(",
    "user_output_path",
    "repository=",
    "ZLC_TUTORIAL_WORKSPACE",
    "schema.point_axes",
)


def _load_notebook():
    notebook = nbformat.read(TUTORIAL, as_version=4)
    nbformat.validate(notebook)
    return notebook


def test_there_is_one_complete_current_user_tutorial() -> None:
    notebooks = sorted(path.name for path in TUTORIALS.glob("*.ipynb"))
    assert notebooks == ["neutral_atom_tutorial.ipynb"]

    notebook = _load_notebook()
    cell_ids = [cell["id"] for cell in notebook.cells]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))

    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    for text in _REQUIRED_MARKDOWN:
        assert text in markdown
    for text in _REQUIRED_CURRENT_CODE:
        assert text in code
    for text in _FORBIDDEN_OLD_OR_RAW_CODE:
        assert text not in code


def test_the_checked_in_tutorial_executes_on_the_virtual_installation(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "ZLC_TUTORIAL_AUTHORED_ROOT",
        str(ROOT),
    )
    monkeypatch.setenv(
        "ZLC_TUTORIAL_REPOSITORY_ROOT",
        str(tmp_path / "tutorial-workspace"),
    )
    notebook = _load_notebook()
    executed = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        allow_errors=False,
    ).execute(cwd=str(ROOT))

    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", ())
        if output.get("output_type") == "error"
    ]
