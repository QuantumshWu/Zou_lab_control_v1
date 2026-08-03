"""The checked-in neutral-atom tutorial is the executable user contract.

The old guard copied selected notebook calls into Python.  That let the copy
stay green when the actual notebook drifted.  This file instead validates and
executes the one shipped notebook itself; product-level physics and data
contracts remain covered by their owning tests.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
TUTORIAL = TUTORIALS / "neutral_atom_tutorial.ipynb"

_REQUIRED_MARKDOWN = (
    "连上装置",
    "构造一个 Camera Measurement",
    "构造 Calibration 与 MOT-field Task",
    "Processor 与完整数据",
    "图形界面",
    "收尾",
)

_REQUIRED_CURRENT_CODE = (
    "Zou_lab_control.api",
    "exp.nodes.calibration",
    "exp.nodes.occupancy",
    "exp.nodes.camera_measurement",
    "exp.nodes.mot_field",
    "exp.nodes.readout_duration_fidelity",
    ".build(",
    "frames_per_cycle",
    "save_frames",
    "model_kind",
    "WorkspacePaths.for_workspace",
    "workspace=WORKSPACE",
    "WORKSPACE.runs_root",
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
    assert "zlc_neutral_atom" not in code


def test_the_checked_in_tutorial_executes_on_the_virtual_installation(
    tmp_path,
    monkeypatch,
) -> None:
    project = (tmp_path / "tutorial-workspace").resolve()
    pulses = project / "pulses"
    pulses.mkdir(parents=True)
    for name in (
        "camera_imaging_address_switch.json",
        "imaging_template.json",
        "probe_template.json",
        "release_recapture.json",
    ):
        shutil.copyfile(ROOT / "pulses" / name, pulses / name)
    monkeypatch.setenv(
        "ZLC_TUTORIAL_PROJECT_ROOT",
        str(project),
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
