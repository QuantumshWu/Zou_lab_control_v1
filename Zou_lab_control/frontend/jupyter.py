"""Notebook generation and execution helpers for Zou lab tutorials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence


from .content.tutorials import (
    BOOTSTRAP_CELL,
    frontend_tutorial_cells,
    neutral_atom_fpga_server_cells,
    neutral_atom_hardware_tutorial_cells,
    neutral_atom_tutorial_cells,
)


@dataclass(frozen=True)
class NotebookBuildResult:
    """Result returned by a notebook writer."""

    path: Path
    cells: int


@dataclass(frozen=True)
class NotebookExecutionResult:
    """Result returned by ``execute_notebook``."""

    source_path: Path
    output_path: Path
    stdout: str


def require_attrs(obj, attrs: Sequence[str], *, name: str = "object") -> None:
    """Fail early when a notebook is using an old package/API."""

    missing = [attr for attr in attrs if not hasattr(obj, attr)]
    if not missing:
        return
    module = sys.modules.get(getattr(obj.__class__, "__module__", "").split(".")[0])
    module_path = getattr(module, "__file__", "<unknown>")
    raise RuntimeError(
        f"{name} is missing required API attributes: {missing}. "
        f"The loaded package is {module_path}. Run the notebook bootstrap cell, "
        "or reinstall with `python -m pip install -e <project-root>` and restart the kernel."
    )


def notebook_setup(*, apply_frontend_style: bool = False) -> None:
    """Validate that the loaded package has the notebook-generation API."""

    module = sys.modules.get("Zou_lab_control.frontend")
    require_attrs(
        module,
        [
            "plot",
            "run",
            "write_frontend_tutorial",
            "write_neutral_atom_fpga_server_tutorial",
            "write_neutral_atom_hardware_tutorial",
            "write_neutral_atom_tutorial",
            "execute_notebook",
        ],
        name="Zou_lab_control.frontend",
    )
    if apply_frontend_style and module is not None and hasattr(module, "apply_style"):
        module.apply_style()


def write_notebook(path: str | Path, cells: Iterable[dict], *, title: str | None = None) -> NotebookBuildResult:
    """Write a UTF-8 notebook from simple cell dictionaries."""

    import nbformat as nbf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    if title:
        notebook.metadata["title"] = str(title)
    nb_cells = []
    for cell in cells:
        kind = cell.get("kind", "markdown")
        source = str(cell.get("source", "")).strip()
        if kind == "code":
            nb_cells.append(nbf.v4.new_code_cell(source))
        elif kind == "markdown":
            nb_cells.append(nbf.v4.new_markdown_cell(source))
        else:
            raise ValueError(f"unknown notebook cell kind {kind!r}.")
    notebook.cells = nb_cells
    nbf.write(notebook, path)
    return NotebookBuildResult(path=path.resolve(), cells=len(nb_cells))


def execute_notebook(path: str | Path, *, output_dir: str | Path | None = None, timeout: int = 180) -> NotebookExecutionResult:
    """Execute a notebook with the current Python environment."""

    path = Path(path)
    output_dir = Path(output_dir) if output_dir is not None else path.parent / "_executed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{path.stem}.executed.ipynb"
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(path),
        "--output",
        output_name,
        "--output-dir",
        str(output_dir),
        f"--ExecutePreprocessor.timeout={int(timeout)}",
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return NotebookExecutionResult(source_path=path.resolve(), output_path=(output_dir / output_name).resolve(), stdout=result.stdout)


def write_frontend_tutorial(path: str | Path = "tutorials/frontend_tutorial.ipynb") -> NotebookBuildResult:
    """Generate the frontend tutorial notebook from package-owned cells."""

    return write_notebook(path, frontend_tutorial_cells(), title="Zou_lab_control.frontend tutorial")


def write_neutral_atom_tutorial(path: str | Path = "tutorials/neutral_atom_tutorial.ipynb") -> NotebookBuildResult:
    """Generate the neutral-atom tutorial notebook from package-owned cells."""

    return write_notebook(path, neutral_atom_tutorial_cells(), title="Zou_lab_control.neutral_atom tutorial")


def write_neutral_atom_hardware_tutorial(path: str | Path = "tutorials/neutral_atom_hardware_quickstart.ipynb") -> NotebookBuildResult:
    """Generate the real-hardware quickstart notebook from package-owned cells."""

    return write_notebook(path, neutral_atom_hardware_tutorial_cells(), title="Zou_lab_control.neutral_atom hardware quickstart")


def write_neutral_atom_fpga_server_tutorial(path: str | Path = "tutorials/neutral_atom_fpga_server.ipynb") -> NotebookBuildResult:
    """Generate the FPGA/Vivado-computer sequencer server notebook."""

    return write_notebook(path, neutral_atom_fpga_server_cells(), title="Zou_lab_control.neutral_atom FPGA sequencer server")




__all__ = [
    "BOOTSTRAP_CELL",
    "NotebookBuildResult",
    "NotebookExecutionResult",
    "execute_notebook",
    "notebook_setup",
    "require_attrs",
    "write_frontend_tutorial",
    "write_neutral_atom_fpga_server_tutorial",
    "write_neutral_atom_hardware_tutorial",
    "write_neutral_atom_tutorial",
    "write_notebook",
]
