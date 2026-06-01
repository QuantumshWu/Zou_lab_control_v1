"""Load notebook tutorial content from package data files."""

from __future__ import annotations

from pathlib import Path
import re

BOOTSTRAP_CELL = r"""
import os
import sys

PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONPATH"] = PROJECT_ROOT

import Zou_lab_control.frontend as zf

zf.notebook_setup()
""".strip()

_CELL_MARKER = re.compile(r"^<!-- cell:(markdown|code) -->\s*$")
_CONTENT_DIR = Path(__file__).resolve().parent / "notebook_templates"


def _load_cells(name: str) -> list[dict]:
    path = _CONTENT_DIR / f"{name}.cells.md"
    text = path.read_text(encoding="utf-8")
    cells: list[dict] = []
    current_kind: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        match = _CELL_MARKER.match(line)
        if match:
            if current_kind is not None:
                cells.append(_cell(current_kind, "\n".join(current_lines).strip()))
            current_kind = match.group(1)
            current_lines = []
        else:
            current_lines.append(line)
    if current_kind is not None:
        cells.append(_cell(current_kind, "\n".join(current_lines).strip()))
    if not cells:
        raise ValueError(f"tutorial content file has no cells: {path}")
    return cells


def _cell(kind: str, source: str) -> dict:
    return {"kind": kind, "source": source.replace("{{BOOTSTRAP_CELL}}", BOOTSTRAP_CELL)}


def frontend_tutorial_cells() -> list[dict]:
    return _load_cells("frontend_tutorial")


def neutral_atom_tutorial_cells() -> list[dict]:
    return _load_cells("neutral_atom_tutorial")


def neutral_atom_hardware_tutorial_cells() -> list[dict]:
    return _load_cells("neutral_atom_hardware_quickstart")


def neutral_atom_fpga_server_cells() -> list[dict]:
    return _load_cells("neutral_atom_fpga_server")


__all__ = [
    "BOOTSTRAP_CELL",
    "frontend_tutorial_cells",
    "neutral_atom_fpga_server_cells",
    "neutral_atom_hardware_tutorial_cells",
    "neutral_atom_tutorial_cells",
]
