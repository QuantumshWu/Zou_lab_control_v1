"""MECHANICAL guard: every shipped ``tutorials/*.ipynb`` that is GENERATED from a
package template (``frontend/content/notebook_templates/*.cells.md`` via
``zf.write_*_tutorial()``) must stay byte-for-byte in sync with its template.

Why this is a test: the tutorials are generated artifacts, but they are also committed
and run by hand.  Editing the ``.ipynb`` directly (instead of the ``.cells.md`` source)
silently diverges the two -- and the next ``write_*_tutorial()`` reverts the hand-edit,
quietly losing fixes.  That is exactly what happened (a racy detection-time scan + a
geometry-mismatched ``write_virtual_run`` lived in the ``.ipynb`` while the template still
shipped the old, broken cells).  pytest was green the whole time because nothing compared
the two.  So pin the single-source-of-truth invariant here: edit the template, regenerate.

If this fails: do NOT hand-edit the notebook.  Edit the matching ``.cells.md`` and run the
writer (e.g. ``python -c "import Zou_lab_control.frontend as zf; zf.write_neutral_atom_tutorial()"``).

``task_console_tutorial.ipynb`` has no template (it is hand-maintained) and is deliberately
not covered here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import Zou_lab_control.frontend as zf
from Zou_lab_control.frontend.content import tutorials as T

REPO_ROOT = Path(__file__).resolve().parents[1]

# (shipped notebook, the cells() source that the writer serialises)
CASES = [
    ("frontend_tutorial.ipynb", T.frontend_tutorial_cells),
    ("neutral_atom_tutorial.ipynb", T.neutral_atom_tutorial_cells),
    ("neutral_atom_hardware_quickstart.ipynb", T.neutral_atom_hardware_tutorial_cells),
    ("neutral_atom_fpga_server.ipynb", T.neutral_atom_fpga_server_cells),
    ("qcmos_live_2d.ipynb", T.neutral_atom_qcmos_live_cells),
]


@pytest.mark.parametrize("name,cells_fn", CASES, ids=[c[0] for c in CASES])
def test_shipped_notebook_matches_its_template(name, cells_fn):
    import nbformat

    path = REPO_ROOT / "tutorials" / name
    assert path.exists(), f"shipped tutorial missing: {path}"
    nb = nbformat.read(path, as_version=4)
    on_disk = [(c.cell_type, c.source.strip()) for c in nb.cells]
    from_template = [
        ("code" if c["kind"] == "code" else "markdown", str(c["source"]).strip())
        for c in cells_fn()
    ]
    assert on_disk == from_template, (
        f"{name} has drifted from its .cells.md template -- do NOT hand-edit the notebook; "
        f"edit the template and regenerate with the matching zf.write_*_tutorial().")


@pytest.mark.parametrize("name,cells_fn", CASES, ids=[c[0] for c in CASES])
def test_no_tutorial_code_cell_fuses_two_init_paths_with_or(name, cells_fn):
    """A GUI-lifecycle sentinel (``mgr.session`` is None until "Init devices") must NEVER be fused with
    a fallback ``na.connect(...)`` via ``or`` in a CODE cell: on any re-eval before Init that ``or``
    would SILENTLY connect a hardcoded config, clobbering a real-hardware session.  The two init paths
    -- GUI-interactive (``exp = mgr.session``) and the headless one-liner (``exp = na.connect(...)``) --
    stay separate.  Prose/comment may say "the session, or connect directly"; executable code may not."""
    for c in cells_fn():
        if c["kind"] == "code":
            assert ".session or " not in str(c["source"]), (
                f"{name}: a code cell fuses the manager session with an `or na.connect(...)` fallback -- "
                f"split the two init paths; put the headless one-liner in prose/comment, never `or`-fused.")


def test_writers_are_idempotent(tmp_path):
    """Each writer round-trips: regenerating a template into a fresh file reproduces the
    same cell sources (no hidden nondeterminism in the generation path)."""
    import nbformat

    writers = [
        (zf.write_frontend_tutorial, "frontend_tutorial.ipynb"),
        (zf.write_neutral_atom_tutorial, "neutral_atom_tutorial.ipynb"),
        (zf.write_neutral_atom_hardware_tutorial, "neutral_atom_hardware_quickstart.ipynb"),
        (zf.write_neutral_atom_fpga_server_tutorial, "neutral_atom_fpga_server.ipynb"),
        (zf.write_neutral_atom_qcmos_live_tutorial, "qcmos_live_2d.ipynb"),
    ]
    for writer, name in writers:
        out = tmp_path / name
        writer(out)
        regenerated = [c.source for c in nbformat.read(out, as_version=4).cells]
        shipped = [c.source for c in nbformat.read(REPO_ROOT / "tutorials" / name, as_version=4).cells]
        assert regenerated == shipped, f"{name}: writer output != shipped notebook"
