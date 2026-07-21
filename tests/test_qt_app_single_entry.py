"""Nobody may build the QApplication except ``ensure_qt_app``.

This is mechanised because it is invisible when you get it wrong.  The high-DPI
attributes (``AA_EnableHighDpiScaling`` / ``AA_UseHighDpiPixmaps``) can only be
set BEFORE the QApplication object exists; ``ensure_qt_app`` sets them and then
constructs it.  Any code that constructs one first silently wins the race, the
attributes become no-ops, and every window opened afterwards is scaled
differently from the one a launcher opens.

Nothing fails when that happens.  The windows just quietly disagree with the
real GUI, which is exactly how a screenshot check can "pass" while showing an
interface the operator never sees — so the rule has to be a test rather than a
note somebody remembers.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The one owner: it sets the high-DPI attributes and then constructs the app.
OWNER = REPO / "zlc_frontend" / "qt_widgets" / "fluent.py"

SEARCH_ROOTS = ("zlc_frontend", "zlc_workbench", "zlc_data", "zlc_neutral_atom",
                "Zou_lab_control", "tests")


def _constructs_qapplication(tree: ast.AST) -> bool:
    """True when this module CALLS QApplication(...) rather than asking for it."""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name == "QApplication":
            return True
    return False


def _python_files():
    for root in SEARCH_ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path
    for name in ("task_console.py", "pulse_gui.py", "figure_viewer.py"):
        candidate = REPO / name
        if candidate.is_file():
            yield candidate


def test_only_ensure_qt_app_constructs_the_application() -> None:
    offenders = []
    for path in _python_files():
        if path == OWNER:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        if _constructs_qapplication(tree):
            offenders.append(str(path.relative_to(REPO)).replace("\\", "/"))

    assert not offenders, (
        "these construct a QApplication instead of calling ensure_qt_app(); the high-DPI "
        "attributes are set before construction and are lost to whoever builds it first:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_the_owner_sets_the_high_dpi_attributes_before_constructing() -> None:
    """The rule above is only worth enforcing while the owner still earns it."""

    source = OWNER.read_text(encoding="utf-8")
    build = source.index("QtWidgets.QApplication(sys.argv)")
    for attribute in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        assert attribute in source, f"{attribute} is no longer set by the owner"
        assert source.index(attribute) < build, (
            f"{attribute} is set AFTER the QApplication is built, where it does nothing"
        )
