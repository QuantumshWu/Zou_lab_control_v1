from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import tomllib


ROOT = Path(__file__).resolve().parents[1]
QT_PACKAGE = ROOT / "zlc_frontend" / "qt_widgets"
WORKBENCH_MODULES = tuple(
    path
    for path in sorted((ROOT / "Zou_lab_control" / "workbench").glob("_*.py"))
    if path.name != "__init__.py"
)
CURRENT_ROOT_QT_LAUNCHERS = tuple(
    ROOT / name for name in ("figure_viewer.py", "pulse_gui.py", "task_console.py")
)
RAW_COMMON_CONTROLS = {
    "QCheckBox",
    "QComboBox",
    "QDoubleSpinBox",
    "QFormLayout",
    "QGroupBox",
    "QLabel",
    "QLineEdit",
    "QPushButton",
    "QScrollArea",
    "QSpinBox",
    "QTabWidget",
}
RAW_NATIVE_DIALOG_PREFIXES = {
    "PyQt5.QtWidgets.QInputDialog",
    "PyQt5.QtWidgets.QMessageBox",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _production_python_files():
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"tests", ".git", "build", "dist"}:
            continue
        yield path


def _import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _imports_qt_surface(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "zlc_frontend.qt_widgets" or node.module.startswith(
                ("zlc_frontend.qt_widgets.", "PyQt5", "qframelesswindow")
            ):
                return True
        elif isinstance(node, ast.Import) and any(
            alias.name.startswith(("zlc_frontend.qt_widgets", "PyQt5", "qframelesswindow"))
            for alias in node.names
        ):
            return True
    return False


def _attribute_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute_path(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _run_fresh(source: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_qt_widgets_is_the_only_current_qt_owner() -> None:
    assert not (ROOT / "Zou_lab_control" / "frontend" / "qt_fluent.py").exists()
    assert not (ROOT / "Zou_lab_control" / "frontend" / "style.py").exists()
    assert not (ROOT / "zlc_frontend" / "qt_board.py").exists()
    assert not (ROOT / "zlc_frontend" / "_matplotlib_render.py").exists()
    assert (QT_PACKAGE / "fluent.py").is_file()
    assert (QT_PACKAGE / "board.py").is_file()
    assert (ROOT / "zlc_frontend" / "render_style.py").is_file()
    assert (
        ROOT
        / "zlc_frontend"
        / "assets"
        / "helvetica-light-587ebe5a59211.ttf"
    ).is_file()

    stale_imports: list[tuple[str, int]] = []
    deep_imports: list[tuple[str, int, str]] = []
    legacy_frontend = ROOT / "Zou_lab_control" / "frontend"
    for path in _production_python_files():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module in {
                    "Zou_lab_control.frontend.qt_fluent",
                    "Zou_lab_control.frontend.style",
                    "zlc_frontend._matplotlib_render",
                    "zlc_frontend.qt_board",
                    "qt_fluent",
                } or node.module.endswith(".qt_fluent"):
                    stale_imports.append((str(path.relative_to(ROOT)), node.lineno))
                if node.module == "Zou_lab_control.frontend" and any(
                    alias.name in {"qt_fluent", "style"} for alias in node.names
                ):
                    stale_imports.append((str(path.relative_to(ROOT)), node.lineno))
                if (
                    legacy_frontend in path.parents
                    and node.level
                    and node.module in {"qt_fluent", "style"}
                ):
                    stale_imports.append((str(path.relative_to(ROOT)), node.lineno))
                if (
                    QT_PACKAGE not in path.parents
                    and node.module.startswith("zlc_frontend.qt_widgets.")
                ):
                    deep_imports.append(
                        (str(path.relative_to(ROOT)), node.lineno, node.module)
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {
                        "Zou_lab_control.frontend.qt_fluent",
                        "Zou_lab_control.frontend.style",
                        "zlc_frontend._matplotlib_render",
                        "zlc_frontend.qt_board",
                    }:
                        stale_imports.append(
                            (str(path.relative_to(ROOT)), node.lineno)
                        )
                    if (
                        QT_PACKAGE not in path.parents
                        and alias.name.startswith("zlc_frontend.qt_widgets.")
                    ):
                        deep_imports.append(
                            (str(path.relative_to(ROOT)), node.lineno, alias.name)
                        )
    assert stale_imports == []
    assert deep_imports == []

    qt_leaks: list[str] = []
    frontend_root = ROOT / "zlc_frontend"
    for path in frontend_root.rglob("*.py"):
        roots = _import_roots(path)
        if roots.intersection({"PyQt5", "qframelesswindow"}):
            if QT_PACKAGE not in path.parents:
                qt_leaks.append(str(path.relative_to(ROOT)))
    assert qt_leaks == []


def test_optional_style_layers_have_no_reverse_dependency() -> None:
    forbidden_qt_roots = {
        "Zou_lab_control",
        "matplotlib",
        "zlc_neutral_atom",
        "zlc_pulse",
        "zlc_workbench",
    }
    for path in QT_PACKAGE.rglob("*.py"):
        assert _import_roots(path).isdisjoint(forbidden_qt_roots), path

    for relative in (
        "zlc_frontend/render_style.py",
        "zlc_frontend/matplotlib_render.py",
    ):
        roots = _import_roots(ROOT / relative)
        assert "PyQt5" not in roots
        assert "qframelesswindow" not in roots
        assert "IPython" not in roots


def test_render_style_is_scoped_and_restores_process_rcparams() -> None:
    import matplotlib

    from zlc_frontend.render_style import (
        DEFAULT_STYLE,
        PALETTE,
        RENDER_TEXT,
        render_style_context,
    )

    assert isinstance(DEFAULT_STYLE["font.sans-serif"], tuple)
    assert isinstance(DEFAULT_STYLE["figure.figsize"], tuple)
    assert isinstance(PALETTE["series"], tuple)
    assert not hasattr(PALETTE["series"], "append")

    original = matplotlib.rcParams["axes.edgecolor"]
    matplotlib.rcParams["axes.edgecolor"] = "magenta"
    try:
        with render_style_context():
            assert matplotlib.rcParams["axes.edgecolor"] == RENDER_TEXT
        assert matplotlib.rcParams["axes.edgecolor"] == "magenta"
    finally:
        matplotlib.rcParams["axes.edgecolor"] = original


def test_product_render_style_contexts_are_serialized_between_threads() -> None:
    import matplotlib

    from zlc_frontend.render_style import RENDER_TEXT, render_style_context

    first_entered = threading.Event()
    second_attempting = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    observed: list[str] = []

    def first() -> None:
        with render_style_context():
            observed.append(matplotlib.rcParams["axes.edgecolor"])
            first_entered.set()
            assert release_first.wait(2.0)

    def second() -> None:
        assert first_entered.wait(2.0)
        second_attempting.set()
        with render_style_context():
            observed.append(matplotlib.rcParams["axes.edgecolor"])
            second_entered.set()

    original = matplotlib.rcParams["axes.edgecolor"]
    matplotlib.rcParams["axes.edgecolor"] = "magenta"
    one = threading.Thread(target=first)
    two = threading.Thread(target=second)
    try:
        one.start()
        two.start()
        assert second_attempting.wait(2.0)
        assert not second_entered.wait(0.1)
        release_first.set()
        one.join(2.0)
        two.join(2.0)
        assert not one.is_alive() and not two.is_alive()
        assert second_entered.is_set()
        assert observed == [RENDER_TEXT, RENDER_TEXT]
        assert matplotlib.rcParams["axes.edgecolor"] == "magenta"
    finally:
        release_first.set()
        one.join(2.0)
        two.join(2.0)
        matplotlib.rcParams["axes.edgecolor"] = original


def test_frontend_and_workbench_roots_remain_headless() -> None:
    result = _run_fresh(
        "import sys\n"
        "import zlc_frontend, zlc_frontend.figure, zlc_frontend.render\n"
        "import zlc_workbench, Zou_lab_control.notebook, Zou_lab_control.workbench\n"
        "assert not any(k == 'PyQt5' or k.startswith('PyQt5.') for k in sys.modules)\n"
        "assert 'qframelesswindow' not in sys.modules\n"
        "assert not any(k == 'matplotlib' or k.startswith('matplotlib.') for k in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr

    qt_result = _run_fresh(
        "import sys\n"
        "import zlc_frontend.qt_widgets as qt\n"
        "assert 'PyQt5' in sys.modules\n"
        "assert not any(k == 'matplotlib' or k.startswith('matplotlib.') for k in sys.modules)\n"
        "assert not any(k == 'IPython' or k.startswith('IPython.') for k in sys.modules)\n"
        "assert qt.FluentSettingRow.__module__ == 'zlc_frontend.qt_widgets.fluent'\n"
        "assert qt.QtImageBoard.__module__ == 'zlc_frontend.qt_widgets.board'\n"
    )
    assert qt_result.returncode == 0, qt_result.stderr

    notebook_result = _run_fresh(
        "import sys\n"
        "import zlc_frontend.notebook_integration\n"
        "assert not any(k == 'matplotlib' or k.startswith('matplotlib.') for k in sys.modules)\n"
        "assert not any(k == 'IPython' or k.startswith('IPython.') for k in sys.modules)\n"
        "assert not any(k == 'PyQt5' or k.startswith('PyQt5.') for k in sys.modules)\n"
    )
    assert notebook_result.returncode == 0, notebook_result.stderr


def test_first_qapplication_creation_is_rejected_from_a_worker_thread() -> None:
    result = _run_fresh(
        "import threading\n"
        "from PyQt5 import QtWidgets\n"
        "from zlc_frontend.qt_widgets import ensure_qt_app\n"
        "out = []\n"
        "def worker():\n"
        "    try:\n"
        "        ensure_qt_app()\n"
        "    except BaseException as error:\n"
        "        out.append((type(error).__name__, str(error)))\n"
        "thread = threading.Thread(target=worker)\n"
        "thread.start(); thread.join()\n"
        "assert out and out[0][0] == 'RuntimeError', out\n"
        "assert QtWidgets.QApplication.instance() is None\n"
    )
    assert result.returncode == 0, result.stderr


def test_qt_and_render_extras_are_independent_and_workbench_is_their_union() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert set(extras["qt"]) == {"PyQt5", "PyQt5-Frameless-Window"}
    assert set(extras["render"]) == {"matplotlib"}
    assert set(extras["workbench"]) == (
        set(extras["analysis"]) | set(extras["render"]) | set(extras["qt"])
    )
    package_data = project["tool"]["setuptools"]["package-data"]
    assert "assets/*.ttf" in package_data["zlc_frontend"]
    assert "assets/*.ttf" not in package_data["Zou_lab_control.frontend"]


def test_w1_w2_w3_take_existing_common_controls_from_qt_widgets() -> None:
    lifecycle_helpers = {
        "center_window_on_primary_screen",
        "ensure_qt_app",
        "release_window",
        "retain_window",
        "screen_fit_window_size",
        "set_fluent_scale",
    }
    qt_shells = tuple(
        path for path in WORKBENCH_MODULES if _imports_qt_surface(_tree(path))
    )
    assert qt_shells
    for path in qt_shells:
        tree = _tree(path)
        imported = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "zlc_frontend.qt_widgets"
            for alias in node.names
        }
        launchers = tuple(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith(("open_", "show_", "launch_"))
            for node in tree.body
        )
        if any(launchers) and not lifecycle_helpers.issubset(imported):
            delegated_calls = {
                call.func.id
                for node, is_launcher in zip(tree.body, launchers)
                if is_launcher
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
            delegated_owners: set[str] = set()
            for node in tree.body:
                if not (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 1
                    and node.module
                ):
                    continue
                owner_path = path.parent / f"{node.module}.py"
                if not owner_path.is_file():
                    continue
                owner_tree = _tree(owner_path)
                owner_imported = {
                    alias.name
                    for owner_node in owner_tree.body
                    if isinstance(owner_node, ast.ImportFrom)
                    and owner_node.module == "zlc_frontend.qt_widgets"
                    for alias in owner_node.names
                }
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    if (
                        local_name in delegated_calls
                        and lifecycle_helpers.issubset(owner_imported)
                    ):
                        delegated_owners.add(local_name)
            assert delegated_owners, path

        import_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "PyQt5":
                for alias in node.names:
                    if alias.name == "QtWidgets":
                        import_aliases[alias.asname or alias.name] = (
                            "PyQt5.QtWidgets"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "PyQt5.QtWidgets":
                        local = alias.asname or "PyQt5"
                        import_aliases[local] = (
                            alias.name if alias.asname else "PyQt5"
                        )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module == "PyQt5.QtWidgets"
            ):
                for alias in node.names:
                    import_aliases[alias.asname or alias.name] = (
                        f"PyQt5.QtWidgets.{alias.name}"
                    )

        raw_calls: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = _attribute_path(node.func)
            if called is None:
                continue
            first, *rest = called.split(".")
            resolved = ".".join((import_aliases.get(first, first), *rest))
            control = resolved.rsplit(".", 1)[-1]
            if (
                resolved.startswith("PyQt5.QtWidgets.")
                and control in RAW_COMMON_CONTROLS
            ):
                raw_calls.append((node.lineno, control))
            if any(
                resolved == prefix or resolved.startswith(f"{prefix}.")
                for prefix in RAW_NATIVE_DIALOG_PREFIXES
            ):
                raw_calls.append((node.lineno, resolved))
        assert raw_calls == [], (path, raw_calls)

        source = path.read_text(encoding="utf-8")
        assert re.search(r"#[0-9A-Fa-f]{3,8}", source) is None, path

    for path in CURRENT_ROOT_QT_LAUNCHERS:
        tree = _tree(path)
        ensure_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "zlc_frontend.qt_widgets"
            and any(alias.name == "ensure_qt_app" for alias in node.names)
        ]
        constructors = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (_attribute_path(node.func) or "").split(".")[-1]
            == "QApplication"
        ]
        assert ensure_imports, path
        assert constructors == [], (path, constructors)


def test_shared_window_retention_has_an_explicit_committed_close_release() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.qt_widgets import (
        ensure_qt_app,
        release_window,
        retain_window,
    )

    from PyQt5 import QtWidgets

    application = ensure_qt_app()
    window = QtWidgets.QWidget()
    retain_window(window)
    retain_window(window)
    registry = application._zlc_retained_windows
    assert registry.count(window) == 1

    release_window(window)
    release_window(window)
    assert window not in registry


def test_fluent_double_spinbox_can_preserve_authoritative_float_values() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.qt_widgets import FluentDoubleSpinBox, ensure_qt_app

    ensure_qt_app()
    value = -1234567890.1234567
    field = FluentDoubleSpinBox(
        length=18,
        allow_minus=True,
        quantize_to_display=False,
    )
    field.setDecimals(9)
    field.setRange(-1e15, 1e15)
    field.setValue(value)
    assert field.value() == value


def test_qt_public_facade_covers_every_production_consumer(monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import zlc_frontend.qt_widgets as qt
    import zlc_frontend.qt_widgets.fluent as fluent_module

    from PyQt5 import QtWidgets

    qt.ensure_qt_app()

    assert all(not name.startswith("_") for name in qt.__all__)
    assert {"board", "fluent", "style"}.isdisjoint(qt.__all__)
    monkeypatch.setattr(
        fluent_module._FluentMessageDialog,
        "exec_",
        lambda _self: QtWidgets.QDialog.Accepted,
    )
    assert qt.fluent_confirm(None, "title", "question")
    monkeypatch.setattr(
        fluent_module._FluentMessageDialog,
        "exec_",
        lambda _self: QtWidgets.QDialog.Rejected,
    )
    assert not qt.fluent_confirm(None, "title", "question")
    monkeypatch.setattr(
        fluent_module.FluentInputDialog,
        "getText",
        lambda self: (self._edit.text(), True),
    )
    assert qt.fluent_text_prompt(
        None,
        "title",
        "prompt",
        text="stable_parameter_id",
    ) == ("stable_parameter_id", True)
    missing: list[tuple[str, int, str]] = []
    for path in _production_python_files():
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "zlc_frontend.qt_widgets":
                continue
            for alias in node.names:
                if alias.name not in qt.__all__:
                    missing.append(
                        (str(path.relative_to(ROOT)), node.lineno, alias.name)
                    )
    assert missing == []


def test_current_user_and_maintainer_docs_name_only_the_new_ui_owners() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "Zou_lab_control" / "frontend" / "README.md",
        ROOT / "docs" / "MAINTAINER_NOTES.md",
        ROOT / "docs" / "task_console_design" / "task_console_design_zh.texbody",
    ]
    paths.extend(
        (ROOT / "Zou_lab_control" / "frontend" / "content" / "manual_templates").glob(
            "*.texbody"
        )
    )
    paths.extend(
        (
            ROOT
            / "Zou_lab_control"
            / "neutral_atom"
            / "content"
            / "manual_templates"
        ).glob("*.texbody")
    )
    forbidden = (
        "qt_fluent",
        r"qt\_fluent",
        "Zou_lab_control.frontend.style",
        "frontend/style.py",
    )
    stale = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                stale.append((str(path.relative_to(ROOT)), marker))
    assert stale == []
