from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
QT_PACKAGE = ROOT / "zlc_frontend" / "qt_widgets"


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


def _render_widget(widget):
    from PyQt5 import QtCore, QtGui

    image = QtGui.QImage(widget.size(), QtGui.QImage.Format_RGBA8888)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    try:
        widget.render(painter)
    finally:
        painter.end()
    return image


def test_frontend_root_is_headless_and_qt_is_confined_to_qt_widgets() -> None:
    result = _run_fresh(
        "import sys\n"
        "import zlc_frontend\n"
        "assert not any(k == 'PyQt5' or k.startswith('PyQt5.') for k in sys.modules)\n"
        "assert not any(k == 'matplotlib' or k.startswith('matplotlib.') for k in sys.modules)\n"
        "import zlc_frontend.qt_widgets as qt\n"
        "assert not any(k == 'PyQt5' or k.startswith('PyQt5.') for k in sys.modules)\n"
        "assert not any(k == 'matplotlib' or k.startswith('matplotlib.') for k in sys.modules)\n"
        "assert qt.FluentSpinBox.__module__ == 'zlc_frontend.qt_widgets.fluent'\n"
        "assert 'PyQt5' in sys.modules\n"
        "assert not any(k == 'matplotlib' or k.startswith('matplotlib.') for k in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr

    frontend_root = ROOT / "zlc_frontend"
    qt_leaks = []
    for path in frontend_root.rglob("*.py"):
        if _import_roots(path).intersection({"PyQt5", "qframelesswindow"}):
            if QT_PACKAGE not in path.parents:
                qt_leaks.append(str(path.relative_to(ROOT)))
    assert qt_leaks == []

    forbidden_domain_roots = {
        "Zou_lab_control",
        "zlc_neutral_atom",
        "zlc_pulse",
        "zlc_workbench",
    }
    for path in QT_PACKAGE.rglob("*.py"):
        assert _import_roots(path).isdisjoint(forbidden_domain_roots), path

    retired = (
        "board.py",
        "frozen_raster.py",
        "panel_host.py",
        "raster_surface.py",
        "figure_surface_host.py",
        "figure_surface_lane.py",
    )
    assert all(not (QT_PACKAGE / name).exists() for name in retired)
    assert not (ROOT / "zlc_frontend" / "render_style.py").exists()
    assert not (ROOT / "zlc_frontend" / "matplotlib_render.py").exists()


def test_qapplication_creation_has_one_owner_and_reuses_zlc_plot() -> None:
    result = _run_fresh(
        "from PyQt5 import QtWidgets\n"
        "from zlc_frontend.qt_widgets import ensure_qt_app\n"
        "from zlc_plot import ensure_qt5_application\n"
        "first = ensure_qt_app()\n"
        "assert ensure_qt_app() is first\n"
        "assert ensure_qt5_application(()) is first\n"
        "assert QtWidgets.QApplication.instance() is first\n"
    )
    assert result.returncode == 0, result.stderr

    worker = _run_fresh(
        "import threading\n"
        "from PyQt5 import QtWidgets\n"
        "from zlc_frontend.qt_widgets import ensure_qt_app\n"
        "out = []\n"
        "def run():\n"
        "    try:\n"
        "        ensure_qt_app()\n"
        "    except BaseException as error:\n"
        "        out.append(type(error).__name__)\n"
        "thread = threading.Thread(target=run)\n"
        "thread.start(); thread.join()\n"
        "assert out == ['RuntimeError'], out\n"
        "assert QtWidgets.QApplication.instance() is None\n"
    )
    assert worker.returncode == 0, worker.stderr


def test_qt_public_facade_covers_every_production_consumer() -> None:
    import zlc_frontend.qt_widgets as qt

    assert all(name and not name.startswith("_") for name in qt.__all__)
    retired = {
        "QtRasterBoard",
        "SinglePanelAggRenderer",
        "RasterPixelRatioObserver",
        "FrozenRasterView",
        "ImageViewportTransform",
    }
    assert retired.isdisjoint(qt.__all__)

    missing: list[tuple[str, int, str]] = []
    deep_imports: list[tuple[str, int, str]] = []
    for path in _production_python_files():
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module == "zlc_frontend.qt_widgets":
                for alias in node.names:
                    if alias.name not in qt.__all__:
                        missing.append(
                            (str(path.relative_to(ROOT)), node.lineno, alias.name)
                        )
            elif (
                node.module.startswith("zlc_frontend.qt_widgets.")
                and QT_PACKAGE not in path.parents
            ):
                deep_imports.append(
                    (str(path.relative_to(ROOT)), node.lineno, node.module)
                )
    assert missing == []
    assert deep_imports == []


def test_fluent_controls_keep_typed_widget_families() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
    from zlc_frontend.qt_widgets import (
        FluentComboBox,
        FluentDoubleSpinBox,
        FluentParameterForm,
        FluentSpinBox,
        FluentSwitch,
        ensure_qt_app,
    )

    ensure_qt_app()
    form = FluentParameterForm(FormSpec((
        FormFieldProps("count", "int", "Count", default=2, minimum=0),
        FormFieldProps("gain", "float", "Gain", default=0.5),
        FormFieldProps(
            "mode",
            "choice",
            "Mode",
            default="a",
            choices=(FormChoice("A", "a"), FormChoice("B", "b")),
        ),
        FormFieldProps("enabled", "bool", "Enabled", default=True),
    )))

    assert isinstance(form.widget_for("count"), FluentSpinBox)
    assert isinstance(form.widget_for("gain"), FluentDoubleSpinBox)
    assert isinstance(form.widget_for("mode"), FluentComboBox)
    assert isinstance(form.widget_for("enabled"), FluentSwitch)


def test_fluent_double_spinbox_preserves_authoritative_float_values() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_frontend.qt_widgets import FluentDoubleSpinBox, ensure_qt_app

    ensure_qt_app()
    value = -1234567890.1234567
    field = FluentDoubleSpinBox(length=18, allow_minus=True)
    field.setDecimals(9)
    field.setRange(-1e15, 1e15)
    field.setValue(value)
    assert field.value() == value


def test_integer_and_double_spinboxes_share_visible_split_arrow_buttons() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore

    import zlc_frontend.qt_widgets.fluent as fluent_module
    from zlc_frontend.qt_widgets import (
        FluentDoubleSpinBox,
        FluentSpinBox,
        ensure_qt_app,
    )
    from zlc_frontend.qt_widgets.style import COMBO_WIDTH

    application = ensure_qt_app()
    for field_type in (FluentSpinBox, FluentDoubleSpinBox):
        field = field_type()
        field.resize(100, 30)
        field.show()
        application.processEvents()
        image = _render_widget(field)
        button_left = field.width() - fluent_module.scaled_px(COMBO_WIDTH)
        split_y = field.height() // 2
        split = [
            image.pixelColor(x, split_y)
            for x in range(button_left, field.width() - 1)
        ]
        below = [
            image.pixelColor(x, split_y + 1)
            for x in range(button_left, field.width() - 1)
        ]
        brightness = lambda color: color.red() + color.green() + color.blue()
        assert sum(map(brightness, split)) > sum(map(brightness, below)) + 500
        field.close()
    application.processEvents(QtCore.QEventLoop.AllEvents, 20)


def test_zlc_plot_is_the_only_plot_font_and_dpr_owner() -> None:
    from zlc_frontend.qt_widgets.style import FONT as QT_FONT
    from zlc_plot import DEFAULTS

    plot_font = (
        ROOT
        / "zlc_plot"
        / "assets"
        / "helvetica-light-587ebe5a59211.ttf"
    )
    assert plot_font.is_file()
    assert not (ROOT / "zlc_frontend" / "assets").exists()
    assert QT_FONT == "Segoe UI"
    assert DEFAULTS.style.fonts.family == "Helvetica Light"

    frontend_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "zlc_frontend").rglob("*.py")
    )
    assert "devicePixelRatioF" not in frontend_source
    assert "set_device_pixel_ratio" not in frontend_source
    assert "RasterPixelRatioObserver" not in frontend_source

    plot_source = (ROOT / "zlc_plot" / "backends.py").read_text(encoding="utf-8")
    assert "_RasterPixelRatioObserver" in plot_source
    assert "set_device_pixel_ratio" in plot_source

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]
    assert "assets/*.ttf" in package_data["zlc_plot"]
    assert "zlc_frontend" not in package_data
