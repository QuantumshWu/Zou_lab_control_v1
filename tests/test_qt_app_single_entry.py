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
import os
import pathlib
import subprocess
import sys
import threading

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The one owner: it sets the high-DPI attributes and then constructs the app.
OWNER = REPO / "zlc_frontend" / "qt_widgets" / "fluent.py"

SEARCH_ROOTS = ("zlc_frontend", "zlc_workbench", "zlc_data", "zlc_neutral_atom",
                "Zou_lab_control", "tests")
SHARED_USER_FLOW = REPO / "tests" / "gui_user_flow.py"


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
    yield from sorted(REPO.glob("*.py"))


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


def test_offscreen_owner_rasterizes_the_declared_ui_font() -> None:
    """The fast-path screenshot must contain glyphs, not only widget boxes."""

    source = r'''
from PyQt5 import QtGui, QtWidgets
from zlc_frontend.qt_widgets import FONT, ensure_qt_app

app = ensure_qt_app()
label = QtWidgets.QLabel("ZLC visible text 123")
label.setStyleSheet(
    f'background: #ffffff; color: #111111; font: 12pt "{FONT}";'
)
label.resize(320, 64)
label.show()
app.processEvents()
image = label.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
dark = 0
for y in range(image.height()):
    for x in range(image.width()):
        if QtGui.qGray(image.pixel(x, y)) < 96:
            dark += 1
print(dark)
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 20, (
        "offscreen Qt painted the widget but omitted its text glyphs"
    )


def test_owner_wake_coalesces_queued_requests_and_replays_only_during_dispatch() -> None:
    """A level-triggered worker wake cannot manufacture empty owner turns."""

    source = r'''
from zlc_frontend.qt_widgets import QtOwnerWake, SerialWorkerWindow, ensure_qt_app

app = ensure_qt_app()
from zlc_workbench.data_figure.window import DataFigureWindow
from zlc_neutral_atom.logic_nodes.readout.calibration.ui.report_window import CalibrationReportWindow
from zlc_neutral_atom.logic_nodes.readout.calibration.ui.workbench_window import CalibrationWorkbenchWindow
from zlc_neutral_atom.logic_nodes.readout.occupancy.ui.workbench_window import OccupancyCellWindow

pending = list(SerialWorkerWindow.__subclasses__())
while pending:
    subclass = pending.pop()
    if "_owner_cycle" in subclass.__dict__:
        raise RuntimeError(f"{subclass.__qualname__} overrides the sealed owner cycle")
    pending.extend(subclass.__subclasses__())

queued_calls = []
queued = QtOwnerWake()
queued.bind(lambda: queued_calls.append("turn"))
queued.request_owner_wake()
queued.request_owner_wake()
app.processEvents()

replay_calls = []
replay = QtOwnerWake()
def callback():
    replay_calls.append("turn")
    if len(replay_calls) == 1:
        replay.request_owner_wake()
replay.bind(callback)
replay.request_owner_wake()
for _ in range(4):
    app.processEvents()

class ExtendedWorkerWindow(SerialWorkerWindow):
    def _drain_owner_completions(self):
        pass

extended = ExtendedWorkerWindow()
extended.request_owner_close()
if not extended.wait_owner_closed(1.0):
    raise RuntimeError("subclass completion hook bypassed application close")
print(len(queued_calls), len(replay_calls))
'''
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1 2"


def test_bulk_compute_cannot_starve_latency_sensitive_owner_work() -> None:
    """Two active Fits leave the existing interaction seam runnable."""

    from zlc_workbench.window_runtime import submit_compute

    release = threading.Event()
    started = (threading.Event(), threading.Event())

    def block(index: int) -> int:
        started[index].set()
        if not release.wait(5.0):
            raise TimeoutError("test did not release bulk compute")
        return index

    bulk = tuple(submit_compute(block, index) for index in range(2))
    try:
        assert all(event.wait(2.0) for event in started)
        interactive = submit_compute(
            lambda: "selector-ready",
            latency_sensitive=True,
        )
        assert interactive.result(timeout=2.0) == "selector-ready"
    finally:
        release.set()
        for future in bulk:
            future.result(timeout=2.0)


def _call_name(node: ast.Call) -> str:
    target = node.func
    return target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")


def _assigned_environment_keys(tree: ast.AST) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            targets = ()
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            owner = target.value
            if not (
                isinstance(owner, ast.Attribute)
                and owner.attr == "environ"
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "os"
            ):
                continue
            key = target.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                result.add(key.value)
        if not isinstance(node, ast.Call) or _call_name(node) not in {
            "putenv",
            "setdefault",
        }:
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str):
                result.add(value)
    return result


def test_every_gui_user_flow_uses_the_same_offscreen_fast_path() -> None:
    """C46: app flows own actions only; app/bootstrap/capture never fork."""

    shared_tree = ast.parse(SHARED_USER_FLOW.read_text(encoding="utf-8"))
    shared_calls = {
        _call_name(node) for node in ast.walk(shared_tree) if isinstance(node, ast.Call)
    }
    assert {"platformName", "grab", "save"} <= shared_calls
    shared_literals = {
        node.value
        for node in ast.walk(shared_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "offscreen" in shared_literals

    flows = sorted(
        path
        for path in (REPO / "tests").glob("*_user_flow.py")
        if path != SHARED_USER_FLOW
    )
    assert flows, "no application-specific GUI user flow is registered"
    forbidden_methods = {
        "setCurrentIndex",
        "setCurrentWidget",
        "setFixedSize",
        "setStyleSheet",
        "resize",
    }
    forbidden_display_environment = {
        "QT_AUTO_SCREEN_SCALE_FACTOR",
        "QT_DEVICE_PIXEL_RATIO",
        "QT_SCALE_FACTOR",
    }
    offenders: list[str] = []
    for path in flows:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_shared = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "gui_user_flow":
                imported_shared.update(alias.name for alias in node.names)
        required = {
            "capture_offscreen_window",
            "configure_offscreen_fast_path",
            "require_offscreen_platform",
        }
        if not required <= imported_shared:
            offenders.append(
                f"{path.name}: does not import the shared offscreen capture/platform owner"
            )
        call_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        calls = {_call_name(node) for node in call_nodes}
        if "ensure_qt_app" not in calls:
            offenders.append(f"{path.name}: bypasses the sole QApplication owner")
        if "configure_offscreen_fast_path" not in calls:
            offenders.append(f"{path.name}: does not select offscreen before app creation")
        elif "ensure_qt_app" in calls:
            configure_line = min(
                node.lineno
                for node in call_nodes
                if _call_name(node) == "configure_offscreen_fast_path"
            )
            ensure_line = min(
                node.lineno
                for node in call_nodes
                if _call_name(node) == "ensure_qt_app"
            )
            if configure_line >= ensure_line:
                offenders.append(
                    f"{path.name}: selects offscreen after ensure_qt_app()"
                )
        if "capture_offscreen_window" not in calls:
            offenders.append(f"{path.name}: does not capture the formal outer window")
        if not any(name == "open" or name.startswith("open_") for name in calls):
            offenders.append(f"{path.name}: does not call a formal GUI composition entry")
        mutated = sorted(calls & forbidden_methods)
        if mutated:
            offenders.append(
                f"{path.name}: mutates product geometry/state directly: {mutated}"
            )
        assigned_environment = _assigned_environment_keys(tree)
        if "QT_QPA_PLATFORM" in assigned_environment:
            offenders.append(
                f"{path.name}: duplicates the shared offscreen platform owner"
            )
        forced = sorted(assigned_environment & forbidden_display_environment)
        if forced:
            offenders.append(f"{path.name}: forces display environment: {forced}")
    assert not offenders, "\n".join(offenders)
