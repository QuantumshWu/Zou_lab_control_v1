"""Print a window's structure as JSON, so two source trees can be compared window-for-window.

The decomposition ahead moves widgets between files and packages for many rounds.  A pure move
must not change what the operator sees, and the cheapest honest proof of that is to BUILD the
window on both sides and diff its structure.  Screenshots would be stronger still but are slow
and noisy; a structural fingerprint is fast enough to run every round, which is what makes it
actually get run.

Two properties make the comparison meaningful:

* it goes through the SAME entry path the launcher uses (``Experiment.task_console()``, not some
  private constructor), so the thing measured is the thing the user double-clicks;
* it records class ``__name__`` only, never the module a class came from.  Moving ``PanelCard``
  from the legacy shell to ``zlc_frontend.qt_widgets`` must read as identical -- that is exactly
  the change being validated.  A fingerprint that included module paths would flag every move as
  a difference and prove nothing.

Comparing two trees needs two PROCESSES: both define a package named ``Zou_lab_control`` and one
interpreter cannot hold both.  Hence ``--repo``: the caller runs this file once per tree and
diffs the JSON.

The target tree is read, never written.  ``ZLC_main`` is the behaviour authority and a probe
that wrote into it would corrupt the very baseline it exists to consult.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

#: Windows this probe knows how to build, named after the root launcher a user double-clicks.
WINDOWS = ("task_console", "pulse_gui", "figure_viewer")

#: How the probe decides WHICH composition a tree's launcher opens.
#:
#: It does not hardcode the answer -- it READS ``<repo>/<window>.py`` and looks for these
#: markers.  That matters because the two trees genuinely disagree: measured 2026-07-20,
#: ``ZLC_main/task_console.py`` calls ``show_task_console(hub=...)`` (the legacy Monitor+Logic
#: console) while this tree's calls ``experiment.task_console(...)`` (a 19-widget rebuilt
#: window).  A probe that assumed either one would measure the wrong thing on the other tree
#: and report a false match.  Deriving it also means the probe follows along on its own the
#: round a launcher is re-pointed, instead of silently comparing old to new.
ENTRY_MARKERS = {
    "task_console": (("show_task_console(", "legacy"),
                     ("experiment.task_console(", "facade"),
                     ("open_task_console(", "workbench_app")),
    "pulse_gui": (("show_pulse_gui(", "legacy"),
                  ("open_pulse_workbench(", "workbench"),
                  ("open_pulse_editor(", "editor_app")),
    "figure_viewer": (("show_figure_viewer(", "legacy"),),
}

#: Run the child in a headless, deterministic environment.  ``MPLBACKEND=Agg`` matters even
#: though this file imports no Matplotlib: the window it builds will, and an interactive
#: backend would try to open a second GUI toolkit underneath Qt.
CHILD_ENV = {
    "QT_QPA_PLATFORM": "offscreen",
    "MPLBACKEND": "Agg",
    "QT_LOGGING_RULES": "qt.qpa.fonts=false",
    "QT_SCALE_FACTOR": "1",
}

_CHILD = r'''
import json, sys, tempfile, pathlib

def describe(widget, depth=0, limit=6):
    """Class name, text where a human would read one, geometry -- and children.

    Class ``__name__`` WITHOUT its module: a widget that moved package is the same widget.
    """
    from PyQt5 import QtWidgets
    node = {"class": type(widget).__name__}
    text = getattr(widget, "text", None)
    if callable(text):
        try:
            value = text()
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            node["text"] = value
    if isinstance(widget, QtWidgets.QTabWidget):
        node["tabs"] = [widget.tabText(i) for i in range(widget.count())]
    size = widget.size()
    node["size"] = [size.width(), size.height()]
    if depth < limit:
        kids = [describe(c, depth + 1, limit) for c in widget.children()
                if isinstance(c, QtWidgets.QWidget)]
        if kids:
            node["children"] = kids
    return node

def qt_app():
    """``ensure_qt_app`` from wherever this tree keeps it (it moved packages mid-migration)."""
    for module in ("zlc_frontend.qt_widgets", "Zou_lab_control.frontend.qt_fluent"):
        try:
            return __import__(module, fromlist=["ensure_qt_app"]).ensure_qt_app()
        except ImportError:
            continue
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

def build(name, kind, connect_devices):
    """Construct exactly what this tree's launcher constructs, minus ``app.exec_()``."""
    app = qt_app()
    room = pathlib.Path(tempfile.mkdtemp(prefix="zlc-fingerprint-"))
    if name == "figure_viewer":
        from Zou_lab_control.frontend import show_figure_viewer
        return app, show_figure_viewer(path=None), None

    if kind == "legacy":
        # The pre-migration composition: the caller owns the hub and hands the shell its
        # catalogs.  Mirrors ZLC_main's launcher body, including that the catalogs are EMPTY
        # without a session -- which is why --no-connect changes the fingerprint and is
        # reported in it rather than being a silent shortcut.
        import Zou_lab_control.neutral_atom as na
        from Zou_lab_control.neutral_atom.core.signals import SignalHub
        catalogs, experiment = {}, None
        if connect_devices:
            experiment = na.connect("virtual", sitemap={"grid_shape": (5, 7)})
            experiment.readout.sitemap(method="box", frames=4, display=False)
            experiment.readout.thresholds(frames=24, display=False)
            catalogs = dict(measurements=experiment.readout.measurement_specs(),
                            processors=experiment.readout.processor_specs(),
                            tasks=experiment.readout.task_specs())
        if name == "task_console":
            from Zou_lab_control.frontend.task_console import (
                default_console_state, show_task_console)
            window = show_task_console(hub=SignalHub(), state=default_console_state(),
                                       session=experiment, **catalogs)
        else:
            from Zou_lab_control.frontend import show_pulse_gui
            window = show_pulse_gui()
        return app, window, experiment

    from Zou_lab_control.notebook import connect
    experiment = connect("virtual", repository=room / "repo", name="fingerprint", seed=0)
    if kind == "workbench_app":
        from zlc_workbench.task_console.app import open_task_console
        window = open_task_console(experiment)
    elif kind == "editor_app":
        # The launcher opens the OFFLINE editor (no session), so the probe does too.
        from zlc_workbench.pulse_editor.app import open_pulse_editor
        window = open_pulse_editor()
    elif name == "task_console":
        window = experiment.task_console()
    elif kind == "workbench":
        from Zou_lab_control.workbench import open_pulse_workbench
        window = open_pulse_workbench(experiment, path=None)
    else:
        window = experiment.pulse_gui()
    return app, window, experiment

name, kind, connect_devices = sys.argv[1], sys.argv[2], sys.argv[3] == "connect"
app, window, experiment = build(name, kind, connect_devices)
try:
    from PyQt5 import QtWidgets
    tree = describe(window)
    flat = {}
    for child in window.findChildren(QtWidgets.QWidget):
        flat[type(child).__name__] = flat.get(type(child).__name__, 0) + 1
    print("@@FINGERPRINT@@" + json.dumps(
        {"window": name, "entry": kind, "connected": connect_devices, "root": tree,
         "widget_total": sum(flat.values()), "by_class": flat},
        sort_keys=True))
finally:
    try:
        # ``window`` may be a BODY inside a frame; closing the body would leak the frame.
        window.window().close()
    finally:
        if experiment is not None:
            experiment.close()
'''


def entry_kind(repo: pathlib.Path, window: str) -> str:
    """Read the tree's own launcher and report which composition it opens.

    Ambiguity is an error rather than a first-match guess: a launcher naming two entries is
    exactly the mid-switch state where picking one silently would make an A/B lie.
    """
    launcher = repo / f"{window}.py"
    if not launcher.is_file():
        raise SystemExit(f"{repo} has no launcher {window}.py to derive the entry from")
    source = launcher.read_text(encoding="utf-8")
    hits = [kind for marker, kind in ENTRY_MARKERS[window] if marker in source]
    if len(hits) != 1:
        raise SystemExit(
            f"{launcher} matches {hits or 'no'} known entry marker(s); teach ENTRY_MARKERS "
            f"about this launcher instead of letting the probe guess")
    return hits[0]


def fingerprint(repo: pathlib.Path, window: str, connect_devices: bool = True) -> dict:
    """Build ``window`` inside ``repo`` in a child process and return its structure."""
    env = dict(os.environ)
    env.update(CHILD_ENV)
    # The target tree must win over any editable install pointing elsewhere.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    argv = [window, entry_kind(repo, window), "connect" if connect_devices else "offline"]
    done = subprocess.run([sys.executable, "-c", _CHILD, *argv],
                          cwd=str(repo), env=env, capture_output=True, text=True, timeout=900)
    marker = "@@FINGERPRINT@@"
    for line in done.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise SystemExit(
        f"probe produced no fingerprint for {window} in {repo}\n"
        f"--- stdout ---\n{done.stdout[-2000:]}\n--- stderr ---\n{done.stderr[-4000:]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("window", choices=WINDOWS)
    parser.add_argument("--repo", type=pathlib.Path, required=True,
                        help="source tree to build the window in (READ ONLY)")
    parser.add_argument("--out", type=pathlib.Path, help="write JSON here instead of stdout")
    parser.add_argument("--no-connect", action="store_true",
                        help="open without a device session (fast structural check; the "
                             "catalogs are empty, which the fingerprint records)")
    args = parser.parse_args(argv)

    repo = args.repo.expanduser().resolve()
    if not (repo / "Zou_lab_control").is_dir():
        raise SystemExit(f"not a source tree (no Zou_lab_control/): {repo}")

    text = json.dumps(fingerprint(repo, args.window, not args.no_connect),
                      indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
