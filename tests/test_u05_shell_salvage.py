"""The legacy GUI's presentation layer moved home; the shells' seams are ledgered.

Direction change, per the user's explicit instruction: the migrated windows are
NOT rebuilt from scratch against the old ones - the OLD GUI code itself is the
skeleton.  Presentation-only modules move verbatim into the target packages, the
legacy paths become pure import shims onto the moved code, and the legacy
windows keep running on that ONE copy.  Visual and behavioural fidelity is then
structural (same code), not something a test has to approximate.

Moved this slice (zero body edits beyond the two seam cuts named below):

    frontend/canvas.py        -> zlc_frontend/live_plot/canvas.py
    frontend/selectors.py     -> zlc_frontend/live_plot/selectors.py
    frontend/ticks.py         -> zlc_frontend/live_plot/ticks.py
    frontend/_validate.py     -> zlc_frontend/live_plot/_validate.py     [renames]
    frontend/param_widgets.py -> zlc_frontend/qt_widgets/param_widgets.py [seam 2]
    Zou_lab_control/_paths.py -> zlc_storage/paths.py

Seams: ``display_path`` now comes from ``zlc_storage.paths``; ``facet``'s one
cross-package import follows ``_readout_math`` into ``zlc_data``.
NOT moved: ``render_loop.py``/``qt_canvas.py`` - zlc_frontend's guards make
qt_widgets the only Qt owner AND matplotlib-free, so the legacy Qt+matplotlib
render model has no legal home there; both stay legacy-side until the shells
adopt the worker-raster model.
Renames: the import-DAG guard reserves ``_positive_float``/``_positive_int``
for ``zlc_storage.canonical``; the moved module renamed them (same bodies) and
the shim rebinds the old names to the SAME objects.

The two big shells - ``task_console.py`` (10k lines) and ``live.py`` (6.3k) -
stay legacy-side because their function bodies lazily import domain code at the
sites listed in TENDRILS below.  That inventory IS the port specification for
wiring the new data plane: each entry is one seam the handoff must cut before
the file can cross the DAG boundary.  Ratchet, both directions: a NEW domain
import in a shell fails here immediately; porting one away must delete its row.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MOVES = {
    "Zou_lab_control.frontend.canvas": "zlc_frontend.live_plot.canvas",
    "Zou_lab_control.frontend.selectors": "zlc_frontend.live_plot.selectors",
    "Zou_lab_control.frontend.ticks": "zlc_frontend.live_plot.ticks",
    "Zou_lab_control.frontend._validate": "zlc_frontend.live_plot._validate",
    "Zou_lab_control.frontend.param_widgets": "zlc_frontend.qt_widgets.param_widgets",
    "Zou_lab_control._paths": "zlc_storage.paths",
    # H1a - the pure numeric/declarative core the shells lazily import.
    "Zou_lab_control._readout_math": "zlc_data.readout_math",
    "Zou_lab_control.neutral_atom.core.facet": "zlc_data.facet",
    "Zou_lab_control.neutral_atom.core.signal_tensor": "zlc_data.signal_tensor",
    "Zou_lab_control.neutral_atom.core.params": "zlc_data.param_decl",
    # H1b - the fit/region core; note plot_region vs the top-level Selection.
    "Zou_lab_control.neutral_atom.core.raster": "zlc_data.raster",
    "Zou_lab_control.neutral_atom.core.selection": "zlc_data.plot_region",
    "Zou_lab_control.neutral_atom.core.fitting": "zlc_data.curve_fitting",
    # H1c - what a saved figure records about where its data came from.
    "Zou_lab_control.neutral_atom.operations.figure_capture": "zlc_data.figure_capture",
}

_MODULE_INFRA = {
    "__name__", "__file__", "__loader__", "__spec__", "__package__",
    "__builtins__", "__cached__", "__doc__",
}


def test_legacy_paths_are_pure_shims():
    for old in MOVES:
        path = ROOT / (old.replace(".", "/") + ".py")
        text = path.read_text(encoding="utf-8")
        assert "MOVED to" in text, f"{path} is not a shim"
        assert len(text.splitlines()) <= 40, f"{path} carries more than forwarding"


def test_every_name_resolves_to_the_identical_object():
    """Legacy imports keep working AND isinstance checks keep passing: same objects."""

    for old, new in MOVES.items():
        old_module = importlib.import_module(old)
        new_module = importlib.import_module(new)
        for key, value in vars(new_module).items():
            if key in _MODULE_INFRA:
                continue
            assert getattr(old_module, key) is value, f"{old}.{key} diverged"


def test_validator_renames_keep_the_old_names_alive_in_the_shim():
    shim = importlib.import_module("Zou_lab_control.frontend._validate")
    moved = importlib.import_module("zlc_frontend.live_plot._validate")
    assert shim._positive_float is moved.positive_float
    assert shim._positive_int is moved.positive_int


def test_moved_modules_import_no_legacy_code():
    """Belt and braces beside the DAG test: the moved copies are legacy-free."""

    for new in MOVES.values():
        path = ROOT / (new.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("Zou_lab_control"), (
                    f"{new} still imports {node.module}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("Zou_lab_control"), (
                        f"{new} still imports {alias.name}"
                    )


# ---------------------------------------------------------------- tendril ledger

#: Every domain import inside the two shells, machine-verified below.  This is
#: the PORT SPECIFICATION for checklist 4: wiring the new data plane means
#: cutting these seams one group at a time; each cut deletes its rows here.
TENDRILS = {
    "Zou_lab_control/frontend/task_console.py": {
        ("neutral_atom.core.signals", "NO_LINEAGE"),
        ("neutral_atom.operations.logic", "Processor"),
        ("neutral_atom.operations.logic", "ProcessorRun"),
        ("neutral_atom.operations.logic", "camera_frame_keys"),
        ("neutral_atom.operations.logic", "contract_shape_label"),
        ("neutral_atom.operations.logic", "describe_shape"),
        ("neutral_atom.operations.measurement", "SWEEP_API_SLOT"),
        ("neutral_atom.operations.measurement", "SWEEP_SCAN_SLOT"),
        ("neutral_atom.operations.measurement", "measurement_slug"),
        ("neutral_atom.operations.measurements.pulse_scan", "_resolve_probe_template"),
        ("neutral_atom.operations.measurements.pulse_scan", "_semantic_api_names"),
        ("neutral_atom.operations.measurements.pulse_scan", "_semantic_scan_names"),
        ("neutral_atom.operations.processors.analysis", "ANALYSIS_ACTIONS"),
        ("neutral_atom.operations.processors.analysis", "ANALYSIS_SPEC_NAME"),
        ("neutral_atom.operations.processors.fit", "FitProcessor"),
        ("neutral_atom.operations.signal_expr", "DEFAULT_SOURCE"),
        ("neutral_atom.operations.signal_expr", "IDENTITY_SOURCE_RE"),
        ("neutral_atom.operations.signal_expr", "NAMESPACE_HELPERS"),
        ("neutral_atom.operations.signal_expr", "SIGNAL_EXPR_HELP"),
        ("neutral_atom.operations.signal_expr", "SignalExpr"),
        ("neutral_atom.operations.signal_expr", "hub_namespace"),
        ("neutral_atom.operations.signal_expr", "is_identity_source"),
        ("neutral_atom.operations.signal_expr", "seed_source_for_slots"),
        ("neutral_atom.operations.task", "DEFAULT_MID_RUN_KEY"),
        ("neutral_atom.timing", "scan_column_spec"),
        ("neutral_atom.timing", "scan_table_template"),
    },
    # live.py: EMPTY.  Its last two seams were cut differently and the contrast is
    # the whole lesson.  ``figure_capture`` was pure description over duck-typed
    # inputs, so it MOVED (see MOVES above).  The DAC-waveform helpers could not:
    # ``pulse_table`` is 3k lines over the port catalog and the streamer geometry,
    # and ``zlc_frontend`` may never import ``zlc_pulse`` at all.  So the STATE now
    # answers for its own waveform (``PulseTableState.analog_bus_samples``) and the
    # render surface asks the object it was already handed - an OBJECT port, pinned
    # by ``test_u06_shell_domain_ports``.  Relocate what is pure; invert what is not.
    "Zou_lab_control/frontend/live.py": set(),
}


def _scan_tendrils(path: Path) -> set[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level >= 2 or module.startswith("Zou_lab_control"):
                target = module.replace("Zou_lab_control.", "")
                if "neutral_atom" in target or target in ("_readout_math", "_paths"):
                    for alias in node.names:
                        found.add((target, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("Zou_lab_control"):
                    found.add((alias.name.replace("Zou_lab_control.", ""), "*"))
    return found


def test_the_shell_tendril_ledger_is_exact():
    for relative, expected in TENDRILS.items():
        actual = _scan_tendrils(ROOT / relative)
        grew = actual - expected
        assert not grew, (
            f"{relative} grew NEW domain tendrils {sorted(grew)}; the shells are "
            "frozen skeleton - new data-logic coupling is not allowed."
        )
        ported = expected - actual
        assert not ported, (
            f"{relative} no longer imports {sorted(ported)}; delete those rows "
            "from TENDRILS in the same commit so the ledger stays exact."
        )
