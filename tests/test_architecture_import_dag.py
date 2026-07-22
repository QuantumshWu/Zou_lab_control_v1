"""Mechanical import-boundary ratchet for the target bounded contexts."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]

#: Every top-level directory holding code this file audits.  Keeping the list in
#: ONE place is what makes :func:`_source_files` able to prove it complete; a new
#: top-level package must be registered here or it is born exempt (which is how
#: ``zlc_workbench`` and ``fpga`` came to be governed by no import rule at all).
SOURCE_ROOTS = (
    "fpga",
    "Zou_lab_control",
    "zlc_data",
    "zlc_frontend",
    "zlc_neutral_atom",
    "zlc_pulse",
    "zlc_storage",
    "zlc_workbench",
)


def _source_files(roots=SOURCE_ROOTS):
    """Every ``.py`` under the code roots, with the roots proven to exist first.

    ``Path.rglob`` on a missing directory yields nothing and raises nothing, so a
    guard that names a root keeps PASSING while silently auditing less the moment
    that root is renamed or deleted.  During a migration that renames roots for a
    living, a guard going quietly vacuous is worse than no guard: it reads as
    evidence.  Proving the roots exist converts that into a red test.
    """

    missing = [root for root in roots if not (ROOT / root).is_dir()]
    assert not missing, (
        f"source roots {missing} no longer exist, so every guard scanning them has "
        "been auditing nothing. Update SOURCE_ROOTS in the SAME commit that removes "
        "a root, so the deletion is a decision rather than a silent gap."
    )
    for root in roots:
        yield from sorted((ROOT / root).rglob("*.py"))


def _sole_definition(name, node_type=ast.ClassDef):
    """Locate the ONE definition of ``name`` by property rather than by path.

    A guard keyed to a hard-coded file goes vacuous the instant that file moves.
    That is not hypothetical: this module used to scan
    ``Zou_lab_control/frontend/data_figure.py`` for a forbidden version field, and
    when the salvage turned that path into a forwarding shim the check kept
    passing against a file which by construction could no longer contain what it
    was looking for.  Searching for the definition follows the code across the
    migration, and requiring EXACTLY ONE site proves in passing that nobody forked
    it into a second copy.
    """

    sites = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, node_type) and node.name == name:
                sites.append((path, node))
    assert len(sites) == 1, (
        f"expected exactly one definition of {name!r}, found "
        f"{[str(path.relative_to(ROOT)) for path, _ in sites]}. Zero means the guard "
        "below is auditing nothing; more than one means the owner was forked."
    )
    return sites[0]


FORBIDDEN = {
    "zlc_data": ("Zou_lab_control", "zlc_frontend", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"),
    "zlc_storage": ("Zou_lab_control", "zlc_data", "zlc_frontend", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"),
    "zlc_pulse": ("Zou_lab_control", "zlc_data", "zlc_frontend", "zlc_neutral_atom", "zlc_workbench"),
    "zlc_frontend": ("Zou_lab_control", "zlc_neutral_atom", "zlc_pulse", "zlc_workbench"),
    "zlc_neutral_atom": ("Zou_lab_control", "zlc_frontend", "zlc_workbench"),
}

CANONICAL_HELPER_NAMES = frozenset(
    {
        "_canonical_nonempty_text",
        "_canonical_text",
        "_exact_map",
        "_exact_tree",
        "_finite_real",
        "_finite_time",
        "_finite_timestamp",
        "_integer",
        "_nonempty_text",
        "_nonnegative_int",
        "_positive_finite",
        "_positive_float",
        "_positive_int",
        "_positive_seconds",
        "_positive_timeout",
        "_require_digest",
        "_sha256",
        "_text",
        "_validate_timeout",
    }
)

ALLOWED_STRIP_CONTEXTS = frozenset(
    {
        (Path("zlc_frontend/form.py"), "parse_number_text"),
        (Path("zlc_frontend/form.py"), "FormChoice.__post_init__"),
        (Path("zlc_frontend/form.py"), "FormFieldProps.__post_init__"),
        (Path("zlc_frontend/form.py"), "FormFieldProps.row_label"),
        # H1e - the salvaged live-plot shell.  Each reads text a HUMAN typed or a
        # saved figure recorded: a panel size ("2x3"), a fit-model key, an axis
        # name, a saved axis label.
        (Path("zlc_data/panel_size.py"), "panel_size_cells"),
        # The domain-ports wiring inherited from the legacy frontend root: the
        # template reader normalises the scan PROGRAM text a human typed into the
        # Scan-tab editor (or a template file carries) -- exactly the external
        # input boundary this guard reserves .strip() for.
        (Path("zlc_workbench/_domain_wiring.py"), "_read_pulse_template"),
        # The board pin map reads the platform's XDC -- a Tcl constraints file written by a
        # human and by Vivado, with arbitrary leading whitespace on every line.  Turning that
        # text into lane identities IS the external input boundary; the identities it yields
        # (lane keys, port labels, pin names) are canonical from here on.
        (Path("zlc_neutral_atom/timing/board_config.py"), "load_board_config"),
        # The runtime compiler (moved whole from the legacy device layer) parses
        # slot REFERENCES ("s0") that a human's plan entries and saved documents
        # carry as text, plus the wire-codec payload fields that mirror them.
        (Path("zlc_neutral_atom/timing/runtime_compiler.py"), "RuntimeBusSegment.from_dict"),
        (Path("zlc_neutral_atom/timing/runtime_compiler.py"), "_slot_ref_index"),
        (Path("zlc_neutral_atom/timing/runtime_compiler.py"), "_pulse_table_bus_segments._emit"),
    # The signal-expression composite moved into qt_widgets (S5-shell(o)); its two
    # normalisation points came with it, so the named entries follow the code.
    (Path("zlc_frontend/qt_widgets/signal_expr_widget.py"), "SignalExprWidget._open_editor"),
    (Path("zlc_frontend/qt_widgets/signal_expr_widget.py"), "SignalExprWidget.values_dict"),
    # The pulse-slot composite moved into qt_widgets (S5-shell(p)); its normalisation
    # points came with it, so the named entries follow the code.
    (Path("zlc_frontend/qt_widgets/pulse_slots_widget.py"), "PulseSlotsWidget._render_program"),
    (Path("zlc_frontend/qt_widgets/pulse_slots_widget.py"), "PulseSlotsWidget.rebuild"),
    (Path("zlc_frontend/qt_widgets/pulse_slots_widget.py"), "PulseSlotsWidget.values_dict"),
    # The panel record sank into zlc_data (S5-shell(t)).  Its one normalisation point reads
    # the ``value = ...`` expression a HUMAN typed into the Setting, so the named entry
    # follows the code rather than the guard being widened.
    (Path("zlc_data/console_records.py"), "PanelConfig.set_source"),
    # The saved-layout record moved into zlc_frontend (S5-shell(w)).  Both normalisation
    # points are exactly the boundary this guard reserves .strip() for: an environment
    # variable a human typed, and a task NAME a human typed on the CLI or in the GUI.
    (Path("zlc_frontend/console_state.py"), "task_files_dir"),
    (Path("zlc_frontend/console_state.py"), "resolve_task_state"),
    # The panel-param catalog moved into zlc_frontend (S5-shell(y)).  Its one normalisation
    # point reads the colormap NAME a human picked in the Setting combo, so the named entry
    # follows the code rather than the guard being widened.
    (Path("zlc_frontend/panel_params.py"), "resolved_cmap"),
    # The six pure helpers left the shell (S5-shell(z)); their normalisation points came with
    # them, and each is the boundary this guard reserves .strip() for -- a panel/logic-node TITLE
    # a human typed (or a saved layout carries), and two line-edit parsers.
    (Path("zlc_data/shape_text.py"), "indexed_unique_name"),
    (Path("zlc_frontend/form.py"), "lenient_float"),
    (Path("zlc_frontend/form.py"), "text_to_python"),
    # The Analysis controls moved into qt_widgets; the fix/seed editor reads the numbers a HUMAN
    # typed into two line edits, which is exactly the boundary this guard reserves .strip() for,
    # so the named entry follows the code rather than the guard being widened.
    (Path("zlc_frontend/qt_widgets/analysis_controls.py"), "_FitFixSeedEditor.values"),
        (Path("zlc_frontend/qt_widgets/fluent.py"), "align_to_resolution"),
        (Path("zlc_frontend/qt_widgets/fluent.py"), "FluentPathEdit._dialog_start"),
        (Path("zlc_frontend/qt_widgets/fluent.py"), "FluentTreeComboBox.current_signal"),
        # Same concern as the line above: this reads what a human TYPED into an
        # editable combo, which is exactly the named-adapter boundary this guard
        # reserves .strip() for.
        (Path("zlc_frontend/qt_widgets/signal_picker.py"), "read_editable_combo"),
        # Salvaged legacy param-form handlers: every one of these reads what a
        # human TYPED into a form field (JSON body, device ref, path, pulse-slot
        # program text) - the exact named-adapter boundary .strip() is reserved for.
        (Path("zlc_frontend/qt_widgets/param_widgets.py"), "JsonHandler.read"),
        (Path("zlc_frontend/qt_widgets/param_widgets.py"), "JsonHandler.is_empty"),
        (Path("zlc_frontend/qt_widgets/param_widgets.py"), "DeviceRefHandler.is_empty"),
        (Path("zlc_frontend/qt_widgets/param_widgets.py"), "PathHandler.is_empty"),
        (Path("zlc_frontend/qt_widgets/param_widgets.py"), "PulseSlotsHandler.is_empty"),
        # Salvaged fit/region core: each of these canonicalises a model key or
        # an axis name that a human typed into a form or a saved recipe - the
        # same named-adapter boundary .strip() is reserved for.
        (Path("zlc_data/curve_fitting.py"), "FitRequest.__post_init__"),
        (Path("zlc_data/curve_fitting.py"), "HistogramFitResult.__post_init__"),
        (Path("zlc_data/curve_fitting.py"), "fit_model"),
        (Path("zlc_data/curve_fitting.py"), "fit_models"),
        (Path("zlc_data/curve_fitting.py"), "fit_histogram"),
        (Path("zlc_data/plot_region.py"), "AxisRange.__post_init__"),
        # W1 - canonicalises the DISPLAY NAME a human typed into the machine token
        # every published signal name derives from: the same named human-input
        # adapter boundary .strip() is reserved for.
        (Path("zlc_data/shape_text.py"), "measurement_slug"),
        # H2a - the value-expression language.  Every one of these reads the
        # expression text a HUMAN typed into a panel or a logic-node source box.
        (Path("zlc_data/signal_expr.py"), "is_identity_source"),
        (Path("zlc_data/signal_expr.py"), "seed_source_for_slots"),
        (Path("zlc_data/signal_expr.py"), "SignalExpr.__init__"),
        (Path("zlc_frontend/qt_widgets/form.py"), "_TextHandler.is_empty"),
        (Path("zlc_frontend/qt_widgets/form.py"), "_IntHandler.read"),
        (Path("zlc_frontend/qt_widgets/form.py"), "_IntHandler.is_empty"),
        (Path("zlc_frontend/qt_widgets/form.py"), "_NumberHandler.read"),
        (Path("zlc_frontend/qt_widgets/form.py"), "_NumberHandler.is_empty"),
        (Path("zlc_frontend/qt_widgets/form.py"), "_LosslessFloatSpinBox.valueFromText"),
        (Path("zlc_frontend/qt_widgets/form.py"), "_LosslessFloatSpinBox.validate"),
        (Path("zlc_frontend/qt_widgets/form.py"), "_FloatHandler.read"),
        (Path("zlc_frontend/qt_widgets/form.py"), "_FloatHandler.is_empty"),
        (Path("zlc_pulse/document.py"), "ScanRecipeProvenance.__post_init__"),
        (Path("zlc_pulse/transport/axi.py"), "VivadoAxiRegisterTransport._parse_read"),
        # The legacy pulse domain moved into zlc_neutral_atom.timing (shims stay behind);
        # the named entries follow the code.  Every one reads text a HUMAN typed - channel
        # labels from config/GUI, scan-slot / API-slot names, a period field's nominal
        # text, a bus-target token, an analog-bus mode word - exactly the named
        # input-adapter boundary .strip() is reserved for.
        (Path("zlc_neutral_atom/timing/ports.py"), "PortSpec.__post_init__"),
        (Path("zlc_neutral_atom/timing/ports.py"), "PortCatalog.__post_init__"),
        (Path("zlc_neutral_atom/timing/ports.py"), "PortCatalog.with_label"),
        (Path("zlc_neutral_atom/timing/ports.py"), "PortCatalog.from_channels"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "_default_scan_name"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "ScanSlot.__post_init__"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "PulseTableState._read_field_nominal"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "PulseTableState._set_bus_target"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "PulseTableState._clear_slot_binding"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "PulseTableState._normalize_analog_bus_modes"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "PulseTableState.set_analog_bus_mode"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "_SafeEval.affine"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "_coerce_bus_value"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "_period_display"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "bus_period_levels"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "eval_time_expr"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "evaluate_scan_table_code"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "is_slot_ref"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "load_scan_table"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "period_index_by_name"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "resolve_pulse_template"),
        (Path("zlc_neutral_atom/timing/pulse_table.py"), "slot_ref_index"),
    }
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            result.add(node.module)
    return result


@pytest.mark.parametrize("package,forbidden", FORBIDDEN.items())
def test_target_package_has_no_reverse_imports(package, forbidden):
    root = ROOT / package
    if not root.exists():
        pytest.skip(f"{package} has not entered its migration slice")
    violations = []
    for path in root.rglob("*.py"):
        for imported in _imports(path):
            if any(imported == prefix or imported.startswith(prefix + ".") for prefix in forbidden):
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")
            if package == "zlc_data" and imported.startswith("zlc_storage"):
                if imported != "zlc_storage.canonical" and not imported.startswith(
                    "zlc_storage.canonical."
                ):
                    violations.append(
                        f"{path.relative_to(ROOT)} imports storage I/O boundary {imported}"
                    )
    assert not violations, "reverse package dependencies:\n" + "\n".join(violations)


def test_zlc_data_import_does_not_initialize_storage_backends():
    code = """
import sys
import zlc_data

forbidden = (
    'zlc_storage.content_store',
    'zlc_storage.durability',
    'zlc_storage.framed_journal',
    'zlc_storage.repository_lease',
)
loaded = tuple(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit('storage backends loaded through zlc_data: ' + ', '.join(loaded))
if 'zlc_storage.canonical' not in sys.modules:
    raise SystemExit('zlc_data did not load its only permitted storage primitive module')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_occupancy_runtime_does_not_import_calibration_analysis_or_repository():
    code = """
import sys
import zlc_neutral_atom.readout.occupancy

forbidden = (
    'scipy',
    'zlc_neutral_atom.readout.analysis',
    'zlc_neutral_atom.readout.calibration_codec',
    'zlc_neutral_atom.readout.calibration_repository',
)
loaded = tuple(
    name for name in forbidden
    if name in sys.modules or any(item.startswith(name + '.') for item in sys.modules)
)
if loaded:
    raise SystemExit('occupancy imported offline calibration machinery: ' + ', '.join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_zlc_pulse_has_no_historical_target_importer():
    forbidden = (
        "pulse_target_from_legacy",
        "Zou_lab_control.neutral_atom.PortCatalog",
        "channel_labels",
        "analog_buses",
        "clk_channels",
    )
    violations = []
    for path in (ROOT / "zlc_pulse").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)} contains {token}")
    assert not violations, (
        "zlc_pulse owns only the current PulseTarget contract; installed legacy "
        "device topology may be projected only at the composition boundary:\n"
        + "\n".join(violations)
    )


def test_canonical_primitive_validators_have_one_owner():
    violations = []
    for package in FORBIDDEN:
        for path in (ROOT / package).rglob("*.py"):
            relative = path.relative_to(ROOT)
            if relative == Path("zlc_storage/canonical.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name not in CANONICAL_HELPER_NAMES:
                    continue
                violations.append(f"{relative}:{node.lineno} defines {node.name}")
    assert not violations, (
        "canonical primitive validators belong to zlc_storage.canonical; "
        "domain modules must import and alias them:\n" + "\n".join(violations)
    )


def test_text_normalization_is_confined_to_named_input_adapters():
    violations = []

    class StripVisitor(ast.NodeVisitor):
        def __init__(self, relative: Path) -> None:
            self.relative = relative
            self.context: list[str] = []

        def visit_ClassDef(self, node):
            self.context.append(node.name)
            self.generic_visit(node)
            self.context.pop()

        def visit_FunctionDef(self, node):
            self.context.append(node.name)
            self.generic_visit(node)
            self.context.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "strip":
                location = (self.relative, ".".join(self.context))
                if location not in ALLOWED_STRIP_CONTEXTS:
                    violations.append(
                        f"{self.relative}:{node.lineno} strips text in {location[1]}"
                    )
            self.generic_visit(node)

    for package in FORBIDDEN:
        for path in (ROOT / package).rglob("*.py"):
            relative = path.relative_to(ROOT)
            if relative == Path("zlc_storage/canonical.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            StripVisitor(relative).visit(tree)
    assert not violations, (
        "machine identities must reject non-canonical text; .strip() is reserved "
        "for explicitly named human/external input adapters:\n" + "\n".join(violations)
    )


def test_pulse_ir_digest_and_affine_formulas_have_one_owner():
    violations = []
    for relative in (
        Path("zlc_pulse/fpga.py"),
        Path("zlc_pulse/artifact.py"),
    ):
        tree = ast.parse(
            (ROOT / relative).read_text(encoding="utf-8"),
            filename=str(relative),
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "canonical_digest":
                continue
            argument = node.args[0]
            if (
                isinstance(argument, ast.Call)
                and isinstance(argument.func, ast.Name)
                and argument.func.id == "target_ir_to_tree"
            ):
                violations.append(f"{relative}:{node.lineno} recomputes TargetIR fingerprint")

    forbidden_helpers = {
        "_apply_affine",
        "_effective",
        "_effective_tick",
        "_exact_integer",
        "_exact_positive_ticks",
    }
    for source in (ROOT / "zlc_pulse").rglob("*.py"):
        relative = source.relative_to(ROOT)
        tree = ast.parse(
            source.read_text(encoding="utf-8"),
            filename=str(relative),
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in forbidden_helpers
            ):
                violations.append(f"{relative}:{node.lineno} defines duplicate {node.name}")
    assert not violations, (
        "TargetIR.fingerprint, ir.evaluate_affine_tick, document._integral_code, "
        "and document._exact_ticks are the sole pulse formula owners:\n"
        + "\n".join(violations)
    )


def test_zlc_data_typed_byte_admission_has_one_owner():
    violations = []
    for source in (ROOT / "zlc_data").rglob("*.py"):
        relative = source.relative_to(ROOT)
        if relative == Path("zlc_data/codec.py"):
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in {"_require_canonical", "_require_typed_canonical"}
            ):
                violations.append(f"{relative}:{node.lineno} defines {node.name}")
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("decode_")
            ):
                calls = {
                    call.func.id
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                }
                if "_require_typed_canonical" not in calls:
                    violations.append(
                        f"{relative}:{node.lineno} bypasses typed canonical-byte admission"
                    )
    assert not violations, (
        "zlc_data.codec owns typed canonical-byte admission; sibling codecs "
        "must delegate:\n" + "\n".join(violations)
    )


def test_zlc_data_codecs_delegate_leaf_invariants_to_typed_owners():
    owner_constructors = {
        "AxisId",
        "AxisLayout",
        "AxisRoleId",
        "AxisSpec",
        "BlockId",
        "CommittedTransform",
        "CoordinateFrameId",
        "CoordinateRangeSelection",
        "DatasetRevision",
        "DatasetRevisionRef",
        "FitNumericPolicy",
        "FitParameterConstraint",
        "FitResultBatch",
        "FitSpec",
        "IndexRangeSelection",
        "IndexSelection",
        "ReductionSpec",
        "StreamGenerationId",
    }
    primitive_prevalidators = {
        "_integer",
        "_text",
        "canonical_text",
        "finite_real",
        "integer",
    }
    sources = (
        ROOT / "zlc_data/codec.py",
        ROOT / "zlc_data/fit_codec.py",
        ROOT / "zlc_data/selection.py",
        ROOT / "zlc_data/transform_codec.py",
    )
    violations = []
    for source in sources:
        relative = source.relative_to(ROOT)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            constructor = None
            if isinstance(call.func, ast.Name):
                constructor = call.func.id
            elif isinstance(call.func, ast.Attribute):
                constructor = call.func.attr
            if constructor not in owner_constructors:
                continue
            for nested in ast.walk(call):
                if nested is call or not isinstance(nested, ast.Call):
                    continue
                helper = None
                if isinstance(nested.func, ast.Name):
                    helper = nested.func.id
                elif isinstance(nested.func, ast.Attribute):
                    helper = nested.func.attr
                if helper in primitive_prevalidators:
                    violations.append(
                        f"{relative}:{nested.lineno} prevalidates {constructor} with {helper}"
                    )
    assert not violations, (
        "codec owns wire structure; typed constructors own their leaf invariants:\n"
        + "\n".join(violations)
    )


def test_dataset_cell_layout_and_source_transform_schema_have_one_owner():
    violations = []
    for source in (
        *(ROOT / "zlc_data").rglob("*.py"),
        ROOT / "zlc_frontend/figure/evaluate.py",
    ):
        relative = source.relative_to(ROOT)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "TransformedSchema"
                and relative != Path("zlc_data/transform.py")
            ):
                violations.append(
                    f"{relative}:{node.lineno} reconstructs TransformedSchema outside its owner"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "AxisLayout"
                and node.func.attr == "product"
                and relative != Path("zlc_data/schema.py")
                and any(
                    isinstance(descendant, ast.Attribute)
                    and descendant.attr == "point_layout"
                    for descendant in ast.walk(node)
                )
            ):
                violations.append(
                    f"{relative}:{node.lineno} reconstructs DatasetSchema.cell_layout"
                )
    assert not violations, (
        "DatasetSchema.cell_layout and zlc_data.transform own source layout/schema "
        "projection respectively:\n" + "\n".join(violations)
    )


def test_zlc_data_does_not_restore_unused_bulk_or_transform_history_surfaces():
    forbidden = {
        "PreviewTransformedData",
        "Reduce",
        "Select",
        "TransformOrigin",
        "TransformRecord",
        "TransformRevision",
        "data_block_from_tree",
        "data_block_to_tree",
        "decode_committed_transform",
        "decode_data_block",
        "decode_selection",
        "decode_transform_record",
        "decode_value",
        "encode_committed_transform",
        "encode_data_block",
        "encode_selection",
        "encode_transform_record",
        "encode_value",
        "preview_transform",
        "transform_record_from_tree",
        "transform_record_to_tree",
        "transformed_schema_from_tree",
        "value_from_tree",
        "value_to_tree",
    }
    violations = []
    for source in (ROOT / "zlc_data").rglob("*.py"):
        relative = source.relative_to(ROOT)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            candidate = None
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                candidate = node.name
            elif isinstance(node, ast.Name):
                candidate = node.id
            elif isinstance(node, ast.Attribute):
                candidate = node.attr
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidate = node.value
            if candidate in forbidden:
                violations.append(f"{relative}:{node.lineno} restores {candidate}")
    assert not violations, (
        "zlc_data persists large frame payloads through bounded artifact chunks; "
        "the data authority must not regrow zero-consumer bulk codecs, edit-history "
        "metadata, or one-field operation wrappers:\n" + "\n".join(violations)
    )


def test_readout_artifacts_do_not_restore_edit_counter_metadata():
    forbidden = {
        "algorithm_id",
        "algorithm_version",
        "default_model_policy",
        "gate_passed",
        "model_id",
        "model_version",
        "policy_version",
        "quality_gate_version",
        "required_model_kinds",
        "CalibrationCapability",
        "CalibrationStage",
        "DefaultModelPolicy",
        "OccupancyModelSelection",
    }
    violations = []
    for source in (ROOT / "zlc_neutral_atom" / "readout").rglob("*.py"):
        relative = source.relative_to(ROOT)
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            candidate = None
            if isinstance(node, ast.Name):
                candidate = node.id
            elif isinstance(node, ast.Attribute):
                candidate = node.attr
            elif isinstance(node, ast.arg):
                candidate = node.arg
            elif isinstance(node, ast.keyword):
                candidate = node.arg
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidate = node.value
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                candidate = node.name
            if candidate in forbidden:
                violations.append(f"{relative}:{node.lineno} restores {candidate}")
    assert not violations, (
        "readout identity is CalibrationArtifactRef + ReadoutModelKind; descriptive "
        "durable format names and CAS bytes replace edit-counter metadata:\n"
        + "\n".join(violations)
    )


GUI_TOOLKITS = frozenset({"matplotlib", "PyQt5", "PySide2", "PySide6", "tkinter"})

#: Only ``zlc_frontend`` renders.  Everything else describes, computes or moves
#: data, and must stay importable in a headless process (a server, a notebook
#: kernel with no display, a CI worker) without dragging a GUI toolkit in.
GUI_FREE_PACKAGES = ("zlc_data", "zlc_storage", "zlc_pulse", "zlc_neutral_atom", "zlc_workbench")


@pytest.mark.parametrize("package", GUI_FREE_PACKAGES)
def test_only_the_frontend_package_may_import_a_gui_toolkit(package):
    """The frontend-neutrality promise, mechanised.

    ``figure_capture`` - what a saved figure records about where its data came
    from - carried exactly this promise as prose plus one legacy test that read
    the module by ``__file__``.  When the module moved into ``zlc_data`` that
    test would have kept passing while checking nothing.  A property worth a
    guard is worth a guard that does not depend on where the file lives, so it
    is stated here once, over every non-rendering package.
    """

    root = ROOT / package
    if not root.exists():
        pytest.skip(f"{package} has not entered its migration slice")
    violations = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        # The ONE sanctioned exception (C14/C20): the workbench plot_bridge zone holds the
        # widgets that marry Qt to matplotlib while the legacy shells are dismantled.  It is
        # reached only by explicit import (never from a package __init__), so every package
        # root here still imports headless; the zone itself empties in the render-purification
        # pass, at which point this exemption is deleted with it.
        if "/plot_bridge" in f"/{rel}":
            continue
        for imported in _imports(path):
            if imported.split(".")[0] in GUI_TOOLKITS:
                violations.append(f"{path.relative_to(ROOT)} imports {imported}")
    assert not violations, (
        f"{package} must stay headless-importable; only zlc_frontend may render: "
        + "; ".join(violations)
    )


def test_notebook_facade_has_no_implicit_current_calibration_state():
    source = (ROOT / "Zou_lab_control/notebook/facade.py").read_text(
        encoding="utf-8"
    )
    assert "current_calibration" not in source, (
        "calibration-dependent notebook requests must receive an explicit typed ref; "
        "do not restore a session current/revision map"
    )


def test_readout_package_root_does_not_eagerly_aggregate_leaf_owners():
    path = ROOT / "zlc_neutral_atom" / "readout" / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not imports, (
        "readout callers import the contracts/model/analysis/repository owner leaf; "
        "the package root must not make a light contract import initialize SciPy"
    )


def test_headless_notebook_import_does_not_load_frontend_renderer():
    code = (
        "import sys; import Zou_lab_control.notebook; "
        "assert 'zlc_frontend.render' not in sys.modules; "
        "assert not any(name == 'matplotlib' or name.startswith('matplotlib.') "
        "or name == 'PyQt5' or name.startswith('PyQt5.') "
        "or name == 'PySide6' or name.startswith('PySide6.') "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


#: The declared single owner of the ContentRef tree and CAS addressing.  It is
#: named -- and therefore skipped -- rather than left outside the scan: the guard
#: below shipped without ``zlc_storage`` among its roots, so "storage has ONE
#: owner" was asserted while the storage package was never looked at.  An owner
#: you exempt is a decision; an owner you forget to scan is a hole.
CONTENT_REF_OWNER = Path("zlc_storage/content_store.py")


def test_content_ref_codec_and_cas_address_have_one_storage_owner():
    violations = []
    for source in _source_files():
        relative = source.relative_to(ROOT)
        if relative == CONTENT_REF_OWNER:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(relative))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                literal_keys = {
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                }
                if len(node.keys) == 2 and literal_keys == {"digest", "size"}:
                    violations.append(
                        f"{relative}:{node.lineno} duplicates the ContentRef tree"
                    )
            if isinstance(node, ast.Set):
                literal_fields = {
                    item.value
                    for item in node.elts
                    if isinstance(item, ast.Constant)
                    and isinstance(item.value, str)
                }
                if len(node.elts) == 2 and literal_fields == {"digest", "size"}:
                    violations.append(
                        f"{relative}:{node.lineno} duplicates the ContentRef field set"
                    )
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name
                in {
                    "content_ref_to_tree",
                    "content_ref_from_tree",
                    "_content_ref_to_tree",
                    "_content_ref_from_tree",
                }
            ):
                violations.append(
                    f"{relative}:{node.lineno} redefines the storage ContentRef codec"
                )
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if function_name != "ContentRef":
                continue
            digest_call, size_call = node.args[:2]
            digest_function_names = {
                call.func.id
                if isinstance(call.func, ast.Name)
                else call.func.attr
                if isinstance(call.func, ast.Attribute)
                else ""
                for call in ast.walk(digest_call)
                if isinstance(call, ast.Call)
            }
            size_uses_len = any(
                isinstance(call.func, ast.Name) and call.func.id == "len"
                for call in ast.walk(size_call)
                if isinstance(call, ast.Call)
            )
            if (
                any("sha256" in name.lower() for name in digest_function_names)
                and size_uses_len
            ):
                violations.append(
                    f"{relative}:{node.lineno} mints a CAS address outside storage"
                )

    for relative in (
        Path("zlc_neutral_atom/artifacts/capture.py"),
        Path("zlc_neutral_atom/readout/calibration_repository.py"),
    ):
        tree = ast.parse(
            (ROOT / relative).read_text(encoding="utf-8"),
            filename=str(relative),
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "target_ref"
            ):
                violations.append(
                    f"{relative}:{node.lineno} slices typed target_ref grammar"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "startswith"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "target_ref"
            ):
                violations.append(
                    f"{relative}:{node.lineno} reparses typed target_ref grammar"
                )

    assert not violations, (
        "ContentRef tree/address belong to zlc_storage and each typed Ref owns its "
        "target_ref grammar:\n" + "\n".join(violations)
    )


def test_artifact_finalizers_do_not_replay_published_payload_digests():
    violations = []
    for relative in (
        Path("zlc_neutral_atom/artifacts/capture.py"),
        Path("zlc_neutral_atom/readout/occupancy_pipeline.py"),
    ):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "digest_components"
            ):
                violations.append(f"{relative}:{node.lineno}")
    assert not violations, (
        "payload content digest belongs to stream publication; artifact/finalizer "
        "code must consume EventRef provenance instead of re-reading arrays:\n"
        + "\n".join(violations)
    )


def test_calibration_numeric_versions_are_not_admission_authority():
    forbidden = (
        "CalibrationNumericBackend",
        "numeric-backend-digest",
        "numeric_backend_digest",
        "backend_digest",
        "frozen_numeric_backend",
        "_sanitized_build_identity",
        "_numeric_backend",
    )
    violations = []
    for path in (ROOT / "zlc_neutral_atom" / "readout").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)} contains {token}")
    assert not violations, (
        "numeric package versions are passive CalibrationReport notes; they "
        "must not regain a digest, replay, model, manifest, or admission role:\n"
        + "\n".join(violations)
    )

    analysis_path = ROOT / "zlc_neutral_atom" / "readout" / "analysis.py"
    source = analysis_path.read_text(encoding="utf-8")
    assert source.count("np.__version__") == 1
    assert source.count("scipy.__version__") == 1, (
        "numeric version notes must be sampled once into CalibrationReport, "
        "never copied into runtime artifact or admission evidence"
    )


def test_current_format_names_do_not_encode_edit_counters():
    pattern = re.compile(
        r"(?:\.v|/v)\d+"
        r"|[\"']schema[\"']\s*:\s*[\"'](?:v\d+|[^\"']*[-_]v\d+)[\"']"
        r"|application/vnd\.zlc\.[^\"']*-v\d+"
        r"|ZLC-CANONICAL-\d+"
    )
    hardware_protocol_allowlist = {
        Path("zlc_pulse/target.py"): ("zlc_pulse.PulseTargetABI/v1",),
        Path("zlc_storage/canonical.py"): ("ZLC-CANONICAL-1",),
        Path("tests/test_zlc_storage_canonical.py"): ("ZLC-CANONICAL-1",),
        Path("docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md"): (
            "zlc_pulse.PulseTargetABI/v1",
            "ZLC-CANONICAL-1",
        ),
        # The progress ledger moved OUT of the design doc (section 22 -> its own file), and the
        # two APPROVED hardware-protocol identities travelled with the historical rows that
        # discuss them.  The allowlist follows the text to its new home; the pattern is not
        # weakened.  Found the commit after the move, because this sweep reads ``git ls-files``
        # and an untracked new document is invisible to it until it is committed.
        Path("docs/MIGRATION_LEDGER_zh.md"): (
            "zlc_pulse.PulseTargetABI/v1",
            "ZLC-CANONICAL-1",
        ),
    }
    ratchet_path = Path(__file__).resolve().relative_to(ROOT)
    roots = (
        "zlc_data",
        "zlc_storage",
        "zlc_pulse",
        "zlc_neutral_atom",
        "zlc_frontend",
        "zlc_workbench",
        "tests",
        "docs",
        "pulses",
    )
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", *roots, ".gitignore"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    violations = []
    for item in tracked.split("\0"):
        if not item:
            continue
        relative = Path(item)
        path = ROOT / relative
        if relative == ratchet_path:
            continue
        # A worktree deletion is already absent from the architecture under
        # review even though ``git ls-files`` retains it until the next commit.
        if not path.is_file():
            continue
        if path.name != ".gitignore" and path.suffix not in {
            ".py",
            ".md",
            ".json",
            ".toml",
        }:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            candidate = line
            for allowed in hardware_protocol_allowlist.get(relative, ()):
                candidate = candidate.replace(allowed, "")
            if pattern.search(candidate):
                violations.append(f"{relative}:{line_number}: {line.strip()}")
    assert not violations, (
        "current-only format and framing names are plain identities; encoded "
        "edit counters are forbidden:\n"
        + "\n".join(violations)
    )


def test_pulse_software_formats_have_no_upgrade_or_edit_counter():
    """The remaining legacy island is current-only until its final H1 cut.

    Its three persisted software values may retain a plain schema name, but
    must not regrow a numeric revision field or a historical parser. Hardware
    ABI and the deployed runtime wire version are intentionally out of scope.
    """

    # The pulse domain moved to zlc_neutral_atom.timing (shims stay behind); the
    # owner pins follow the code, and the deleted-upgrader ratchet covers BOTH homes.
    for timing_root in (
        ROOT / "Zou_lab_control" / "neutral_atom" / "timing",
        ROOT / "zlc_neutral_atom" / "timing",
    ):
        upgrader = timing_root / "legacy_pulse_upgrade.py"
        assert not upgrader.exists(), "historical pulse upgrade code must stay deleted"

    timing_root = ROOT / "zlc_neutral_atom" / "timing"
    owners = (
        (timing_root / "ports.py", "PortCatalog"),
        (timing_root / "sequence_model.py", "PulseSequence"),
        (timing_root / "pulse_table.py", "PulseTableState"),
    )
    violations = []
    for path, class_name in owners:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        owner = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        for node in ast.walk(owner):
            if isinstance(node, ast.Constant) and node.value == "version":
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} persists a version field"
                )
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if any(isinstance(target, ast.Name) and target.id == "version" for target in targets):
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} defines a version attribute"
                    )

    for path in (ROOT / "Zou_lab_control").rglob("*.py"):
        if "legacy_pulse_upgrade" in path.read_text(encoding="utf-8"):
            violations.append(f"{path.relative_to(ROOT)} imports the deleted upgrader")

    assert not violations, (
        "legacy pulse software formats are current-only; use their plain schema "
        "name and reject every other shape:\n" + "\n".join(violations)
    )


def test_legacy_frontend_persisted_formats_have_no_edit_counter_or_single_role():
    """Owners are resolved by DEFINITION SITE, never by path -- see ``_sole_definition``.

    The two console records follow their class definitions, so the guard survives
    a salvage instead of being silently orphaned by it.  Persisted figures now use
    the exact current ``zlc_frontend.figure_archive`` envelope and owner codecs;
    the deleted frontend record therefore has no compatibility anchor here.
    """

    console_state_path, console_state_node = _sole_definition("TaskConsoleState")
    panel_config_path, panel_config_node = _sole_definition("PanelConfig")

    owners = (
        (console_state_path, console_state_node, {"version"}),
        (panel_config_path, panel_config_node, {"role"}),
    )
    violations = []
    for path, root, forbidden in owners:
        for node in ast.walk(root):
            candidate = None
            if isinstance(node, ast.Name):
                candidate = node.id
            elif isinstance(node, ast.Attribute):
                candidate = node.attr
            elif isinstance(node, ast.arg):
                candidate = node.arg
            elif isinstance(node, ast.keyword):
                candidate = node.arg
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                candidate = node.value
            if candidate in forbidden:
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} contains {candidate!r}"
                )

    assert not violations, (
        "frontend persisted records are current-only: FigureArchive and "
        "TaskConsole keep a plain schema, while single-value panel roles and "
        "numeric edit counters stay deleted:\n" + "\n".join(violations)
    )
