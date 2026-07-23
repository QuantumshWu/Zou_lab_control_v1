"""The desktop composition package wires the render layer's domain ports.

This is the call the legacy frontend package's ``__init__`` promised to
``zlc_workbench`` at Z0 ("zlc_workbench inherits exactly this call"): the one
module that sees both the window shells and the pulse domain registers the two
:mod:`zlc_frontend.domain_ports` implementations, so a saved pulse figure
replays and the pulse-slots form reads templates -- without ``zlc_frontend``
ever importing the pulse compiler itself.

Both factories stay LAZY on purpose: importing this package must not pay for
the 3k-line pulse compiler; the domain loads on the first replay/template read.
"""

from __future__ import annotations

from zlc_frontend.domain_ports import (
    PulseTemplateRows,
    register_pulse_state_factory,
    register_pulse_template_reader,
)


def _pulse_state_from_dict(data):
    from zlc_neutral_atom.timing.pulse_table import PulseTableState

    return PulseTableState.from_dict(data)


def _slot_target(document, field) -> str:
    """Return the shared, index-based target spelling used by ``slot_label``."""

    period_indices = {
        period.period_id: index for index, period in enumerate(document.periods)
    }
    if field.kind == "duration":
        return str(period_indices[field.period_id])
    if field.kind == "dac":
        return f"{field.port}@{period_indices[field.period_id]}"
    return str(field.port)


def _value_in_parameter_unit(document, parameter) -> int | float:
    value, authored_unit = document.field_value(parameter.field)
    if parameter.field.kind not in ("duration", "delay"):
        return value
    from zlc_pulse import TIME_UNIT_TO_NS

    return (
        float(value)
        * TIME_UNIT_TO_NS[authored_unit]
        / TIME_UNIT_TO_NS[parameter.unit]
    )


def _api_column_specs(document):
    """Derive starter ranges from the current PulseDocument's real fields."""

    from zlc_data.scan_template import ScanColumnSpec
    from zlc_pulse import FIELD_DAC, TIME_UNIT_TO_NS

    result = []
    for parameter in document.api_parameters:
        value = _value_in_parameter_unit(document, parameter)
        if parameter.field.kind == FIELD_DAC:
            port = document.target.by_key[parameter.field.port]
            assert port.signed_range is not None
            lo, hi = port.signed_range
            result.append(
                ScanColumnSpec(
                    parameter.parameter_id,
                    float(lo),
                    float(hi),
                    is_dac=True,
                    unit="value",
                    label=parameter.parameter_id,
                )
            )
            continue
        tick = float(document.time_step_ns) / TIME_UNIT_TO_NS[parameter.unit]
        result.append(
            ScanColumnSpec(
                parameter.parameter_id,
                tick,
                max(float(value) * 2.0, 100.0 * tick),
                unit=parameter.unit,
                label=parameter.parameter_id,
            )
        )
    return tuple(result)


def _read_pulse_template(path) -> PulseTemplateRows:
    """Resolve a saved template and describe its slots as plain rows.

    The derivation is pulse-domain (bus signed range, unit table, clock tick),
    so it runs on THIS side of the seam; the form gets str/float rows it can
    draw without importing the compiler.
    """

    import hashlib
    from pathlib import Path

    from zlc_neutral_atom.pulse_programs import DEFAULT_PROBE_PULSE_PATH
    from zlc_pulse import load_pulse_document
    from zlc_storage.paths import PROJECT_ROOT, project_path
    from zlc_workbench.pulse_editor.scan_workspace import scan_column_specs

    source_text = str(path or "").strip() or DEFAULT_PROBE_PULSE_PATH
    document = load_pulse_document(source_text)
    api_rows = tuple(
        (
            parameter.parameter_id,
            parameter.parameter_id,
            parameter.field.kind,
            _slot_target(document, parameter.field),
            parameter.unit,
            _value_in_parameter_unit(document, parameter),
        )
        for parameter in document.api_parameters
    )
    scan_rows = tuple(
        (
            parameter.parameter_id,
            parameter.field.kind,
            _slot_target(document, parameter.field),
            parameter.unit,
            parameter.label,
        )
        for parameter in document.scan_parameters
    )
    code = (
        ""
        if document.scan_recipe is None
        else str(document.scan_recipe.source)
    )
    if not code.strip() and document.scan_table is not None:
        code = ("scan_table = np.array("
                + repr([list(row) for row in document.scan_table.rows])
                + ", dtype=float)")
    # ``program_id`` is the stable identity of the source document, not a digest
    # of its current contents.  A content digest changes precisely when a slot is
    # inserted/moved/edited and would force the frontend to throw away every
    # unaffected ``(program_id, slot_id)`` row.  Project-relative identity also
    # survives moving the checkout; external files retain their canonical path.
    project_root = PROJECT_ROOT.resolve(strict=False)
    candidate = Path(source_text)
    source_path = candidate if candidate.is_file() else None
    if source_path is None:
        for base in (Path("pulses"), project_path("pulses")):
            shipped = base / candidate.name
            if shipped.is_file():
                source_path = shipped
                break
    if source_path is None:
        source_key = "declared:" + Path(source_text).as_posix()
    else:
        source_path = source_path.resolve(strict=False)
        try:
            source_key = "project:" + source_path.relative_to(project_root).as_posix()
        except ValueError:
            source_key = "external:" + source_path.as_posix()
    program_id = hashlib.sha256(
        source_key.encode("utf-8")
    ).hexdigest()
    api_columns = _api_column_specs(document)
    scan_columns = scan_column_specs(document)
    return PulseTemplateRows(api_rows=api_rows, scan_rows=scan_rows,
                             api_columns=api_columns, scan_columns=scan_columns,
                             program=code, program_id=program_id)


register_pulse_state_factory(_pulse_state_from_dict)
register_pulse_template_reader(_read_pulse_template)
