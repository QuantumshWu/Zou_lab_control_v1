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


def _unique_names(candidates):
    from zlc_data.shape_text import measurement_slug

    names, counts = [], {}
    for candidate in candidates:
        base = measurement_slug(candidate) or "scan_parameter"
        counts[base] = counts.get(base, 0) + 1
        names.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return names


def _semantic_api_names(state) -> list[str]:
    """Meaningful public coordinates for API handles.

    ``a1``/``a2`` are mutation handles, not experiment vocabulary; a sweep
    publishes the bound pulse field (``da_x``, ``probe_duration``, ...) while
    the handle stays available for ``PulseTableState.set_api``.
    """

    from zlc_neutral_atom.timing.pulse_table import scan_target_label

    candidates: list[str] = []
    for slot in state.api_slots:
        target = str(slot.target)
        if slot.kind == "dac":
            candidate = target.split("@", 1)[0]
        elif slot.kind == "duration":
            try:
                period = state.periods[int(target)]
                candidate = f"{period.name or f'period_{target}'}_duration"
            except (IndexError, TypeError, ValueError):
                candidate = f"period_{target}_duration"
        elif slot.kind == "delay":
            candidate = f"{target}_delay"
        else:
            candidate = scan_target_label(state, slot.kind, target)
        candidates.append(candidate)
    return _unique_names(candidates)


def _read_pulse_template(path) -> PulseTemplateRows:
    """Resolve a saved template and describe its slots as plain rows.

    The derivation is pulse-domain (bus signed range, unit table, clock tick),
    so it runs on THIS side of the seam; the form gets str/float rows it can
    draw without importing the compiler.
    """

    import hashlib
    import json

    from zlc_neutral_atom.timing.pulse_table import (
        PROBE_TEMPLATE_PATH,
        resolve_fireable_template,
        scan_column_spec,
        single_imaging_template,
    )

    state = resolve_fireable_template(path, default_name=PROBE_TEMPLATE_PATH,
                                      default_factory=single_imaging_template)
    api_names = _semantic_api_names(state)
    api_rows = tuple(
        (slot.name, api_names[index], str(slot.kind), str(slot.target), str(slot.unit),
         float(state._read_api_field(slot)))
        for index, slot in enumerate(state.api_slots)
    )
    scan_names = list(state.scan_names)
    scan_rows = tuple(
        (scan_names[i], str(s.kind), str(s.target), str(s.unit), s.label)
        for i, s in enumerate(state.scan_slots)
    )
    code = str(getattr(state, "scan_code", "") or "")
    if not code.strip() and getattr(state, "scan_table", None):
        code = ("scan_table = np.array("
                + repr([list(row) for row in state.scan_table]) + ", dtype=float)")
    payload = state.to_dict()
    program_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    api_columns = tuple(
        scan_column_spec(coordinate, "dac" if kind == "dac" else "duration",
                         unit=(unit or "ns"))
        for _handle, coordinate, kind, _target, unit, _current in api_rows
    )
    scan_columns = tuple(
        scan_column_spec(coordinate, kind, unit=(unit or "ns"))
        for coordinate, kind, _target, unit, _label in scan_rows
    )
    return PulseTemplateRows(api_rows=api_rows, scan_rows=scan_rows,
                             api_columns=api_columns, scan_columns=scan_columns,
                             program=code, program_id=program_id)


register_pulse_state_factory(_pulse_state_from_dict)
register_pulse_template_reader(_read_pulse_template)
