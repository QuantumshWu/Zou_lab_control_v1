"""Pulse-table scan measurement (auto-discovered).

The pulse template is the single source of two intentionally different sweep mechanisms:

* ``scan_slot`` sweeps the template's hardware scan slots.  One complete
  ``(points, slots)`` table is attached to :class:`PulseTableState`, uploaded once, and streamed
  by the sequencer.
* ``api_slot`` sweeps the template's API handles.  Each row is resolved into one finite pulse and
  fired in software; API values remain in their declared native units.

There is no camera role and no frame relay.  A separately running measurement/processor pipeline
publishes the chosen ``y`` signal; :class:`~..logic.PulseScanNode` samples one fresh,
lineage-coherent update per point.  Both mechanisms share the same coordinate and signal-consumer
contract; only their sequencer execution strategy differs.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ...timing import (
    PROBE_TEMPLATE_PATH,
    PulseTableState,
    evaluate_scan_table_code,
    resolve_fireable_template,
    scan_table_template,
    scan_target_label,
    single_imaging_template,
    snap_scan_table,
)
from ...core.params import ParamDecl
from ..measurement import (
    NODE_META_KEY,
    PULSE_SCAN_NODE,
    PULSE_SWEEP_KINDS,
    SWEEP_API_SLOT,
    SWEEP_SCAN_SLOT,
    MeasurementSpec,
)
from ..measurement import measurement_slug
from ..measurement_registry import measurement
from ..signal_expr import SignalExpr

#: The generic scan needs a compact, inspectable pulse program to sweep.  The shipped probe
#: template is that default; acquiring and reducing the external y signal remain the independent
#: producer pipeline's responsibility.
DEFAULT_PROBE_TEMPLATE = PROBE_TEMPLATE_PATH

#: Default y source: the loading ``rate`` published by a Judge-occupancy processor (the common
#: case -- sweep a pulse parameter, watch the loading fraction).  The operator re-picks it in
#: the form / notebook to any running node's signal.
DEFAULT_Y_SOURCE: Mapping[str, object] = {"inputs": ["rate"], "source": "value = signal"}

class _Axis:
    """The scan-axis the live node reads (``values`` + plot labels)."""

    def __init__(self, values, *, label: str = "x", unit: str = ""):
        self.values = np.asarray(values, dtype=float).reshape(-1)
        self.label = str(label)
        self.unit = str(unit)


def _resolve_probe_template(template: str, sequencer=None) -> PulseTableState:
    """The PROBE-flavoured fireable-template loader: the ONE timing-layer loader
    (:func:`~...timing.resolve_fireable_template` -- resolve + force the hardware tick, so
    author == snap == fire) with this measurement's single-image defaults bound
    (``PROBE_TEMPLATE_PATH`` / :func:`single_imaging_template`).  All the load/snap
    mechanics live in the timing helper; only the default name/factory are chosen here."""
    return resolve_fireable_template(template, default_name=DEFAULT_PROBE_TEMPLATE,
                                     default_factory=single_imaging_template, sequencer=sequencer)


def _unique_names(candidates: Sequence[str]) -> list[str]:
    """Normalize coordinate candidates and disambiguate them deterministically."""

    names: list[str] = []
    counts: dict[str, int] = {}
    for candidate in candidates:
        base = measurement_slug(candidate) or "scan_parameter"
        counts[base] = counts.get(base, 0) + 1
        names.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return names


def _semantic_scan_names(state: PulseTableState) -> list[str]:
    """Return the authoritative public identities without deriving a second namespace."""

    return list(state.scan_names)


def _semantic_api_names(state: PulseTableState) -> list[str]:
    """Meaningful public coordinates for API handles.

    ``a1``/``a2`` are mutation handles, not experiment vocabulary.  A sweep therefore publishes
    the bound pulse field (``da_x``, ``probe_duration``, ``camera_delay``), while retaining the API
    handle separately for :meth:`PulseTableState.set_api`.
    """

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


def _scan_table_arrays(
    state: PulseTableState, scan_code: str | None,
) -> tuple[list[str], list[np.ndarray], tuple[int, ...] | None]:
    """Attach one hardware-legal scan table to ``state`` and expose its semantic axes.

    Source precedence is deliberate: an explicit task-console program, then the program persisted
    in the selected template, then the template's already-materialised table, and finally the shared
    starter program.  Whichever source wins is snapped once and written back to the same
    :class:`PulseTableState` that the node later passes to ``sequencer.prepare``.
    """

    if not state.scan_slots:
        raise ValueError(
            "Pulse scan selected a scan-slot sweep, but the pulse template has no scan slot. "
            "Bind one in the Pulse GUI or select the API-slot sweep strategy.")

    code = str(scan_code or "").strip() or str(getattr(state, "scan_code", "") or "").strip()
    scan_shape: tuple[int, ...] | None = None
    if code:
        table, scan_shape = evaluate_scan_table_code(code, len(state.scan_slots))
    elif state.scan_table:
        table = [list(row) for row in state.scan_table]
    else:
        code = scan_table_template("column_stack", state.scan_column_specs())
        table, scan_shape = evaluate_scan_table_code(code, len(state.scan_slots))

    snapped = snap_scan_table(
        table, state.scan_slots, time_step_ns=state.time_step_ns,
        dac_ranges=state.scan_slot_dac_ranges())
    state.set_scan_table(snapped)
    state.scan_code = code
    matrix = np.asarray(state.scan_table, dtype=float)
    arrays = [matrix[:, index].copy() for index in range(matrix.shape[1])]
    return _semantic_scan_names(state), arrays, scan_shape


def _api_table_arrays(
    state: PulseTableState, program: str | None,
) -> tuple[list[str], list[str], list[np.ndarray], tuple[int, ...] | None]:
    """Evaluate one API-slot sweep program without hardware-table snapping.

    Returns public semantic coordinate names, private API mutation handles, aligned columns, and
    optional logical point geometry.  Keeping the two identities separate prevents opaque ``aN``
    handles from leaking into the signal namespace.
    """

    if not state.api_slots:
        raise ValueError(
            "Pulse scan selected an API-slot sweep, but the pulse template has no API slot.")
    code = str(program or "").strip()
    if not code:
        code = scan_table_template("column_stack", state.api_column_specs())
    table, scan_shape = evaluate_scan_table_code(code, len(state.api_slots))
    matrix = np.asarray(table, dtype=float)
    arrays = [matrix[:, index].copy() for index in range(matrix.shape[1])]
    return (
        _semantic_api_names(state),
        [str(slot.name) for slot in state.api_slots],
        arrays,
        scan_shape,
    )


def _label_for_first_scan_slot(state: PulseTableState) -> tuple[str, str]:
    """``(axis label, axis unit)`` for the FIRST scan slot (drives the live plot's x label).

    The unit is the slot's NATIVE unit -- it is NOT assumed to be time: a duration slot's column
    is whole clock ticks in nanoseconds, a DAC slot's column is signed integer code (LSB, 0 = 0 V).
    Empty / no scan slot -> the axis is point INDEX."""

    if not state.scan_slots:
        return ("point", "")
    slot = state.scan_slots[0]
    # Readable axis name: the slot's own label, else a prettified "<bus> @ <period name>" /
    # "<period name> duration" (never the opaque "bus@<index>") via the single-source helper.
    name = slot.label or scan_target_label(state, slot.kind, slot.target)
    if slot.kind == "dac":
        return (name, "LSB")
    if slot.kind == "duration":
        return (name, "ns")                       # the snapped scan-table column is in ns
    return (name, str(getattr(slot, "unit", "") or ""))


def _label_for_first_api_slot(state: PulseTableState) -> tuple[str, str]:
    """Human axis label and native unit for the first swept API handle."""

    if not state.api_slots:
        return ("point", "")
    slot = state.api_slots[0]
    return (
        scan_target_label(state, slot.kind, slot.target),
        str(getattr(slot, "unit", "") or ""),
    )


class PulseScanPlan:
    """Immutable run inputs for :class:`~..logic.PulseScanNode`.

    ``sweep_kind`` chooses exactly one execution strategy.  ``scan_names`` are public semantic
    coordinates; ``api_handles`` are present only for an API sweep and are private mutation keys.
    A scan-slot plan's ``pulse_state`` already contains the complete snapped hardware table.
    """

    def __init__(self, pulse_state: PulseTableState, scan_names: Sequence[str],
                 scan_arrays: Sequence[np.ndarray], sequencer, *,
                 axis_label: str = "point", axis_unit: str = "", y_key: str = "signal",
                 y_expr=None, scan_shape: tuple[int, ...] | None = None,
                 sweep_kind: str = SWEEP_SCAN_SLOT, api_handles: Sequence[str] = ()):
        self.pulse_state = pulse_state
        self.scan_names = list(scan_names)
        self.scan_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in scan_arrays]
        if len(self.scan_names) != len(self.scan_arrays) or not self.scan_arrays:
            raise ValueError("pulse scan plan needs one semantic axis per bound scan-table column.")
        lengths = {int(a.size) for a in self.scan_arrays}
        if len(lengths) != 1 or next(iter(lengths), 0) < 1:
            raise ValueError("pulse scan coordinate columns must be non-empty and aligned.")
        self.sweep_kind = str(sweep_kind)
        if self.sweep_kind not in PULSE_SWEEP_KINDS:
            raise ValueError(
                f"pulse sweep kind must be one of {PULSE_SWEEP_KINDS}, got {self.sweep_kind!r}.")
        self.api_handles = [str(name) for name in api_handles]
        if self.sweep_kind == SWEEP_API_SLOT:
            if len(self.api_handles) != len(self.scan_arrays):
                raise ValueError("an API-slot sweep needs one API handle per coordinate column.")
        elif self.api_handles:
            raise ValueError("a scan-slot sweep cannot carry API mutation handles.")
        if self.sweep_kind == SWEEP_SCAN_SLOT \
                and next(iter(lengths)) != len(self.pulse_state.scan_table):
            raise ValueError(
                "scan-slot axes must all have the same length as pulse_state.scan_table.")
        primary = self.scan_arrays[0]
        self.axis = _Axis(primary, label=axis_label, unit=axis_unit)
        self.axis_label = str(axis_label)
        self.axis_unit = str(axis_unit)
        self.sequencer = sequencer
        # The OUTPUT signal name (user-set #7): the node publishes y under this key.
        self.y_key = str(y_key or "signal").strip() or "signal"
        # Optional logical grid geometry.  The unique y signal remains physical
        # ``(R,P,1)``; SignalSchema.point_shape carries this geometry so a view
        # can unflatten P without publishing a duplicate array.
        self.scan_shape = scan_shape
        self.y_expr = y_expr if isinstance(y_expr, SignalExpr) else SignalExpr.from_value(y_expr)


@measurement(order=30)
def pulse_scan(readout) -> MeasurementSpec:
    """Sweep either scan slots or API slots and record an external y per point."""

    s = readout.session

    def build(*, template: str = DEFAULT_PROBE_TEMPLATE, pulse_slots: Mapping | None = None,
              y=None, y_name: str = "signal") -> PulseScanPlan:
        state = _resolve_probe_template(template, sequencer=s.devices.sequencer)
        spec = dict(pulse_slots or {})
        unknown = sorted(set(spec) - {"program_id", "api", "sweep_kind", "program"})
        if unknown:
            raise ValueError(
                f"unsupported pulse-scan field(s) {unknown}; expected only program_id, api, "
                "sweep_kind, program.")
        api_values: Mapping[str, float] = dict(spec.get("api") or {})
        sweep_kind = str(spec.get("sweep_kind") or "").strip()
        if not sweep_kind:
            sweep_kind = SWEEP_SCAN_SLOT if state.scan_slots else (
                SWEEP_API_SLOT if state.api_slots else "")
        if sweep_kind not in PULSE_SWEEP_KINDS:
            raise ValueError(
                f"pulse sweep kind must be one of {PULSE_SWEEP_KINDS}; got {sweep_kind!r}.")
        program = str(spec.get("program") or "")

        # Apply the FIXED api values ONCE; reject typos (set_api raises for unknown handle names).
        for name, value in api_values.items():
            if str(value).strip() == "":
                continue                                       # leave the template's value
            state.set_api(name, float(value))

        api_handles: list[str] = []
        if sweep_kind == SWEEP_SCAN_SLOT:
            scan_names, scan_arrays, scan_shape = _scan_table_arrays(state, program)
            axis_label, axis_unit = _label_for_first_scan_slot(state)
        else:
            scan_names, api_handles, scan_arrays, scan_shape = _api_table_arrays(state, program)
            axis_label, axis_unit = _label_for_first_api_slot(state)
        y_expr = SignalExpr.from_value(y if y is not None else DEFAULT_Y_SOURCE)
        return PulseScanPlan(
            state, scan_names, scan_arrays, s.devices.sequencer,
            axis_label=axis_label, axis_unit=axis_unit, y_key=y_name, y_expr=y_expr,
            scan_shape=scan_shape, sweep_kind=sweep_kind, api_handles=api_handles)

    params = (
        ParamDecl("template", "Pulse template", "path", default=DEFAULT_PROBE_TEMPLATE,
                  path_mode="file", base_dir="pulses", file_filter="Pulse program (*.json);;All files (*)",
                  tooltip="The complete hardware pulse program. Its fixed API values and persisted "
                          "scan-slot program populate the form below."),
        ParamDecl("pulse_slots", "Slots", "pulse_slots", default={}, depends_on="template",
                  tooltip="One resting value per API slot plus one sweep program. Select scan-slot "
                          "execution for a single FPGA table or API-slot execution for one finite "
                          "software-submitted pulse per row; all selected columns advance together."),
        ParamDecl("y", "Signal (y)", "signal_expr", default=dict(DEFAULT_Y_SOURCE),
                  tooltip="The y recorded per point.  Pick a signal published by ANOTHER running node "
                          "(e.g. a Judge-occupancy processor's loading 'rate') and combine via value = ... . "
                          "Start the selected producer/processor pipeline first; Pulse scan owns only "
                          "the sequencer and samples each fresh lineage once."),
        ParamDecl("y_name", "Output name", "text", default="signal",
                  tooltip="Name of the OUTPUT signal this scan publishes (the y curve), e.g. "
                          "'loading_rate' or 'survival' -- so a plot picks a meaningful name."),
    )
    return MeasurementSpec(
        name="Pulse scan", key="pulse_scan", params=params,
        result_labels=("Pulse parameter", "Signal"),
        x_key="param", y_key="signal", build=build,
        # The console builds the dedicated sequencer-driving, signal-consuming PulseScanNode.
        metadata={NODE_META_KEY: PULSE_SCAN_NODE})
