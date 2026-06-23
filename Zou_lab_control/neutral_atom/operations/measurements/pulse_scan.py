"""Generic "Pulse scan" measurement (auto-discovered).

A pulse template the user loads exposes TWO kinds of named handle, which are DIFFERENT
mechanisms and stay separate:

* **API slots** (``a1``, ``a2`` ...) — one numeric value the operator wants to FIX (not sweep)
  before the run.  Each slot is set ONCE at build time via :meth:`PulseTableState.set_api`.
* **Scan slots** (``s0``, ``s1`` ...) — the parameters the run SWEEPS (the x axis).  The user
  gives a list / ``linspace`` expression PER scan slot; per point the run resolves the scan
  slots to that row via :meth:`PulseTableState.with_slots_resolved` (the SAME named-slot
  resolver the hardware scan + pulse GUI use) and fires.  ``x`` = the scan points (1-D = the
  one scan slot's values; 2-D = two slots advanced in lockstep against a common index).

The ``y`` is DECOUPLED: pulse-scan does not reduce its own frames.  It is a device-driving
logic node that fires the pulse, publishes the camera ``frame`` it produces, and then reads
``y`` from a **source expression** that subscribes to a signal published by ANOTHER running
node (e.g. a Judge-occupancy processor's loading ``rate``).  So the readout pipeline lives in
its own node (calibration, detection) and pulse-scan just sweeps + records its output -- the
same decoupling the rest of the console uses.  The driving node is :class:`~..logic.PulseScanNode`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...core.analysis import positive_int
from ...timing import (
    PulseTableState,
    evaluate_scan_table_code,
    scan_table_template,
    single_imaging_template,
    snap_scan_table,
)
from ..measurement import MeasurementSpec, ParamDecl
from ..measurement_registry import measurement
from ..signal_expr import SignalExpr

#: A pulse scan images ONCE per point, so the default pulse is the SINGLE-image probe template
#: (one camera trigger) -- NOT the Calibrate task's long-short-long bracket.
DEFAULT_PROBE_TEMPLATE = "pulses/probe_template.json"

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


def _resolve_probe_template(template: str) -> PulseTableState:
    """Load the pulse template the operator picked: the given path if real, else the shipped
    template of that name in ``pulses/``, else the in-memory single-image default."""

    text = str(template or "").strip() or DEFAULT_PROBE_TEMPLATE
    path = Path(text)
    if path.is_file():
        return PulseTableState.load(path)
    name = path.name
    for base in (Path("pulses"), Path(__file__).resolve().parents[4] / "pulses"):
        shipped = base / name
        if shipped.is_file():
            return PulseTableState.load(shipped)
    return single_imaging_template()


def _scan_table_arrays(state: PulseTableState, scan_code: str) -> tuple[list[str], list[np.ndarray]]:
    """Evaluate the scan PROGRAM into per-slot arrays (one array per bound scan slot, in slot
    order).

    The program builds ONE ``(N_points x n_slots)`` table -- the slots advance in LOCKSTEP, one
    row per scan point -- which is then snapped to the hardware grid via the SAME ``snap_scan_table``
    the pulse GUI + server use (durations -> whole clock ticks, DAC -> integer codes in range).
    The points are NOT set per slot in isolation: the whole table is one object, so correlations
    between slots are expressed by how the columns are built.  Returns ``(slot names, columns)``."""

    names = [f"s{i}" for i in range(len(state.scan_slots))]
    if not names:
        raise ValueError(
            "the pulse template has no scan slot bound -- in the pulse GUI Edit tab, click the dot "
            "next to a duration or DAC value to bind a scan slot (sN), then give a scan program.")
    table = evaluate_scan_table_code(scan_code, len(names))            # N x n_slots, slot-native units
    snapped = snap_scan_table(table, state.scan_slots, time_step_ns=state.time_step_ns,
                              dac_ranges=state.scan_slot_dac_ranges())  # hardware-legal table
    columns = list(zip(*snapped))                                      # one column per slot, length N
    arrays = [np.asarray(col, dtype=float) for col in columns]
    return names, arrays


def _label_for_first_scan_slot(state: PulseTableState) -> tuple[str, str]:
    """``(axis label, axis unit)`` for the FIRST scan slot (drives the live plot's x label).

    The unit is the slot's NATIVE unit -- it is NOT assumed to be time: a duration slot's column
    is whole clock ticks in nanoseconds, a DAC slot's column is signed integer code (LSB, 0 = 0 V).
    Empty / no scan slot -> the axis is point INDEX."""

    if not state.scan_slots:
        return ("point", "")
    slot = state.scan_slots[0]
    name = slot.label or f"{slot.kind} {slot.target}"
    if slot.kind == "dac":
        return (name, "LSB")
    if slot.kind == "duration":
        return (name, "ns")                       # the snapped scan-table column is in ns
    return (name, str(getattr(slot, "unit", "") or ""))


class PulseScanPlan:
    """Everything :class:`~..logic.PulseScanNode` needs to drive a pulse-template scan.

    API slots are already FIXED on ``base_state`` (``set_api``); ``scan_names``/``scan_arrays``
    are the bound scan slots + their per-point sweep values (the x axis), resolved per point via
    ``with_slots_resolved`` (no clearing, no period editing).  ``y_expr`` is the DECOUPLED y
    source -- a :class:`~..signal_expr.SignalExpr` over signals published by other running nodes.
    ``settle`` (optional) lets a HEADLESS caller step the consumer inline for single-threaded
    determinism; in the GUI it is ``None`` and the node waits for the consumer's own thread to
    republish the y signals (a per-signal version bump)."""

    def __init__(self, base_state: PulseTableState, scan_names: Sequence[str],
                 scan_arrays: Sequence[np.ndarray], camera, sequencer, *,
                 axis_label: str = "point", axis_unit: str = "", n_frames: int = 1,
                 y_expr=None, settle=None):
        self.base_state = base_state
        self.scan_names = list(scan_names)
        self.scan_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in scan_arrays]
        self.axis = _Axis(self.scan_arrays[0] if self.scan_arrays else [0.0],
                          label=axis_label, unit=axis_unit)
        self.axis_label = str(axis_label)
        self.axis_unit = str(axis_unit)
        self.camera = camera
        self.sequencer = sequencer
        self.n_frames = positive_int(n_frames, "n_frames")
        self.y_expr = y_expr if isinstance(y_expr, SignalExpr) else SignalExpr.from_value(y_expr)
        self.settle = settle


@measurement(order=30)
def pulse_scan(readout) -> MeasurementSpec:
    """Pulse-template scan: set every API slot's value, give scan-points per scan slot (x),
    sweep the points, and record y from a source signal published by another running node."""

    s = readout.session

    def build(*, template: str = DEFAULT_PROBE_TEMPLATE, pulse_slots: Mapping | None = None,
              y=None, n_frames: int = 1, **_ignored) -> PulseScanPlan:
        state = _resolve_probe_template(template)
        spec = dict(pulse_slots or {})
        api_values: Mapping[str, float] = dict(spec.get("api") or {})
        scan_code = str(spec.get("scan_code") or "")

        # Apply api values ONCE; reject typos (set_api raises for unknown handle names).
        for name, value in api_values.items():
            if str(value).strip() == "":
                continue                                       # leave the template's value
            state.set_api(name, float(value))
        # The scan POINTS are ONE table: the scan program builds an (N_points x n_slots) array
        # (one row per point, one column per slot) -- the slots advance in LOCKSTEP, snapped to the
        # hardware grid in each slot's NATIVE unit (ns ticks for a duration, integer code for a DAC).
        # Empty program -> the column_stack default for the bound slot count.
        if not scan_code.strip():
            scan_code = scan_table_template("column_stack", len(state.scan_slots))
        scan_names, scan_arrays = _scan_table_arrays(state, scan_code)
        axis_label, axis_unit = _label_for_first_scan_slot(state)
        y_expr = SignalExpr.from_value(y if y is not None else DEFAULT_Y_SOURCE)
        return PulseScanPlan(
            state, scan_names, scan_arrays, s.devices.camera, s.devices.sequencer,
            axis_label=axis_label, axis_unit=axis_unit, n_frames=n_frames, y_expr=y_expr)

    params = (
        ParamDecl("template", "Pulse template", "path", default=DEFAULT_PROBE_TEMPLATE,
                  path_mode="file", base_dir="pulses", file_filter="Pulse program (*.json);;All files (*)",
                  tooltip="The pulse program fired each point.  Its api / scan slots populate the "
                          "auto-form below (one numeric input per api slot + ONE scan-table program)."),
        ParamDecl("pulse_slots", "Slots", "pulse_slots", default={}, depends_on="template",
                  tooltip="One numeric input per API slot (the fixed operator-set value), and ONE scan "
                          "program that builds the (N_points x n_slots) scan table -- one row per point, "
                          "one column per scan slot, advanced in lockstep (this is x).  Same model as the "
                          "pulse GUI Scan tab (column_stack / grid templates)."),
        ParamDecl("y", "Signal (y)", "signal_expr", default=dict(DEFAULT_Y_SOURCE),
                  tooltip="The y recorded per point.  Pick a signal published by ANOTHER running node "
                          "(e.g. a Judge-occupancy processor's loading 'rate') and combine via value = ... .  "
                          "Start that producer FIRST; pulse-scan fires the pulse, publishes the frame it "
                          "produces, then reads this signal back for the point's y."),
        ParamDecl("n_frames", "Frames / point", "int", default=1, lo=1, hi=1000,
                  tooltip="Camera triggers (frames) acquired per scan point before the y signal is read."),
    )
    return MeasurementSpec(
        name="Pulse scan", key="pulse_scan", params=params,
        result_labels=("Pulse parameter", "Signal"),
        x_key="param", y_key="signal", build=build,
        # The console builds a device-driving PulseScanNode (not a ScannedMeasurementNode):
        # it fires + publishes a frame + reads y from a subscribed signal, per point.
        metadata={"node": "pulse_scan"})
