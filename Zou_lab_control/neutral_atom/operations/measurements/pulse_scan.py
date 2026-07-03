"""Generic "Pulse scan" measurement (auto-discovered).

A pulse template the user loads exposes TWO kinds of named handle:

* **API slots** (``a1``, ``a2`` ...) — one numeric value the operator wants to FIX before the
  run.  Each slot is set ONCE at build time via :meth:`PulseTableState.set_api`.
* **Scan slots** (``s0``, ``s1`` ...) — handles bound to a HARDWARE scan column (the FPGA scan
  table), resolved per point via :meth:`PulseTableState.with_slots_resolved` (the SAME named-slot
  resolver the hardware scan + pulse GUI use).

A pulse scan runs in ONE of three :data:`SCAN_MODES` -- a single ``scan_mode`` in the spec picks
WHAT a single shared scan table sweeps and HOW the device runs it:

* ``"scan"`` — sweep the bound hardware scan slots (one ``(N x n_slots)`` table snapped to the
  FPGA grid, slots advancing in lockstep).  ``x`` = the scan points (1-D = one slot; 2-D = a grid).
* ``"api"`` — SOFTWARE sweep of the API slots (the analogue table, native units, no snap): per
  point the node ``set_api``'s that row, fires, waits the pulse + an extra settle, then advances.
* ``"none"`` — no sweep: fire ONE point at the fixed api values (a single-point measurement).

The fixed api values are ALWAYS applied first; the mode only decides whether (and how) to sweep.

The ``y`` is DECOUPLED: pulse-scan does not reduce its own frames.  It is a device-driving
logic node that fires the pulse, publishes the camera ``frame`` it produces, and then reads
``y`` from a **source expression** that subscribes to a signal published by ANOTHER running
node (e.g. a Judge-occupancy processor's loading ``rate``).  So the readout pipeline lives in
its own node (calibration, detection) and pulse-scan just sweeps + records its output -- the
same decoupling the rest of the console uses.  The driving node is :class:`~..logic.PulseScanNode`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ...timing import (
    PulseTableState,
    evaluate_scan_table_code,
    resolve_pulse_template,
    scan_table_template,
    scan_target_label,
    single_imaging_template,
    snap_scan_table,
)
from ..measurement import MeasurementSpec, ParamDecl
from ..measurement_registry import measurement
from ..signal_expr import SignalExpr

#: The THREE pulse-scan modes -- the SINGLE source for the name strings the widget toggle, the
#: build() dispatch, and the tests all share (so "none"/"api"/"scan" is typed in exactly one place):
#:   * ``"none"`` -- no sweep; fire ONE point at the fixed api values (a single-point measurement).
#:   * ``"api"``  -- SOFTWARE sweep of the API slots (load -> on_pulse -> settle -> next), no snap.
#:   * ``"scan"`` -- HARDWARE sweep of the bound scan slots (the FPGA scan table), snapped to the grid.
SCAN_MODE_NONE = "none"
SCAN_MODE_API = "api"
SCAN_MODE_SCAN = "scan"
SCAN_MODES: tuple[str, str, str] = (SCAN_MODE_NONE, SCAN_MODE_API, SCAN_MODE_SCAN)

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
    """Load the pulse template the operator picked via the ONE shared resolver: the given path if
    real, else the same-named file shipped under the project ``pulses/`` folder, else the in-memory
    single-image default."""

    return resolve_pulse_template(template, default_name=DEFAULT_PROBE_TEMPLATE,
                                  default_factory=single_imaging_template)


def _scan_table_arrays(
    state: PulseTableState, scan_code: str,
) -> tuple[list[str], list[np.ndarray], tuple[int, int] | None]:
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
    table, scan_shape = evaluate_scan_table_code(scan_code, len(names))   # N x n_slots + optional grid shape
    snapped = snap_scan_table(table, state.scan_slots, time_step_ns=state.time_step_ns,
                              dac_ranges=state.scan_slot_dac_ranges())  # hardware-legal table
    columns = list(zip(*snapped))                                      # one column per slot, length N
    arrays = [np.asarray(col, dtype=float) for col in columns]
    return names, arrays, scan_shape


def _api_table_arrays(state: PulseTableState, api_code: str) -> tuple[list[str], list[np.ndarray], tuple[int, ...] | None]:
    """Evaluate the SOFTWARE api-sweep program into per-api-slot arrays (one column per API slot,
    in slot order).

    The api sweep is the software analogue of the hardware scan table: ONE ``(N_points x n_api)``
    program (the api slots advance in lockstep, one row per point).  Values are in each slot's
    NATIVE unit (``set_api`` interprets them: a duration api slot's value in its unit, a DAC api
    slot's value in signed LSB) -- there is NO hardware snap here (set_api snaps a duration to the
    clock grid itself).  Returns ``(api names, columns, declared scan_shape or None)`` -- the same
    optional grid declaration the hardware table carries, so a multi-axis api sweep (e.g. the x/y/z
    coil-field grid) facets into the SAME grid display a hardware scan does."""

    names = [s.name for s in state.api_slots]
    if not names:
        raise ValueError(
            "the pulse template has no API slot to sweep -- in the pulse GUI Edit tab, click a "
            "duration / DAC cell to its API (purple) state to expose a named handle aN, then give "
            "an api-sweep program here.")
    table, scan_shape = evaluate_scan_table_code(api_code, len(names))   # N x n_api + optional grid shape
    columns = list(zip(*table))                                          # one column per api slot
    arrays = [np.asarray(col, dtype=float) for col in columns]
    return names, arrays, scan_shape


def _resolve_scan_mode(spec: Mapping, state: PulseTableState) -> str:
    """The pulse-scan mode ("none" | "api" | "scan") -- ONE place that picks WHAT is swept.

    A form built on the new schema sets ``scan_mode`` explicitly.  A LEGACY saved task (pre-toggle)
    has no ``scan_mode`` key but may carry the old separate programs (``scan_code`` for the hardware
    scan + ``api_scan`` for the software api sweep).  Infer it ONCE here (no dual runtime path):
      * a hardware ``scan_code`` + bound scan slots -> "scan" (the old default when slots existed);
      * else an old ``api_scan`` program -> "api";
      * else "none" (nothing to sweep -> a single fixed point)."""

    mode = str(spec.get("scan_mode") or "").strip().lower()
    if mode in SCAN_MODES:
        return mode
    # ---- legacy inference (no scan_mode key) -- map the OLD two-table schema onto a single mode.
    legacy_scan = str(spec.get("scan_code") or "").strip()
    legacy_api = str(spec.get("api_scan") or "").strip()
    if state.scan_slots and legacy_scan:
        return SCAN_MODE_SCAN
    if legacy_api:
        return SCAN_MODE_API
    # An old task that bound scan slots but left the program blank still meant a hardware scan
    # (the build seeded the column_stack default), so honour that too.
    if state.scan_slots:
        return SCAN_MODE_SCAN
    if state.api_slots and legacy_scan:
        # the only program present is under scan_code but there are no scan slots -> it was an api one
        return SCAN_MODE_API
    return SCAN_MODE_NONE


def _label_for_first_api_slot(state: PulseTableState) -> tuple[str, str]:
    """``(axis label, unit)`` for the FIRST API slot (drives the x label of an api-only sweep)."""

    if not state.api_slots:
        return ("point", "")
    slot = state.api_slots[0]
    name = slot.name + ": " + scan_target_label(state, slot.kind, slot.target)
    return (name, str(getattr(slot, "unit", "") or ""))


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
                 axis_label: str = "point", axis_unit: str = "", y_key: str = "signal",
                 y_expr=None, scan_shape: tuple[int, ...] | None = None, settle=None,
                 api_names: Sequence[str] = (), api_arrays: Sequence[np.ndarray] = (),
                 extra_delay_s: float = 0.0):
        self.base_state = base_state
        self.scan_names = list(scan_names)
        self.scan_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in scan_arrays]
        # SOFTWARE api-slot sweep (the analogue of the hardware scan table): per point the node
        # deep-copies the base state, calls set_api for each api column, fires, waits the pulse +
        # extra_delay_s, then advances.  The DEVICE owns the inter-point wait (sequencer.settle).
        self.api_names = list(api_names)
        self.api_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in api_arrays]
        self.extra_delay_s = max(0.0, float(extra_delay_s))
        # The whole-sweep count is NOT carried on the plan: a pulse-scan pass IS a whole sweep, so the
        # measurement's ONE ``repeat`` knob (0 = ∞), injected by the console into PulseScanNode, is the
        # sweep count.  PulseScanNode bounds the number of point-by-point passes by that ``repeat``.
        # The x axis is the swept dimension: the hardware scan slots if any, else the software api
        # sweep, else a single point.
        primary = self.scan_arrays[0] if self.scan_arrays else (
            self.api_arrays[0] if self.api_arrays else [0.0])
        self.axis = _Axis(primary, label=axis_label, unit=axis_unit)
        self.axis_label = str(axis_label)
        self.axis_unit = str(axis_unit)
        self.camera = camera
        self.sequencer = sequencer
        # The OUTPUT signal name (user-set #7): the node publishes y under this key.
        self.y_key = str(y_key or "signal").strip() or "signal"
        # Optional grid shape (n0, n1) for a 2-D scan: the node publishes the raw ``<y_key>_grid``
        # ``(repeat, n0, n1)`` block (a pure reshape of its raw y, no combine) so a 2-D image panel
        # reduces the repeat axis then shows the (n0, n1) map.  None = 1-D.
        self.scan_shape = scan_shape
        self.y_expr = y_expr if isinstance(y_expr, SignalExpr) else SignalExpr.from_value(y_expr)
        self.settle = settle


@measurement(order=30, devices=["camera"])
def pulse_scan(readout) -> MeasurementSpec:
    """Pulse-template scan: set every API slot's value, give scan-points per scan slot (x),
    sweep the points, and record y from a source signal published by another running node.

    The camera is a declared device ROLE (``devices=["camera"]``): the base appends the camera
    dropdown to this form and injects the RESOLVED device into ``build`` -- so this factory
    never spells ``choices=camera_names()`` or ``s.devices[name]`` itself (any camera in the
    config sweeps the same way, and a new device domain needs no edit here)."""

    s = readout.session

    def build(*, template: str = DEFAULT_PROBE_TEMPLATE, pulse_slots: Mapping | None = None,
              camera=None,
              y=None, y_name: str = "signal", **_ignored) -> PulseScanPlan:
        state = _resolve_probe_template(template)
        spec = dict(pulse_slots or {})
        api_values: Mapping[str, float] = dict(spec.get("api") or {})
        scan_code = str(spec.get("scan_code") or "")
        extra_delay_s = max(0.0, float(spec.get("extra_delay") or 0.0))
        # The mode is the SINGLE place that picks WHAT is swept: "scan" (hardware scan slots, snap),
        # "api" (software api sweep, no snap), "none" (one fixed point).  ONE scan_code (the active
        # mode's program); the form never emits two simultaneous tables.
        scan_mode = _resolve_scan_mode(spec, state)

        # Apply the FIXED api values ONCE; reject typos (set_api raises for unknown handle names).
        # (An api slot named in the api SWEEP below is driven per point instead, so its fixed value
        # here is just the resting seed.)
        for name, value in api_values.items():
            if str(value).strip() == "":
                continue                                       # leave the template's value
            state.set_api(name, float(value))

        scan_names, scan_arrays, scan_shape = [], [], None
        api_names, api_arrays = [], []
        if scan_mode == SCAN_MODE_SCAN:
            # HARDWARE scan slots (x on the FPGA scan table): ONE (N_points x n_slots) program,
            # snapped to the hardware grid, slots advancing in lockstep.  A 2-slot grid program
            # also declares scan_shape (n0, n1) -> a 2-D scan map.  Empty program with bound slots
            # -> the column_stack default.
            if not scan_code.strip():
                scan_code = scan_table_template("column_stack", state.scan_column_specs())
            scan_names, scan_arrays, scan_shape = _scan_table_arrays(state, scan_code)
        elif scan_mode == SCAN_MODE_API:
            # SOFTWARE api sweep (the analogue, for api slots): the SAME single scan_code, read as
            # an (N_points x n_api) table in native units (no hardware snap; set_api snaps a duration
            # itself).  Empty program -> the column_stack default for the api columns.
            if not scan_code.strip():
                scan_code = scan_table_template("column_stack", state.api_column_specs())
            api_names, api_arrays, scan_shape = _api_table_arrays(state, scan_code)
        # scan_mode == "none": no sweep -> a single point at the fixed api values (scan/api arrays
        # stay empty; PulseScanPlan's primary axis is then the lone [0.0] point).

        # x axis label: the hardware scan slot if scanning, the swept API slot if api-sweeping, else
        # (none) the fixed point's first api slot (a meaningful single-point label).
        if scan_arrays:
            axis_label, axis_unit = _label_for_first_scan_slot(state)
        else:
            axis_label, axis_unit = _label_for_first_api_slot(state)
        y_expr = SignalExpr.from_value(y if y is not None else DEFAULT_Y_SOURCE)
        # ``camera`` is ALREADY the resolved device the base injected from the dropdown choice
        # (the readout qCMOS, or a monitor sensor for a MOT sweep) -- the pure-grabber contract
        # is identical, so the scan engine never cares which sensor it arms.
        scan_camera = camera
        return PulseScanPlan(
            state, scan_names, scan_arrays, scan_camera, s.devices.sequencer,
            axis_label=axis_label, axis_unit=axis_unit, y_key=y_name, y_expr=y_expr,
            scan_shape=scan_shape, api_names=api_names, api_arrays=api_arrays,
            extra_delay_s=extra_delay_s)

    params = (
        ParamDecl("template", "Pulse template", "path", default=DEFAULT_PROBE_TEMPLATE,
                  path_mode="file", base_dir="pulses", file_filter="Pulse program (*.json);;All files (*)",
                  tooltip="The pulse program fired each point.  Its api / scan slots populate the "
                          "auto-form below (one numeric input per api slot + ONE scan-table program)."),
        # (the "Camera" dropdown is appended by the base from devices=["camera"] -- not hand-rolled here)
        ParamDecl("pulse_slots", "Slots", "pulse_slots", default={}, depends_on="template",
                  tooltip="One numeric input per API slot (the fixed operator-set value), then a "
                          "Scan: [None | API | Scan] mode toggle that picks WHAT one shared scan "
                          "table sweeps: API sweeps the api slots in software (native units, no "
                          "snap); Scan sweeps the bound hardware scan slots (snapped to the FPGA "
                          "grid); None fires a single fixed point.  Same table model as the pulse "
                          "GUI Scan tab (column_stack / grid templates) -- one row per point, one "
                          "column per slot, advanced in lockstep (this is x)."),
        ParamDecl("y", "Signal (y)", "signal_expr", default=dict(DEFAULT_Y_SOURCE),
                  tooltip="The y recorded per point.  Pick a signal published by ANOTHER running node "
                          "(e.g. a Judge-occupancy processor's loading 'rate') and combine via value = ... .  "
                          "Start that producer FIRST; pulse-scan fires the pulse each point and reads this "
                          "UPSTREAM signal back for the point's y (it does not own the camera)."),
        ParamDecl("y_name", "Output name", "text", default="signal",
                  tooltip="Name of the OUTPUT signal this scan publishes (the y curve), e.g. "
                          "'loading_rate' or 'survival' -- so a plot picks a meaningful name."),
    )
    return MeasurementSpec(
        name="Pulse scan", key="pulse_scan", params=params,
        result_labels=("Pulse parameter", "Signal"),
        x_key="param", y_key="signal", build=build,
        # The console builds a device-driving PulseScanNode (not a ScannedMeasurementNode):
        # it fires + publishes a frame + reads y from a subscribed signal, per point.
        metadata={"node": "pulse_scan"})
