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
from ...timing import PulseTableState, single_imaging_template
from ...timing.sequence import snap_seconds_to_clock
from ..measurement import MeasurementSpec, ParamDecl
from ..measurement_registry import measurement
from ..signal_expr import SignalExpr

# A duration scan slot's points are entered in microseconds; a DAC scan slot's points are
# unitless signed LSB codes.  An API slot's value is entered in the slot's own unit (set
# at bind time -- ns for time fields, signed LSB for DAC).
_TIME_KINDS = ("duration", "delay")
_TIME_UNIT = "us"

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


def _evaluate_points_expression(expr: str) -> np.ndarray:
    """Evaluate a 1-D points expression in a small numpy namespace and return a clean float
    array.  Accepts ``np.linspace(0, 50, 11)`` / ``linspace(0, 50, 11)`` / ``[0, 5, 10]``
    / ``range(0, 50, 5)``.  Raises a CLEAR error on something the user can fix."""

    text = str(expr or "").strip()
    if not text:
        raise ValueError("scan-slot points expression is empty -- give a list or linspace(...).")
    namespace = {"np": np, "numpy": np, "linspace": np.linspace, "arange": np.arange,
                 "logspace": np.logspace, "array": np.array, "list": list, "range": range}
    try:
        out = eval(text, {"__builtins__": {}}, namespace)   # noqa: S307 - local experiment tool
    except Exception as exc:
        raise ValueError(f"scan-slot points expression {text!r} did not evaluate: {exc}") from exc
    arr = np.asarray(out, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"scan-slot points {text!r} produced 0 points.")
    return arr


def _scan_slot_values(state: PulseTableState, scan_spec: Mapping[str, str]) -> tuple[list[str], list[np.ndarray]]:
    """For each declared scan slot ``sN``, return its evaluated array, in slot order.  Every
    scan slot in ``state`` must have an entry in ``scan_spec`` (so every swept parameter has
    been given points -- nothing silently freezes at its nominal); each array must have the
    SAME length (the sweep advances every slot in lockstep)."""

    names = [f"s{i}" for i, _ in enumerate(state.scan_slots)]
    arrays: list[np.ndarray] = []
    for slot_name in names:
        expr = str(scan_spec.get(slot_name, "")).strip()
        if not expr:
            raise ValueError(f"scan slot {slot_name!r} has no points expression "
                             "(give a list or linspace(...) for every scan slot the template has).")
        arrays.append(_evaluate_points_expression(expr))
    if len({a.size for a in arrays}) > 1:
        sizes = {n: a.size for n, a in zip(names, arrays)}
        raise ValueError(
            "every scan slot needs the SAME number of points (advanced in lockstep); "
            f"got {sizes}.  Use the SAME N in every linspace(...,N) / list of N values.")
    return names, arrays


def _label_for_first_scan_slot(state: PulseTableState) -> tuple[str, str]:
    """``(axis label, axis unit)`` for the first scan slot of the template (drives the live
    plot's x label).  Empty -> the axis is point INDEX."""

    if not state.scan_slots:
        return ("point", "")
    slot = state.scan_slots[0]
    name = slot.label or f"{slot.kind} {slot.target}"
    if slot.kind == "duration":
        return (f"{name}", _TIME_UNIT)
    if slot.kind == "dac":
        return (f"{name}", "LSB")
    return (name, "")


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
        scan_spec: Mapping[str, str] = dict(spec.get("scan") or {})

        # Apply api values ONCE; reject typos (set_api raises for unknown handle names).
        for name, value in api_values.items():
            if str(value).strip() == "":
                continue                                       # leave the template's value
            state.set_api(name, float(value))
        # Resolve every scan slot's points expression to a 1-D array (all the same length).
        # A duration slot's unit is ALWAYS "ns" inside ``ScanSlot`` (bind_field pins it), but
        # the operator enters microseconds in the form (the natural unit for an exposure);
        # convert us -> ns here and snap each time value to the sequencer clock grid so the
        # fired durations land on whole ticks and the plotted x is what actually ran.
        scan_names, scan_arrays = _scan_slot_values(state, scan_spec)
        clock = float(getattr(s.devices.sequencer, "clock_hz", 0.0) or 0.0)
        for k, name in enumerate(scan_names):
            slot = state.scan_slots[int(name[1:])]
            if slot.kind == "duration":
                arr_ns = scan_arrays[k] * 1000.0                    # us -> ns
                if clock > 0.0:
                    arr_ns = np.array([snap_seconds_to_clock(v * 1e-9, clock) * 1e9 for v in arr_ns],
                                      dtype=float)
                scan_arrays[k] = arr_ns

        axis_label, axis_unit = _label_for_first_scan_slot(state)
        y_expr = SignalExpr.from_value(y if y is not None else DEFAULT_Y_SOURCE)
        return PulseScanPlan(
            state, scan_names, scan_arrays, s.devices.camera, s.devices.sequencer,
            axis_label=axis_label, axis_unit=axis_unit, n_frames=n_frames, y_expr=y_expr)

    params = (
        ParamDecl("template", "Pulse template", "path", default=DEFAULT_PROBE_TEMPLATE,
                  path_mode="file", base_dir="pulses", file_filter="Pulse program (*.json);;All files (*)",
                  tooltip="The pulse program fired each point.  Its api / scan slots populate the "
                          "auto-form below (one numeric input per api slot, one points expression per scan slot)."),
        ParamDecl("pulse_slots", "Slots", "pulse_slots", default={}, depends_on="template",
                  tooltip="One numeric input per API slot (the fixed operator-set value), one "
                          "points expression per scan slot (the sweep -- list or linspace(...); this is x)."),
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
