"""Generic "Pulse scan" measurement (auto-discovered).

A pulse template the user loads exposes TWO kinds of named handle:

* **API slots** (``a1``, ``a2`` ...) — one numeric value the operator wants to FIX (not sweep)
  before the run.  Each slot is set ONCE at build time via :meth:`PulseTableState.set_api`.
* **Scan slots** (``s0``, ``s1`` ...) — parameters the run SWEEPS.  The user gives a list /
  ``linspace`` expression PER scan slot; the run advances every scan slot in lockstep one row
  at a time, edits the loaded template by binding each slot to its current value, recompiles
  ``to_sequence()`` and fires.  The axis is point index (and the FIRST scan slot's values).

So a "Pulse scan" measurement is parameterised purely by the loaded template — the operator
sees one numeric input per api slot and one points-expression per scan slot, with no measure-
ment-side hard-coded slot names.  Each point reduces its frames to one ``y`` value via the
same ``TrapCalibration`` contract the live readout uses (virtual==real; only the camera frames
differ).  The y reduction is pluggable: ``counts`` (mean per-site readout signal), ``loading``
(mean occupancy), or ``fidelity`` (single-shot Otsu model fidelity, reusing the detection-time
scan's reducer).

The node only needs ``measure(value, index)`` + ``axis.values`` + ``reducer`` (it is duck
typed by :class:`~..logic.ScannedMeasurementNode`), so this reuses the whole live-scan node
(NaN-preallocated curve, cooperative Stop, cumulative publish) without a new node class.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...core.analysis import positive_int
from ...timing import PulseTableState, single_imaging_template
from ...timing.sequence import snap_seconds_to_clock
from ..measurement import MeasurementSpec, OtsuFidelityReducer, ParamDecl
from ..measurement_registry import measurement

# A duration scan slot's points are entered in microseconds; a DAC scan slot's points are
# unitless signed LSB codes.  An API slot's value is entered in the slot's own unit (set
# at bind time -- ns for time fields, signed LSB for DAC).
_TIME_KINDS = ("duration", "delay")
_TIME_UNIT = "us"
_US_TO_S = 1e-6

#: A pulse scan images ONCE per point, so the default pulse is the SINGLE-image probe template
#: (one camera trigger) -- NOT the Calibrate task's long-short-long bracket.
DEFAULT_PROBE_TEMPLATE = "pulses/probe_template.json"


class _Axis:
    """The scan-axis the live node reads (``values`` + plot labels).  Just point INDEX
    (an integer running from 0) so a sweep with more than one scan slot has one unambiguous
    x; the first scan slot's actual swept values are published as ``<x_key>`` by the parent
    node (this is the axis the curve is drawn against)."""

    def __init__(self, values, *, label: str = "x", unit: str = ""):
        self.values = np.asarray(values, dtype=float).reshape(-1)
        self.label = str(label)
        self.unit = str(unit)


class _LoadingReducer:
    """y = mean occupancy across sites (the loading fraction) from one readout frame."""

    n_series = 1

    def __init__(self, labels):
        self.labels = labels

    def reduce(self, frames, calibration):
        occ = np.asarray(calibration.detect(frames[0]).occupied, dtype=float)
        return float(np.nanmean(occ)) if occ.size else float("nan")


class _CountsReducer:
    """y = mean per-site readout signal (``detect.counts``) pooled over the point's frames."""

    n_series = 1

    def __init__(self, labels):
        self.labels = labels

    def reduce(self, frames, calibration):
        counts = np.vstack([np.asarray(calibration.detect(f).counts, dtype=float) for f in frames])
        return float(np.nanmean(counts)) if counts.size else float("nan")


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


class PulseScanMeasurement:
    """Software per-point scan over an arbitrary pulse template.

    API slots and SCAN slots are DIFFERENT mechanisms and are kept separate:

    * ``api_values`` are fixed values applied ONCE to ``base_state`` (``set_api``) -- they are
      NOT swept.
    * the scan slots already bound on ``base_state`` (``s0``, ``s1`` ...) are the swept axes;
      ``scan_arrays`` carries one value array per scan slot.  Per point ``i`` the measurement
      resolves the scan slots to that row via :meth:`PulseTableState.with_slots_resolved`
      ({s0: row0, s1: row1, ...}) -- the SAME named-slot resolver the hardware scan path and
      the pulse GUI use -- so the slots stay slots (no clearing, no per-period hacking, no
      collision with an api slot on the same period).

    ``scan_arrays`` values are in the slot's resolved unit (ns for a duration slot, signed LSB
    code for a DAC slot), so they go straight into ``with_slots_resolved``."""

    def __init__(self, base_state: PulseTableState, scan_names: Sequence[str],
                 scan_arrays: Sequence[np.ndarray], camera, sequencer, calibration, reducer, *,
                 n_frames: int = 1, shots_per_point: int = 1, axis_label: str = "point",
                 axis_unit: str = ""):
        self.base_state = base_state
        self.scan_names = list(scan_names)
        self.scan_arrays = [np.asarray(a, dtype=float).reshape(-1) for a in scan_arrays]
        # The axis the live node plots: the FIRST scan slot's values when there is one
        # (everyone-in-lockstep means the others sweep against the same x); else the point
        # INDEX (nothing to sweep, single shot per point).
        if self.scan_arrays:
            values = self.scan_arrays[0]
        else:
            values = np.array([0.0])
        self.axis = _Axis(values, label=axis_label, unit=axis_unit)
        self.camera = camera
        self.sequencer = sequencer
        self.calibration = calibration
        self.reducer = reducer
        self.n_frames = positive_int(n_frames, "n_frames")
        self.shots_per_point = positive_int(shots_per_point, "shots_per_point")
        # A driving node (ScannedMeasurementNode) sets this to its Stop event so a Stop
        # interrupts a wedged trigger DURING a point (the camera honours stop=).
        self.stop_event = None

    def measure(self, value: float, index: int | None = None) -> np.ndarray:
        # Resolve the bound scan slots to THIS point's row via the named-slot resolver
        # ``with_slots_resolved`` -- the SAME mechanism the hardware scan + pulse GUI use.  The
        # api slots stay fixed (already set on base_state); the scan slots are replaced by this
        # row's values (ns for a duration slot, signed code for a DAC slot).  No clearing of
        # slots, no per-period editing -- so an api slot on the same period is never clobbered.
        idx = 0 if index is None else int(index)
        slots = {name: float(arr[idx]) for name, arr in zip(self.scan_names, self.scan_arrays)}
        resolved = self.base_state.with_slots_resolved(slots)
        sequence = resolved.to_sequence(name="pulse_scan")
        rows = []
        for _ in range(self.shots_per_point):
            frames = self.camera.acquire(self.n_frames, sequence=sequence, sequencer=self.sequencer,
                                         stop=self.stop_event)
            rows.append(np.atleast_1d(np.asarray(self.reducer.reduce(frames, self.calibration), dtype=float)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(np.vstack(rows), axis=0)


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


@measurement(order=30)
def pulse_scan(readout) -> MeasurementSpec:
    """Pulse-template scan: set every API slot's value, give scan-points per scan slot,
    sweep the points in lockstep, reduce frames to y."""

    s = readout.session

    def build(*, template: str = DEFAULT_PROBE_TEMPLATE, pulse_slots: Mapping | None = None,
              shots: int = 10, reduce: str = "counts", **_ignored):
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
        # convert us -> ns here so ``measure()`` can pass ``unit="ns"`` straight through.  Also
        # snap each time value to the sequencer clock grid so the fired durations land on
        # whole ticks and the plotted x is what actually ran (mirrors ScannedMeasurement).
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
        mode = str(reduce).lower()
        if mode == "loading":
            n_frames, reducer, require_thr = 1, _LoadingReducer((axis_label, "Loading fraction", "Loading fraction")), True
        elif mode == "fidelity":
            reducer = OtsuFidelityReducer()
            reducer.labels = (axis_label, "Fidelity", "Fidelity")
            n_frames, require_thr = positive_int(shots, "shots"), False
        else:  # "counts"
            n_frames, reducer, require_thr = 1, _CountsReducer((axis_label, "Mean counts", "Mean counts")), False
        shots_per_point = 1 if mode == "fidelity" else positive_int(shots, "shots")
        calibration = s.require_calibration(require_thresholds=require_thr)
        return PulseScanMeasurement(
            state, scan_names, scan_arrays, s.devices.camera, s.devices.sequencer, calibration,
            reducer, n_frames=n_frames, shots_per_point=shots_per_point,
            axis_label=axis_label, axis_unit=axis_unit)

    params = (
        ParamDecl("template", "Pulse template", "path", default=DEFAULT_PROBE_TEMPLATE,
                  path_mode="file", base_dir="pulses", file_filter="Pulse program (*.json);;All files (*)",
                  tooltip="The pulse program loaded each point.  Its api / scan slots populate the "
                          "auto-form below (one numeric input per api slot, one points expression per scan slot)."),
        ParamDecl("pulse_slots", "Slots", "pulse_slots", default={}, depends_on="template",
                  tooltip="One numeric input per API slot (the operator-set value the slot binds), one "
                          "points expression per scan slot (the sweep -- list or linspace(...))."),
        ParamDecl("shots", "Shots / point", "int", default=10, lo=1, hi=100_000,
                  tooltip="Frames averaged per point (counts/loading), or pooled per point (fidelity)."),
        ParamDecl("reduce", "Reduce", "choice", default="counts", choices=("counts", "loading", "fidelity"),
                  tooltip="frames -> y: counts = mean per-site signal; loading = mean occupancy; "
                          "fidelity = single-shot Otsu model fidelity.  (Recapture survival vs trap-off "
                          "is the dedicated Temperature measurement.)"),
    )
    return MeasurementSpec(
        name="Pulse scan", key="pulse_scan", params=params,
        result_labels=("Pulse parameter", "Signal"),
        x_key="param", y_key="signal", build=build)
