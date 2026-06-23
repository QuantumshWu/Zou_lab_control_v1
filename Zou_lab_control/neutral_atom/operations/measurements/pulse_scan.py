"""Generic "Pulse scan" measurement (auto-discovered).

Sweep ANY parameter of a loaded pulse template -- a period DURATION, a channel DELAY, or a
DAC bus level -- by, PER POINT, editing the template (:class:`~...timing.PulseParam`),
recompiling ``to_sequence()`` and firing.  This is the NON-slot SOFTWARE scan the experimenter
asked for: no hardware scan slot is needed, so it sweeps things the streaming scan-slot engine
cannot (a channel delay; a duration that is not slot-bound).  Each point reduces its frames to
one y value via the SAME ``TrapCalibration`` contract the live readout uses (virtual==real;
only the camera frames differ).

The y reduction is pluggable: ``counts`` (mean per-site readout signal), ``loading`` (mean
occupancy), or ``fidelity`` (single-shot Otsu model fidelity, reusing the detection-time
scan's reducer).  Recapture survival vs trap-off time is its own purpose-built measurement
(``Temperature``), which fires a single same-atom two-image bracket -- the generic per-point
scan deliberately does not reconstruct that bracket.

The node only needs ``measure(value, index)`` + ``axis.values`` + ``reducer`` (it is duck
typed by :class:`~..logic.ScannedMeasurementNode`), so this reuses the whole live-scan node
(NaN-preallocated curve, cooperative Stop, cumulative publish) without a new node class.
"""

from __future__ import annotations

import warnings

import numpy as np

from ...core.analysis import positive_int
from ...timing import PulseParam, enumerate_pulse_params
from ...timing.sequence import snap_seconds_to_clock
from ..measurement import MeasurementSpec, OtsuFidelityReducer, ParamDecl, axis_range_tuple
from ..measurement_registry import measurement

# A time param's scan range is entered in microseconds; DAC is unitless signed LSB code.
_TIME_KINDS = ("duration", "delay")
_TIME_UNIT = "us"
_US_TO_S = 1e-6


class _Axis:
    """Minimal scan-axis the live node reads (``values`` + plot labels).  Not a hardware
    ``ScanAxis`` -- this scan edits the template per point, it binds no scan slot."""

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


class PulseScanMeasurement:
    """Software per-point scan: edit ``base_state`` with ``pulse_param`` to each value, compile,
    acquire ``n_frames``, reduce to y.  Mirrors :meth:`ScannedMeasurement.measure` (same camera /
    calibration contract) but rebuilds the sequence from the edited template each point instead
    of pushing a hardware scan slot -- so it can sweep duration / delay / DAC of any program."""

    def __init__(self, base_state, pulse_param: PulseParam, values, camera, sequencer,
                 calibration, reducer, *, n_frames: int = 1, shots_per_point: int = 1):
        self.base_state = base_state
        self.pulse_param = pulse_param
        self.axis = _Axis(values, label=str(reducer.labels[0]))
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
        sequence = self.pulse_param.apply(self.base_state, float(value)).to_sequence(name="pulse_scan")
        rows = []
        for _ in range(self.shots_per_point):
            frames = self.camera.acquire(self.n_frames, sequence=sequence, sequencer=self.sequencer,
                                         stop=self.stop_event)
            rows.append(np.atleast_1d(np.asarray(self.reducer.reduce(frames, self.calibration), dtype=float)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(np.vstack(rows), axis=0)


def _split_target(scan_target: str) -> tuple[str, str]:
    """Parse the UI's ``"kind:target"`` token (e.g. ``"duration:2"``, ``"delay:probe"``,
    ``"dac:da@0"``) into ``(kind, target)``; raise a clear error if no target is chosen."""

    token = str(scan_target or "").strip()
    kind, sep, target = token.partition(":")
    if not sep or not target:
        raise ValueError("pick a scan target (its dropdown is populated from the pulse template).")
    return kind.strip().lower(), target.strip()


def _label_for(state, kind: str, target: str, unit: str) -> str:
    """Human axis label for the chosen param (reuse the dropdown's own label), + unit."""

    label = next((lab for k, t, lab in enumerate_pulse_params(state) if k == kind and t == target),
                 f"{kind}:{target}")
    return f"{label} ({unit})" if unit else label


#: A pulse scan images ONCE per point, so its default pulse is the SINGLE-image probe
#: template (one camera trigger) -- NOT the Calibrate task's long-short-long bracket (three
#: triggers).  The user can still point ``template`` at any pulse program.
DEFAULT_PROBE_TEMPLATE = "pulses/probe_template.json"


def _resolve_probe_template(template: str):
    """Load the single-image probe pulse: the given path if real, else the shipped probe
    template of that name in ``pulses/``, else the in-memory single-image default -- never the
    3-trigger cali bracket (a pulse scan reads ONE frame per point)."""
    from pathlib import Path

    from ...timing import PulseTableState, single_imaging_template
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


@measurement(order=30)
def pulse_scan(readout) -> MeasurementSpec:
    """Generic pulse-parameter scan (auto-discovered into the task console Add Panel)."""

    s = readout.session

    def build(*, template: str = DEFAULT_PROBE_TEMPLATE, scan_target: str = "",
              scan=(0.0, 50.0, 11), shots: int = 10, reduce: str = "counts", **_ignored):
        kind, target = _split_target(scan_target)
        state = _resolve_probe_template(template)
        lo, hi, points = axis_range_tuple(scan, "scan")
        is_time = kind in _TIME_KINDS
        # A duration/delay range must be >= 0: negative/zero time snaps to one clock tick
        # (snap_seconds_to_clock's min_ticks floor), silently collapsing distinct points onto
        # the same x.  Reject it up front rather than plot a misleading, repeated axis.  (A DAC
        # target legitimately scans negative signed-LSB codes, so this guard is time-only.)
        if is_time and (lo < 0 or hi < 0):
            raise ValueError(f"a {kind} scan range must be >= 0 (got {lo}..{hi} {_TIME_UNIT}); "
                             "negative durations/delays snap to one clock tick and collapse the axis.")
        values = np.asarray(np.linspace(float(lo), float(hi), int(points)), dtype=float)
        unit = _TIME_UNIT if is_time else ""
        if is_time:
            # snap time values to the sequencer clock grid (the ONE place a software duration
            # scan quantizes), so the fired durations land on whole ticks and the plotted x is
            # what actually ran -- mirroring ScannedMeasurement's duration-axis snap.
            clock = float(getattr(s.devices.sequencer, "clock_hz", 0.0) or 0.0)
            if clock > 0.0:
                values = np.array([snap_seconds_to_clock(v * _US_TO_S, clock) / _US_TO_S for v in values],
                                  dtype=float)
        param = PulseParam(kind, target, unit=(unit or "ns"))
        x_label = _label_for(state, kind, target, unit)

        mode = str(reduce).lower()
        if mode == "loading":
            n_frames, reducer, require_thr = 1, _LoadingReducer((x_label, "Loading fraction", "Loading fraction")), True
        elif mode == "fidelity":
            reducer = OtsuFidelityReducer()
            reducer.labels = (x_label, "Fidelity", "Fidelity")
            n_frames, require_thr = positive_int(shots, "shots"), False
        else:  # "counts"
            n_frames, reducer, require_thr = 1, _CountsReducer((x_label, "Mean counts", "Mean counts")), False
        # counts/loading average the point over `shots` independent shots; fidelity pools `shots`
        # frames in ONE reduce, so its per-point shot count is 1.
        shots_per_point = 1 if mode == "fidelity" else positive_int(shots, "shots")
        calibration = s.require_calibration(require_thresholds=require_thr)
        return PulseScanMeasurement(state, param, values, s.devices.camera, s.devices.sequencer,
                                    calibration, reducer, n_frames=n_frames, shots_per_point=shots_per_point)

    params = (
        ParamDecl("template", "Pulse template", "path", default=DEFAULT_PROBE_TEMPLATE,
                  path_mode="file", base_dir="pulses", file_filter="Pulse program (*.json);;All files (*)",
                  tooltip="The pulse program loaded each point (a single-image probe by default); its "
                          "periods / channels / DAC buses populate the Target dropdown."),
        ParamDecl("scan_target", "Target", "pulse_param", default="", required=True, depends_on="template",
                  tooltip="WHICH parameter of the template to sweep -- a period duration, a channel "
                          "delay, or a DAC bus level.  The choices come from the template above."),
        ParamDecl("scan", "Scan range", "axis_range", default=(0.0, 50.0, 11), unit="us", lo=-1e6, hi=1e6,
                  tooltip="Sweep min / max / number of points.  Microseconds for a duration or delay; "
                          "unitless signed LSB code for a DAC target."),
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
