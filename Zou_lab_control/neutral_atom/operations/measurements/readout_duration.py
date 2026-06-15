"""Built-in measurement: readout duration -> fidelity (auto-discovered).

Sweeps the detection exposure and reduces each point to an Otsu single-shot
fidelity.  The factory reuses ``ReadoutSubsystem.build_detection_scan`` (the SAME
builder ``exp.readout.detection_time_scan(...)`` uses), so GUI and API agree.
"""

from __future__ import annotations

import numpy as np

from ...core.analysis import positive_int
from ..measurement import MeasurementSpec, ParamDecl, axis_range_tuple
from ..measurement_registry import measurement


@measurement(order=20)
def readout_duration_fidelity(readout) -> MeasurementSpec:
    def build(*, duration=(2.0, 5000.0, 11), shots=60, site=None, **_ignored):
        d_min_us, d_max_us, points = axis_range_tuple(duration, "duration")
        times = np.linspace(float(d_min_us) * 1e-6, float(d_max_us) * 1e-6, int(points))
        site_val = None if site in (None, "", -1) else int(site)
        return readout.build_detection_scan(times, shots=positive_int(shots, "shots"), site=site_val, pulse=None)

    params = (
        ParamDecl("duration", "Detection time", "axis_range", default=(2.0, 5000.0, 11), unit="us",
                  lo=1e-3, hi=1e6, tooltip="Readout-duration sweep min/max (us) and number of points."),
        ParamDecl("shots", "Shots / point", "int", default=60, lo=1, hi=100_000,
                  tooltip="Frames pooled per point for the Otsu fidelity estimate."),
        ParamDecl("site", "Site (optional)", "int", default=None, lo=0, hi=100_000,
                  tooltip="Restrict the fidelity to one site index; leave blank to pool all sites."),
    )
    return MeasurementSpec(
        name="Readout duration -> fidelity",
        params=params,
        result_labels=("Detection time (s)", "Fidelity"),
        x_key="dur_detection_time",
        y_key="dur_fidelity",
        build=build,
    )
