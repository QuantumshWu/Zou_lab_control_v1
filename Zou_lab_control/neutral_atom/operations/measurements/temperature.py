"""Built-in measurement: release-recapture temperature (auto-discovered).

Sweeps the trap-off time and reduces each point to survival; the optional fit
(``metadata['fit']``) recovers the temperature.  The factory receives the readout
subsystem so its ``build`` closure captures the session and reuses
``ReadoutSubsystem.build_temperature_scan`` -- the SAME builder the notebook
one-liner (``exp.readout.temperature(...)``) uses, so GUI and API cannot drift.
"""

from __future__ import annotations

import numpy as np

from ...core.analysis import positive_int
from ...timing import imaging_channel_kwargs
from ..measurement import MeasurementSpec, ParamDecl, axis_range_tuple
from ..measurement_registry import measurement
from ..temperature import build_release_recapture_pulse


@measurement(order=10)
def temperature_release_recapture(readout) -> MeasurementSpec:
    s = readout.session

    def build(*, t_off=(0.0, 300.0, 13), shots=16, capture_radius=6.0, per_site=False, **_ignored):
        t_min_us, t_max_us, points = axis_range_tuple(t_off, "t_off")
        t_off_s = np.linspace(float(t_min_us) * 1e-6, float(t_max_us) * 1e-6, int(points))
        # Target the channels the bound sequencer actually exposes (real configs
        # name them ch00..chNN -> probe=ch03 etc.): otherwise the builder's
        # trap/probe/emCCD placeholder roles aren't in the channel list and it
        # raises on a real streamer.  Same single source as the imaging path; {}
        # on a virtual/named sequencer keeps the builder's placeholder defaults.
        kw = imaging_channel_kwargs(s.devices.sequencer)
        role_kwargs = {k: kw[k] for k in ("trap_channel", "probe_channel", "trigger_channel") if k in kw}
        state = build_release_recapture_pulse(channels=list(s.devices.sequencer.channels), **role_kwargs)
        from ...devices import bind_pulse  # lazy: keep operations->devices off import-time graph

        pulse = bind_pulse(s.devices.sequencer, state)
        return readout.build_temperature_scan(
            t_off_s, pulse=pulse, shots=positive_int(shots, "shots"), per_site=bool(per_site),
        )

    params = (
        ParamDecl("t_off", "Trap-off time", "axis_range", default=(0.0, 300.0, 13), unit="us",
                  lo=0.0, hi=1e4, tooltip="Trap-off sweep min/max (us) and number of points."),
        ParamDecl("shots", "Shots / point", "int", default=16, lo=1, hi=100_000,
                  tooltip="Loadings averaged at each t_off (more = lower survival noise)."),
        ParamDecl("capture_radius", "Capture radius", "float", default=6.0, unit="um", required=True,
                  lo=1e-3, hi=1e3, tooltip="Trap capture radius (um); fixes the temperature scale for the fit."),
        ParamDecl("per_site", "Per-site survival", "bool", default=False,
                  tooltip="Report one survival column per site (else the array mean)."),
    )
    grid = None
    trap = getattr(s.devices, "trap_array", None)
    grid_shape = getattr(trap, "grid_shape", None)
    if grid_shape is not None:
        grid = (int(grid_shape[0]), int(grid_shape[1]))
    return MeasurementSpec(
        name="Temperature",
        key="temperature",                       # signals: temperature_t_off / temperature_survival
        params=params,
        result_labels=("Trap-off time", "Survival"),
        x_key="t_off",
        y_key="survival",
        build=build,
        grid_shape=grid,
        # The fit needs the capture radius in METRES; the GUI param is um, so
        # record the converter the consumer applies before calling fit_temperature.
        metadata={"fit": "fit_temperature", "fit_param": "capture_radius", "fit_param_scale": 1e-6},
    )
