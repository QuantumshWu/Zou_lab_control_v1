"""Built-in measurement: grey-molasses two-photon detuning scan (auto-discovered).

Grey molasses cools best when the two-photon (Raman) detuning delta between the cooling and repump beams
is on resonance (delta = 0).  That detuning is NOT a laser knob -- it is PRODUCED by the RF sideband
source (see ``devices.base.RFSourceDevice``).  So to find the optimum an operator sweeps the RF's
``two_photon_detuning_gamma`` and, at each detuning, does a release-recapture at a FIXED trap-off and
reads the recapture rate (survival): a colder cloud (better cooling) recaptures more, so the survival
curve PEAKS at the best detuning.  From that peak survival one reads off the coldest tuning (and, with a
fixed t_off, its temperature via the same ballistic model as ``na.fit_temperature``).

Architecturally this is the DEVICE-knob sibling of the ``Temperature`` measurement: it reuses the SAME
release-recapture pulse + :class:`SurvivalReducer` + scan engine, but its swept axis is a
:class:`~..measurement.DeviceControlAxis` over the RF control instead of a pulse duration slot -- so a
detuning scan and a temperature scan share one acquisition (``ReadoutSubsystem.build_device_survival_scan``).
It declares ``devices=["rf"]`` so the console/notebook inject the chosen RF source.
"""

from __future__ import annotations

import numpy as np

from ...core.analysis import positive_int
from ...core.params import ParamDecl
from ..measurement import SCAN_TIER_COUPLED, SCAN_TIER_KEY, MeasurementSpec, axis_range_tuple
from ..measurement_registry import measurement
# Reuse the release-recapture template preparation the Temperature measurement already owns (channel
# mapping onto this sequencer + exposure match to the calibration) -- same release-recapture pulse,
# built once, so the two measurements cannot drift.
from .temperature import (
    DEFAULT_RR_TEMPLATE,
    _match_imaging_exposure,
    _resolve_release_recapture_template,
)

RF_DETUNING_KEY = "two_photon_detuning_gamma"


def _rf_detuning_writer(rf):
    """The ``write(value)`` closure for a scan: route each swept detuning through the RF device's OWN
    validated runtime-control setter (never poke the attribute directly), so virtual and real hardware
    apply it identically and the device owns the delta<->frequency conversion."""
    if rf is None:
        raise ValueError("the grey-molasses detuning scan needs an RF source (devices=['rf']).")
    for control in rf.runtime_controls():
        if control.decl.key == RF_DETUNING_KEY and control.writable:
            return lambda value, c=control: c.setter(rf, float(value))
    raise ValueError(f"the selected RF source {type(rf).__name__} has no writable {RF_DETUNING_KEY} control.")


@measurement(order=11, devices=["rf"])
def grey_molasses_detuning(readout) -> MeasurementSpec:
    s = readout.session

    def build(*, rf=None, template=DEFAULT_RR_TEMPLATE, detuning=(-0.4, 0.4, 21), t_off=20.0,
              shots=16, per_site=False, **_ignored):
        d_min, d_max, points = axis_range_tuple(detuning, "detuning")
        detuning_gamma = np.linspace(float(d_min), float(d_max), int(points))
        write = _rf_detuning_writer(rf)
        # The SAME release-recapture pulse the Temperature measurement fires, channel-mapped to this
        # sequencer and imaged at the calibration exposure (a mismatch floors survival, #H3v-2).  Its
        # trap-off is held FIXED here (the plan) -- the RF detuning is what sweeps.
        cam = getattr(s._device_set, "camera", None)
        state = _resolve_release_recapture_template(
            template, s._device_set.sequencer,
            trigger_channel=getattr(cam, "primary_trigger_channel", None),
        )
        cal = s.require_calibration(require_thresholds=True)
        expo = cal.readout_exposure(fallback=None)
        if expo:
            _match_imaging_exposure(state, float(expo))
        from ...devices import bind_pulse  # lazy: keep operations->devices off import-time graph

        pulse = bind_pulse(s._device_set.sequencer, state)
        return readout.build_device_survival_scan(
            detuning_gamma, write, pulse=pulse, t_off_s=float(t_off) * 1e-6,
            label="Two-photon detuning", unit="Γ", y_label="Recapture rate",
            shots=positive_int(shots, "shots"), per_site=bool(per_site),
        )

    params = (
        ParamDecl("template", "Pulse template", "path", default=DEFAULT_RR_TEMPLATE,
                  path_mode="file", base_dir="pulses", file_filter="Pulse program (*.json);;All files (*)",
                  tooltip="The release-recapture pulse fired at each detuning (image -> trap-off -> image); "
                          "its trap-off is held fixed while the RF detuning sweeps."),
        ParamDecl("detuning", "Two-photon detuning", "axis_range", default=(-0.4, 0.4, 21), unit="Γ",
                  lo=-50.0, hi=50.0, tooltip="RF two-photon detuning sweep min/max (in linewidths Γ) and "
                          "number of points; δ = 0 (RF on the 6.834682 GHz hyperfine line) is the grey-"
                          "molasses resonance where survival peaks."),
        ParamDecl("t_off", "Trap-off time", "float", default=20.0, unit="us", lo=0.0, hi=1e4,
                  tooltip="The FIXED release-recapture trap-off time (µs).  Pick it so a cold cloud "
                          "recaptures most atoms while a hot one loses many, so the survival-vs-detuning "
                          "peak is sharp."),
        ParamDecl("shots", "Shots / point", "int", default=16, lo=1, hi=100_000,
                  tooltip="Loadings averaged at each detuning (more = lower survival noise)."),
        ParamDecl("per_site", "Per-site survival", "bool", default=False,
                  tooltip="Report one survival column per site (else the array mean)."),
    )
    return MeasurementSpec(
        name="Grey molasses detuning",
        key="gm_detuning",                       # signals: gm_detuning_detuning / gm_detuning_recapture
        params=params,
        result_labels=("Two-photon detuning", "Recapture rate"),
        x_key="detuning",
        y_key="recapture",                       # survival == recapture rate; distinct from Temperature
        build=build,
        # Like Temperature this is a COUPLED release-recapture scan (two frames of one loading -> a
        # survival point, reduced inline), but its axis writes a DEVICE control per point (the RF
        # detuning) rather than a pulse slot -- so the console builds it as a plain
        # ScannedMeasurementNode (no "node":"pulse_scan").  The optimum grey-molasses detuning is the
        # survival PEAK; convert the peak survival to temperature with na.fit_temperature if wanted.
        metadata={"analysis_fit": "optimum = argmax(survival); "
                                  "operations.temperature.fit_temperature(capture_radius=...) for T",
                  SCAN_TIER_KEY: SCAN_TIER_COUPLED},
    )
