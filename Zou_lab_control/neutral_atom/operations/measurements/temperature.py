"""Built-in measurement: release-recapture temperature (auto-discovered).

This is a COUPLED pulse-scan special case (#H3u-4): like the generic ``Pulse scan`` it starts from a
SELECTABLE pulse ``template`` (the release-recapture program, shipped as ``pulses/release_recapture.json``
and editable in the pulse GUI), and its swept axis is a named DURATION slot of that template -- the
trap-off period (scan slot ``s0``).  What makes it the SPECIAL case (not the generic decoupled
``PulseScanNode``) is the COUPLED physics: each point fires ONE loading with TWO camera triggers
(image-before, trap-off, image-after) and the y is reduced INLINE by a :class:`SurvivalReducer` over
those two frames -- survival cannot come from a separately-running node, so it is not the generic
acquire(1)-per-point decoupled scan.  The factory reuses ``ReadoutSubsystem.build_temperature_scan`` --
the SAME builder the notebook one-liner (``exp.readout.temperature(...)``) uses -- so GUI and API agree.
"""

from __future__ import annotations

import numpy as np

from ...core.analysis import positive_int
from ...core.params import ParamDecl
from ...timing import PulseTableState
from ..measurement import SCAN_TIER_COUPLED, SCAN_TIER_KEY, MeasurementSpec, axis_range_tuple
from ..measurement_registry import measurement
from ..temperature import build_release_recapture_pulse
from ._coupled_template import resolve_coupled_template

DEFAULT_RR_TEMPLATE = "pulses/release_recapture.json"


def _match_imaging_exposure(state: PulseTableState, exposure_s: float) -> None:
    """Set every image-EXPOSE period of the release-recapture template to ``exposure_s`` so the survival
    frames are imaged at the SAME exposure the readout thresholds were calibrated at.  A threshold is
    exposure-specific (bright counts scale with the imaging time), so a mismatch leaves empty sites above
    a too-low threshold = a false-positive floor that stops survival reaching 0 even when all atoms are
    physically gone (#H3v-2).  The trap-off scan slot + settle/recapture periods are left untouched."""
    ns = float(exposure_s) * 1e9
    for i, period in enumerate(state.periods):
        if "expose" in str(period.name):
            state.set_period_duration(i, ns, unit="ns")


def _resolve_release_recapture_template(template: str, sequencer, *, trigger_channel: str | None = None) -> PulseTableState:
    """Load the release-recapture pulse the operator SELECTED and map its role channels onto whatever
    THIS sequencer exposes (the shared coupled-template resolver).  A template whose channels already
    match the sequencer is honoured AS-IS (tuned durations kept); otherwise (a role-named template on a
    real ``ch00..`` streamer) the standard release-recapture is rebuilt on the session channels via
    ``imaging_channel_kwargs``.  ``trigger_channel`` is the CAMERA's ``primary_trigger_channel``.  The
    result keeps the trap-off duration scan slot ``build_temperature_scan`` requires.  ``missing_policy=
    "raise"``: a template the operator NAMED but that resolves to no file fails LOUD (the selection step
    is real, never a silent fall-back)."""
    return resolve_coupled_template(
        template, sequencer,
        default_name=DEFAULT_RR_TEMPLATE,
        default_factory=build_release_recapture_pulse,
        role_keys=("trap_channel", "probe_channel", "trigger_channel"),
        trigger_channel=trigger_channel,
        missing_policy="raise",                      # a NAMED-but-absent template fails LOUD
        fallback_to_loaded_channels=True,            # the factory has no usable channels=None default
        bind_period=None,                            # the trap-off scan slot already ships in the template
    )


@measurement(order=10)
def temperature_release_recapture(readout) -> MeasurementSpec:
    s = readout.session

    def build(*, template=DEFAULT_RR_TEMPLATE, t_off=(0.0, 300.0, 13), shots=16, per_site=False, **_ignored):
        t_min_us, t_max_us, points = axis_range_tuple(t_off, "t_off")
        t_off_s = np.linspace(float(t_min_us) * 1e-6, float(t_max_us) * 1e-6, int(points))
        # The SELECTED release-recapture pulse, channel-mapped to this sequencer (real ch00.. vs
        # virtual roles).  Its trap-off duration slot s0 is the t_off scan axis.  The CAMERA owns
        # the capture-trigger line now, so its channel is threaded in for a real chNN streamer.
        # This measurement is INTENTIONALLY pinned to the readout (science) camera -- it images
        # single-atom survival, which a MOT monitor cannot do -- so it declares NO ``devices=[...]``
        # role.  The generic pulse scan is a separate sequencer consumer and carries no camera role.
        cam = getattr(s.devices, "camera", None)
        state = _resolve_release_recapture_template(
            template, s.devices.sequencer,
            trigger_channel=getattr(cam, "primary_trigger_channel", None),
        )
        # Image the survival frames at the SAME exposure the thresholds were calibrated at (recorded on
        # the calibration metadata): a threshold is exposure-specific, so a mismatch floors the survival
        # at the readout false-positive rate and it never reaches 0 (#H3v-2).  A high-SNR calibration
        # exposure then yields a clean survival -> 0 at large t_off (the real ballistic loss).
        cal = s.require_calibration(require_thresholds=True)
        # The authoritative reader of the calibration frame contract's exposure; fallback=None means
        # an UNstamped calibration does not force an exposure match (leave the template windows as-is).
        expo = cal.readout_exposure(fallback=None)
        if expo:
            _match_imaging_exposure(state, float(expo))
        from ...devices import bind_pulse  # lazy: keep operations->devices off import-time graph

        pulse = bind_pulse(s.devices.sequencer, state)
        return readout.build_temperature_scan(
            t_off_s, pulse=pulse, shots=positive_int(shots, "shots"), per_site=bool(per_site),
        )

    params = (
        ParamDecl("template", "Pulse template", "path", default=DEFAULT_RR_TEMPLATE,
                  path_mode="file", base_dir="pulses", file_filter="Pulse program (*.json);;All files (*)",
                  tooltip="The release-recapture pulse fired each point (image -> trap-off -> image): a "
                          "SELECTABLE template (edit it in the pulse GUI) whose trap-off duration slot is "
                          "swept by 'Trap-off time'.  This is the COUPLED pulse-scan special case -- the "
                          "two frames of ONE loading become a survival point, so y is reduced inline, not "
                          "read from a separate node."),
        ParamDecl("t_off", "Trap-off time", "axis_range", default=(0.0, 300.0, 13), unit="us",
                  lo=0.0, hi=1e4, tooltip="Trap-off sweep min/max (us) and number of points -- the "
                          "template's trap-off duration scan slot."),
        ParamDecl("shots", "Shots / point", "int", default=16, lo=1, hi=100_000,
                  tooltip="Loadings averaged at each t_off (more = lower survival noise)."),
        ParamDecl("per_site", "Per-site survival", "bool", default=False,
                  tooltip="Report one survival column per site (else the array mean)."),
    )
    return MeasurementSpec(
        name="Temperature",
        key="temperature",                       # signals: temperature_t_off / temperature_survival
        params=params,
        result_labels=("Trap-off time", "Survival"),
        x_key="t_off",
        y_key="survival",
        build=build,
        # The measurement ACQUIRES survival vs t_off; turning that curve into a TEMPERATURE is a
        # post-hoc ANALYSIS that needs the trap capture radius (a known geometry, NOT an acquisition
        # knob, NOT a sim parameter): ``na.fit_temperature(scan.x, scan.y, capture_radius=<metres>)``.
        # So ``capture_radius`` lives with the FIT, never as an acquisition ParamDecl (it changes
        # nothing about what is fired/read).
        # ``scan_tier="coupled"`` marks this as the pulse-scan SPECIAL case (#H3u-4): it is built from a
        # selectable pulse template + a swept duration slot like the generic ``Pulse scan``, but its y is
        # reduced INLINE over the two coupled frames of one loading (NOT routed through a decoupled
        # ``PulseScanNode``), so the console builds it as a plain ScannedMeasurementNode (no "node":
        # "pulse_scan" key).  The single source the docs / boundary test read for the tier.
        metadata={"analysis_fit": "operations.temperature.fit_temperature(capture_radius=...)",
                  SCAN_TIER_KEY: SCAN_TIER_COUPLED},
    )
