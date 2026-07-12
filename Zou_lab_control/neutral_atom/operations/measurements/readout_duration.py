"""Built-in measurement: readout duration -> fidelity (auto-discovered).

This is the SECOND COUPLED pulse-scan special case (the sibling of ``Temperature``, #H3v-1): like the
generic ``Pulse scan`` -- and exactly like ``Temperature`` -- it starts from a SELECTABLE imaging pulse
``template`` (the single-image readout program, shipped as ``pulses/probe_template.json`` and editable
in the pulse GUI), and its swept axis is the READOUT DURATION of that template (how long fluorescence is
collected).  What makes it the SPECIAL case (not the generic decoupled ``PulseScanNode``) is the COUPLED
reduce: each point pools the point's frames and otsu-splits the per-site COUNTS into a single-shot
fidelity INLINE (:class:`OtsuFidelityReducer`) -- a per-point frame-set statistic a single-frame
processor cannot see, so it is irreducibly coupled, NOT an acquire(1)-per-point decoupled scan.

The ONE honest physical difference from ``Temperature`` (which sweeps a STREAMER trap-pulse duration slot
``s0``): the readout duration is the camera GATE-OPEN window, so the sweep is realised on the camera
(the imaging ``template`` supplies the imaging-light pulse + camera trigger fired each point, and the
readout window is configured per point).  Both are still "load a readout template, sweep ONE of its
durations, reduce each point inline" -- the same coupled tier (MAINTAINER_NOTES §19).

The factory reuses ``ReadoutSubsystem.build_detection_scan`` -- the SAME builder the notebook one-liner
(``exp.readout.detection_time(...)``) uses -- so GUI and API agree.
"""

from __future__ import annotations

import numpy as np

from ...core.analysis import positive_int
from ...core.params import ParamDecl
from ...timing import PROBE_TEMPLATE_PATH, PulseTableState, single_imaging_template
from ..measurement import SCAN_TIER_COUPLED, SCAN_TIER_KEY, MeasurementSpec, axis_range_tuple
from ..measurement_registry import measurement
from ._coupled_template import resolve_coupled_template

# The SAME shipped single-image probe program the generic Pulse-scan defaults to -- its
# path is typed ONCE, in the timing layer beside the in-memory factory (single source).
DEFAULT_IMAGING_TEMPLATE = PROBE_TEMPLATE_PATH


def _resolve_imaging_template(template: str, sequencer, *, trigger_channel: str | None = None) -> PulseTableState:
    """Load the SINGLE-image readout pulse the operator selected and map its role channels onto whatever
    THIS sequencer exposes -- the SAME shared coupled-template resolver ``temperature`` uses.  A file
    whose channels already match the sequencer is honoured as-is (tuned durations kept); otherwise (a
    role-named template on a real ``ch00..`` streamer, or no file) the standard single-image template is
    rebuilt on this sequencer's channels via ``imaging_channel_kwargs``.  ``trigger_channel`` is the
    CAMERA's ``primary_trigger_channel``.  ``missing_policy="fabricate"``: a missing/unnamed template
    falls back to the standard single-image program.  The resolver binds a duration SCAN slot on the IMAGE
    period -- the readout window the scan sweeps (the fidelity analogue of temperature's trap-off slot
    ``s0``)."""
    return resolve_coupled_template(
        template, sequencer,
        default_name=DEFAULT_IMAGING_TEMPLATE,
        default_factory=single_imaging_template,
        role_keys=("trap_channel", "cooling_channel", "probe_channel", "trigger_channel"),
        trigger_channel=trigger_channel,
        missing_policy="fabricate",                  # a missing/unnamed template fabricates the default
        fallback_to_loaded_channels=False,           # single_imaging_template uses its own role defaults
        bind_period=lambda name: "image" in name,    # bind the IMAGE-period readout window as the scan slot
        bind_label="Detection time", bind_unit="s",
    )


@measurement(order=20)
def readout_duration_fidelity(readout) -> MeasurementSpec:
    s = readout.session

    def build(*, template=DEFAULT_IMAGING_TEMPLATE, duration=(2.0, 20000.0, 11), shots=60, site=None, **_ignored):
        d_min_us, d_max_us, points = axis_range_tuple(duration, "duration")
        times = np.linspace(float(d_min_us) * 1e-6, float(d_max_us) * 1e-6, int(points))
        site_val = None if site in (None, "", -1) else int(site)
        # The SELECTED single-image readout pulse, channel-mapped to this sequencer (real ch00.. vs
        # virtual roles): the imaging-light pulse + camera trigger fired each scan point.  The readout
        # DURATION (the camera gate-open window) is the swept axis -- build_detection_scan configures it
        # per point through this bound pulse (the coupled OtsuFidelityReducer reduces each point inline).
        # The CAMERA owns the capture-trigger line now, so its channel is threaded in for a real streamer.
        # INTENTIONALLY pinned to the readout (science) camera -- readout-fidelity imaging needs the
        # science sensor, not a MOT monitor -- so it declares NO ``devices=[...]`` role.  The generic
        # pulse scan is a separate sequencer consumer and carries no camera role at all.
        cam = getattr(s._device_set, "camera", None)
        state = _resolve_imaging_template(
            template, s._device_set.sequencer,
            trigger_channel=getattr(cam, "primary_trigger_channel", None),
        )
        from ...devices import bind_pulse  # lazy: keep operations->devices off the import-time graph

        pulse = bind_pulse(s._device_set.sequencer, state)
        return readout.build_detection_scan(
            times, shots=positive_int(shots, "shots"), site=site_val, pulse=pulse,
        )

    params = (
        ParamDecl("template", "Pulse template", "path", default=DEFAULT_IMAGING_TEMPLATE,
                  path_mode="file", base_dir="pulses", file_filter="Pulse program (*.json);;All files (*)",
                  tooltip="The single-image readout pulse fired each point (load -> image): a SELECTABLE "
                          "template (edit it in the pulse GUI) whose readout-duration window is swept by "
                          "'Detection time'.  This is the COUPLED pulse-scan special case -- each point's "
                          "frame set is otsu-split into a single-shot fidelity inline, not read from a "
                          "separate node."),
        ParamDecl("duration", "Detection time", "axis_range", default=(2.0, 20000.0, 11), unit="us",
                  lo=1e-3, hi=1e6, tooltip="Readout-duration sweep min/max (us) and number of points -- "
                          "the camera gate-open window the imaging template is read at.  The default sweeps up "
                          "to the ~20 ms qCMOS working point, where single-shot fidelity rises from ~0.5 (too "
                          "short to tell bright from dark) to near its ceiling (bright/dark counts fully separate)."),
        ParamDecl("shots", "Shots / point", "int", default=60, lo=1, hi=100_000,
                  tooltip="Frames pooled per point for the Otsu fidelity estimate."),
        ParamDecl("site", "Site (optional)", "int", default=None, lo=0, hi=100_000,
                  tooltip="Restrict the fidelity to one site index; leave blank to pool all sites."),
    )
    return MeasurementSpec(
        name="Fidelity vs duration",             # distinct from the one-shot "Readout fidelity" PROCESSOR
        key="readout",                           # signals: readout_detection_time / readout_fidelity
        params=params,
        result_labels=("Detection time", "Fidelity"),
        x_key="detection_time",
        y_key="fidelity",
        build=build,
        # ``scan_tier="coupled"`` marks this as the pulse-scan SPECIAL case (sibling of Temperature):
        # built from a selectable pulse template + a swept readout duration, but its y is reduced INLINE
        # over each point's frame set (NOT routed through a decoupled PulseScanNode).  The single source
        # the docs / boundary test read for the tier.
        metadata={"analysis_fit": "operations.fidelity.characterize_readout(...)",
                  SCAN_TIER_KEY: SCAN_TIER_COUPLED},
    )
