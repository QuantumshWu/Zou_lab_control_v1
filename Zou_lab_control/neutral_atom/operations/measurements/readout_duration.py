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

from pathlib import Path

import numpy as np

from Zou_lab_control._paths import resolve_under_project
from ...core.analysis import positive_int
from ...timing import PulseTableState, imaging_channel_kwargs, single_imaging_template
from ..measurement import MeasurementSpec, ParamDecl, axis_range_tuple
from ..measurement_registry import measurement

DEFAULT_IMAGING_TEMPLATE = "pulses/probe_template.json"


def _resolve_imaging_template(template: str, sequencer, *, trigger_channel: str | None = None) -> PulseTableState:
    """Load the SINGLE-image readout pulse the operator selected and map its role channels onto whatever
    THIS sequencer exposes -- the SAME pattern ``temperature`` uses for the release-recapture pulse.  A
    file whose channels are already all on the sequencer is honoured as-is (its tuned durations kept);
    otherwise (a role-named template on a real ``ch00..`` streamer, or no file) the standard
    single-image template is rebuilt on this sequencer's channels via ``imaging_channel_kwargs`` -- the
    SAME role->channel single source the imaging path uses.  ``trigger_channel`` is the CAMERA's
    ``capture_trigger_channels[0]`` (the sequencer no longer owns it): a real chNN streamer needs it to
    map the camera trigger onto a real channel.  Either way the result is a one-trigger load->image
    program whose ``image`` duration is the readout window."""
    chans = list(getattr(sequencer, "channels", ()) or ())
    text = str(template or "").strip()
    state = None
    if text:
        path = Path(text)
        if not path.is_file():
            path = resolve_under_project(text)
        if path.is_file():
            loaded = PulseTableState.load(path)
            if chans and all(c in chans for c in loaded.channels):
                state = loaded                       # channels already match -> honour the template
    if state is None:
        kw = imaging_channel_kwargs(sequencer, trigger_channel=trigger_channel)
        role_kwargs = {k: kw[k] for k in
                       ("trap_channel", "cooling_channel", "probe_channel", "trigger_channel") if k in kw}
        state = single_imaging_template(channels=chans or None, **role_kwargs)
    # Bind a duration SCAN slot on the IMAGE period -- the readout window the scan sweeps (the imaging
    # light-on time on the streamer; the camera gate is matched per point by build_detection_scan).  This
    # is the fidelity analogue of temperature's trap-off slot ``s0``; bind_field is idempotent, so a
    # template that already ships the slot is honoured unchanged.
    img_idx = next((i for i, p in enumerate(state.periods) if "image" in str(p.name)),
                   len(state.periods) - 1)
    state.bind_field("duration", str(img_idx), label="Detection time", unit="s")
    return state


@measurement(order=20)
def readout_duration_fidelity(readout) -> MeasurementSpec:
    s = readout.session

    def build(*, template=DEFAULT_IMAGING_TEMPLATE, duration=(2.0, 5000.0, 11), shots=60, site=None, **_ignored):
        d_min_us, d_max_us, points = axis_range_tuple(duration, "duration")
        times = np.linspace(float(d_min_us) * 1e-6, float(d_max_us) * 1e-6, int(points))
        site_val = None if site in (None, "", -1) else int(site)
        # The SELECTED single-image readout pulse, channel-mapped to this sequencer (real ch00.. vs
        # virtual roles): the imaging-light pulse + camera trigger fired each scan point.  The readout
        # DURATION (the camera gate-open window) is the swept axis -- build_detection_scan configures it
        # per point through this bound pulse (the coupled OtsuFidelityReducer reduces each point inline).
        # The CAMERA owns the capture-trigger line now, so its channel is threaded in for a real streamer.
        cam = getattr(s.devices, "camera", None)
        cam_trig = getattr(cam, "capture_trigger_channels", None)
        state = _resolve_imaging_template(
            template, s.devices.sequencer, trigger_channel=(cam_trig[0] if cam_trig else None),
        )
        from ...devices import bind_pulse  # lazy: keep operations->devices off the import-time graph

        pulse = bind_pulse(s.devices.sequencer, state)
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
        ParamDecl("duration", "Detection time", "axis_range", default=(2.0, 5000.0, 11), unit="us",
                  lo=1e-3, hi=1e6, tooltip="Readout-duration sweep min/max (us) and number of points -- "
                          "the camera gate-open window the imaging template is read at."),
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
                  "scan_tier": "coupled"},
    )
