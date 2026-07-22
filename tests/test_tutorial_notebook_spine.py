"""The calls the neutral-atom tutorial teaches, exercised end to end.

The tutorial went stale without anything going red: it was written against a
facade that has since been deleted, and no test ever ran its calls, so the
first person to find out would have been a reader typing them in.  This pins
the spine -- the sequence a reader actually follows -- so that removing or
reshaping any step of it fails here first.

It is deliberately the WHOLE arc rather than one call per test: what the
tutorial promises is that these steps compose, and a capture reference that
no calibration will accept would pass every isolated check.

Scope note: the spine ends at per-shot detection, which is where the rewritten
notebook ends.  Per-site thresholds, scans and the temperature fit are real
capabilities that belong in this guard, but the notebook's text for them has
not been rewritten onto this facade yet; asserting calls the tutorial does not
make would pin a contract nobody is reading.  They join when that section does.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import Zou_lab_control.notebook as zlc
from zlc_neutral_atom.artifacts import CaptureArtifactRef
from zlc_neutral_atom.readout.sitemap import load_packaged_sitemap_pulse


def _single_readout_event(document):
    """The imaging document with only its SECOND trigger window left armed.

    The tutorial reaches detection this way and the reason is physical: the
    imaging template brackets a shot with three frames, while an occupancy
    decision is about one readout event.  Keeping the middle window is what
    makes the capture carry the single event detection requires.
    """

    trigger_index = document.target.raw_lanes.index("ch11")
    periods, runs, previous = [], 0, False
    for period in document.periods:
        states = list(period.states)
        high = bool(states[trigger_index])
        if high and not previous:
            runs += 1
        states[trigger_index] = int(high and runs == 2)
        periods.append(replace(period, states=tuple(states)))
        previous = high
    assert runs == 3, "the bracket is what makes this worth narrowing"
    return replace(document, name="spine-readout", periods=tuple(periods), repeat=None)


ROOT = Path(__file__).resolve().parents[1]
IMAGING_PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


def test_the_tutorial_spine_runs_on_the_virtual_installation() -> None:
    assert IMAGING_PULSE.exists(), "the tutorial names this pulse by path"

    with tempfile.TemporaryDirectory() as workspace:
        with zlc.connect("virtual", repository=Path(workspace) / "ws") as exp:
            # The tutorial opens by asking what the installation offers, so the
            # reader binds roles rather than device names.
            roles = set(exp.device_catalog)
            assert {"camera", "sequencer"} <= roles

            # Look before you run: the descriptor answers "what will this do?"
            # off the same request object that is about to be executed, which is
            # the habit the tutorial is teaching.
            request = exp.readout.capture_request(IMAGING_PULSE)
            descriptor = exp.inspect(request)
            assert descriptor.camera_role == "camera"
            assert descriptor.expected_frames > 0

            reference = exp.run(request)
            assert isinstance(reference, CaptureArtifactRef)

            # What was run is recoverable from the reference alone -- the point
            # of handing back a reference instead of an array.
            artifact = exp.readout.load_capture(reference)
            assert artifact.frame_source.schema.physical_shape == descriptor.output_shape
            assert artifact.pulse_evidence is not None

            # The one-liner the tutorial offers once the long form is understood.
            assert isinstance(exp.readout.capture(IMAGING_PULSE), CaptureArtifactRef)

            # Site calibration: the tutorial's second act, and the input every
            # per-site quantity downstream is expressed against.
            calibration = exp.readout.sitemap(frames=6)
            report = exp.readout.load_calibration_report(calibration)
            assert report.labels.n_sites > 0
            assert len(report.psf_fits) == report.labels.n_sites

            # Third act: is this site loaded, on this shot?  A pulse document is
            # ordinary data the tutorial edits in place -- no editor involved.
            shot = exp.readout.capture(
                _single_readout_event(load_packaged_sitemap_pulse()),
                trigger_channel="ch11",
                readout_events_per_repeat=1,
            )
            occupancy = exp.readout.detect(
                exp.readout.detection_request(shot, calibration)
            )
            assert isinstance(occupancy, zlc.OccupancyArtifactRef)
            assert exp.readout.load_occupancy(occupancy) is not None
