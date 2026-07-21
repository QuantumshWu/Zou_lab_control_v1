"""The calls the neutral-atom tutorial teaches, exercised end to end.

The tutorial went stale without anything going red: it was written against a
facade that has since been deleted, and no test ever ran its calls, so the
first person to find out would have been a reader typing them in.  This pins
the spine -- the sequence a reader actually follows -- so that removing or
reshaping any step of it fails here first.

It is deliberately the WHOLE arc rather than one call per test: what the
tutorial promises is that these steps compose, and a capture reference that
no calibration will accept would pass every isolated check.

Scope note: the spine ends at the calibration report.  Detection, per-site
thresholds, scans and the temperature fit are taught later in the notebook and
belong to the same guard, but the notebook's own text for them has not been
rewritten onto this facade yet; adding assertions for calls the tutorial does
not yet make would pin a contract nobody is reading.  They join when that
section does.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import Zou_lab_control.notebook as zlc
from zlc_neutral_atom.artifacts import CaptureArtifactRef


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
            assert descriptor.estimated_peak_bytes < request.pipeline_memory_limit_bytes

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
            assert report.labels is not None
            assert report.model is not None
