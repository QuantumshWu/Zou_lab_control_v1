"""The operator flows that the tutorial spine does not reach, run for real.

`test_tutorial_notebook_spine` already walks capture -> sitemap -> detect, which
is the arc a reader follows.  What it does not touch is the rest of what an
operator does in a session: watch a camera free-running, reduce a region of it
to a number, fit a committed capture, and project one into a figure.  Those four
had no end-to-end guard at all, so a break in any of them would first show up
when somebody tried it.

Deliberately one test over one connection: these steps compose (the fit and the
figure both need a real committed capture), and checking them separately would
pass while the composition was broken.  Calibration is left to the spine — it is
the slow step, and running it twice buys nothing.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import Zou_lab_control.notebook as zlc
from zlc_data.fit_model import fit_model_catalog
from zlc_data.selection import Selection

ROOT = Path(__file__).resolve().parents[1]
IMAGING_PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


def test_monitor_roi_fit_and_figure_run_on_the_virtual_installation() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        with zlc.connect("virtual", repository=Path(workspace) / "ws") as exp:
            capture = exp.run(exp.readout.capture_request(IMAGING_PULSE))

            # --- watching a camera: the request freezes WITHOUT starting hardware,
            # so the descriptor is answerable before anything is armed.
            monitor = exp.readout.camera_monitor_request()
            assert monitor.roi is None, "a plain monitor reduces nothing"
            assert exp.readout.inspect_camera_monitor(monitor) is not None

            # --- a region reduced to a scalar.  The axes come from the capture that
            # was just committed, so the ROI names real axes rather than guessed ones.
            frame_axes = exp.readout.load_capture(capture).frame_source.schema.cell_schema.data_axes
            assert len(frame_axes) == 2, "a camera frame carries two spatial axes"
            y_axis, x_axis = frame_axes
            roi = Selection.rectangle(
                x_axis.axis_id, y_axis.axis_id, 0, 16, 0, 16, coordinate_frame=None)
            assert exp.readout.camera_monitor_request(roi=roi).roi == roi

            # --- fitting.  A 2-D model resolves its own axes on a frame; a 1-D model
            # is AMBIGUOUS there and must be told which axis, which is the domain
            # refusing to silently pick one rather than a gap.
            models = {model.model_id for model in fit_model_catalog()}
            assert {"radial_gaussian_center", "gaussian_offset"} <= models
            assert exp.fit(capture, model="radial_gaussian_center") is not None
            assert exp.fit(
                capture, model="gaussian_offset", fit_axis_ids=(x_axis.axis_id,)
            ) is not None

            # --- and the same capture projected into a figure: what the viewer shows
            # is a view OF this artifact, not a document reopened from disk.
            figure = exp.figure(capture)
            assert figure.document.datasets, "a figure names the data it draws"
