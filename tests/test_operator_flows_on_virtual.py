"""The operator flows that the tutorial spine does not reach, run for real.

`test_tutorial_notebook_spine` already walks capture -> sitemap -> detect, which
is the arc a reader follows.  This test keeps the independent committed
capture -> Fit -> Figure notebook composition under one real virtual session.

Area/Cross/Fit signals published by a live Figure are exercised at the
TaskConsole/Figure boundary.  They are deliberately not Camera Measurement
request fields: display selection must never reconfigure acquisition.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import Zou_lab_control.notebook as zlc
from zlc_data.fit_model import fit_model_catalog

ROOT = Path(__file__).resolve().parents[1]
IMAGING_PULSE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


def test_fit_and_figure_run_on_the_virtual_installation() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        with zlc.connect("virtual", repository=Path(workspace) / "ws") as exp:
            capture = exp.run(exp.readout.capture_request(IMAGING_PULSE))

            # The axes come from the committed capture rather than being guessed
            # from array rank or copied into a Camera Measurement request.
            frame_axes = exp.readout.load_capture(capture).frame_source.schema.cell_schema.data_axes
            assert len(frame_axes) == 2, "a camera frame carries two spatial axes"
            _y_axis, x_axis = frame_axes

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
