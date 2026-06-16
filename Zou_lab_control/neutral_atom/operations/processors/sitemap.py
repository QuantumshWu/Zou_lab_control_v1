"""Detect trap sites from frames and show them -- a one-shot data-processing action.

This is the frame -> sites -> display primitive surfaced in the task console: read
grouped frames from a folder, DETECT the site centers (and, for method='psf', fit
per-site PSF weights), and publish the centers + the detection template + a per-site
brightness so the default 'sites' atom view draws the detected circles over the
template image.  ``run`` DRIVES ``ReadoutSubsystem.sitemap_from_dir`` (which itself
runs ``calibrate_sitemap_from_images``/``find_site_centers``) -- it re-implements no
detection math, and the calibration it establishes becomes the session calibration
that the readout-fidelity action and live detection then reuse.

Only data source is a saved frames folder, so a virtual run (na.write_virtual_run
output) and a real run traverse the identical path.
"""

from __future__ import annotations

import numpy as np

from ..processor import ParamDecl, ProcessorContext, ProcessorSpec
from ..processor_registry import processor


@processor(order=5)   # detection comes before characterization in the catalog
def detect_sites(readout) -> ProcessorSpec:
    """Detect the site map from a saved frames folder and display the sites.

    Publishes ``site_centers`` (N, 2), ``sitemap_frame`` (the averaged template the
    sites were detected on, as the underlay) and ``site_brightness`` (per-site signal,
    the circle colour); the default view is the 'sites' atom map."""

    params = (
        ParamDecl("data_dir", "Frames folder", "text", default="", required=True,
                  tooltip="Folder of saved frames (na.write_virtual_run output, or a real run)."),
        ParamDecl("prefix", "Frame prefix", "text", default="img"),
        ParamDecl("method", "Readout method", "choice", default="psf", choices=("box", "psf")),
        ParamDecl("psf_half_width", "PSF half-width", "int", default=3, lo=1, hi=15),
    )

    def run(ctx: ProcessorContext) -> dict:
        p = ctx.params
        # ``readout`` captured from the factory -> console stays decoupled.
        result = readout.sitemap_from_dir(
            str(p["data_dir"]), prefix=str(p.get("prefix", "img")),
            method=str(p["method"]), psf_half_width=int(p["psf_half_width"]), display=False)
        cal = readout.current
        template = np.asarray(result.average_image, dtype=float)
        return {
            "site_centers": np.asarray(cal.centers, dtype=float),       # (N, 2) detected centers
            "sitemap_frame": template,                                  # underlay image
            "site_brightness": np.asarray(cal.signals(template), dtype=float),  # per-site colour
        }

    return ProcessorSpec(
        name="Detect sites",
        params=params,
        run=run,
        # NB: no generic "n_sites" scalar -- it collides with the fidelity processor
        # on the shared hub (the dup-result_keys guard); len(site_centers) IS N.
        result_keys=("site_centers", "sitemap_frame", "site_brightness"),
        summary_keys=(),
        default_kind="sites",            # the atom map (camera underlay + per-site circles)
        default_value_key="site_brightness",
        # tell the console which published signals are the sites' centers + underlay
        metadata={"centers_key": "site_centers", "image_key": "sitemap_frame"},
    )
