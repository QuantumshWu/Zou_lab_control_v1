"""Exact named memory envelope for the interactive physical SiteMap."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("shape,dtype,sites", [
    ((13, 17), "uint8", 1),
    ((40, 64), "uint16", 127),
    ((40, 64), "float32", 128),
    ((40, 64), "float64", 129),
    ((512, 768), "uint16", 4096),
])
def test_interactive_site_map_peak_is_the_named_four_phase_max(
    shape,
    dtype,
    sites,
):
    from zlc_frontend.image_raster import (
        estimate_evaluated_image_retained_nbytes,
        estimate_indexed8_raster_peak_nbytes,
    )
    from zlc_frontend.occupancy_render import (
        estimate_interactive_site_map_peak_nbytes,
    )

    height, width = shape
    itemsize = np.dtype(dtype).itemsize
    pixels = height * width
    source_peak = 97_531_337
    fixed = 1 << 20
    sample = estimate_evaluated_image_retained_nbytes(
        height,
        width,
        value_itemsize=itemsize,
    )
    payload_sites = 34 * sites
    front = fixed + sample + payload_sites + 2 * pixels
    site_workspace = max(409_600, 16 * sites, 3 * sites)
    indexed = estimate_indexed8_raster_peak_nbytes(
        height,
        width,
        value_itemsize=itemsize,
        retained_fronts=2,
        retained_sample_fronts=2,
    )
    expected = max(
        source_peak + 2 * front,
        2 * front + fixed + sample + 18 * sites + payload_sites + site_workspace,
        indexed + 3 * fixed + sample + 120 * sites,
        3 * front + 40 * sites,
    )
    assert estimate_interactive_site_map_peak_nbytes(
        shape,
        dtype,
        sites,
        source_projection_peak_upper_bound_bytes=source_peak,
    ) == expected


def test_interactive_site_map_budget_rejects_shape_guessing_and_complex_frames():
    from zlc_frontend.occupancy_render import (
        estimate_interactive_site_map_peak_nbytes,
    )

    with pytest.raises(ValueError, match="two positive"):
        estimate_interactive_site_map_peak_nbytes(
            (10,),
            "uint16",
            4,
            source_projection_peak_upper_bound_bytes=1,
        )
    with pytest.raises(TypeError, match="real numeric"):
        estimate_interactive_site_map_peak_nbytes(
            (10, 12),
            "complex64",
            4,
            source_projection_peak_upper_bound_bytes=1,
        )
