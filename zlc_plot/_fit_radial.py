"""Capability boundary for the regular-image radial fit.

The numerical implementation remains in :mod:`zlc_plot.fit`; this private
module only gives the solver capability an explicit owner seam so a model
cannot reach the regular-image path by an implicit model-id whitelist.
"""

from __future__ import annotations

from .fit import _fit_regular_radial_image as fit_regular_image_radial

__all__ = ["fit_regular_image_radial"]
