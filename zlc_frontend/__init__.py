"""Shared headless form contracts for Zou Lab Control frontends.

Plot specifications, projection, interaction, fitting and rendering belong to
``zlc_plot``.  Qt widgets remain lazily available from
``zlc_frontend.qt_widgets``; importing this package never imports Qt,
Matplotlib, or a plot runtime.
"""

from .form import FormChoice, FormFieldProps, FormSpec

__all__ = ["FormChoice", "FormFieldProps", "FormSpec"]
