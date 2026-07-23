"""Canonical project-relative paths of the shipped pulse programs.

This module owns names only.  Loading, compiling, binding, and execution remain
with their respective domain owners.
"""

from __future__ import annotations


DEFAULT_PROBE_PULSE_PATH = "pulses/probe_template.json"
DEFAULT_MOT_FIELD_PULSE_PATH = "pulses/mot_field_template.json"
DEFAULT_RELEASE_RECAPTURE_PULSE_PATH = "pulses/release_recapture.json"


__all__ = [
    "DEFAULT_MOT_FIELD_PULSE_PATH",
    "DEFAULT_PROBE_PULSE_PATH",
    "DEFAULT_RELEASE_RECAPTURE_PULSE_PATH",
]
