"""Project pulse-catalog keys shared by neutral-atom Logic Nodes.

The editable PulseDocuments themselves live only in the repository-level
``pulses/`` catalog.  This module owns their stable relative names so one Logic
Node never imports another Logic Node merely to reuse a default recipe path.
"""

CALIBRATION_PULSE_PATH = "pulses/imaging_template.json"
MOT_FIELD_PULSE_PATH = "pulses/mot_field_template.json"
PROBE_PULSE_PATH = "pulses/probe_template.json"
RELEASE_RECAPTURE_PULSE_PATH = "pulses/release_recapture.json"


__all__ = [
    "CALIBRATION_PULSE_PATH",
    "MOT_FIELD_PULSE_PATH",
    "PROBE_PULSE_PATH",
    "RELEASE_RECAPTURE_PULSE_PATH",
]
