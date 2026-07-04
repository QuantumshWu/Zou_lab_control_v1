"""Camera capture-trigger utilities -- the CAMERA layer's reading of a pulse sequence.

A camera is externally triggered: it reads out exactly one frame per rising edge on
its OWN capture-trigger channel(s).  ``count_trigger_pulses`` is how a camera (or the
virtual atom device, when it parses frame windows) interprets a ``PulseSequence`` --
counting its capture edges.  It lives in the camera layer, NOT in ``timing``/the
sequencer: the pulse streamer is a *pure* sequencer (it compiles + streams whatever
sequence it is handed and knows nothing about which channel gates a camera).  Every
caller passes the CAMERA's own ``capture_trigger_channels`` -- the trigger channel is a
property of the camera device, set when it is constructed, never read off the sequencer.
"""

from __future__ import annotations

from typing import Sequence

from ..timing import PulseSequence, channel_names

#: The capture-trigger channel a camera defaults to when none is configured.  A CAMERA-layer
#: default (which TTL line the qCMOS/EMCCD external trigger sits on), not a sequencer concept.
DEFAULT_CAMERA_TRIGGER_CHANNELS = ("emCCD",)


def count_trigger_pulses(
    sequence: PulseSequence,
    *,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> int:
    """Count rising camera-trigger pulses (one frame each) in ``sequence`` on ``trigger_channels``."""

    channels = set(channel_names(trigger_channels, "trigger_channels"))
    base_count = sum(1 for pulse in sequence.base_pulses() if pulse.value and pulse.channel in channels)
    if sequence.repeat_forever:
        return base_count
    return base_count * int(sequence.repeat_count)


__all__ = [
    "DEFAULT_CAMERA_TRIGGER_CHANNELS",
    "count_trigger_pulses",
]
