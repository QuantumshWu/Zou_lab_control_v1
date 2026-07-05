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


def base_cycle_trigger_pulses(
    sequence: PulseSequence,
    *,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> int:
    """Camera-trigger pulses in ONE base cycle (before any repeat) -- the number of camera windows a
    SINGLE atom loading is imaged through.  A bracket carries >1 (release-recapture = 2 windows around a
    readout, long-short-long imaging = 3); an ordinary single-trigger shot carries 1.  This is the
    repeat-invariant shape of the pulse: a continuous (``repeat_forever``) live monitor and a finite
    ``.repeated(N)`` program have the SAME base cycle, so this distinguishes 'one loading, N windows'
    (a bracket) from 'N independent single-window shots' (a repeated grab) -- which the raw fired-trigger
    total cannot, since both total N."""
    channels = set(channel_names(trigger_channels, "trigger_channels"))
    return sum(1 for pulse in sequence.base_pulses() if pulse.value and pulse.channel in channels)


def count_trigger_pulses(
    sequence: PulseSequence,
    *,
    trigger_channels: Sequence[str] = DEFAULT_CAMERA_TRIGGER_CHANNELS,
) -> int:
    """Count rising camera-trigger pulses (one frame each) in the FIRED ``sequence`` (base cycle x
    repeats) on ``trigger_channels`` -- how many frames the streamer's triggers gate in total."""
    base_count = base_cycle_trigger_pulses(sequence, trigger_channels=trigger_channels)
    if sequence.repeat_forever:
        return base_count
    return base_count * int(sequence.repeat_count)


__all__ = [
    "DEFAULT_CAMERA_TRIGGER_CHANNELS",
    "base_cycle_trigger_pulses",
    "count_trigger_pulses",
]
