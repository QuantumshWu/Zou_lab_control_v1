"""Camera capture-trigger utilities -- the CAMERA layer's reading of a pulse sequence.

A camera is externally triggered: it reads out exactly one frame per rising edge on
its OWN capture-trigger channel(s).  ``count_trigger_pulses`` is how a camera (or the
virtual atom device, when it parses frame windows) interprets a ``PulseSequence`` --
counting its capture edges.  It lives in the camera layer, NOT in ``timing``/the
sequencer: the pulse streamer is a *pure* sequencer (it compiles + streams whatever
sequence it is handed and knows nothing about which channel gates a camera).  Every
caller passes the CAMERA's own counting line(s) -- the trigger channel is a property of
the camera device, set when it is constructed, never read off the sequencer.

``trigger_channels`` NORMALIZATION (the ONE rule, applied by every function here --
callers pass the value through verbatim, never re-spell the fallback):

* ``None`` (unspecified)  -> :data:`DEFAULT_CAMERA_TRIGGER_CHANNELS`, the conventional
  camera line.  This default is applied HERE and nowhere else.
* ``()`` (an EMPTY tuple) -> "this camera counts on NO line": a FREE-RUNNING sensor
  (``trigger_source == SOFTWARE_TRIGGER``) exposes on its own clock and never consults
  its trigger input, so its edge counts are honestly ZERO -- never silently collapsed
  back to the default line.  ``CameraDevice.effective_trigger_channels`` is what
  declares ``()`` (identically on the virtual and real backends).
* anything else           -> used as given.

``resolve_camera_trigger_channels`` applies the same rule starting from a CAMERA object
(reading its ``effective_trigger_channels``), and the ``*_camera_*`` wrappers below
compose it with the counting functions -- so "count this camera's edges in this
sequence" is a single call at every call site.
"""

from __future__ import annotations

from typing import Sequence

from ..timing import PulseSequence, channel_names

#: The capture-trigger channel a camera defaults to when none is configured.  A CAMERA-layer
#: default (which TTL line the qCMOS/EMCCD external trigger sits on), not a sequencer concept.
DEFAULT_CAMERA_TRIGGER_CHANNELS = ("emCCD",)


def normalize_trigger_channels(trigger_channels: "Sequence[str] | None") -> tuple[str, ...]:
    """THE 'unspecified -> default line' rule (see the module docstring): ``None`` becomes
    :data:`DEFAULT_CAMERA_TRIGGER_CHANNELS`; an EMPTY sequence stays empty (a free-running
    camera's honest "no counting line"); anything else is name-validated and passed through."""
    if trigger_channels is None:
        return DEFAULT_CAMERA_TRIGGER_CHANNELS
    return channel_names(trigger_channels, "trigger_channels", allow_empty=True)


def resolve_camera_trigger_channels(camera) -> tuple[str, ...]:
    """The line(s) ``camera`` counts capture edges on -- the ONE camera->channels resolution.

    Reads the camera's :attr:`~.base.CameraDevice.effective_trigger_channels` (its declared
    wiring, ``()`` while free-running); an object that declares neither that nor a plain
    ``capture_trigger_channels`` falls back to the default line via
    :func:`normalize_trigger_channels` -- so no call site ever re-spells the fallback."""
    channels = getattr(camera, "effective_trigger_channels", None)
    if channels is None:
        channels = getattr(camera, "capture_trigger_channels", None)
    return normalize_trigger_channels(channels)


def base_cycle_trigger_pulses(
    sequence: PulseSequence,
    *,
    trigger_channels: "Sequence[str] | None" = None,
) -> int:
    """Camera-trigger pulses in ONE base cycle (before any repeat) -- the number of camera windows a
    SINGLE atom loading is imaged through.  A bracket carries >1 (release-recapture = 2 windows around a
    readout, long-short-long imaging = 3); an ordinary single-trigger shot carries 1.  This is the
    repeat-invariant shape of the pulse: a continuous (``repeat_forever``) live monitor and a finite
    ``.repeated(N)`` program have the SAME base cycle, so this distinguishes 'one loading, N windows'
    (a bracket) from 'N independent single-window shots' (a repeated grab) -- which the raw fired-trigger
    total cannot, since both total N.  ``trigger_channels`` follows the module normalization rule:
    ``None`` -> the default line, ``()`` -> no counting line -> 0."""
    channels = set(normalize_trigger_channels(trigger_channels))
    if not channels:
        return 0
    return sum(1 for pulse in sequence.base_pulses() if pulse.value and pulse.channel in channels)


def count_trigger_pulses(
    sequence: PulseSequence,
    *,
    trigger_channels: "Sequence[str] | None" = None,
) -> int:
    """Count rising camera-trigger pulses (one frame each) in the FIRED ``sequence`` (base cycle x
    repeats) on ``trigger_channels`` -- how many frames the streamer's triggers gate in total.
    ``trigger_channels`` follows the module normalization rule (``None`` -> default, ``()`` -> 0)."""
    base_count = base_cycle_trigger_pulses(sequence, trigger_channels=trigger_channels)
    if sequence.repeat_forever:
        return base_count
    return base_count * int(sequence.repeat_count)


def trigger_pulse_offsets(
    sequence: PulseSequence,
    *,
    trigger_channels: "Sequence[str] | None" = None,
) -> tuple[float, ...]:
    """Return every finite capture-edge offset in authoritative playback order.

    Legacy :class:`PulseSequence` exposes ``effective_pulses()`` while the
    pulse-owned virtual playback projection deliberately exposes only its
    already-expanded ``base_pulses()``.  This camera-owned adapter seam accepts
    both without teaching a camera backend either pulse representation.
    """

    if bool(getattr(sequence, "repeat_forever", False)):
        raise ValueError("an unbounded pulse train has no finite trigger schedule")
    channels = set(normalize_trigger_channels(trigger_channels))
    if not channels:
        return ()
    effective = getattr(sequence, "effective_pulses", None)
    if callable(effective):
        return tuple(
            sorted(
                float(pulse.start)
                for pulse in effective()
                if pulse.value and pulse.channel in channels
            )
        )
    base = tuple(sequence.base_pulses())
    repeats = int(getattr(sequence, "repeat_count", 1))
    if repeats < 1:
        raise ValueError("pulse repeat_count must be positive")
    base_offsets = tuple(
        sorted(
            float(pulse.start)
            for pulse in base
            if pulse.value and pulse.channel in channels
        )
    )
    if repeats == 1 or not base_offsets:
        return base_offsets
    repeat_period = getattr(sequence, "repeat_period", None)
    if repeat_period is None:
        repeat_period = max(
            (float(pulse.start) + float(pulse.duration) for pulse in base),
            default=0.0,
        )
    period = float(repeat_period)
    if period <= 0.0:
        raise ValueError("repeated pulse playback requires a positive period")
    return tuple(
        offset + repeat * period
        for repeat in range(repeats)
        for offset in base_offsets
    )


def base_cycle_camera_trigger_pulses(camera, sequence: PulseSequence) -> int:
    """:func:`base_cycle_trigger_pulses` on ``camera``'s own counting line(s) -- the one-call
    spelling of "how many windows does ONE base cycle image through THIS camera"."""
    return base_cycle_trigger_pulses(
        sequence, trigger_channels=resolve_camera_trigger_channels(camera))


def count_camera_trigger_pulses(camera, sequence: PulseSequence) -> int:
    """:func:`count_trigger_pulses` on ``camera``'s own counting line(s) -- the one-call
    spelling of "how many frames does this FIRED sequence gate on THIS camera"."""
    return count_trigger_pulses(
        sequence, trigger_channels=resolve_camera_trigger_channels(camera))


def camera_trigger_pulse_offsets(camera, sequence: PulseSequence) -> tuple[float, ...]:
    """Finite capture-edge offsets on ``camera``'s declared trigger line."""

    return trigger_pulse_offsets(
        sequence,
        trigger_channels=resolve_camera_trigger_channels(camera),
    )


__all__ = [
    "DEFAULT_CAMERA_TRIGGER_CHANNELS",
    "base_cycle_camera_trigger_pulses",
    "base_cycle_trigger_pulses",
    "camera_trigger_pulse_offsets",
    "count_camera_trigger_pulses",
    "count_trigger_pulses",
    "normalize_trigger_channels",
    "resolve_camera_trigger_channels",
    "trigger_pulse_offsets",
]
