"""Shared resolver for the COUPLED-tier measurement pulse templates.

Both coupled measurements -- ``temperature`` (release-recapture) and
``readout_duration`` (fidelity-vs-duration) -- load a SELECTABLE pulse template and map
its role channels onto whatever the live sequencer exposes, in the exact same four steps:

  1. resolve the operator-selected template file (or a default when none is named);
  2. if the template's channels are already all on the sequencer, honour it AS-IS (its
     tuned durations are kept);
  3. otherwise rebuild the standard program on this sequencer's channels via
     ``imaging_channel_kwargs`` -- the SAME role->channel single source the imaging path
     uses -- so a role-named template still fires on a real ``chNN`` streamer;
  4. (optionally) bind a duration SCAN slot the coupled scan sweeps.

That framework was copied into both measurements; this is its single source (AGENTS §2,
§5 #15 -- a copied mapping silently drifts).  The measurements differ ONLY in their
``default_name`` / ``default_factory`` / ``role_keys`` / bind target -- and, crucially, in
their MISSING-FILE policy, which had ALREADY drifted between the two copies: temperature
fails LOUD on a named-but-absent file (the selection step is real), while readout_duration
silently fabricates a default.  That divergence is now an EXPLICIT ``missing_policy``
parameter each caller passes, never a silent accident.
"""

from __future__ import annotations

from typing import Callable, Sequence

from ...timing import PulseTableState, imaging_channel_kwargs, resolve_pulse_template


def resolve_coupled_template(
    template: str,
    sequencer,
    *,
    default_name: str,
    default_factory: Callable[..., PulseTableState],
    role_keys: Sequence[str],
    trigger_channel: str | None = None,
    missing_policy: str = "fabricate",
    fallback_to_loaded_channels: bool = True,
    bind_period: Callable[[str], bool] | None = None,
    bind_label: str = "Detection time",
    bind_unit: str = "s",
) -> PulseTableState:
    """Resolve a coupled-measurement pulse template onto the live sequencer (single source).

    ``default_factory`` builds the standard program when the template must be (re)built on the
    sequencer's channels; ``role_keys`` selects which ``imaging_channel_kwargs`` role channels to
    pass to it.  ``trigger_channel`` is the CAMERA's ``capture_trigger_channels[0]`` (the sequencer
    no longer owns the capture line) -- threaded into the rebuild so a real ``chNN`` streamer maps
    the camera trigger onto a real channel.

    ``missing_policy`` makes the (previously drifted) missing-file behaviour EXPLICIT:
      * ``"raise"``     -- an operator-NAMED template that resolves to no file fails LOUD
                           (``FileNotFoundError``); only an unnamed default may fabricate.
      * ``"fabricate"`` -- a missing/unnamed template silently falls back to ``default_factory``.

    ``bind_period`` (optional) is a predicate over period NAMES: when given, the first matching
    period's ``duration`` is bound as a scan slot (``bind_label`` / ``bind_unit``) -- the readout
    window the fidelity scan sweeps.  ``None`` binds nothing (temperature's trap-off slot already
    ships in its template).
    """
    policy = str(missing_policy).lower()
    if policy not in ("raise", "fabricate"):
        raise ValueError("missing_policy must be 'raise' or 'fabricate'.")
    named = bool(str(template or "").strip())

    def _missing() -> PulseTableState:
        if policy == "raise" and named:
            raise FileNotFoundError(f"pulse template not found: {str(template).strip()}")
        return default_factory()

    loaded = resolve_pulse_template(template, default_name=default_name, default_factory=_missing)
    chans = list(getattr(sequencer, "channels", ()) or ())
    if chans and all(c in chans for c in loaded.channels):
        state = loaded                                   # channels already match -> honour the template
    else:
        kw = imaging_channel_kwargs(sequencer, trigger_channel=trigger_channel)
        role_kwargs = {k: kw[k] for k in role_keys if k in kw}
        # When the sequencer exposes no channels, ``fallback_to_loaded_channels`` decides the rebuild
        # channel set: True reuses the loaded template's channels (release-recapture, whose factory has
        # no usable None default), False passes None so the factory uses its OWN role defaults
        # (single-image template).  This per-measurement choice is kept EXPLICIT, not silently merged.
        fallback = list(loaded.channels) if fallback_to_loaded_channels else None
        state = default_factory(channels=chans or fallback, **role_kwargs)

    if bind_period is not None:
        # bind the scan slot on the first matching period (fallback: the last period) -- the readout
        # window the coupled scan sweeps.  bind_field is idempotent, so a template that already ships
        # the slot is honoured unchanged.
        idx = next((i for i, p in enumerate(state.periods) if bind_period(str(p.name))),
                   len(state.periods) - 1)
        state.bind_field("duration", str(idx), label=bind_label, unit=bind_unit)
    return state


__all__ = ["resolve_coupled_template"]
