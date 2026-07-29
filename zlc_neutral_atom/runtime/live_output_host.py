"""Headless attachment of one prepared command's live Dataset output."""

from __future__ import annotations

from collections.abc import Callable

from zlc_neutral_atom.dataset_output import LiveDatasetOutputOwner
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane, SignalProducer
from zlc_neutral_atom.runtime.preview import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
    LiveDatasetViewSpec,
)
from zlc_neutral_atom.runtime.signal_source import SignalEventSource
from .live_dataset import LiveDatasetPort, _ExactDeltaLivePort


class LiveDatasetHost:
    """Attach at most one live Dataset slot to one hosted producer generation."""

    __slots__ = ("_data_plane", "_event_source", "_node", "_opened")

    def __init__(
        self,
        node: SignalProducer,
        data_plane: SignalDataPlane,
        *,
        event_source: SignalEventSource | None = None,
    ) -> None:
        if not isinstance(node, SignalProducer):
            raise TypeError("node must implement SignalProducer")
        if not isinstance(data_plane, SignalDataPlane):
            raise TypeError("data_plane must be SignalDataPlane")
        if event_source is not None and not isinstance(
            event_source,
            SignalEventSource,
        ):
            raise TypeError("event_source must implement SignalEventSource")
        self._node = node
        self._data_plane = data_plane
        self._event_source = event_source
        self._opened = False

    def open_live_dataset(
        self,
        spec: LiveDatasetViewSpec,
        *,
        output_owner: LiveDatasetOutputOwner,
        retain_on_terminal: bool = True,
    ) -> LiveDatasetPort:
        """Create and attach the sole slot; it interprets no domain semantics."""

        if self._opened:
            raise RuntimeError("one hosted start may attach only one live Dataset")
        slot = LiveDatasetPort(
            spec,
            retain_on_terminal=retain_on_terminal,
            output_owner=output_owner,
        )
        self._attach(slot)
        return slot

    def open_exact_dataset(
        self,
        spec: ExactDatasetPreviewSpec,
        *,
        projection: object,
    ) -> ExactDatasetPreviewPort:
        """Attach the sole internal exact-delta projection for this start."""

        if self._opened:
            raise RuntimeError("one hosted start may attach only one live Dataset")
        slot = _ExactDeltaLivePort(spec, projection)
        self._attach(slot)
        return slot

    def factory(
        self,
        *,
        output_owner: LiveDatasetOutputOwner,
        retain_on_terminal: bool = True,
    ) -> Callable[[LiveDatasetViewSpec], LiveDatasetPort]:
        """Return the factory shape used by prepared live/preview commands."""

        return lambda spec: self.open_live_dataset(
            spec,
            output_owner=output_owner,
            retain_on_terminal=retain_on_terminal,
        )

    def _attach(self, slot) -> None:
        if self._opened:
            raise RuntimeError("one hosted start may attach only one live Dataset")
        if not callable(getattr(slot, "freeze_live_outputs", None)):
            raise TypeError("live slot exposes no typed Dataset materializer")
        try:
            # Fresh slots cannot receive producer updates before this host starts
            # the command.  Install the sole listener first so attachment is one
            # atomic transition, never a plane slot with no wake path.
            slot.set_change_listener(
                lambda: self._data_plane.mark_changed(self._node, slot)
            )
            self._data_plane.attach(
                self._node,
                slot,
                event_source=self._event_source,
            )
        except BaseException:
            slot.close()
            raise
        self._opened = True

    def detach_after_failed_start(self) -> None:
        if self._opened:
            self._data_plane.detach_live(self._node)


def start_with_live_output(
    command,
    node: SignalProducer,
    data_plane: SignalDataPlane,
    *,
    start: Callable[[object, LiveDatasetHost], object],
):
    """Run one capability-owned start adapter with generic slot cleanup."""

    if not callable(start):
        raise TypeError("start must be callable")
    live_host = LiveDatasetHost(
        node,
        data_plane,
        event_source=(
            command if isinstance(command, SignalEventSource) else None
        ),
    )
    try:
        return start(command, live_host)
    except BaseException:
        live_host.detach_after_failed_start()
        raise


__all__ = ["LiveDatasetHost", "start_with_live_output"]
