"""Domain-neutral attachment of one prepared command's live Dataset output."""

from __future__ import annotations

from collections.abc import Callable
import uuid

from zlc_frontend.figure import DatasetId
from zlc_neutral_atom.dataset_output import LiveDatasetOutputOwner
from zlc_neutral_atom.runtime.preview import LiveDatasetViewSpec
from zlc_storage import canonical_text
from zlc_workbench.live_slot import LiveDatasetSlot


class ConsoleLiveDatasetHost:
    """Attach at most one live Dataset slot to one Console node generation."""

    __slots__ = ("_data_plane", "_dataset_id", "_node", "_opened")

    def __init__(self, node, data_plane, *, dataset_namespace: str) -> None:
        namespace = canonical_text(dataset_namespace, "dataset_namespace")
        self._node = node
        self._data_plane = data_plane
        self._dataset_id = DatasetId(
            f"console-{namespace}-{uuid.uuid4().hex}"
        )
        self._opened = False

    def open_live_dataset(
        self,
        spec: LiveDatasetViewSpec,
        *,
        output_owner: LiveDatasetOutputOwner,
        retain_on_terminal: bool = True,
    ) -> LiveDatasetSlot:
        """Create and attach the sole slot; it interprets no domain semantics."""

        if self._opened:
            raise RuntimeError("one Console start may attach only one live Dataset")
        slot = LiveDatasetSlot(
            spec,
            dataset_id=self._dataset_id,
            retain_on_terminal=retain_on_terminal,
            output_owner=output_owner,
        )
        try:
            self._data_plane.attach(self._node, slot)
            slot.set_change_listener(
                lambda: self._data_plane.mark_changed(self._node)
            )
        except BaseException:
            slot.close()
            raise
        self._opened = True
        return slot

    def factory(
        self,
        *,
        output_owner: LiveDatasetOutputOwner,
        retain_on_terminal: bool = True,
    ) -> Callable[[LiveDatasetViewSpec], LiveDatasetSlot]:
        """Return the factory shape used by prepared live/preview commands."""

        return lambda spec: self.open_live_dataset(
            spec,
            output_owner=output_owner,
            retain_on_terminal=retain_on_terminal,
        )

    def attach_live_output(self, live_output) -> None:
        """Attach an application-owned live source that already implements the slot."""

        if self._opened:
            raise RuntimeError("one Console start may attach only one live Dataset")
        if not callable(getattr(live_output, "freeze_live_outputs", None)):
            raise TypeError("live output exposes no typed Dataset materializer")
        try:
            self._data_plane.attach(self._node, live_output)
            live_output.set_change_listener(
                lambda: self._data_plane.mark_changed(self._node)
            )
        except BaseException:
            live_output.close()
            raise
        self._opened = True

    def detach_after_failed_start(self) -> None:
        if self._opened:
            self._data_plane.detach_live(self._node)


def start_with_console_live_output(
    command,
    node,
    host,
    *,
    start: Callable[[object, ConsoleLiveDatasetHost], object],
):
    """Run one capability-owned start adapter with generic slot cleanup."""

    if not callable(start):
        raise TypeError("start must be callable")
    namespace = node.spec.key.stable_definition_id
    live_host = ConsoleLiveDatasetHost(
        node,
        host.data_plane,
        dataset_namespace=namespace,
    )
    try:
        return start(command, live_host)
    except BaseException:
        live_host.detach_after_failed_start()
        raise


__all__ = ["ConsoleLiveDatasetHost", "start_with_console_live_output"]
