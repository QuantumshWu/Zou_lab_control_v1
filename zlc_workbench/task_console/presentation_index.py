"""TaskConsole-only presentation sidecars for headless signal values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from zlc_frontend.figure_outputs import FigureOutputPresentation
from zlc_frontend.site_map import SiteMapPresentation
from zlc_neutral_atom.processing.signal_plane import (
    SignalValue,
    signal_revision_identity,
)
from zlc_storage import canonical_text


SignalPresentation = SiteMapPresentation | FigureOutputPresentation


@dataclass(frozen=True, slots=True)
class _PresentationEntry:
    revision_identity: tuple[object, ...]
    presentation: SignalPresentation


class ConsolePresentationIndex:
    """Own UI metadata without contaminating neutral runtime values.

    A presentation is accepted only beside the exact immutable signal revision
    it describes.  The same signal route may later publish another revision;
    callers must explicitly admit the matching sidecar for that revision.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _PresentationEntry] = {}
        self._final_projectors: dict[
            str,
            Callable[[object, object, Mapping[str, object]], Mapping[str, SignalPresentation]],
        ] = {}

    def publish(
        self,
        values: Mapping[str, SignalValue],
        presentations: Mapping[str, SignalPresentation],
    ) -> None:
        if not isinstance(values, Mapping) or not isinstance(presentations, Mapping):
            raise TypeError("presentation publication requires mappings")
        values = dict(values)
        presentations = dict(presentations)
        if not set(presentations).issubset(values):
            raise ValueError("presentation route has no published signal")
        admitted: dict[str, _PresentationEntry] = {}
        for name, presentation in presentations.items():
            name = canonical_text(name, "presented signal name")
            value = values[name]
            if not isinstance(value, SignalValue) or value.name != name:
                raise TypeError("presentation value differs from its signal route")
            if not isinstance(
                presentation,
                (SiteMapPresentation, FigureOutputPresentation),
            ):
                raise TypeError("signal presentation has an unknown type")
            admitted[name] = _PresentationEntry(
                signal_revision_identity(value),
                presentation,
            )
        self._entries.update(admitted)

    def presentation_for(self, value: SignalValue) -> SignalPresentation | None:
        if not isinstance(value, SignalValue):
            raise TypeError("value must be SignalValue")
        entry = self._entries.get(value.name)
        if (
            entry is None
            or entry.revision_identity != signal_revision_identity(value)
        ):
            return None
        return entry.presentation

    def withdraw_signals(self, names) -> None:
        for name in tuple(names):
            self._entries.pop(canonical_text(name, "presented signal name"), None)

    def register_final_projector(
        self,
        producer_id: str,
        projector: Callable[
            [object, object, Mapping[str, object]],
            Mapping[str, SignalPresentation],
        ],
    ) -> None:
        identity = canonical_text(producer_id, "final projector producer_id")
        if not callable(projector):
            raise TypeError("final projector must be callable")
        self._final_projectors[identity] = projector

    def project_final(
        self,
        producer_id: str,
        command: object,
        result: object,
        outputs: Mapping[str, object],
    ) -> Mapping[str, SignalPresentation]:
        identity = canonical_text(producer_id, "final projector producer_id")
        projector = self._final_projectors.get(identity)
        if projector is None:
            return {}
        projected = projector(command, result, outputs)
        if not isinstance(projected, Mapping):
            raise TypeError("final presentation projector must return a mapping")
        result_map = dict(projected)
        if not set(result_map).issubset(outputs):
            raise ValueError("FINAL presentation has no matching domain output")
        if any(
            not isinstance(value, (SiteMapPresentation, FigureOutputPresentation))
            for value in result_map.values()
        ):
            raise TypeError("FINAL presentation has an unknown type")
        return result_map

    def unregister_final_projector(self, producer_id: str) -> None:
        self._final_projectors.pop(
            canonical_text(producer_id, "final projector producer_id"),
            None,
        )

    def clear(self) -> None:
        self._entries.clear()
        self._final_projectors.clear()


__all__ = ["ConsolePresentationIndex", "SignalPresentation"]
