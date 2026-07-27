"""TaskConsole-only presentation sidecars for headless signal values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

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


@dataclass(frozen=True, slots=True)
class _PresentationRoute:
    """The visible and newest candidate metadata for one signal route.

    ``SignalDataPlane`` may deliberately keep revision N visible while a
    causal descendant for N+1 is still pending.  Frontend metadata therefore
    has the same two-stage lifecycle: publishing N+1 must not overwrite the
    presentation that still belongs to visible N.
    """

    visible: _PresentationEntry | None = None
    candidate: _PresentationEntry | None = None


@dataclass(frozen=True, slots=True)
class _PreparedPresentationReplacement:
    """Fully validated route replacements for one owner transaction."""

    routes: Mapping[str, _PresentationRoute]

    def __post_init__(self) -> None:
        routes = dict(self.routes)
        if any(not isinstance(route, _PresentationRoute) for route in routes.values()):
            raise TypeError("prepared presentation replacement has another type")
        object.__setattr__(self, "routes", MappingProxyType(routes))


class ConsolePresentationIndex:
    """Own UI metadata without contaminating neutral runtime values.

    A presentation is accepted only beside the exact immutable signal revision
    it describes.  Candidate publication and consumer-visible promotion are
    separate facts, just as they are in :class:`SignalDataPlane`: a newer
    candidate never invalidates metadata for an older coherent front that is
    still on screen.
    """

    def __init__(self) -> None:
        self._routes: dict[str, _PresentationRoute] = {}
        self._final_projectors: dict[
            str,
            Callable[[object, object, Mapping[str, object]], Mapping[str, SignalPresentation]],
        ] = {}

    def prepare_publish(
        self,
        values: Mapping[str, SignalValue],
        presentations: Mapping[str, SignalPresentation],
        *,
        withdraw=(),
    ) -> _PreparedPresentationReplacement:
        """Validate a complete sidecar replacement without mutating the index."""

        if not isinstance(values, Mapping) or not isinstance(presentations, Mapping):
            raise TypeError("presentation publication requires mappings")
        values = dict(values)
        presentations = dict(presentations)
        if not set(presentations).issubset(values):
            raise ValueError("presentation route has no published signal")
        withdrawn = {
            canonical_text(name, "presented signal name")
            for name in tuple(withdraw)
        }
        overlap = withdrawn.intersection(presentations)
        if overlap:
            raise ValueError(
                "one presentation transaction cannot publish and withdraw "
                + ", ".join(sorted(overlap))
            )
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
        replacements: dict[str, _PresentationRoute] = {}
        for name, entry in admitted.items():
            route = self._routes.get(name, _PresentationRoute())
            for retained in (route.visible, route.candidate):
                if (
                    retained is not None
                    and retained.revision_identity == entry.revision_identity
                    and retained.presentation != entry.presentation
                ):
                    raise ValueError(
                        "one signal revision has conflicting presentations"
                    )
            if (
                route.visible is not None
                and route.visible.revision_identity == entry.revision_identity
            ):
                replacements[name] = _PresentationRoute(
                    visible=entry,
                    candidate=route.candidate,
                )
            else:
                # The data plane keeps at most one not-yet-visible candidate
                # per route.  Replacing it here mirrors that exact ownership;
                # intermediate candidates can never later become visible.
                replacements[name] = _PresentationRoute(
                    visible=route.visible,
                    candidate=entry,
                )
        for name in withdrawn:
            route = self._routes.get(name)
            if route is None:
                continue
            # Withdrawal is staged until the neutral visible front no longer
            # carries this route.  A shell callback between data withdrawal
            # and the next freeze therefore still sees matching metadata for
            # the exact pixels/value it can consume.
            replacements[name] = _PresentationRoute(
                visible=route.visible,
                candidate=None,
            )
        return _PreparedPresentationReplacement(replacements)

    def commit_prepared(
        self,
        prepared: _PreparedPresentationReplacement,
    ) -> None:
        """Install an already-valid sidecar replacement with no validation."""

        if not isinstance(prepared, _PreparedPresentationReplacement):
            raise TypeError("prepared replacement must come from this index")
        self._routes.update(prepared.routes)

    def publish(
        self,
        values: Mapping[str, SignalValue],
        presentations: Mapping[str, SignalPresentation],
    ) -> None:
        self.commit_prepared(
            self.prepare_publish(values, presentations)
        )

    def reconcile_visible(self, values: Mapping[str, SignalValue]) -> None:
        """Promote sidecars beside the exact immutable consumer front.

        This is the presentation half of ``SignalDataPlane.freeze()``.  It
        retains a newer candidate while an older causal component is visible,
        promotes that candidate when its exact revision becomes visible, and
        retires the superseded visible entry in the same owner transaction.
        """

        if not isinstance(values, Mapping):
            raise TypeError("visible signal values must be a mapping")
        visible_values = dict(values)
        for name, route in tuple(self._routes.items()):
            value = visible_values.get(name)
            if value is not None and not isinstance(value, SignalValue):
                raise TypeError("visible signal front contains another value type")
            identity = (
                None if value is None else signal_revision_identity(value)
            )
            candidate = route.candidate
            visible = route.visible
            if candidate is not None and candidate.revision_identity == identity:
                visible = candidate
                candidate = None
            elif visible is not None and visible.revision_identity != identity:
                visible = None
            if visible is None and candidate is None:
                self._routes.pop(name, None)
                continue
            self._routes[name] = _PresentationRoute(
                visible=visible,
                candidate=candidate,
            )

    def presentation_for(self, value: SignalValue) -> SignalPresentation | None:
        if not isinstance(value, SignalValue):
            raise TypeError("value must be SignalValue")
        route = self._routes.get(value.name)
        if route is None:
            return None
        identity = signal_revision_identity(value)
        for entry in (route.visible, route.candidate):
            if entry is not None and entry.revision_identity == identity:
                return entry.presentation
        return None

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
        self._routes.clear()
        self._final_projectors.clear()


__all__ = ["ConsolePresentationIndex", "SignalPresentation"]
