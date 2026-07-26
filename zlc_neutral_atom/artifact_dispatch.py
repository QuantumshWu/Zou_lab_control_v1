"""Frozen dispatch from durable artifact references to their owning capability.

Artifact leaves own reference types, codecs, repositories, and any special Figure
projection.  Application code receives this immutable projection of already-discovered
owners; it never imports a concrete Logic node, registers a handler, or dispatches by a
user-supplied string.

``format_id`` is only the discriminator inside a durable reference envelope.  Runtime
dispatch is by the owner-declared Python reference type, while decoding consults the
fixed map built once during Experiment composition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from zlc_storage import canonical_text, exact_mapping

from .artifact_dataset_source import ArtifactDatasetSource


@dataclass(frozen=True, slots=True)
class ArtifactCapability:
    """One artifact owner's bound, node-neutral application operations."""

    format_id: str
    source_label: str
    reference_type: type
    project_dataset: Callable[..., ArtifactDatasetSource] | None = None
    project_figure: Callable[..., object] | None = None
    reference_to_tree: Callable[[object], object] | None = None
    reference_from_tree: Callable[[object], object] | None = None
    admit_dataset_content: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "format_id",
            canonical_text(self.format_id, "artifact capability format_id"),
        )
        object.__setattr__(
            self,
            "source_label",
            canonical_text(self.source_label, "artifact capability source_label"),
        )
        if not isinstance(self.reference_type, type):
            raise TypeError("artifact capability reference_type must be a type")
        if self.project_dataset is None and self.project_figure is None:
            raise ValueError("artifact capability must project a Dataset or Figure")
        for name in (
            "project_dataset",
            "project_figure",
            "reference_to_tree",
            "reference_from_tree",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"artifact capability {name} must be callable or None")
        if (self.reference_to_tree is None) != (self.reference_from_tree is None):
            raise ValueError("artifact reference codec functions must be supplied together")
        if self.project_dataset is not None and self.reference_to_tree is None:
            raise ValueError(
                "Dataset artifact capability requires its owner reference codec"
            )
        if not isinstance(self.admit_dataset_content, bool):
            raise TypeError("admit_dataset_content must be bool")
        if self.admit_dataset_content and self.project_dataset is None:
            raise ValueError("only a Dataset artifact can require content admission")

    def owns(self, reference: object) -> bool:
        return isinstance(reference, self.reference_type)


class ArtifactDispatch:
    """Immutable operations projected from core and discovered artifact owners."""

    __slots__ = ("_by_format", "_capabilities")

    def __init__(self, capabilities: tuple[ArtifactCapability, ...]) -> None:
        values = tuple(capabilities)
        if not values or any(not isinstance(item, ArtifactCapability) for item in values):
            raise ValueError("ArtifactDispatch requires ArtifactCapability values")
        formats = tuple(item.format_id for item in values)
        if len(set(formats)) != len(formats):
            raise ValueError("artifact capability format ids must be unique")
        reference_types = tuple(item.reference_type for item in values)
        if len(set(reference_types)) != len(reference_types):
            raise ValueError("artifact reference types must have one owner")
        self._capabilities = values
        self._by_format = MappingProxyType(
            {
                item.format_id: item
                for item in values
                if item.reference_to_tree is not None
            }
        )

    def _owner(self, reference: object) -> ArtifactCapability:
        owners = tuple(item for item in self._capabilities if item.owns(reference))
        if len(owners) != 1:
            raise TypeError(
                "artifact reference is not owned by exactly one composed capability"
            )
        return owners[0]

    def can_project_dataset(self, reference: object) -> bool:
        try:
            owner = self._owner(reference)
        except TypeError:
            return False
        return owner.project_dataset is not None

    def can_project_figure(self, reference: object) -> bool:
        try:
            owner = self._owner(reference)
        except TypeError:
            return False
        return owner.project_figure is not None

    def source_label(self, reference: object) -> str:
        return self._owner(reference).source_label

    def project_dataset(
        self,
        reference: object,
        *,
        materialize: bool,
        abort_check: Callable[[], None] | None = None,
    ) -> ArtifactDatasetSource:
        owner = self._owner(reference)
        projector = owner.project_dataset
        if projector is None:
            raise TypeError("artifact owner does not expose a Dataset source")
        result = projector(
            reference,
            materialize=bool(materialize),
            abort_check=abort_check,
        )
        if not isinstance(result, ArtifactDatasetSource):
            raise TypeError("artifact Dataset projector returned another value type")
        if bool(materialize):
            result.require_owned_snapshot()
        return result

    def admit_dataset_reference(self, reference: object) -> ArtifactDatasetSource:
        """Revalidate a saved derived artifact against its source owner's policy."""

        owner = self._owner(reference)
        if owner.project_dataset is None:
            raise TypeError("artifact reference is not a Dataset source")
        return self.project_dataset(
            reference,
            materialize=owner.admit_dataset_content,
        )

    def project_figure(
        self,
        reference: object,
        *,
        output: str | None,
        materialize: bool,
    ) -> object:
        owner = self._owner(reference)
        projector = owner.project_figure
        if projector is None:
            raise TypeError("artifact owner has no special Figure projection")
        return projector(
            reference,
            output=output,
            materialize=bool(materialize),
        )

    def encode_dataset_reference(self, reference: object) -> dict[str, object]:
        owner = self._owner(reference)
        if owner.project_dataset is None or owner.reference_to_tree is None:
            raise TypeError("artifact reference is not a durable Dataset source")
        payload = {
            "format": owner.format_id,
            "reference": owner.reference_to_tree(reference),
        }
        if self.decode_dataset_reference(payload) != reference:
            raise ValueError("artifact owner reference codec does not round-trip")
        return payload

    def decode_dataset_reference(self, tree: object) -> object:
        data = exact_mapping(
            tree,
            {"format", "reference"},
            "artifact Dataset reference",
            discriminator=None,
        )
        format_id = canonical_text(data["format"], "artifact reference format")
        try:
            owner = self._by_format[format_id]
        except KeyError as error:
            raise ValueError("artifact Dataset reference format is not composed") from error
        if owner.project_dataset is None or owner.reference_from_tree is None:
            raise ValueError("artifact reference format is not a Dataset source")
        reference = owner.reference_from_tree(data["reference"])
        if not owner.owns(reference):
            raise TypeError("artifact owner codec returned another reference type")
        canonical = {
            "format": owner.format_id,
            "reference": owner.reference_to_tree(reference),
        }
        if canonical != data:
            raise ValueError("artifact Dataset reference is typed but non-canonical")
        return reference


__all__ = ["ArtifactCapability", "ArtifactDispatch"]
