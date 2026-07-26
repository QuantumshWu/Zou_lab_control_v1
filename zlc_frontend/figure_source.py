"""One exact immutable dataset source for rendering and Figure operations."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import OwnedSnapshot
from zlc_storage import canonical_text

from .site_map import SiteMapPresentation


@dataclass(frozen=True, slots=True)
class FigureSource:
    """Snapshot plus the semantic contract and optional SiteMap presentation.

    ``source_contract_id`` is optional for render-only callers.  Figure output
    materialisation requires it only when the derived output contract is a
    projection of the source contract (currently Area data).  Keeping that fact
    on the frontend source lets the frontend publish a complete output
    presentation; a Workbench never has to reconstruct Figure contracts.
    """

    snapshot: OwnedSnapshot
    site_map: SiteMapPresentation | None = None
    source_contract_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("Figure source requires OwnedSnapshot")
        if self.site_map is not None:
            if not isinstance(self.site_map, SiteMapPresentation):
                raise TypeError("site_map must be SiteMapPresentation or None")
            if self.site_map.site_state_input.ref != self.snapshot.ref:
                raise ValueError("SiteMap and Figure source revisions disagree")
        if self.source_contract_id is not None:
            object.__setattr__(
                self,
                "source_contract_id",
                canonical_text(
                    self.source_contract_id,
                    "Figure source contract id",
                ),
            )

    @property
    def session_identity(self) -> tuple[object, ...]:
        """Source lineage whose change must not reuse another render session."""

        ref = self.snapshot.ref
        return (
            ref.block_id,
            ref.stream_generation,
            ref.schema_fingerprint,
            (
                None
                if self.site_map is None
                else (
                    self.site_map.presentation_kind,
                    self.site_map.site_geometry_identity,
                )
            ),
        )


__all__ = ["FigureSource"]
