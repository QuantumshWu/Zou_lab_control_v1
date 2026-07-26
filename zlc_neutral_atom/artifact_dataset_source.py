"""Exact Dataset source projected by one admitted artifact owner.

This value is the only node-neutral seam between durable experiment artifacts
and consumers such as Fit or Figure.  Artifact owners remain responsible for
finding their physical Dataset, its exact revision identity, and (when
requested) materialising an owned immutable snapshot.  Consumers never inspect
artifact-specific storage fields to recover those facts again.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import DatasetRevisionRef, DatasetSchema, OwnedSnapshot


@dataclass(frozen=True, slots=True)
class ArtifactDatasetSource:
    """One owner-admitted Dataset revision, optionally materialised.

    ``snapshot is None`` is the metadata-only form used for preflight and
    renderer-free Figure documents.  A non-``None`` snapshot must own exactly
    ``ref`` and ``schema``; there is no lazy callback or repository handle in
    this value.
    """

    schema: DatasetSchema
    ref: DatasetRevisionRef
    snapshot: OwnedSnapshot | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("artifact Dataset source schema must be DatasetSchema")
        if not isinstance(self.ref, DatasetRevisionRef):
            raise TypeError("artifact Dataset source ref must be DatasetRevisionRef")
        if self.ref.schema_fingerprint != self.schema.fingerprint:
            raise ValueError("artifact Dataset source ref differs from its schema")
        if self.snapshot is None:
            return
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("artifact Dataset source snapshot must be OwnedSnapshot")
        if self.snapshot.ref != self.ref:
            raise ValueError("artifact Dataset source snapshot has another revision")
        if self.snapshot.block.schema != self.schema:
            raise ValueError("artifact Dataset source snapshot has another schema")

    def require_owned_snapshot(self) -> OwnedSnapshot:
        """Return the exact materialised snapshot or reject metadata-only use."""

        if self.snapshot is None:
            raise RuntimeError("artifact Dataset source was not materialised")
        return self.snapshot


__all__ = ["ArtifactDatasetSource"]
