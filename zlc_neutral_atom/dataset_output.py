"""Exact named Dataset outputs published by neutral-atom applications.

Output owners freeze the catalog-visible bare name together with the immutable
Dataset and its join identity.  Desktop shells route these values; they do not
reconstruct arrays, axes, coverage, or lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, Protocol, TypeVar, runtime_checkable

from zlc_data import OwnedSnapshot, dataset_revision_ref_to_tree
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    DatasetPreviewSnapshot,
    MonitorCoverage,
    MonitorDatasetSnapshot,
)
from zlc_storage import canonical_digest, canonical_text, sha256_text


def _bare_name(value: str) -> str:
    name = canonical_text(value, "dataset output name")
    if "/" in name or name.startswith("@"):
        raise ValueError("dataset output name must be bare, not namespaced")
    return name


@dataclass(frozen=True, slots=True)
class FinalDatasetOutput:
    """One owner-materialized FINAL Dataset under one bare output name."""

    name: str
    snapshot: OwnedSnapshot
    join_digest: str

    def __post_init__(self) -> None:
        name = _bare_name(self.name)
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        digest = sha256_text(self.join_digest, "join_digest")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "join_digest", digest)


@dataclass(frozen=True, slots=True)
class LiveDatasetOutput:
    """One owner-projected live Dataset with coverage in its own geometry."""

    name: str
    snapshot: OwnedSnapshot
    coverage: DatasetCoverage | MonitorCoverage
    join_digest: str

    def __post_init__(self) -> None:
        name = _bare_name(self.name)
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if not isinstance(self.coverage, (DatasetCoverage, MonitorCoverage)):
            raise TypeError("coverage must be DatasetCoverage or MonitorCoverage")
        total = (
            self.snapshot.block.schema.repeat_axis.size
            * self.snapshot.block.schema.point_layout.storage_size
        )
        if self.coverage.total_cells != total:
            raise ValueError("live coverage differs from projected Dataset geometry")
        digest = sha256_text(self.join_digest, "join_digest")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "join_digest", digest)


class LiveDatasetOutputOwner(Protocol):
    """Application owner that names/materializes one live producer front.

    The Workbench may retain this owner beside its process-local live slot, but
    it never receives an open projection callable and never interprets a
    measurement Dataset itself.
    """

    def live_dataset_outputs(
        self,
        frozen: DatasetPreviewSnapshot | MonitorDatasetSnapshot,
    ) -> Mapping[str, LiveDatasetOutput]: ...


@runtime_checkable
class LiveDatasetSnapshotSource(Protocol):
    """Narrow process-local source exposed to a desktop live-slot host.

    Acquisition implementations own ingestion and physical materialization.
    A Workbench may only freeze the already-defined current revision and close
    its borrowed lifetime; it must not branch on Camera/runtime dataset types.
    """

    def freeze_current(self) -> MonitorDatasetSnapshot: ...

    def close(self) -> None: ...


ResultT = TypeVar("ResultT")


class FinalDatasetOutputOwner(Protocol, Generic[ResultT]):
    """Prepared application that materializes its own successful result.

    This is deliberately an object capability, not a caller-supplied projector
    callback.  The prepared command retains the exact repository/request state
    needed to interpret ``result``; a desktop host only schedules this method
    off the Qt thread and routes the returned named values.
    """

    def final_dataset_outputs(
        self,
        result: ResultT,
    ) -> Mapping[str, FinalDatasetOutput]: ...


def single_live_dataset_output(
    output_name: str,
    frozen: DatasetPreviewSnapshot | MonitorDatasetSnapshot,
) -> LiveDatasetOutput:
    """Publish an identity live view whose geometry was not transformed."""

    name = _bare_name(output_name)
    if isinstance(frozen, MonitorDatasetSnapshot):
        if frozen.head is None:
            raise RuntimeError("monitor dataset has no accepted event head")
        join_digest = frozen.head.payload_digest
    elif isinstance(frozen, DatasetPreviewSnapshot):
        join_digest = canonical_digest(
            {
                "owner": "zlc_neutral_atom.identity-live-output",
                "output_name": name,
                "revision": dataset_revision_ref_to_tree(frozen.ref),
                "coverage": {
                    "written_cells": frozen.coverage.written_cells,
                    "total_cells": frozen.coverage.total_cells,
                },
            }
        )
    else:
        raise TypeError(
            "live output requires MonitorDatasetSnapshot or DatasetPreviewSnapshot"
        )
    return LiveDatasetOutput(
        name,
        frozen.snapshot,
        frozen.coverage,
        join_digest,
    )


def final_dataset_join_digest(
    *,
    owner: str,
    output_name: str,
    source_identity: object,
    snapshot: OwnedSnapshot,
) -> str:
    """Return the canonical join identity shared by every FINAL owner."""

    owner_name = canonical_text(owner, "final output owner")
    name = _bare_name(output_name)
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    return canonical_digest(
        {
            "owner": "zlc_neutral_atom.final-dataset-output",
            "domain_owner": owner_name,
            "output_name": name,
            "source_identity": source_identity,
            "dataset": dataset_revision_ref_to_tree(snapshot.ref),
        }
    )


__all__ = [
    "FinalDatasetOutput",
    "FinalDatasetOutputOwner",
    "LiveDatasetOutput",
    "LiveDatasetOutputOwner",
    "LiveDatasetSnapshotSource",
    "final_dataset_join_digest",
    "single_live_dataset_output",
]
