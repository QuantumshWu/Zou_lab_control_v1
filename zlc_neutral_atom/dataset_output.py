"""Exact named Dataset outputs published by neutral-atom applications.

Output owners freeze the catalog-visible bare name together with the immutable
Dataset.  Desktop shells route these values; they do not
reconstruct arrays, axes, coverage, or lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from zlc_data import OwnedSnapshot
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    DatasetPreviewSnapshot,
    MonitorCoverage,
    MonitorDatasetSnapshot,
)
from zlc_storage import canonical_text
from .output_name import bare_output_name


@dataclass(frozen=True, slots=True)
class DatasetOutputDeclaration:
    """One owner-paired public output name and semantic contract identity."""

    name: str
    contract_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            bare_output_name(self.name, kind="dataset output"),
        )
        object.__setattr__(
            self,
            "contract_id",
            canonical_text(self.contract_id, "dataset output contract id"),
        )


@dataclass(frozen=True, slots=True)
class FinalDatasetOutput:
    """One owner-materialized FINAL Dataset under one bare output name."""

    declaration: DatasetOutputDeclaration
    snapshot: OwnedSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def contract_id(self) -> str:
        return self.declaration.contract_id


@dataclass(frozen=True, slots=True)
class LiveDatasetOutput:
    """One owner-projected live Dataset with coverage in its own geometry."""

    declaration: DatasetOutputDeclaration
    snapshot: OwnedSnapshot
    coverage: DatasetCoverage | MonitorCoverage

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if not isinstance(self.coverage, (DatasetCoverage, MonitorCoverage)):
            raise TypeError("coverage must be DatasetCoverage or MonitorCoverage")
        total = (
            self.snapshot.block.schema.repeat_axis.size
            * self.snapshot.block.schema.point_table.row_count
        )
        if self.coverage.total_cells != total:
            raise ValueError("live coverage differs from projected Dataset geometry")

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def contract_id(self) -> str:
        return self.declaration.contract_id


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


def single_live_dataset_output(
    declaration: DatasetOutputDeclaration,
    frozen: DatasetPreviewSnapshot | MonitorDatasetSnapshot,
) -> LiveDatasetOutput:
    """Publish an identity live view whose geometry was not transformed."""

    if not isinstance(declaration, DatasetOutputDeclaration):
        raise TypeError("declaration must be DatasetOutputDeclaration")
    if isinstance(frozen, MonitorDatasetSnapshot):
        if frozen.head is None:
            raise RuntimeError("monitor dataset has no accepted event head")
    elif isinstance(frozen, DatasetPreviewSnapshot):
        pass
    else:
        raise TypeError(
            "live output requires MonitorDatasetSnapshot or DatasetPreviewSnapshot"
        )
    return LiveDatasetOutput(
        declaration,
        frozen.snapshot,
        frozen.coverage,
    )


__all__ = [
    "DatasetOutputDeclaration",
    "FinalDatasetOutput",
    "LiveDatasetOutput",
    "LiveDatasetOutputOwner",
    "LiveDatasetSnapshotSource",
    "single_live_dataset_output",
]
