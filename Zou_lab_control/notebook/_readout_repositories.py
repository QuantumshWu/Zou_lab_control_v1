"""Explicit application-owned resources for built-in readout capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
        CalibrationRepository,
    )
    from zlc_neutral_atom.logic_nodes.readout.calibration.sitemap import (
        SitemapAcquisitionProfile,
    )
    from zlc_neutral_atom.logic_nodes.readout.occupancy.repository import (
        OccupancyRepository,
    )
    from zlc_neutral_atom.logic_nodes.pulse_scan.repository import ScanRepository


@dataclass
class ReadoutApplicationResources:
    """Lifecycle and installation facts owned by static readout composition."""

    calibration_repository_path: Path
    occupancy_repository_path: Path
    sitemap_profiles: Mapping[str, "SitemapAcquisitionProfile"]
    camera_signal_association_authorities: Mapping[str, object]
    scan_repository: "ScanRepository"
    _calibration_repository: "CalibrationRepository | None" = None
    _occupancy_repository: "OccupancyRepository | None" = None

    def __post_init__(self) -> None:
        self.calibration_repository_path = self.calibration_repository_path.resolve()
        self.occupancy_repository_path = self.occupancy_repository_path.resolve()
        self.sitemap_profiles = MappingProxyType(dict(self.sitemap_profiles))
        self.camera_signal_association_authorities = MappingProxyType(
            dict(self.camera_signal_association_authorities)
        )

    def calibration_repository(self) -> "CalibrationRepository":
        repository = self._calibration_repository
        if repository is not None:
            return repository
        from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
            CalibrationRepository,
        )

        repository = CalibrationRepository(self.calibration_repository_path)
        self._calibration_repository = repository
        return repository

    def occupancy_repository(self) -> "OccupancyRepository":
        repository = self._occupancy_repository
        if repository is not None:
            return repository
        from zlc_neutral_atom.logic_nodes.readout.occupancy.repository import (
            OccupancyRepository,
        )

        repository = OccupancyRepository(self.occupancy_repository_path)
        self._occupancy_repository = repository
        return repository

    def close(self) -> tuple[Exception, ...]:
        """Close only repositories that this resource owner materialized."""

        failures: list[Exception] = []
        for repository in (
            self._occupancy_repository,
            self._calibration_repository,
            self.scan_repository,
        ):
            if repository is None:
                continue
            try:
                repository.close()
            except Exception as error:
                failures.append(error)
        return tuple(failures)


__all__ = ["ReadoutApplicationResources"]
