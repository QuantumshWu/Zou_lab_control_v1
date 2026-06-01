"""Base class for organic experiment subsystems."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..session import NeutralAtomSession


class ExperimentSubsystem:
    """Base class for stateful subsystems attached to one experiment session.

    Subsystems are larger organic capabilities, not one method per file.  For
    example, readout owns sitemap calibration, threshold calibration, detection,
    and readout-fidelity scans because those actions share the same
    calibration state and camera-readout assumptions.
    """

    def __init__(self, session: "NeutralAtomSession"):
        self._session = session

    @property
    def session(self) -> "NeutralAtomSession":
        return self._session


__all__ = ["ExperimentSubsystem"]
