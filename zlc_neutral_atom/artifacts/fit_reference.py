"""Artifact-owned path identity for one persisted fit result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from zlc_storage import canonical_text


FIT_RESULT_ARTIFACT_NAMESPACE = "fit-result"


@dataclass(frozen=True, order=True, slots=True)
class FitResultArtifactRef:
    """Path to ``fit.json`` relative to the configured Fit output root."""

    record_path: str

    def __post_init__(self) -> None:
        value = canonical_text(self.record_path, "record_path")
        if "\\" in value:
            raise ValueError("record_path must use POSIX separators")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("record_path must stay beneath the Fit output root")
        if path.name != "fit.json" or len(path.parts) != 2:
            raise ValueError("record_path must be '<run-name>/fit.json'")
        object.__setattr__(self, "record_path", path.as_posix())

    @property
    def target_ref(self) -> str:
        return f"{FIT_RESULT_ARTIFACT_NAMESPACE}/{self.record_path}"


__all__ = [
    "FIT_RESULT_ARTIFACT_NAMESPACE",
    "FitResultArtifactRef",
]
