"""Typed artifact outputs published by Logic-node owners."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import canonical_text

from .output_name import bare_output_name


@dataclass(frozen=True, slots=True)
class ArtifactOutputDeclaration:
    """One FINAL artifact reference contract, distinct from Dataset signals."""

    name: str
    contract_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            bare_output_name(self.name, kind="artifact output"),
        )
        object.__setattr__(
            self,
            "contract_id",
            canonical_text(self.contract_id, "artifact output contract id"),
        )


__all__ = ["ArtifactOutputDeclaration"]
