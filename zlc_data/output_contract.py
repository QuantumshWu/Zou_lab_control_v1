"""Stable semantic ids for explicit authoritative Dataset projections."""

from __future__ import annotations

from zlc_storage.canonical import canonical_text


AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID = "authoritative-area-selection"


def projected_dataset_output_contract_id(
    source_contract_id: str,
    projection_id: str,
) -> str:
    """Derive the producer contract of one explicit Dataset projection."""

    source = canonical_text(source_contract_id, "source output contract id")
    projection = canonical_text(projection_id, "Dataset projection id")
    return f"{source}.projection.{projection}"


__all__ = [
    "AUTHORITATIVE_AREA_SELECTION_PROJECTION_ID",
    "projected_dataset_output_contract_id",
]
