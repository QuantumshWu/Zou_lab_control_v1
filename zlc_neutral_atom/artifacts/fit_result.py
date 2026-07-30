"""Execute and persist Fit results without a repository authority layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from zlc_data import FitCancelled, FitDeadlineExceeded, FitResultBatch, FitSpec
from zlc_data.fit import (
    bind_fit,
    decode_fit_result_batch,
    encode_fit_result_batch,
    validate_fit_result_source_binding,
)
from zlc_storage import canonical_text, decode, encode, exact_mapping
from zlc_storage.durability import atomic_write_bytes, durable_mkdir, durable_makedirs
from zlc_storage.paths import resolve_under

from zlc_neutral_atom.artifact_dispatch import ArtifactDispatch
from .fit_reference import FitResultArtifactRef


FIT_RESULT_RECORD_SCHEMA = "zlc_neutral_atom.FitResultRecord"


def _require_fit_active(
    cancel_check: Callable[[], bool] | None,
    deadline_monotonic: float | None,
) -> None:
    if cancel_check is not None and cancel_check():
        raise FitCancelled("fit was cancelled during source materialization")
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise FitDeadlineExceeded("fit expired during source materialization")


@dataclass(frozen=True, slots=True)
class SavedFitResult:
    """One cold-opened Fit result and its exact Dataset artifact source."""

    reference: FitResultArtifactRef
    source_artifact_ref: object
    result: FitResultBatch

    def __post_init__(self) -> None:
        if not isinstance(self.reference, FitResultArtifactRef):
            raise TypeError("reference must be FitResultArtifactRef")
        if self.source_artifact_ref is None:
            raise TypeError("source_artifact_ref must be a Dataset artifact reference")
        if not isinstance(self.result, FitResultBatch):
            raise TypeError("result must be FitResultBatch")


def execute_fit(
    artifacts: ArtifactDispatch,
    source: object,
    spec: FitSpec,
    *,
    cancel_check: Callable[[], bool] | None = None,
    deadline_monotonic: float | None = None,
) -> FitResultBatch:
    """Fit one exact FINAL Dataset artifact and return the ordinary data result."""

    if not isinstance(artifacts, ArtifactDispatch):
        raise TypeError("artifacts must be ArtifactDispatch")
    if not isinstance(spec, FitSpec):
        raise TypeError("spec must be FitSpec")
    _require_fit_active(cancel_check, deadline_monotonic)
    projected = artifacts.project_dataset(
        source,
        materialize=True,
        abort_check=lambda: _require_fit_active(cancel_check, deadline_monotonic),
    )
    _require_fit_active(cancel_check, deadline_monotonic)
    return bind_fit(spec, projected.schema).run(
        projected.require_owned_snapshot(),
        cancel_check=cancel_check,
        deadline_monotonic=deadline_monotonic,
    )


def _run_name(label: str | None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    if label is None:
        return f"{stamp}-{suffix}"
    authored = canonical_text(label, "fit label")
    safe = "-".join(part for part in authored.replace("_", "-").split() if part)
    safe = "".join(character for character in safe if character.isalnum() or character == "-")
    if not safe:
        raise ValueError("fit label has no filesystem-safe characters")
    return f"{stamp}-{safe[:48]}-{suffix}"


def write_fit_result(
    fits_root: str | Path,
    artifacts: ArtifactDispatch,
    source_artifact_ref: object,
    result: FitResultBatch,
    *,
    label: str | None = None,
) -> FitResultArtifactRef:
    """Write ``fit.json`` last and return its typed relative path."""

    if not isinstance(artifacts, ArtifactDispatch):
        raise TypeError("artifacts must be ArtifactDispatch")
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    source = artifacts.project_dataset(source_artifact_ref, materialize=False)
    validate_fit_result_source_binding(result, source.ref, source.schema)
    root = Path(fits_root).expanduser().resolve()
    durable_makedirs(root)
    run_directory = resolve_under(root, _run_name(label))
    durable_mkdir(run_directory)
    record_path = run_directory / "fit.json"
    record = {
        "schema": FIT_RESULT_RECORD_SCHEMA,
        "source": artifacts.encode_dataset_reference(source_artifact_ref),
        "result": decode(encode_fit_result_batch(result)),
    }
    payload = encode(record)
    if encode(exact_mapping(decode(payload), set(record), FIT_RESULT_RECORD_SCHEMA)) != payload:
        raise ValueError("fit record failed its canonical round-trip")
    atomic_write_bytes(record_path, payload)
    return FitResultArtifactRef(record_path.relative_to(root).as_posix())


def load_fit_result(
    fits_root: str | Path,
    reference: FitResultArtifactRef,
    *,
    artifacts: ArtifactDispatch,
) -> SavedFitResult:
    """Cold-open one complete Fit record and validate its source binding."""

    if not isinstance(reference, FitResultArtifactRef):
        raise TypeError("reference must be FitResultArtifactRef")
    if not isinstance(artifacts, ArtifactDispatch):
        raise TypeError("artifacts must be ArtifactDispatch")
    root = Path(fits_root).expanduser().resolve()
    record_path = resolve_under(root, reference.record_path)
    payload = record_path.read_bytes()
    record = exact_mapping(
        decode(payload),
        {"schema", "source", "result"},
        FIT_RESULT_RECORD_SCHEMA,
    )
    if encode(record) != payload:
        raise ValueError("fit record is not canonical current format")
    source_ref = artifacts.decode_dataset_reference(record["source"])
    source = artifacts.project_dataset(source_ref, materialize=False)
    result = decode_fit_result_batch(encode(record["result"]))
    validate_fit_result_source_binding(result, source.ref, source.schema)
    return SavedFitResult(reference, source_ref, result)


__all__ = [
    "FIT_RESULT_RECORD_SCHEMA",
    "SavedFitResult",
    "execute_fit",
    "load_fit_result",
    "write_fit_result",
]
