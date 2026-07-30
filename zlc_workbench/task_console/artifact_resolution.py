"""Domain-neutral admission of explicitly selected FINAL artifact inputs."""

from __future__ import annotations

from collections.abc import Callable

from .input_binding import ResolvedArtifactInput


def resolve_producer_final_artifact(binding: ResolvedArtifactInput) -> object:
    """Return the selected producer's exact completed Artifact value."""

    if not isinstance(binding, ResolvedArtifactInput):
        raise TypeError("binding must be ResolvedArtifactInput")
    producer = binding.producer
    if producer is None:
        raise RuntimeError("artifact selection names no producing Logic node")
    if producer.running:
        raise RuntimeError("the selected artifact-producing Logic node is running")
    if not producer.artifact_resolved:
        raise RuntimeError(
            "the selected artifact-producing Logic node has no successful "
            "current FINAL Artifact"
        )
    return producer.artifact


def resolve_final_or_saved_artifact(
    binding: ResolvedArtifactInput,
    *,
    load_saved: Callable[[object], object],
    extract_reference: Callable[[object], object],
) -> object:
    """Resolve either the exact producer result or one explicit saved record."""

    if not isinstance(binding, ResolvedArtifactInput):
        raise TypeError("binding must be ResolvedArtifactInput")
    if not callable(load_saved) or not callable(extract_reference):
        raise TypeError("saved artifact loader/extractor must be callable")
    if binding.producer is not None:
        return resolve_producer_final_artifact(binding)
    path = binding.selection.reference_path
    if path is None:
        raise RuntimeError("saved artifact input lost its exact path")
    return extract_reference(load_saved(path))


__all__ = [
    "resolve_final_or_saved_artifact",
    "resolve_producer_final_artifact",
]
