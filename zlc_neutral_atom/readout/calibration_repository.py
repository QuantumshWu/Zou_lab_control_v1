"""Content-addressed repository for immutable calibration artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zlc_storage import (
    ContentAddressedStore,
    ContentRef,
    ContentSizeLimitError,
    decode,
    encode,
    sha256_digest,
)

from .calibration import (
    DEFAULT_CALIBRATION_RESOURCE_POLICY,
    CalibrationArtifact,
    CalibrationResourceExceeded,
    CalibrationResourcePolicy,
    CalibrationResourceSummary,
    calibration_resource_summary,
    validate_calibration_artifact_resources,
    validate_calibration_artifact_source_compatibility,
    validate_calibration_resource_summary,
)
from .calibration_codec import (
    CALIBRATION_ARTIFACT_SCHEMA,
    decode_calibration_artifact,
    encode_calibration_artifact,
)
from .calibration_reference import CalibrationArtifactRef


CALIBRATION_MANIFEST_SCHEMA = "zlc_neutral_atom.calibration-manifest.v1"
_CALIBRATION_NAMESPACE = "calibration"


def _canonical_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


class CalibrationRepository:
    """Standalone/offline CAS for immutable calibration artifacts.

    ``put`` is not a runtime final-commit authority.  A later Task integration
    must wrap publication in the run commit protocol instead of treating this
    convenience method as evidence that a hardware run committed safely.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        repository_id: str = "zlc-neutral-calibration",
        resource_policy: CalibrationResourcePolicy = DEFAULT_CALIBRATION_RESOURCE_POLICY,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.repository_id = _canonical_text(repository_id, "repository_id")
        if not isinstance(resource_policy, CalibrationResourcePolicy):
            raise TypeError("resource_policy must be CalibrationResourcePolicy")
        self.resource_policy = resource_policy
        self._store = ContentAddressedStore(self.root / "content")

    def put(self, artifact: CalibrationArtifact) -> CalibrationArtifactRef:
        if not isinstance(artifact, CalibrationArtifact):
            raise TypeError("put requires CalibrationArtifact")
        validate_calibration_artifact_resources(artifact, self.resource_policy)
        resource_summary = calibration_resource_summary(artifact)
        artifact_payload = encode_calibration_artifact(artifact)
        if len(artifact_payload) > self.resource_policy.max_artifact_blob_bytes:
            raise CalibrationResourceExceeded(
                "calibration artifact blob exceeds repository resource policy"
            )
        artifact_blob = self._store.put_blob(artifact_payload)
        manifest_payload = _manifest_payload(
            repository_id=self.repository_id,
            artifact_blob=artifact_blob,
            artifact_fingerprint=artifact.fingerprint,
            resource_summary=resource_summary,
        )
        if len(manifest_payload) > self.resource_policy.max_manifest_bytes:
            raise CalibrationResourceExceeded(
                "calibration manifest exceeds repository resource policy"
            )
        digest = sha256_digest(manifest_payload)
        stored = self._store.publish_manifest(
            _CALIBRATION_NAMESPACE,
            manifest_payload,
            expected_digest=digest,
        )
        if stored.content.digest != digest:
            raise RuntimeError("published calibration manifest digest changed")
        return CalibrationArtifactRef(self.repository_id, digest)

    def load(self, reference: CalibrationArtifactRef) -> CalibrationArtifact:
        """Structurally load for offline inspection; does not resolve source capture."""

        self._validate_reference(reference)
        try:
            manifest_payload = self._store.read_manifest(
                _CALIBRATION_NAMESPACE,
                reference.manifest_digest,
                max_bytes=self.resource_policy.max_manifest_bytes,
            )
        except ContentSizeLimitError as exc:
            raise CalibrationResourceExceeded(
                "calibration manifest exceeds repository resource policy"
            ) from exc
        data = _manifest_from_tree(decode(manifest_payload))
        if data["repository_id"] != self.repository_id:
            raise ValueError("CalibrationArtifact belongs to another repository")
        declared_summary = _resource_summary_from_tree(data["resource_summary"])
        validate_calibration_resource_summary(
            declared_summary,
            self.resource_policy,
        )
        blob = _content_ref_from_tree(data["artifact_blob"])
        if blob.size > self.resource_policy.max_artifact_blob_bytes:
            raise CalibrationResourceExceeded(
                "calibration artifact blob exceeds repository resource policy"
            )
        try:
            artifact_payload = self._store.read_blob(
                blob,
                max_bytes=self.resource_policy.max_artifact_blob_bytes,
            )
        except ContentSizeLimitError as exc:
            raise CalibrationResourceExceeded(
                "calibration artifact blob exceeds repository resource policy"
            ) from exc
        artifact = decode_calibration_artifact(
            artifact_payload,
            resource_policy=self.resource_policy,
        )
        if artifact.fingerprint != data["artifact_fingerprint"]:
            raise ValueError("calibration artifact fingerprint differs from manifest")
        computed_summary = calibration_resource_summary(artifact)
        if computed_summary != declared_summary:
            raise ValueError("calibration resource summary differs from artifact content")
        rebuilt = _manifest_payload(
            repository_id=self.repository_id,
            artifact_blob=blob,
            artifact_fingerprint=artifact.fingerprint,
            resource_summary=computed_summary,
        )
        if (
            rebuilt != manifest_payload
            or sha256_digest(rebuilt) != reference.manifest_digest
        ):
            raise ValueError("CalibrationArtifact manifest is not canonical")
        return artifact

    def load_source_verified(
        self,
        reference: CalibrationArtifactRef,
        *,
        capture_resolver,
    ) -> CalibrationArtifact:
        """Load and exact-rederive declared source compatibility.

        The result is not runtime authority: this standalone repository cannot
        prove capture final commit or calibration derivation.  Physical runtime
        paths must additionally require the trusted calibration Task's committed
        evidence wrapper/reference.
        """

        artifact = self.load(reference)
        validate_calibration_artifact_source_compatibility(
            artifact,
            capture_resolver,
        )
        return artifact

    def has(self, reference: CalibrationArtifactRef) -> bool:
        self._validate_reference(reference)
        return self._store.has_manifest(
            _CALIBRATION_NAMESPACE,
            reference.manifest_digest,
        )

    def _validate_reference(self, reference: CalibrationArtifactRef) -> None:
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        if reference.repository_id != self.repository_id:
            raise ValueError("CalibrationArtifactRef belongs to another repository")


def _content_ref_to_tree(reference: ContentRef) -> dict[str, object]:
    return {"digest": reference.digest, "size": reference.size}


def _content_ref_from_tree(tree: Any) -> ContentRef:
    if not isinstance(tree, dict) or set(tree) != {"digest", "size"}:
        raise ValueError("calibration content reference has an unknown field set")
    return ContentRef(tree["digest"], tree["size"])


def _manifest_payload(
    *,
    repository_id: str,
    artifact_blob: ContentRef,
    artifact_fingerprint: str,
    resource_summary: CalibrationResourceSummary,
) -> bytes:
    return encode(
        {
            "schema": CALIBRATION_MANIFEST_SCHEMA,
            "repository_id": repository_id,
            "artifact_schema": CALIBRATION_ARTIFACT_SCHEMA,
            "artifact_blob": _content_ref_to_tree(artifact_blob),
            "artifact_fingerprint": artifact_fingerprint,
            "resource_summary": _resource_summary_to_tree(resource_summary),
        }
    )


def _manifest_from_tree(tree: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "repository_id",
        "artifact_schema",
        "artifact_blob",
        "artifact_fingerprint",
        "resource_summary",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CalibrationArtifact manifest has an unknown field set")
    if tree["schema"] != CALIBRATION_MANIFEST_SCHEMA:
        raise ValueError("unsupported CalibrationArtifact manifest schema")
    if tree["artifact_schema"] != CALIBRATION_ARTIFACT_SCHEMA:
        raise ValueError("CalibrationArtifact manifest names another artifact schema")
    _canonical_text(tree["repository_id"], "repository_id")
    _sha256(tree["artifact_fingerprint"], "artifact_fingerprint")
    _content_ref_from_tree(tree["artifact_blob"])
    _resource_summary_from_tree(tree["resource_summary"])
    return tree


def _resource_summary_to_tree(
    summary: CalibrationResourceSummary,
) -> dict[str, int]:
    if not isinstance(summary, CalibrationResourceSummary):
        raise TypeError("summary must be CalibrationResourceSummary")
    return {
        "site_count": summary.site_count,
        "model_count": summary.model_count,
        "kernel_elements": summary.kernel_elements,
        "max_sampled_pixels_per_model": summary.max_sampled_pixels_per_model,
        "total_sampled_pixels_all_models": summary.total_sampled_pixels_all_models,
    }


def _resource_summary_from_tree(tree: Any) -> CalibrationResourceSummary:
    fields = {
        "site_count",
        "model_count",
        "kernel_elements",
        "max_sampled_pixels_per_model",
        "total_sampled_pixels_all_models",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("calibration resource summary has an unknown field set")
    if any(type(tree[field]) is not int for field in fields):
        raise ValueError("calibration resource summary fields must be canonical integers")
    return CalibrationResourceSummary(
        tree["site_count"],
        tree["model_count"],
        tree["kernel_elements"],
        tree["max_sampled_pixels_per_model"],
        tree["total_sampled_pixels_all_models"],
    )


__all__ = [
    "CALIBRATION_MANIFEST_SCHEMA",
    "CalibrationRepository",
]
