"""Single owner identity for raw camera sample-to-value projection."""

from zlc_storage import canonical_digest


CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT = canonical_digest(
    {
        "owner": "zlc_neutral_atom.acquisition.camera.CameraDatasetEventAdapter",
        "operator": "camera-sample.image-identity",
    }
)


__all__ = ["CAMERA_DATASET_IDENTITY_OPERATOR_FINGERPRINT"]
