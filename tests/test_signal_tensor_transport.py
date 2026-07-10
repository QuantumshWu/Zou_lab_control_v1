"""Core contract for the mandatory-leading-axis signal transport.

These tests intentionally stop at the neutral-atom core boundary.  Producer and
frontend migrations are covered separately; the transport itself must never
infer repeat/point semantics from ndarray rank.
"""

from __future__ import annotations

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.core.signal_tensor import SignalSchema, SignalTensor, TensorPatch
from Zou_lab_control.neutral_atom.core.signals import NO_LINEAGE, SignalHub
from Zou_lab_control.neutral_atom.operations.signal_expr import NAMESPACE_HELPERS, hub_namespace


def test_schema_requires_both_nonempty_logical_layouts():
    for kwargs in (
        {"point_shape": ()},
        {"data_shape": ()},
        {"point_shape": (0,)},
        {"data_shape": (3, -1)},
    ):
        with pytest.raises(ValueError):
            SignalSchema(**kwargs)
    with pytest.raises(TypeError):
        SignalSchema(dtype=object)
    with pytest.raises(ValueError):
        SignalSchema(repeat_capacity=0)


def test_unregistered_raw_publish_has_one_deterministic_external_schema():
    hub = SignalHub()
    original = np.arange(6, dtype=np.int16).reshape(2, 3)
    hub.publish({"raw": original})

    # Raw rank never implies repeat or points: the WHOLE 2x3 array is one datum.
    assert hub.latest("raw").shape == (1, 1, 2, 3)
    assert hub.latest_logical("raw").shape == (1, 1, 2, 3)
    np.testing.assert_array_equal(hub.latest_logical("raw")[0, 0], original)
    schema = hub.schema("raw")
    assert schema.point_shape == (1,)
    assert schema.data_shape == (2, 3)
    assert schema.metadata["origin"] == "external-single-datum"
    hub.publish({"raw": original + 10})
    np.testing.assert_array_equal(hub.latest("raw")[0, 0], original + 10)
    with pytest.raises(ValueError):
        hub.publish({"raw": np.zeros((3, 2), dtype=np.int16)})

    hub.publish({"scalar": 2.5})
    scalar = hub.latest("scalar")
    assert isinstance(scalar, np.ndarray) and scalar.shape == (1, 1, 1)
    assert scalar[0, 0, 0] == pytest.approx(2.5)


def test_unregistered_signal_tensor_carries_explicit_repeat_point_meaning():
    hub = SignalHub()
    schema = SignalSchema(point_shape=(3,), data_shape=(2,), dtype=np.float32,
                          repeat_capacity=4)
    logical = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    hub.publish({"typed": SignalTensor.from_value(logical, schema)})
    assert hub.latest("typed").shape == (4, 3, 2)
    np.testing.assert_array_equal(hub.latest_logical("typed"), logical)


def test_registered_image_keeps_physical_hw_axes_and_restores_logical_shape():
    hub = SignalHub()
    schema = SignalSchema(point_shape=(1,), data_shape=(4, 5), dtype=np.uint16,
                          repeat_capacity=2, label="camera image", unit="counts")
    hub.register_signal("frame", schema)
    logical = np.arange(40, dtype=np.uint16).reshape(2, 1, 4, 5)
    hub.publish({"frame": logical}, provenance=17)

    canonical = hub.latest("frame")
    assert canonical.shape == (2, 1, 4, 5)
    restored = hub.latest_logical("frame")
    assert restored.shape == (2, 1, 4, 5)
    np.testing.assert_array_equal(restored, logical)
    tensor = hub.latest_tensor("frame")
    assert tensor.provenance == 17 and tensor.schema is hub.schema("frame")
    assert not tensor.data.flags.writeable and not tensor.valid.flags.writeable


def test_multidimensional_point_layout_unflattens_only_the_point_axis():
    hub = SignalHub()
    schema = SignalSchema(point_shape=(2, 3), data_shape=(4,), dtype=np.float64,
                          repeat_capacity=2)
    logical = np.arange(48, dtype=float).reshape(2, 2, 3, 4)
    hub.register_signal("grid_scan", schema)
    hub.publish({"grid_scan": logical})
    assert hub.latest("grid_scan").shape == (2, 6, 4)
    np.testing.assert_array_equal(hub.latest_logical("grid_scan"), logical)


def test_variable_rank_image_patch_preserves_trailing_axes():
    hub = SignalHub()
    schema = SignalSchema(point_shape=(3,), data_shape=(2, 4), dtype=np.uint16,
                          repeat_capacity=1)
    hub.register_signal("images", schema, initialize=True)
    image = np.arange(8, dtype=np.uint16).reshape(2, 4)
    hub.publish_patch("images", TensorPatch.point(0, 1, image))
    block = hub.latest("images")
    assert block.shape == (1, 3, 2, 4)
    np.testing.assert_array_equal(block[0, 1], image)
    np.testing.assert_array_equal(hub.latest_valid("images"), [[False, True, False]])


def test_multidimensional_patch_history_reverses_without_flattening_data_axes():
    hub = SignalHub(history=4)
    schema = SignalSchema(point_shape=(2,), data_shape=(2, 3), dtype=np.float64,
                          repeat_capacity=1)
    hub.register_signal("images", schema, initialize=True)
    first = np.arange(6, dtype=np.float64).reshape(2, 3)
    second = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
    hub.publish_patch("images", TensorPatch.point(0, 0, first), provenance=10)
    hub.publish_patch("images", TensorPatch.point(0, 1, second), provenance=11)

    history = hub.history_tensor("images")
    assert history.data.shape == (2, 1, 2, 2, 3)
    assert history.logical().shape == (2, 1, 2, 2, 3)
    np.testing.assert_array_equal(history.data[0, 0, 0], first)
    assert np.isnan(history.data[0, 0, 1]).all()
    np.testing.assert_array_equal(history.data[1, 0, 1], second)
    at_first = hub.snapshot_at(10, tensors=True)["images"]
    assert at_first.data.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(at_first.data[0, 0], first)
    assert np.isnan(at_first.data[0, 1]).all()


def test_registered_boundary_rejects_wrong_shape_dtype_and_control_objects():
    hub = SignalHub()
    schema = SignalSchema(point_shape=(3,), data_shape=(2,), dtype=np.float64,
                          repeat_capacity=1)
    hub.register_signal("curve", schema)
    with pytest.raises(ValueError, match="schema requires physical"):
        hub.publish({"curve": np.zeros((3,), dtype=np.float64)})
    with pytest.raises(TypeError, match="does not match registered dtype"):
        hub.publish({"curve": np.zeros((1, 3, 2), dtype=np.float32)})
    with pytest.raises(TypeError, match="Control text"):
        hub.publish({"stage": "acquiring"})
    assert hub.names() == []
    assert sorted(hub.registered_names()) == ["curve"]


def test_multi_signal_publish_is_preflighted_before_any_value_lands():
    hub = SignalHub()
    scalar = SignalSchema(dtype=np.float64, repeat_capacity=1)
    vector = SignalSchema(point_shape=(1,), data_shape=(2,), dtype=np.float64,
                          repeat_capacity=1)
    hub.register_signal("a", scalar)
    hub.register_signal("b", vector)
    hub.publish({"a": 1.0, "b": np.array([[2.0, 3.0]])})
    before_a = hub.latest("a")
    before_b = hub.latest("b")
    before_versions = hub.signal_versions()
    before_global = hub.version

    with pytest.raises(ValueError):
        hub.publish({"a": 9.0, "b": np.array([4.0, 5.0, 6.0])})  # wrong declared data shape

    np.testing.assert_array_equal(hub.latest("a"), before_a)
    np.testing.assert_array_equal(hub.latest("b"), before_b)
    assert hub.signal_versions() == before_versions
    assert hub.version == before_global


def test_schema_change_is_explicit_version_and_clears_old_history():
    hub = SignalHub()
    first = SignalSchema(dtype=np.float64, repeat_capacity=1)
    hub.register_signal("value", first)
    hub.publish({"value": 1.0})
    assert hub.latest_tensor("value").schema_version == 1

    second = SignalSchema(point_shape=(1,), data_shape=(2,), dtype=np.float64,
                          repeat_capacity=1)
    with pytest.raises(ValueError, match="different schema"):
        hub.register_signal("value", second)
    hub.register_signal("value", second, replace=True)
    assert "value" not in hub.names()
    hub.publish({"value": np.array([[3.0, 4.0]])})
    tensor = hub.latest_tensor("value")
    assert tensor.schema_version == 2 and tensor.shape == (1, 1, 2)
    assert hub.history("value").shape == (1, 1, 1, 2)


def test_schema_metadata_change_is_an_explicit_new_version_too():
    hub = SignalHub()
    first = SignalSchema(point_shape=(2,), data_shape=(1,), dtype=np.float64,
                         repeat_capacity=1,
                         metadata={"point_coords": np.array([1.0, 2.0])})
    changed = SignalSchema(point_shape=(2,), data_shape=(1,), dtype=np.float64,
                           repeat_capacity=1,
                           metadata={"point_coords": np.array([3.0, 4.0])})
    hub.register_signal("scan", first)
    with pytest.raises(ValueError, match="different schema"):
        hub.register_signal("scan", changed)
    hub.register_signal("scan", changed, replace=True)
    hub.publish({"scan": np.array([[10.0], [20.0]])})
    assert hub.latest_tensor("scan").schema_version == 2


def test_patch_history_and_snapshot_at_reconstruct_versions_without_shape_guessing():
    hub = SignalHub(history=8)
    schema = SignalSchema(point_shape=(4,), data_shape=(1,), dtype=np.float64,
                          repeat_capacity=1)
    hub.register_signal("scan", schema, initialize=True)
    for point in range(4):
        hub.publish_patch(
            "scan", TensorPatch.point(0, point, np.array([10.0 + point])),
            provenance=point + 1)

    history = hub.history_tensor("scan")
    assert history.data.shape == (4, 1, 4, 1)  # update is explicit and is NOT a signal axis
    assert history.versions == (1, 2, 3, 4)
    assert history.provenance == (1, 2, 3, 4)
    np.testing.assert_allclose(history.data[-1, 0, :, 0], [10, 11, 12, 13])
    assert np.isnan(history.data[0, 0, 1:, 0]).all()

    at_two = hub.snapshot_at(2, tensors=True)["scan"]
    np.testing.assert_allclose(at_two.data[0, :2, 0], [10, 11])
    assert np.isnan(at_two.data[0, 2:, 0]).all()
    np.testing.assert_array_equal(at_two.valid, [[True, True, False, False]])


def test_forward_checkpoint_advances_on_eviction_and_replays_retained_patch_states():
    hub = SignalHub(history=3)
    schema = SignalSchema(point_shape=(4,), data_shape=(1,), dtype=np.float64,
                          repeat_capacity=1)
    hub.register_signal("scan", schema, initialize=True)
    updates = ((0, 10.0), (1, 11.0), (2, 12.0), (3, 13.0), (0, 99.0))
    for provenance, (point, value) in enumerate(updates, start=1):
        hub.publish_patch(
            "scan", TensorPatch.point(0, point, np.array([value])),
            provenance=provenance,
        )

    history = hub.history_tensor("scan")
    assert history.versions == (3, 4, 5)
    assert history.provenance == (3, 4, 5)
    np.testing.assert_allclose(history.data[0, 0, :3, 0], [10.0, 11.0, 12.0])
    assert np.isnan(history.data[0, 0, 3, 0])
    np.testing.assert_allclose(history.data[1, 0, :, 0], [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_allclose(history.data[2, 0, :, 0], [99.0, 11.0, 12.0, 13.0])

    at_three = hub.snapshot_at(3, tensors=True)["scan"]
    np.testing.assert_allclose(at_three.data[0, :3, 0], [10.0, 11.0, 12.0])
    assert np.isnan(at_three.data[0, 3, 0])


def test_patch_rejects_stale_version_broadcast_and_dtype_change():
    hub = SignalHub()
    schema = SignalSchema(point_shape=(2,), data_shape=(3,), dtype=np.float32,
                          repeat_capacity=1)
    hub.register_signal("x", schema, initialize=True)
    hub.publish_patch("x", TensorPatch.point(0, 0, np.ones(3, dtype=np.float32), expected_version=0))
    with pytest.raises(RuntimeError, match="stale tensor patch"):
        hub.publish_patch("x", TensorPatch.point(0, 1, np.ones(3, dtype=np.float32), expected_version=0))
    with pytest.raises(ValueError, match="broadcasting is forbidden"):
        hub.publish_patch("x", TensorPatch.point(0, 1, np.array([1.0], dtype=np.float32)))
    with pytest.raises(TypeError, match="patch dtype"):
        hub.publish_patch("x", TensorPatch.point(0, 1, np.ones(3, dtype=np.float64)))
    assert hub.signal_versions()["x"] == 1


def test_signal_expression_namespace_exposes_canonical_and_schema_driven_views():
    hub = SignalHub()
    schema = SignalSchema(point_shape=(1,), data_shape=(2, 3), dtype=np.float64,
                          repeat_capacity=1)
    hub.register_signal("img", schema)
    hub.publish({"img": np.arange(6, dtype=float).reshape(1, 1, 2, 3)})
    ns = hub_namespace(hub)

    assert ns["img"].shape == (1, 1, 2, 3)
    assert ns["latest"]("img").shape == (1, 1, 2, 3)
    assert ns["logical"]("img").shape == (1, 1, 2, 3)
    assert ns["tensor"]("img").schema.data_shape == (2, 3)
    assert ns["valid"]("img").shape == (1, 1)
    assert ns["history"]("img").shape == (1, 1, 1, 2, 3)
    assert ns["history_logical"]("img").shape == (1, 1, 1, 2, 3)
    assert set(NAMESPACE_HELPERS) == {
        "history", "history_logical", "latest", "tensor", "logical", "valid", "schema",
        "names", "shot", "np", "numpy", "math",
    }


def test_no_lineage_values_stay_latest_in_coherent_snapshot():
    hub = SignalHub()
    hub.publish({"free": 1.0})
    assert hub.latest_tensor("free").provenance == NO_LINEAGE
    hub.publish({"free": 2.0})
    assert hub.snapshot_at(999)["free"][0, 0, 0] == pytest.approx(2.0)
