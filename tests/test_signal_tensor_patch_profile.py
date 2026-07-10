"""Deterministic complexity benchmark for cumulative scan transport.

Wall-clock thresholds are noisy on CI.  The transport exposes exact copy/journal
byte counters instead, which directly prove the complexity class that matters:
P point patches copy O(P*D), never P full O(P*D) snapshots.
"""

from __future__ import annotations

import tracemalloc

import numpy as np

from Zou_lab_control.neutral_atom.core.signal_tensor import SignalSchema, TensorPatch
from Zou_lab_control.neutral_atom.core.signals import SignalHub


def test_cumulative_scan_patch_storage_is_linear_not_quadratic():
    points = 512
    data_items = 4
    hub = SignalHub(history=points)
    schema = SignalSchema(point_shape=(points,), data_shape=(data_items,), dtype=np.float64,
                          repeat_capacity=1)
    hub.register_signal("scan", schema, initialize=True)

    for point in range(points):
        row = np.full(data_items, float(point), dtype=np.float64)
        hub.publish_patch("scan", TensorPatch.point(0, point, row), provenance=point + 1)

    stats = hub.storage_stats("scan")
    row_bytes = data_items * np.dtype(np.float64).itemsize
    one_block_bytes = points * row_bytes
    quadratic_snapshot_bytes = points * one_block_bytes

    assert stats["patch_updates"] == points
    assert stats["full_updates"] == 0
    assert stats["bytes_copied_in"] == points * row_bytes
    # One current block + one forward-replay base checkpoint + one row delta per
    # retained version (+ tiny masks).
    assert stats["storage_nbytes"] < quadratic_snapshot_bytes // 50
    assert stats["journal_payload_nbytes"] < points * (row_bytes + 4)

    latest = hub.latest("scan")
    assert latest.shape == (1, points, data_items)
    np.testing.assert_array_equal(latest[0, :, 0], np.arange(points, dtype=float))


def test_history_materializes_only_on_explicit_read_and_does_not_change_store_size():
    points = 64
    hub = SignalHub(history=points)
    schema = SignalSchema(point_shape=(points,), data_shape=(1,), dtype=np.float64,
                          repeat_capacity=1)
    hub.register_signal("scan", schema, initialize=True)
    for point in range(points):
        hub.publish_patch("scan", TensorPatch.point(0, point, np.array([point], dtype=float)))

    stored_before = hub.storage_stats("scan")["storage_nbytes"]
    history = hub.history("scan")
    assert history.shape == (points, 1, points, 1)
    assert hub.storage_stats("scan")["storage_nbytes"] == stored_before


def test_explicit_history_read_has_one_materialized_output_not_list_plus_stack():
    points = 1024
    updates = 64
    hub = SignalHub(history=updates)
    schema = SignalSchema(point_shape=(points,), data_shape=(1,), dtype=np.float64,
                          repeat_capacity=1)
    hub.register_signal("scan", schema, initialize=True)
    for point in range(updates):
        hub.publish_patch(
            "scan", TensorPatch.point(0, point, np.array([point], dtype=float)),
            provenance=point + 1,
        )

    tracemalloc.start()
    history = hub.history_tensor("scan")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    output_bytes = history.data.nbytes + history.valid.nbytes
    assert history.versions == tuple(range(1, updates + 1))
    assert history.provenance == tuple(range(1, updates + 1))
    np.testing.assert_array_equal(history.data[-1, 0, :updates, 0], np.arange(updates))
    # The result itself dominates.  Building a list of N state copies and then
    # stacking it peaks near 2x; direct preallocation stays well below 1.5x.
    assert peak < output_bytes * 1.5, f"history peak {peak:,} B for {output_bytes:,} B output"
