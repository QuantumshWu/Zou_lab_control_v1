import inspect
import threading
import time

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.core.signals import (
    SignalCursor,
    SignalHistoryGap,
    SignalHub,
    SignalSchema,
)
from Zou_lab_control.neutral_atom.core.signal_tensor import TensorStore


def _scalar(value):
    return np.asarray(value, dtype=np.float64).reshape(1, 1, 1)


def test_single_signal_cursor_consumes_every_update_not_latest():
    hub = SignalHub(history=8)
    cursors = hub.signal_cursors(["y"])
    assert cursors["y"].position == SignalCursor(0, 0)

    schema = SignalSchema(dtype=np.float64, repeat_capacity=1)
    hub.register_signal("y", schema)
    for provenance, value in ((10, 1.0), (11, 2.0), (12, 3.0)):
        hub.publish({"y": _scalar(value)}, provenance=provenance)

    seen = []
    for _ in range(3):
        update = hub.next_coherent_update(["y"], cursors, timeout=0)
        assert update is not None
        seen.append((update.provenance, float(update.tensors["y"].data[0, 0, 0])))
        cursors = dict(update.cursors)

    assert seen == [(10, 1.0), (11, 2.0), (12, 3.0)]
    assert hub.next_coherent_update(["y"], cursors, timeout=0) is None


def test_multi_signal_cursor_selects_earliest_common_provenance():
    hub = SignalHub(history=8)
    schema = SignalSchema(dtype=np.float64, repeat_capacity=1)
    hub.register_signal("a", schema)
    hub.register_signal("b", schema)
    cursors = hub.signal_cursors(["a", "b"])

    hub.publish({"a": _scalar(1)}, provenance=1)
    hub.publish({"a": _scalar(2)}, provenance=2)
    hub.publish({"b": _scalar(20)}, provenance=2)

    update = hub.next_coherent_update(["a", "b"], cursors, timeout=0)
    assert update is not None and update.provenance == 2
    assert float(update.tensors["a"].data[0, 0, 0]) == 2
    assert float(update.tensors["b"].data[0, 0, 0]) == 20
    assert update.cursors["a"].version == 2  # unmatched provenance=1 was deliberately passed
    assert update.cursors["b"].version == 1

    cursors = dict(update.cursors)
    hub.publish({"b": _scalar(30)}, provenance=3)
    hub.publish({"a": _scalar(3)}, provenance=3)
    update = hub.next_coherent_update(["a", "b"], cursors, timeout=0)
    assert update is not None and update.provenance == 3
    assert float(update.tensors["a"].data[0, 0, 0]) == 3
    assert float(update.tensors["b"].data[0, 0, 0]) == 30


def test_coherent_session_discards_other_stream_prefix_by_first_stream_order():
    hub = SignalHub(history=8)
    schema = SignalSchema(dtype=np.float64, repeat_capacity=1)
    hub.register_signal("a", schema)
    hub.register_signal("b", schema)
    subscription = hub.signal_cursors(["a", "b"])

    for provenance in (1, 2, 3):
        hub.publish({"a": _scalar(provenance)}, provenance=provenance)
    # Both 2 and 3 are common, but stream a defines order: selecting 2 must
    # deliberately pass b's earlier provenance 3, so it cannot reappear later.
    hub.publish({"b": _scalar(30)}, provenance=3)
    hub.publish({"b": _scalar(20)}, provenance=2)

    update = hub.next_coherent_update(["a", "b"], subscription, timeout=0)
    assert update is not None and update.provenance == 2
    assert update.cursors["a"].version == 2
    assert update.cursors["b"].version == 2
    assert hub.next_coherent_update(["a", "b"], update.cursors, timeout=0) is None


def test_cursor_reports_bounded_history_gap_instead_of_skipping_to_latest():
    hub = SignalHub(history=2)
    schema = SignalSchema(dtype=np.float64, repeat_capacity=1)
    hub.register_signal("y", schema)
    cursor = hub.signal_cursors(["y"])
    for value in (1, 2, 3):
        hub.publish({"y": _scalar(value)}, provenance=value)

    with pytest.raises(SignalHistoryGap, match="earliest retained=2"):
        hub.next_coherent_update(["y"], cursor, timeout=0)


def test_stateless_cursor_replay_is_rejected_instead_of_reintroducing_reverse_scans():
    hub = SignalHub(history=4)
    hub.register_signal("y", SignalSchema(dtype=np.float64, repeat_capacity=1))
    with pytest.raises(TypeError, match="stateless SignalCursor replay is not supported"):
        hub.next_coherent_update(["y"], {"y": SignalCursor(1, 0)}, timeout=0)


def test_subscription_transport_has_no_stateless_rescan_or_reverse_undo_path():
    hub_source = inspect.getsource(SignalHub.next_coherent_update)
    store_source = inspect.getsource(TensorStore)
    assert "update_refs_after" not in store_source
    assert "snapshot_at_version" not in hub_source
    assert "_undo" not in store_source
    assert "reversed(self._updates)" not in store_source


def test_cursor_cannot_cross_explicit_schema_version_replace():
    hub = SignalHub(history=4)
    first = SignalSchema(dtype=np.float64, repeat_capacity=1)
    hub.register_signal("y", first)
    hub.publish({"y": _scalar(1)}, provenance=1)
    cursor = hub.signal_cursors(["y"])

    changed = SignalSchema(data_shape=(2,), dtype=np.float64, repeat_capacity=1)
    hub.register_signal("y", changed, replace=True)
    hub.publish({"y": np.asarray([[[2.0, 3.0]]])}, provenance=2)

    with pytest.raises(SignalHistoryGap, match="schema changed"):
        hub.next_coherent_update(["y"], cursor, timeout=0)


@pytest.mark.parametrize("reset", ["remove", "clear"])
def test_cursor_cannot_cross_removed_and_recreated_same_named_stream(reset):
    hub = SignalHub(history=4)
    schema = SignalSchema(dtype=np.float64, repeat_capacity=1)
    hub.register_signal("y", schema)
    hub.publish({"y": _scalar(1)}, provenance=1)
    cursor = hub.signal_cursors(["y"])

    if reset == "remove":
        hub.remove_signals(["y"])
    else:
        hub.clear()
    hub.register_signal("y", schema)
    hub.publish({"y": _scalar(2)}, provenance=2)
    # A coincidentally equal store version must not hide the generation change,
    # even after enough new updates arrive to advance beyond the old version.
    hub.publish({"y": _scalar(3)}, provenance=3)

    with pytest.raises(SignalHistoryGap, match="schema changed"):
        hub.next_coherent_update(["y"], cursor, timeout=0)


def test_cursor_generation_tombstone_also_applies_to_raw_auto_registration():
    hub = SignalHub(history=4)
    hub.publish({"y": 1.0}, provenance=1)
    cursor = hub.signal_cursors(["y"])
    hub.remove_signals(["y"])
    hub.publish({"y": 2.0}, provenance=2)
    hub.publish({"y": 3.0}, provenance=3)

    with pytest.raises(SignalHistoryGap, match="schema changed"):
        hub.next_coherent_update(["y"], cursor, timeout=0)


def test_publish_rejects_provenance_below_the_no_lineage_sentinel():
    hub = SignalHub()
    hub.register_signal("y", SignalSchema(dtype=np.float64, repeat_capacity=1))
    with pytest.raises(ValueError, match="non-negative source-shot"):
        hub.publish({"y": _scalar(1)}, provenance=-2)
    assert hub.names() == []


def test_consumer_history_reservation_layers_over_producer_policy():
    hub = SignalHub(history=8)
    schema = SignalSchema(dtype=np.float64, repeat_capacity=1)
    hub.register_signal("y", schema, history=3)
    cursors = hub.signal_cursors(["y"])

    lease = hub.reserve_history(["y"], 20)
    assert hub.history_limit("y") == 20
    for value in range(20):
        hub.publish({"y": _scalar(value)}, provenance=value)

    seen = []
    for _ in range(20):
        update = hub.next_coherent_update(["y"], cursors, timeout=0)
        assert update is not None
        seen.append(int(update.tensors["y"].data[0, 0, 0]))
        cursors = dict(update.cursors)
    assert seen == list(range(20))

    lease.release()
    assert hub.history_limit("y") == 3


def test_reserved_cursor_is_lossless_while_producer_publishes_concurrently():
    count = 256
    hub = SignalHub(history=8)
    schema = SignalSchema(dtype=np.float64, repeat_capacity=1)
    hub.register_signal("y", schema, history=8)
    lease = hub.reserve_history(["y"], count)
    cursors = hub.signal_cursors(["y"])
    started = threading.Event()

    def produce():
        started.set()
        for value in range(count):
            hub.publish({"y": _scalar(value)}, provenance=value)

    producer = threading.Thread(target=produce)
    producer.start()
    assert started.wait(1.0)
    seen = []
    for _ in range(count):
        update = hub.next_coherent_update(["y"], cursors, timeout=2.0)
        assert update is not None
        seen.append(int(update.tensors["y"].data[0, 0, 0]))
        cursors = dict(update.cursors)
    producer.join(timeout=2.0)
    lease.release()

    assert not producer.is_alive()
    assert seen == list(range(count))


def _consume_backlog_seconds(count: int, *, coherent_pair: bool) -> float:
    names = ["a", "b"] if coherent_pair else ["a"]
    hub = SignalHub(history=count)
    schema = SignalSchema(dtype=np.float64, repeat_capacity=1)
    for name in names:
        hub.register_signal(name, schema)
    subscription = hub.signal_cursors(names)
    for value in range(count):
        hub.publish(
            {name: _scalar(value + index) for index, name in enumerate(names)},
            provenance=value,
        )

    started = time.perf_counter()
    for provenance in range(count):
        update = hub.next_coherent_update(names, subscription, timeout=0)
        assert update is not None and update.provenance == provenance
    return time.perf_counter() - started


@pytest.mark.parametrize("coherent_pair", [False, True])
def test_stateful_subscription_backlog_scaling_is_near_linear(coherent_pair):
    """128..2048 backlog replay must scale with updates, not remaining_backlog².

    Medians suppress timer noise.  From 256 to 2048, linear work grows 8x and
    the deleted stateless implementation grew about 68x; 24x leaves generous
    loaded-CI headroom while still mechanically rejecting the old complexity.
    """

    counts = (128, 256, 512, 1024, 2048)
    elapsed = {
        count: float(np.median([
            _consume_backlog_seconds(count, coherent_pair=coherent_pair)
            for _ in range(3)
        ]))
        for count in counts
    }
    assert elapsed[2048] / elapsed[256] < 24.0, elapsed
    per_update = [elapsed[count] / count for count in counts[1:]]
    assert max(per_update) / min(per_update) < 4.0, elapsed
