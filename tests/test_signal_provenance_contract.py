"""#shot-clock: SOURCE-shot provenance contract.

Every value the hub stores carries the source-shot id it derives from: an acquiring node mints a fresh id
per real shot; a reactive processor INHERITS the id of the input it consumed.  This is what lets the
console assemble one coherent cross-producer display shot (the sitemap underlay == the 2D image ==
occupancy), instead of mixing the latest of each independently-threaded producer.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

os.environ.setdefault("ZLC_VIRTUAL_SLEEP_SCALE", "0")


def test_hub_provenance_primitives_and_coherent_read():
    from Zou_lab_control.neutral_atom.core.signals import SignalHub, NO_LINEAGE
    h = SignalHub()
    assert h.latest_provenance("missing") == NO_LINEAGE          # soft-fail, never raises
    s1, s2 = h.next_source_shot(), h.next_source_shot()
    assert (s1, s2) == (1, 2)                                     # strictly monotonic, distinct
    h.publish({"frame_0": np.full((2, 2), 1.0)}, provenance=s1)
    h.publish({"derived": np.array([1.0])}, provenance=s1)        # a consumer inheriting s1
    h.publish({"rate": 7.0})                                      # free-running scalar -> NO_LINEAGE
    h.publish({"frame_0": np.full((2, 2), 2.0)}, provenance=s2)     # frame advances; derived lags at s1
    assert h.latest_provenance("frame_0") == s2
    assert h.latest_provenance("derived") == s1
    assert h.latest_provenance("rate") == NO_LINEAGE
    assert h.provenance_map() == {"frame_0": s2, "derived": s1, "rate": NO_LINEAGE}
    # coherent read at s1: the ahead frame is held back to s1, the scalar shows latest
    snap = h.snapshot_at(s1)
    assert snap["frame_0"][0, 0, 0, 0] == 1.0
    assert snap["derived"][0, 0, 0] == 1.0
    assert snap["rate"][0, 0, 0] == 7.0
    assert h.snapshot_at(None)["frame_0"][0, 0, 0, 0] == 2.0      # None == latest of each
    # clear() / remove_signals() drop the parallel provenance store in lockstep
    h.remove_signals(["frame_0"])
    assert "frame_0" not in h.provenance_map()
    h.clear()
    assert h.provenance_map() == {}


def test_snapshot_at_never_substitutes_latest_for_an_evicted_lineage():
    from Zou_lab_control.neutral_atom.core.signals import SignalHistoryGap, SignalHub

    hub = SignalHub(history=2)
    for provenance in (1, 2, 3):
        hub.publish({"frame_0": np.array([float(provenance)])}, provenance=provenance)

    with pytest.raises(SignalHistoryGap, match="cannot substitute its latest value"):
        hub.snapshot_at(1)
    assert hub.snapshot_at(2)["frame_0"][0, 0, 0] == 2.0
