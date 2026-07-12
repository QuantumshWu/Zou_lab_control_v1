"""MECHANICAL guard: host AXI bursts must NEVER cross a 4 KB address boundary.

Real-hardware root cause (the ``pulse_test`` scan glitch): ``_burst_runs`` coalesced
address-contiguous writes into INCR bursts capped only at ``burst_max`` beats and NEVER
split them at a 4 KB (4096-byte) boundary.  AXI4 (IHI0022, A3.4.1) forbids a single burst
from crossing a 4 KB boundary, and Vivado ``create_hw_axi_txn ... -burst INCR`` issues ONE
transaction per run with no auto-split -- so a straddling run's post-boundary beats are
misaligned on hardware.

The dense scan region (``num_slots`` words per point, points adjacent) is where this shows:
with the default geometry the scan region base sits 256 B into a 4 KB page, so uploading a
2000-point / 3-slot scan produced SEVEN illegal crossing bursts -- the first corrupts scan
point 240, then every +256 points (240, 496, 752, ...), which is exactly the user's "the
swept value jumps at ~260 points, recurring."  4096 / (4 bytes * num_slots) = 256 scan
points per 4 KB, uniquely fixing the period; no other structure (bank_size=2048 points,
burst_max=64 points) has a 256-point period.

These tests exercise the REAL host geometry (``image.default_params`` / ``region_bases`` /
``scan_bank_words``) and the REAL ``_burst_runs``; a local copy of the OLD (pre-fix) inner
loop pins the 7-crossings-at-240/+256 signature so the guard can never silently pass on a
no-op fix.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from fpga.pulse_streamer.host import image as im
from zlc_pulse.transport import VivadoAxiRegisterTransport


_AXI_BURST_BOUNDARY_BYTES = 4096


# --------------------------------------------------------------------------- helpers
def _make_session(tmp_path, burst_max):
    return VivadoAxiRegisterTransport(
        state_dir=tmp_path, tcl_executor=lambda *a: "ok\n", burst_max=burst_max
    )


def _crossings(runs):
    """Runs (base_byte, [values]) whose first and last beat lie in different 4 KB pages."""
    B = _AXI_BURST_BOUNDARY_BYTES
    out = []
    for base, vals in runs:
        last = base + 4 * (len(vals) - 1)
        if base // B != last // B:
            out.append((base, len(vals)))
    return out


def _flatten(runs):
    """Expand runs back to the (byte_addr, value) beat list they encode (INCR, stride 4)."""
    out = []
    for base, vals in runs:
        for k, v in enumerate(vals):
            out.append((base + 4 * k, v))
    return out


def _old_burst_runs(pending, burst_max):
    """The PRE-FIX inner loop (no 4 KB clause) -- kept here to LOCK the 7-crossings
    signature so the real-fixture regression can never silently pass on a no-op."""
    runs = []
    i = 0
    n = len(pending)
    while i < n:
        base, val = pending[i]
        vals = [val]
        j = i + 1
        while j < n and len(vals) < burst_max and pending[j][0] == base + 4 * len(vals):
            vals.append(pending[j][1])
            j += 1
        runs.append((base, vals))
        i = j
    return runs


def _scan_pending(n_points, n_slots_used):
    """The exact (byte_addr, value) writes for chunk 0 of an ``n_points`` scan with
    ``n_slots_used`` swept slots, via the REAL packer -- addresses match hardware."""
    p = im.default_params()

    class _Prog:
        scan_points = [[(i * 1000 + s) for s in range(n_slots_used)] for i in range(n_points)]
        slot_count = n_slots_used

    words = im.scan_bank_words(_Prog(), p, 0, target_bank=0)
    return [(off * 4, words[off]) for off in sorted(words)], p


# ------------------------------------------------------------------- invariant tests
@pytest.mark.parametrize("burst_max", [1, 7, 64, 256])
def test_no_burst_crosses_a_4kb_boundary_and_data_is_preserved(tmp_path, burst_max):
    """For a dense scan upload, EVERY coalesced burst stays inside one 4 KB page, no run
    exceeds burst_max beats, each run is strictly stride-4 contiguous, and the beats --
    values AND addresses, in order -- are byte-identical to the queued writes."""
    s = _make_session(tmp_path, burst_max)
    pending, _ = _scan_pending(2000, 3)
    runs = s._burst_runs(pending)

    assert _crossings(runs) == []                                   # the whole point
    for base, vals in runs:
        assert len(vals) <= burst_max                               # beat cap intact
        # strictly address-contiguous within a run
        assert base % 4 == 0
    assert _flatten(runs) == list(pending)                          # order + data exact


def test_real_pulse_test_scan_geometry_had_seven_crossings_now_zero(tmp_path):
    """LOCK the user's exact reported scenario: the default geometry's 2000x(<=4)-slot scan
    produced 7 illegal crossings -- first at scan point 240, then every +256 -- and the fix
    drives that to 0 with byte-identical data.  Uses the OLD inner loop to prove the 7 were
    real (guards against a silent no-op fix)."""
    pending, p = _scan_pending(2000, 3)
    scan_base_word = im.region_bases(p)["scan"]

    old = _old_burst_runs(pending, burst_max=256)
    old_cross = _crossings(old)
    # translate each crossing's 4 KB boundary byte back to a scan-point index
    B = _AXI_BURST_BOUNDARY_BYTES
    pts = []
    for base, n in old_cross:
        boundary_byte = ((base // B) + 1) * B
        pts.append((boundary_byte // 4 - scan_base_word) // p.scan_words)
    assert pts == [240, 496, 752, 1008, 1264, 1520, 1776]          # the observed signature

    s = _make_session(tmp_path, 256)
    new = s._burst_runs(pending)
    assert _crossings(new) == []
    assert _flatten(new) == list(pending)


def test_matches_the_saved_pulse_test_scan_file_if_present(tmp_path):
    """If the actual saved artifact the user ran is in the tree, the fix must yield 0
    crossings on it too (belt-and-suspenders over the deterministic geometry test)."""
    npy = REPO_ROOT / "pulses" / "pulse_test_scan.npy"
    if not npy.exists():
        pytest.skip("pulses/pulse_test_scan.npy not in this checkout")
    arr = np.load(npy)
    p = im.default_params()

    class _Prog:
        scan_points = [list(r) for r in arr]
        slot_count = arr.shape[1]

    words = im.scan_bank_words(_Prog(), p, 0, target_bank=0)
    pending = [(off * 4, words[off]) for off in sorted(words)]
    s = _make_session(tmp_path, 256)
    assert _crossings(s._burst_runs(pending)) == []


# --------------------------------------------------------------- synthetic edge cases
def test_synthetic_unaligned_bases_and_boundary_edges(tmp_path):
    """A run that starts just before a 4 KB boundary and extends past it splits at the
    boundary; a run whose base is exactly on a boundary, and one word past, both stay legal;
    all preserve order + data."""
    s = _make_session(tmp_path, 256)
    B = _AXI_BURST_BOUNDARY_BYTES

    # 200 contiguous words starting 64 words (256 B) before a 4 KB boundary -> must split
    start = B - 64 * 4
    pending = [(start + 4 * k, 0xB000 + k) for k in range(200)]
    runs = s._burst_runs(pending)
    assert _crossings(runs) == []
    assert _flatten(runs) == pending
    # the split lands exactly ON the boundary
    assert any(base == B for base, _ in runs)

    # base exactly on a boundary, and base one word past -- both never cross
    for base in (B, B + 4):
        p2 = [(base + 4 * k, k) for k in range(300)]
        r2 = s._burst_runs(p2)
        assert _crossings(r2) == []
        assert _flatten(r2) == p2


def test_order_dependent_len1_writes_still_not_merged(tmp_path):
    """The 4 KB clause must not disturb the existing insertion-order invariant: a repeated
    same-address sequence (COMMAND 0 then cmd) stays two len-1 writes in order, and
    non-contiguous addresses stay separate."""
    s = _make_session(tmp_path, 256)
    assert s._burst_runs([(0x4, 0), (0x4, 2)]) == [(0x4, [0]), (0x4, [2])]
    assert s._burst_runs([(0x0, 9), (0x10, 8)]) == [(0x0, [9]), (0x10, [8])]
