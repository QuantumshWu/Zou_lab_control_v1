"""Durability contracts for the shared append-only canonical journal."""

from __future__ import annotations

import pytest
import zlc_storage.framed_journal as journal_module

from zlc_storage import FramedJournal, JournalCorruptionError
from zlc_storage.file_lock import FileLockBusy


def test_framed_journal_is_idempotent_and_rejects_conflicting_ids(tmp_path):
    path = tmp_path / "journal.zlcj"
    with FramedJournal.open_exclusive(path) as journal:
        assert journal.append("record-one", {"value": 1})
        assert not journal.append("record-one", {"value": 1})
        with pytest.raises(ValueError, match="conflicting content"):
            journal.append("record-one", {"value": 2})
    with FramedJournal.open_exclusive(path) as reopened:
        assert reopened.records() == (("record-one", {"value": 1}),)


def test_framed_journal_repairs_only_a_torn_final_frame(tmp_path):
    path = tmp_path / "journal.zlcj"
    with FramedJournal.open_exclusive(path) as journal:
        journal.append("complete", {"value": "durable"})
    durable_size = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(b"ZLC")

    with FramedJournal.open_exclusive(path) as reopened:
        assert reopened.records() == (("complete", {"value": "durable"}),)
    assert path.stat().st_size == durable_size


def test_framed_journal_fails_closed_on_complete_frame_corruption(tmp_path):
    path = tmp_path / "journal.zlcj"
    with FramedJournal.open_exclusive(path) as journal:
        journal.append("record", {"value": "durable"})
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(JournalCorruptionError, match="digest mismatch"):
        FramedJournal.open_exclusive(path)


def test_framed_journal_has_one_lifetime_owner_and_reopens_after_close(tmp_path):
    path = tmp_path / "journal.zlcj"
    owner = FramedJournal.open_exclusive(path)
    try:
        owner.append("record", {"value": 1})
        with pytest.raises(FileLockBusy, match="already held"):
            FramedJournal.open_exclusive(path)
    finally:
        owner.close()
    with FramedJournal.open_exclusive(path) as reopened:
        assert reopened.records() == (("record", {"value": 1}),)


def test_framed_journal_decodes_each_existing_record_once_per_scan(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "journal.zlcj"
    expected = tuple(
        (f"record-{index}", {"index": index})
        for index in range(4)
    )
    with FramedJournal.open_exclusive(path) as journal:
        for record_id, value in expected:
            journal.append(record_id, value)

    decode_calls = 0
    real_decode = journal_module.decode

    def counting_decode(payload):
        nonlocal decode_calls
        decode_calls += 1
        return real_decode(payload)

    monkeypatch.setattr(journal_module, "decode", counting_decode)

    with FramedJournal.open_exclusive(path) as journal:
        # The lifetime session scans each existing frame exactly once.
        assert decode_calls == len(expected)
        assert journal.records() == expected
        assert decode_calls == len(expected)

        assert journal.append("record-next", {"index": len(expected)})
        assert decode_calls == len(expected) + 1
        assert not journal.append("record-next", {"index": len(expected)})
        assert decode_calls == len(expected) + 1
        assert journal.records() == expected + (
            ("record-next", {"index": len(expected)}),
        )
