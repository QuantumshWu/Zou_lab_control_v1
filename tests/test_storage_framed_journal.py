"""Durability contracts for the shared append-only canonical journal."""

from __future__ import annotations

import threading

import pytest
import zlc_storage.framed_journal as journal_module

from zlc_storage import FramedJournal, JournalCorruptionError


def test_framed_journal_is_idempotent_and_rejects_conflicting_ids(tmp_path):
    path = tmp_path / "journal.zlcj"
    journal = FramedJournal(path)
    assert journal.append("record-one", {"value": 1})
    assert not journal.append("record-one", {"value": 1})
    with pytest.raises(ValueError, match="conflicting content"):
        journal.append("record-one", {"value": 2})
    assert FramedJournal(path).records() == (("record-one", {"value": 1}),)


def test_framed_journal_repairs_only_a_torn_final_frame(tmp_path):
    path = tmp_path / "journal.zlcj"
    journal = FramedJournal(path)
    journal.append("complete", {"value": "durable"})
    durable_size = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(b"ZLC")

    reopened = FramedJournal(path)
    assert reopened.records() == (("complete", {"value": "durable"}),)
    assert path.stat().st_size == durable_size


def test_framed_journal_fails_closed_on_complete_frame_corruption(tmp_path):
    path = tmp_path / "journal.zlcj"
    journal = FramedJournal(path)
    journal.append("record", {"value": "durable"})
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(JournalCorruptionError, match="digest mismatch"):
        FramedJournal(path)


def test_framed_journal_serializes_concurrent_appenders(tmp_path):
    journal = FramedJournal(tmp_path / "journal.zlcj")
    errors = []

    def append(index: int) -> None:
        try:
            journal.append(f"record-{index}", {"index": index})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=append, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(journal.records()) == 16


def test_framed_journal_decodes_each_existing_record_once_per_scan(
    tmp_path,
    monkeypatch,
):
    journal = FramedJournal(tmp_path / "journal.zlcj")
    expected = tuple(
        (f"record-{index}", {"index": index})
        for index in range(4)
    )
    for record_id, value in expected:
        journal.append(record_id, value)

    decode_calls = 0
    real_decode = journal_module.decode

    def counting_decode(payload):
        nonlocal decode_calls
        decode_calls += 1
        return real_decode(payload)

    monkeypatch.setattr(journal_module, "decode", counting_decode)

    assert journal.records() == expected
    assert decode_calls == len(expected)

    decode_calls = 0
    validated = []
    assert journal.append_checked(
        "record-next",
        {"index": len(expected)},
        validated.append,
    )
    expected_after_append = expected + (
        ("record-next", {"index": len(expected)}),
    )
    assert validated == [expected_after_append]
    assert decode_calls == len(expected) + 1  # existing records + candidate normalization

    decode_calls = 0
    validated.clear()
    assert not journal.append_checked(
        "record-next",
        {"index": len(expected)},
        validated.append,
    )
    assert validated == [expected_after_append]
    assert decode_calls == len(expected_after_append)
