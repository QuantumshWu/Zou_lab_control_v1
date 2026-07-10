"""Saved-frame runs fail loudly before group/shot alignment can shift."""

from __future__ import annotations

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.operations.imageio import index_run


def _write(folder, numbers, *, shape=(4, 5)):
    folder.mkdir()
    for number in numbers:
        np.save(folder / f"img{number}.npy", np.full(shape, number, dtype=np.uint16))


def test_index_run_rejects_missing_frame_instead_of_shifting_later_groups(tmp_path):
    folder = tmp_path / "gap"
    _write(folder, [1, 2, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError, match="missing frame numbers.*3"):
        index_run(folder, "img", shots_per_group=4, short_shot=3, ref_shots=(1, 2, 4))


def test_index_run_rejects_partial_final_group(tmp_path):
    folder = tmp_path / "partial"
    _write(folder, range(1, 7))
    with pytest.raises(ValueError, match="incomplete loading"):
        index_run(folder, "img", shots_per_group=4, short_shot=3, ref_shots=(1, 2, 4))


def test_index_run_rejects_non_one_based_numbering(tmp_path):
    folder = tmp_path / "offset"
    _write(folder, range(2, 10))
    with pytest.raises(ValueError, match="contiguous from 1"):
        index_run(folder, "img", shots_per_group=4, short_shot=3, ref_shots=(1, 2, 4))


def test_run_index_checks_every_frame_shape_on_read(tmp_path):
    folder = tmp_path / "shape"
    _write(folder, range(1, 5))
    np.save(folder / "img3.npy", np.zeros((3, 5), dtype=np.uint16))
    run = index_run(folder, "img", shots_per_group=4, short_shot=3, ref_shots=(1, 2, 4))
    with pytest.raises(ValueError, match="img3.npy has shape"):
        list(run.short_frames())
