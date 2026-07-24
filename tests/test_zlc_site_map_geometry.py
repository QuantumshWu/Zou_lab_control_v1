import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from zlc_frontend.site_map import immutable_site_state, site_ring_radius


ROOT = Path(__file__).resolve().parents[1]


def test_site_map_fact_and_exact_view_owners_are_headless() -> None:
    for name in ("site_map.py", "site_map_render.py"):
        tree = ast.parse((ROOT / "zlc_frontend" / name).read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        assert roots.isdisjoint({"PyQt5", "matplotlib"})

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import zlc_frontend.site_map\n"
                "import zlc_frontend.site_map_render\n"
                "assert not any(name == 'PyQt5' or name.startswith('PyQt5.') "
                "for name in sys.modules)\n"
                "assert not any(name == 'matplotlib' or name.startswith('matplotlib.') "
                "for name in sys.modules)\n"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_site_ring_radius_finds_nearest_pair_across_workspace_blocks():
    centers = np.column_stack(
        (np.arange(260, dtype=float) * 100.0, np.zeros(260, dtype=float))
    )
    centers[128, 0] = centers[127, 0] + 10.0

    assert site_ring_radius(centers) == pytest.approx(3.0)


def test_site_ring_radius_preserves_floor_and_duplicate_center_semantics():
    assert site_ring_radius(np.empty((0, 2))) == pytest.approx(1.5)
    assert site_ring_radius(np.asarray(((2.0, 3.0), (2.0, 3.0)))) == pytest.approx(1.5)
    assert site_ring_radius(np.asarray(((0.0, 0.0), (1.0, 0.0)))) == pytest.approx(1.5)
    assert site_ring_radius(np.asarray(((0.0, 0.0), (20.0, 0.0)))) == pytest.approx(6.0)


def test_site_ring_radius_rejects_non_site_matrix_and_bounds_nonfinite_input():
    with pytest.raises(ValueError, match=r"shape \(sites, 2\)"):
        site_ring_radius(np.zeros((3, 3)))

    assert site_ring_radius(np.asarray(((0.0, 0.0), (np.nan, 1.0)))) == pytest.approx(1.5)


def test_immutable_site_state_owns_exact_dtype_shape_and_validity():
    centers = np.asarray(((1, 2), (3, 4), (5, 6), (7, 8)))
    occupied = np.asarray((False, True, False, False), dtype=bool)
    validity = np.asarray((True, True, False, True), dtype=bool)

    frozen_centers, frozen_occupied, frozen_validity = immutable_site_state(
        centers,
        occupied,
        validity,
        site_count=4,
    )
    assert frozen_centers.dtype == np.dtype("<f8")
    assert frozen_centers.shape == (4, 2)
    np.testing.assert_array_equal(frozen_occupied, occupied)
    np.testing.assert_array_equal(frozen_validity, validity)
    assert not frozen_centers.flags.writeable
    assert not frozen_occupied.flags.writeable
    assert not frozen_validity.flags.writeable


def test_immutable_site_state_rejects_noncanonical_or_ambiguous_values():
    with pytest.raises(ValueError, match="canonical False"):
        immutable_site_state(
            np.zeros((2, 2)),
            np.asarray((False, True), dtype=bool),
            np.asarray((True, False), dtype=bool),
            site_count=2,
        )
    with pytest.raises(TypeError, match="bool dtype"):
        immutable_site_state(
            np.zeros((2, 2)),
            np.asarray((0, 1)),
            np.asarray((True, True)),
            site_count=2,
        )
    with pytest.raises(ValueError, match="expected"):
        immutable_site_state(
            np.zeros((2, 2)),
            np.asarray((False,), dtype=bool),
            np.asarray((True, True), dtype=bool),
            site_count=2,
        )
