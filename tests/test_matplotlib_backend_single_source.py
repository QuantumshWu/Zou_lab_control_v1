"""The matplotlib backend is pinned to the non-interactive Agg in ONE place -- ``conftest.py`` --
for the whole test session.

If the session is left on the GUI ``qtagg`` backend, every ``plt.figure()`` spawns a real Qt
``FigureManagerQT``; those managers are destroyed at interpreter shutdown racing with the
QApplication teardown, which intermittently segfaults the process at exit (exit 139) even though
every test passed.  Pinning Agg once in conftest makes the backend deterministic.  This guard
fails if any test or import leaves the session on an interactive backend, so the single-source pin
can never silently regress back to the per-file ``matplotlib.use("Agg")`` sprinkling it replaced.
"""

from __future__ import annotations

import matplotlib


def test_session_matplotlib_backend_is_agg():
    backend = matplotlib.get_backend()
    assert backend.lower() == "agg", (
        f"the test session must run on the Agg backend pinned in conftest, got {backend!r}; "
        "a GUI backend reintroduces the flaky Qt-FigureManager teardown segfault"
    )
