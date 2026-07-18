import os
from pathlib import Path
import subprocess
import sys

import pytest

# Pin the non-interactive Agg backend for the WHOLE test session, in ONE place (this conftest
# loads before any test module).  Otherwise matplotlib lazily resolves to the GUI 'qtagg' backend
# the first time a test creates a figure under PyQt5, and every plt.figure() then spawns a real Qt
# FigureManager.  The suite USED to set Agg only in scattered per-file `matplotlib.use("Agg")`
# calls, so any test that ran before the first such call (alphabetically earlier) created live Qt
# managers; those are destroyed at interpreter shutdown racing with the QApplication teardown -> an
# INTERMITTENT process-exit segfault (exit 139) even though every test passed.  Pinning Agg here is
# the single source that makes the backend deterministic and kills that flaky teardown crash.  The
# GUI panel canvas (EmbeddedFigureCanvas, an explicit FigureCanvasQTAgg) is unaffected: it imports
# the Qt backend directly, independent of this default.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
root_text = str(REPO_ROOT)
if sys.path[0] != root_text:
    sys.path.insert(0, root_text)

ACTIVE_TEST_MANIFEST = REPO_ROOT / "tests" / "migration_active_tests.txt"


def load_active_test_paths(
    manifest: Path = ACTIVE_TEST_MANIFEST,
) -> tuple[Path, ...]:
    """Validate and return the migration test authority as canonical paths."""

    manifest = Path(manifest).resolve()
    try:
        manifest.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise pytest.UsageError(
            f"migration test manifest escapes the repository: {manifest}"
        ) from exc
    if not manifest.is_file():
        raise pytest.UsageError(
            f"migration test manifest does not exist: {manifest}"
        )

    active: list[Path] = []
    seen: dict[str, int] = {}
    for line_number, raw in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        relative = Path(entry)
        canonical_entry = relative.as_posix()
        if (
            entry != canonical_entry
            or relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 2
            or relative.parts[0] != "tests"
            or not relative.name.startswith("test_")
            or relative.suffix != ".py"
        ):
            raise pytest.UsageError(
                f"{manifest}:{line_number}: invalid active test path {entry!r}"
            )
        previous = seen.get(entry)
        if previous is not None:
            raise pytest.UsageError(
                f"{manifest}:{line_number}: duplicate active test path {entry!r} "
                f"(first declared on line {previous})"
            )
        path = (REPO_ROOT / relative).resolve()
        try:
            path.relative_to(REPO_ROOT / "tests")
        except ValueError as exc:
            raise pytest.UsageError(
                f"{manifest}:{line_number}: active test escapes tests/: {entry!r}"
            ) from exc
        if not path.is_file():
            raise pytest.UsageError(
                f"{manifest}:{line_number}: active test does not exist: {entry!r}"
            )
        seen[entry] = line_number
        active.append(path)
    if not active:
        raise pytest.UsageError("migration active test manifest is empty")
    return tuple(active)


def pytest_configure(config):
    allowed = frozenset(load_active_test_paths())
    requested: set[Path] = set()
    for argument in config.args:
        file_part = str(argument).split("::", 1)[0]
        candidate = Path(file_part)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        candidate = candidate.resolve()
        if candidate.is_dir():
            raise pytest.UsageError(
                "pytest directory/automatic discovery is disabled during migration; "
                "enumerate paths from tests/migration_active_tests.txt"
            )
        if candidate.suffix != ".py" or not candidate.name.startswith("test_"):
            raise pytest.UsageError(
                f"pytest invocation is not an explicit test file: {argument!r}"
            )
        if candidate not in allowed:
            raise pytest.UsageError(
                "pytest invocation is restricted to "
                "tests/migration_active_tests.txt; unauthorized path: "
                f"{candidate}"
            )
        requested.add(candidate)
    if not requested:
        raise pytest.UsageError(
            "pytest requires explicit paths from tests/migration_active_tests.txt"
        )
    config._zlc_active_test_paths = allowed
    config._zlc_requested_test_paths = frozenset(requested)


def pytest_ignore_collect(collection_path, config):
    """Reject historical tests before pytest imports their modules."""

    path = Path(str(collection_path)).resolve()
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    allowed = config._zlc_active_test_paths
    return path not in allowed


def pytest_collection_modifyitems(config, items):
    allowed = config._zlc_active_test_paths
    collected = {Path(str(item.path)).resolve() for item in items}
    unauthorized = sorted(str(path) for path in collected if path not in allowed)
    if unauthorized:
        raise pytest.UsageError(
            "pytest collected tests outside tests/migration_active_tests.txt: "
            + ", ".join(unauthorized)
        )
    if config._zlc_requested_test_paths == allowed:
        missing = sorted(str(path) for path in allowed - collected)
        if missing:
            raise pytest.UsageError(
                "active manifest files collected no tests: " + ", ".join(missing)
            )


def tracked_repo_files(pattern: str) -> tuple[Path, ...]:
    """Return repository fixtures selected from Git's tracked-file set only.

    Test coverage must not change when a developer has ignored experiment files
    next to the committed fixtures.  Repository-level tests may depend on Git,
    but they must never discover their input set with a filesystem glob.
    """

    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", pattern],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        REPO_ROOT / relative
        for relative in result.stdout.splitlines()
        if relative
    )


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.close("all")


def pulse_backend_completion_for(artifact, *, transport_id="test-transport"):
    """One valid typed hardware-backend receipt for current pulse unit tests."""

    from fpga.pulse_streamer.host.image import STATUS_DONE
    from zlc_pulse import (
        AUTONOMOUS_TABLE_READ_RECIPE,
        POST_TERMINAL_TAIL_WAIT_RECIPE,
        STATIC_STATUS_READ_RECIPE,
        AutonomousTableTerminalEvidence,
        PostTerminalTailEvidence,
        PulseBackendCompletion,
        PulseExecutionForm,
        StaticOnceTerminalEvidence,
    )

    if artifact.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        cursor = len(artifact.target_ir.scan_points) - 1
        terminal = AutonomousTableTerminalEvidence(
            AUTONOMOUS_TABLE_READ_RECIPE,
            transport_id,
            STATUS_DONE,
            cursor,
            STATUS_DONE,
            cursor,
            False,
            2,
        )
    else:
        terminal = StaticOnceTerminalEvidence(
            STATIC_STATUS_READ_RECIPE,
            transport_id,
            STATUS_DONE,
            STATUS_DONE,
            False,
            2,
        )
    required_ticks = artifact.max_configured_output_delay_ticks
    elapsed_ns = int(
        (required_ticks * 1_000_000_000 + artifact.target_ir.clock_hz - 1)
        // artifact.target_ir.clock_hz
    )
    tail = PostTerminalTailEvidence(
        terminal.fingerprint,
        POST_TERMINAL_TAIL_WAIT_RECIPE,
        required_ticks,
        artifact.target_ir.clock_hz,
        elapsed_ns,
    )
    return PulseBackendCompletion(terminal, tail)
