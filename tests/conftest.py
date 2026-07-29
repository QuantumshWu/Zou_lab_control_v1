import os
from pathlib import Path
import subprocess
import sys

# Pin the non-interactive Agg backend for the whole test session before any test
# imports Matplotlib.  Production rendering uses explicit OO Agg figures and
# releases each renderer through its owner; tests must not import pyplot later
# as a second global figure manager merely to perform cleanup.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
root_text = str(REPO_ROOT)
if sys.path[0] != root_text:
    sys.path.insert(0, root_text)

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


def private_pulse_backend_snapshot(
    *,
    state: str,
    raw_lane_count: int,
    artifact=None,
    scan_point_count: int | None = None,
    cursor: int | None = None,
    cursor_sample_count: int = 0,
    underflow_observed: bool = False,
):
    """Build the current private backend observation used by pulse test doubles.

    This is only a fixture factory.  Production ``PulseServerSnapshot`` remains
    the sole validator and public codec owner for these physical facts.
    """

    point_count = (
        len(artifact.target_ir.scan_points)
        if scan_point_count is None and artifact is not None
        else 0 if scan_point_count is None else int(scan_point_count)
    )
    sampled_cursor = cursor if cursor_sample_count else None
    is_safe = state == "SAFE"
    return {
        "schema": "zlc_pulse.DeployedStreamerSessionSnapshot",
        "state": state,
        "prepared_artifact_digest": (
            None if artifact is None else artifact.fingerprint
        ),
        "scan_points": point_count,
        "last_confirmed_cursor": sampled_cursor,
        "cursor_sample_count": cursor_sample_count,
        "underflow_observed": underflow_observed,
        "safe_status_word": 0 if is_safe else None,
        "safe_clock_enable_words": (
            (0,) * ((raw_lane_count + 31) // 32) if is_safe else None
        ),
    }
