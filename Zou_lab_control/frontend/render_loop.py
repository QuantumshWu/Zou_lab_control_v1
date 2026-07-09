"""The console's ONE background render thread -- the frontend/backend decoupling seam.

The task console's live pipeline is two-phase (``BaseLivePlot.compose`` writes the
offscreen Agg buffer, ``present`` schedules the Qt paint).  compose -- the numpy
display prep, the matplotlib artist updates and the Agg rasterisation -- is pure CPU
work that never touches Qt widget state, so it runs HERE, off the GUI thread.  The
GUI thread keeps only what genuinely must be there: dirtiness scheduling, the Qt
paint of the finished front buffer, and every user interaction.  A slow render then
only lowers the PLOT frame rate; buttons, combos and drags stay responsive because
the GUI thread never executes a multi-hundred-ms compose burst again (the old
per-tick budget/rotor machinery re-scheduled that burst but never removed it).

Ownership protocol (the whole thread-safety story):

* The GUI thread submits at most ONE job at a time (``submit`` refuses while busy;
  the console simply skips that tick -- dirty panels retry on the next one, which is
  the natural frame-skip back-pressure).
* While a job is in flight the GUI thread must not touch any live figure.  Every
  GUI entry point that does (a mouse press/wheel on a canvas, a Setting edit, a
  structural rebuild, the Edit-tab snapshot, save/close) first calls
  :meth:`barrier`, which drops nothing and simply waits for the in-flight job to
  finish -- bounded by one batch (tens of ms), idempotent and free when idle.
* Results come back on the GUI thread through the queued ``job_done`` signal; the
  console presents every composed panel together there, preserving the board's
  one-coherent-shot flip.
"""

from __future__ import annotations

import threading

from PyQt5 import QtCore


class RenderLoop(QtCore.QObject):
    """One daemon worker executing submitted render jobs strictly one at a time."""

    #: Emitted from the worker thread with the job's return value (or the raised
    #: exception object); auto-queued to the GUI thread by Qt cross-thread rules.
    job_done = QtCore.pyqtSignal(object)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._job = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._stopping = False
        self._thread = threading.Thread(target=self._loop, name="zlc-render", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ GUI side
    @property
    def busy(self) -> bool:
        """True while a job is queued or running (the console skips submitting)."""
        return not self._idle.is_set()

    def submit(self, job) -> bool:
        """Queue ``job`` (a zero-arg callable) for the worker; False when busy/stopped.

        Refusing while busy IS the frame-skip policy: the caller's dirty state is
        version-gated, so the work is simply retried next tick."""
        if self._stopping:
            return False
        with self._lock:
            if not self._idle.is_set():
                return False
            self._job = job
            self._idle.clear()
        self._wake.set()
        return True

    def barrier(self, timeout: float = 5.0) -> bool:
        """Wait until the worker is idle -- the GUI must hold this before touching any
        figure the worker composes.  Idempotent and free when idle; bounded by one
        batch when busy.  Returns False only on a pathological timeout (a wedged
        render), in which case the caller proceeds -- a rare visual glitch beats a
        frozen UI."""
        return self._idle.wait(timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker (console shutdown).  Any in-flight job finishes first."""
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout)

    # --------------------------------------------------------------- worker side
    def _loop(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stopping:
                self._idle.set()
                return
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                self._idle.set()
                continue
            try:
                result = job()
            except Exception as exc:  # surfaced to the GUI slot, never a dead thread
                result = exc
            # Mark idle BEFORE emitting: the GUI slot presents the composed panels and
            # may immediately barrier()/submit() -- it must see the worker as free.
            self._idle.set()
            self.job_done.emit(result)


__all__ = ["RenderLoop"]
