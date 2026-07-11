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
  the console simply skips that tick -- dirty panels retry, which is the natural
  frame-skip back-pressure).
* A job stays "busy" from ``submit`` until the GUI thread has CONSUMED its result
  (``deliver_pending``) -- NOT merely until the worker finished computing it.  A
  finished-but-undelivered batch still owns its figures: its present/structural
  pass has not run, so marking the loop idle any earlier would let the next tick
  re-submit the SAME figures while that pass is still queued (two threads on one
  Agg buffer -- the exact race the first cut of this loop shipped with).
* While a job is in flight the GUI thread must not touch any live figure.  Every
  GUI entry point that does (a mouse press/wheel on a canvas, a Setting edit, a
  structural rebuild, the Edit-tab snapshot, save/close) first calls
  :meth:`barrier`, which waits out the compose AND delivers the finished batch to
  the consumer itself -- so the caller returns to a settled board with nothing in
  flight or pending.  Free when idle; bounded by one batch when busy.
* Delivery is single-consumption: the queued ``job_done`` signal and ``barrier``
  both land in :meth:`deliver_pending`; whichever runs first hands the result to
  the consumer, the other finds nothing and no-ops (never a double present).
"""

from __future__ import annotations

import threading
import time

from PyQt5 import QtCore

#: The render worker thread's name -- ONE source, shared with the canvas barrier so a
#: compose-phase re-entry (a stock ``canvas.draw()`` that fires mid-compose) can recognise
#: it is already ON the worker and must never wait on its own in-flight job.
RENDER_THREAD_NAME = "zlc-render"


class RenderLoop(QtCore.QObject):
    """One daemon worker executing submitted render jobs strictly one at a time.

    ``consumer`` is the GUI-side batch handler (the console's ``_on_render_batch``):
    it receives each job's return value (or the exception the job raised) on the GUI
    thread, exactly once per job."""

    #: Emitted from the worker thread AFTER the job's result is published on the loop;
    #: auto-queued to the GUI thread (this QObject lives there).  Carries nothing: the
    #: GUI slot pulls the result through :meth:`deliver_pending`, so a barrier() that
    #: already consumed it leaves this delivery a harmless no-op.
    job_done = QtCore.pyqtSignal()

    def __init__(self, consumer, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._consumer = consumer
        self._job = None
        self._result = None
        self._has_result = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._published = threading.Event()   # worker finished computing; result awaits delivery
        self._stopping = False
        self.job_done.connect(self.deliver_pending)
        self._thread = threading.Thread(target=self._loop, name=RENDER_THREAD_NAME, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ GUI side
    @property
    def busy(self) -> bool:
        """True from ``submit`` until the GUI CONSUMED the finished batch.  The console
        skips submitting while busy -- including the window where the worker is done but
        the batch's present/structural pass is still queued behind this tick."""
        return not self._idle.is_set()

    def submit(self, job) -> bool:
        """Queue ``job`` (a zero-arg callable) for the worker; False when busy/stopped.

        Refusing while busy IS the frame-skip policy: the caller's dirty state is
        version-gated, so the work is simply retried on a later tick."""
        if self._stopping:
            return False
        with self._lock:
            if not self._idle.is_set():
                return False
            self._job = job
            self._idle.clear()
        self._wake.set()
        return True

    def deliver_pending(self) -> bool:
        """GUI thread: hand the published result to the consumer and mark the loop idle
        -- the ONE consumption point.  Both the queued ``job_done`` delivery and
        :meth:`barrier` land here; whichever runs first wins, the other no-ops.
        Returns True iff a result was actually delivered."""
        with self._lock:
            if not self._has_result:
                return False
            result, self._result, self._has_result = self._result, None, False
            self._published.clear()
        # Idle BEFORE the consumer runs: the consumer is GUI-thread code (no tick can
        # interleave on this same thread), and any barrier() nested inside it must see
        # the loop as free instead of waiting 5 s on its own delivery.
        self._idle.set()
        self._consumer(result)
        return True

    def barrier(self, timeout: float = 5.0) -> bool:
        """GUI thread: settle the loop -- wait out any composing job AND deliver its
        finished batch before returning, so the caller owns every figure with nothing
        in flight OR pending (a pending batch delivered later would stomp whatever the
        caller is about to do to those figures).  Idempotent and free when idle.
        Returns False only on a pathological timeout (a wedged render), in which case
        the caller proceeds -- a rare visual glitch beats a frozen UI.

        Short-circuit when called ON the render worker itself: a compose path that
        re-enters here (a stock ``canvas.draw()`` triggered mid-compose -- e.g. an mpl
        selector ``set_active`` doing an ``update_background`` draw) already OWNS the
        figure it is composing, so it must never wait on its own in-flight job (the 5 s
        drag self-deadlock: worker compose -> canvas.draw() -> barrier waits on its own
        job).  It has the figure; proceed immediately."""
        if threading.current_thread() is self._thread:
            return True
        deadline = time.monotonic() + timeout
        while True:
            self.deliver_pending()
            if self._idle.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._published.wait(remaining)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker (console shutdown).  Any in-flight job finishes first; a
        finished-but-undelivered batch is DISCARDED (its only effect would be painting
        panels that are about to be torn down)."""
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout)
        with self._lock:
            self._result, self._has_result = None, False
            self._published.clear()
        self._idle.set()

    # --------------------------------------------------------------- worker side
    def _loop(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stopping:
                return
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            try:
                result = job()
            except Exception as exc:  # surfaced to the consumer, never a dead thread
                result = exc
            with self._lock:
                self._result, self._has_result = result, True
            self._published.set()
            self.job_done.emit()


__all__ = ["RenderLoop"]
