"""The one owner of "a plot host is shutting down but has not stopped yet"."""

from __future__ import annotations

from zlc_plot import RasterPlotHost


class RetiringPlotHosts:
    """Hold the plot hosts whose worker outlived their close request.

    ``RasterPlotHost.close`` marks the host closing, drains and cancels every
    unstarted task and wakes the worker before it releases the lock, so a
    zero-timeout attempt is always complete as a *request* and only ever
    incomplete as an *observation*.  That observation lives here as membership
    in one collection: "still retiring" is derived from the set, never stored
    as a boolean some later line has to remember to clear, so it cannot latch.

    Nothing here ever blocks, which is what makes a Qt owner turn a legal
    caller; no other module may attempt a plot-host close.
    """

    def __init__(self) -> None:
        self._hosts: set[RasterPlotHost] = set()

    def retire(self, host: RasterPlotHost) -> None:
        """Request ``host``'s shutdown and record whether it has stopped.

        ``RasterPlotHost.close`` short-circuits once the host is already
        closing, so an owner may — and must — call this on *every* entrance to
        its teardown: the request is idempotent and re-asking is the only thing
        that survives an earlier entrance dying on a raising cleanup edge.
        Nothing is returned, so no caller can keep a per-call copy of a fact
        that only this collection owns.
        """

        self._attempt(host)

    def poll(self) -> bool:
        """Re-attempt every outstanding host; True once none remain."""

        for host in tuple(self._hosts):
            self._attempt(host)
        return not self._hosts

    def _attempt(self, host: RasterPlotHost) -> None:
        # Membership is revoked by the same call that makes it untrue, so no
        # window exists in which two objects disagree about this host.
        if host.close(timeout=0.0):
            self._hosts.discard(host)
        else:
            self._hosts.add(host)

    def __bool__(self) -> bool:
        return bool(self._hosts)


__all__ = ["RetiringPlotHosts"]
