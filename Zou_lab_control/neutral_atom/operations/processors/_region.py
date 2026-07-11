"""Per-panel analysis-processor base: consume a data SOURCE coherently + a REGION signal latest-wins.

An analysis node (an ROI reducer or a curve fit) is owned by ONE plot panel and consumes TWO signals:
the data SOURCE (a frame / scan curve / distribution) and the panel's own ``<slug>_region`` control
signal -- the drawn :class:`~...core.selection.Selection`, encoded by
:func:`...core.selection.encode_region`.  The two are consumed DIFFERENTLY, dictated by the hub's
coherence model:

* the SOURCE is a data-plane stream -- consumed COHERENTLY (each frame replayed exactly once, in
  publication order), so the node computes once per new frame just like any :class:`Processor`;
* the REGION is a control-plane, latest-wins input -- published from the GUI with NO physical lineage
  (``NO_LINEAGE``), so it can NEVER be a lineage-coherent co-input of the frame
  (:meth:`SignalHub.next_coherent_update` only advances MULTIPLE names together at a SHARED provenance
  id).  It is therefore drained to its LATEST value each pass, independently of the source journal.

A re-drag republishes the region signal; the node recomputes on the next source frame, OR immediately
on the LAST source block if only the region moved -- so the node genuinely reacts to BOTH a new frame
and a new region.  ``consumes`` lists BOTH names (the flow graph, provenance and self-loop guard see
the true dependency), while the coherent journal this node advances is the source stream alone.

This is the sole retarget mechanism: there is no console-side ``set_selection`` / ``set_fit_request``
plumbing.  A drag is just a republish of ``<slug>_region``; the reactive worker re-computes.
"""

from __future__ import annotations

from ...core.selection import Selection, decode_region, region_bins
from ...core.signals import NO_LINEAGE
from ..logic import DEFAULT_SOURCE, FRAME_0, Processor, SignalExpr
from ..signal_expr import ProcessorSignalSnapshot


class RegionProcessor(Processor):
    """A :class:`Processor` that reduces / fits a per-panel REGION of a SOURCE block.

    Subclasses implement :meth:`transform` using :meth:`current_region` (the latest
    :class:`Selection`) and :meth:`current_region_bins` (the panel's bins, for a hist fit)."""

    def __init__(self, hub, *, source_expr=None, region=None, selection=None, prefix: str = ""):
        expr = source_expr if isinstance(source_expr, SignalExpr) else SignalExpr.from_value(source_expr)
        if not expr.inputs:
            expr = SignalExpr([FRAME_0], DEFAULT_SOURCE)
        self.source_expr = expr
        self.source_names = tuple(expr.inputs)
        self.region_name = str(region) if region else ""
        consumes = (*self.source_names, self.region_name) if self.region_name else self.source_names
        super().__init__(hub, consumes=consumes, prefix=prefix)
        # The SOURCE journal is the coherent data plane: replace the base's all-consumes subscription
        # (which would starve, waiting for the NO_LINEAGE region to share a frame's provenance) with a
        # source-only coherent subscription.
        self._input_subscription = self.hub.signal_cursors(self.source_names)
        self._last_source_provenance = NO_LINEAGE
        # The region: (a) a REGION SIGNAL (the console's drag path) -- seed from its LATEST value (the
        # drag published it BEFORE this node was created), then subscribe for later re-drags; (b) an
        # INLINE Selection (the manual catalog form's typed bounds -- no hub signal, fixed for the
        # node's life); (c) neither -> the empty selection = the whole-frame default.
        self._region: Selection = Selection()
        self._region_bins: int | None = None
        self._region_subscription = None
        if self.region_name:
            self._seed_region_from_hub()
            self._region_subscription = self.hub.signal_cursors((self.region_name,))
        elif selection is not None:
            self._region = selection if isinstance(selection, Selection) else Selection.from_dict(selection)
            self._region_bins = region_bins(self._region)

    def _seed_region_from_hub(self) -> None:
        """Adopt the region signal's LATEST value (also the gap self-heal re-seed)."""
        try:
            tensor = self.hub.latest_tensor(self.region_name)
        except KeyError:
            return
        if tensor is not None and tensor.version:
            self._region = decode_region(tensor.data)
            self._region_bins = region_bins(self._region)

    # ------------------------------------------------------------------ region access
    def current_region(self) -> Selection:
        """The latest region as a :class:`Selection` (empty = the whole frame)."""
        return self._region

    def current_region_bins(self) -> int | None:
        """The panel's ``bins`` carried on the latest region (a hist fit's single-source binning), or
        ``None`` when the region is not a distribution / carries no bins."""
        return self._region_bins

    def _drain_region(self) -> bool:
        """Advance the region cursor to its LATEST published value (latest-wins).  Returns whether the
        region moved this pass.

        SELF-HEALING on :class:`SignalHistoryGap` (the journal was overrun, or the signal was
        removed/recreated under this subscription): the region is a latest-wins CONTROL input, so a
        gap is never an error here -- re-seed from the signal's current latest value and open a fresh
        subscription.  Without this the gap would propagate out of ``shot()`` into the worker's error
        backoff loop and freeze the node forever."""
        if not self.region_name or self._region_subscription is None:
            return False
        from ...core.signal_tensor import SignalHistoryGap
        moved = False
        while True:
            try:
                update = self.hub.next_coherent_update(
                    (self.region_name,), self._region_subscription, timeout=0)
            except SignalHistoryGap:
                self._seed_region_from_hub()                   # latest-wins: adopt whatever is current
                self._region_subscription = self.hub.signal_cursors((self.region_name,))
                return True
            if update is None:
                break
            self._region_subscription = update.cursors
            tensor = update.tensors[self.region_name]
            self._region = decode_region(tensor.data)
            self._region_bins = region_bins(self._region)
            moved = True
        return moved

    # ------------------------------------------------------------------ consumption
    def new_inputs(self):
        """Advance the region latest-wins, then the source coherently.  A new source frame recomputes
        with the latest region; a region-only change recomputes on the LAST source block (inheriting its
        shot id, so the re-region publish joins the same shot lineage as the frame it re-analyses)."""
        region_moved = self._drain_region()
        update = self.hub.next_coherent_update(
            self.source_names, self._input_subscription, timeout=0)
        if update is not None:
            self._input_subscription = update.cursors
            self._input_tensors = dict(update.tensors)
            self._last_source_provenance = int(update.provenance)
            self._current_source_shot = (
                None if update.provenance == NO_LINEAGE else int(update.provenance))
            return ProcessorSignalSnapshot(self._input_tensors, provenance=int(update.provenance))
        if region_moved and self._input_tensors:
            self._current_source_shot = (
                None if self._last_source_provenance == NO_LINEAGE else int(self._last_source_provenance))
            return ProcessorSignalSnapshot(
                self._input_tensors, provenance=int(self._last_source_provenance))
        return None
