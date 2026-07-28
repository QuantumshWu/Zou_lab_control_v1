"""Pure authored state for a single-value METER panel.

The four dataset-evaluated `ViewIntent` values need frontend display state;
three already owned a module here (`image_display`, `curve_display`,
`histogram_display`).  METER's state was
stranded as a private class inside one workbench window, which meant a board
could render a METER panel but could not hold a display state for it - the live
board's panel-to-revision mapping had no slot for it at all, and every other
consumer that wanted one would have had to reach into a GUI module.

There is deliberately **no** ``meter_display_form_spec``: a METER shows one
number for one exact source-aware panel address, and it has no authored display
parameter to edit.  Inventing an empty form so the file "matches its siblings"
would put an empty tab in every Setting popup.  When a real METER knob appears
(a unit or a format, say), it belongs here beside the state, and the form arrives
with it.
PULSE is deliberately outside that dataset-state set: it is fed by an authored
pulse document and its x-only view state belongs to the document renderer seam,
not to a DataBlock evaluator or this METER owner.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import AxisSourceRef
from zlc_storage import nonnegative_integer

from .fit_projection import canonical_panel_focus_address


@dataclass(frozen=True, slots=True)
class MeterDisplayState:
    """Which panel a METER occupies and which exact value it is showing."""

    panel_index: int
    expected_address: tuple[tuple[AxisSourceRef, int], ...] | None
    revision: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.panel_index, bool) or not isinstance(self.panel_index, int):
            raise TypeError("meter panel_index must be a non-negative integer")
        if self.panel_index < 0:
            raise ValueError("meter panel_index must be a non-negative integer")
        if self.expected_address is not None:
            object.__setattr__(
                self,
                "expected_address",
                canonical_panel_focus_address(self.expected_address),
            )
        object.__setattr__(
            self,
            "revision",
            nonnegative_integer(self.revision, "meter display revision"),
        )


__all__ = ["MeterDisplayState"]
