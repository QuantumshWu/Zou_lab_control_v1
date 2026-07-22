"""Confocal-style PyQt editor for pulse values and previews.

Executable operations use an optional generation-bound command port supplied by the
Workbench composition.  The editor never receives or constructs a sequencer adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import os
import re

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_neutral_atom.timing.clock import (DEFAULT_TIME_STEP_NS,
                                           default_clock_hz as _default_hardware_clock_hz)
from zlc_neutral_atom.timing.ports import PORT_CLOCK, PORT_DAC, PortCatalog
from zlc_neutral_atom.timing.pulse_table import (
    DELAY_MAX_TICKS,
    SCAN_SLOT_KINDS,
    UNITS_TO_NS,
    PulsePeriod,
    PulseTableState,
    ScanSlot,
    default_pulse_name,
    is_slot_ref,
    bus_signed_range,
    bus_zero_code,
    load_scan_table,
    slot_ref_index,
    slot_var,
    snap_scan_table,
    _analog_bus_value_at_tick,
    analog_bus_ticks as _analog_bus_ticks,
)
from zlc_data.panel_size import PANEL_SIZES
from zlc_data.scan_template import scan_table_template
from zlc_data.shape_text import slot_label   # naming a bound field is grammar, not GUI
from zlc_storage.paths import project_path   # the ONE owner of where the project root is
# The pulse RENDER (state -> figure) lives in the plot layer (live.py); the editor CONSUMES it -- it does
# not own the render.  ``bus_signed_bounds`` / ``bus_display_label`` are shared render+editor helpers that
# also live there now (single source; the editor's ``_bus_signed_bounds`` / ``_bus_display_label`` names
# alias them below so the many editor call sites are unchanged).
from zlc_frontend.qt_widgets import (
    ACCENT,
    BG,
    FONT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    TEXT,
    WINDOW_SCREEN_FRACTION,
    YELLOW,
    ElidedLabel,
    fluent_message,
    FluentButton,
    FluentCheckBox,
    FluentCodeEdit,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentScanDot,
    FluentScanLineEdit,
    FluentScrollArea,
    FluentStatusDot,
    FluentSwitch,
    FluentTabWidget,
    Metrics,
    ensure_qt_app,
    launch_fluent_window,
    fluent_font_size,
    fluent_scrollbar_stylesheet,
    fluent_scrollbar_thickness,
    fluent_text_width,
    fluent_widget_stylesheet,
    format_compact_number,
    mark_scan_field,
    measure_text_width,
    scaled_px,
    window_pad,
    screen_fit_window_size,
    set_fluent_scale,
    align_to_resolution,
    batched_updates,
    signals_blocked as _signals_blocked,
)



TIME_UNITS = ["ns", "us", "ms", "s", "str (ns)"]
# User-selectable duration units.  "str (ns)" is the internal unit for a SCAN-BOUND
# duration ("s0" expression); it is set automatically when you bind via the scan dot and
# is NOT offered in the dropdown for a normal (unbound) duration.
DURATION_UNITS = ["ns", "us", "ms", "s"]
# TTL AND DAC-bus delays are both TRUE physical delays now (out[t] = in[t-d], first frame
# correct, no modulo), event-scheduled per bit, bounded only by the 32-bit field (~42.9 s at
# 20 ns/tick) -- so ms / s are valid units for both and the ranges match (a negative TTL delay's
# global shift can reach the buses, no more range mismatch).
DELAY_UNITS = ["ns", "us", "ms", "s"]


def _delay_eligible_position(channel_key: str) -> int | None:
    """Hardware bit position of a ``chNN`` channel, else None (cannot tell)."""
    text = str(channel_key)
    if text.startswith("ch") and text[2:].isdigit():
        return int(text[2:])
    return None


try:  # the eligible-channel count is a fixed hardware fact (board layout)
    from zlc_neutral_atom.timing.streamer_geometry import (
        DEFAULT_FPGA_CHANNEL_COUNT as _FPGA_CH, delay_eligible_channel_count as _elig_count,
        hardware_channel_names as _hw_channel_names)
    NUM_DELAY_CHANNELS = _elig_count(_FPGA_CH)
    DEFAULT_CHANNEL_NAMES = _hw_channel_names(_FPGA_CH)  # ["ch00", ...] for the board's channel count
except Exception:  # pragma: no cover - host tooling optional
    NUM_DELAY_CHANNELS = 10 ** 9   # unknown -> allow everything (host/RTL still gate)
    DEFAULT_CHANNEL_NAMES = []      # unknown board -> caller must supply channels explicitly

try:  # the configured event-FIFO depths (changes-in-flight caps); shown in delay tips
    from zlc_neutral_atom.timing.streamer_geometry import (
        EVT_FIFO_DEPTH as _EVT_DEPTH,           # per TTL channel
        BUS_EVT_FIFO_DEPTH as _BUS_EVT_DEPTH,   # per DA bus bit
    )
except Exception:  # pragma: no cover - host tooling optional
    _EVT_DEPTH = 64
    _BUS_EVT_DEPTH = 64
# Unit->ns factors are owned by the timing layer (pulse_table.UNITS_TO_NS) -- import it
# rather than keep a second near-identically-named copy that could silently drift.
UNIT_TO_NS = UNITS_TO_NS
ROW_HEIGHT = 30
CHANNEL_LABEL_WIDTH = 100
TIME_UNIT_WIDTH = 60
HIDE_BUTTON_WIDTH = 26
PANEL_TOP_HEIGHT = 178   # name row + Duration label + value + unit (all four panels share it)
CHANNEL_ROW_SPACING = 4
PERIOD_CARD_WIDTH = 158
DEFAULT_WINDOW_RATIO = WINDOW_SCREEN_FRACTION   # the ONE shared screen-fraction (qt_widgets), == task console
DEFAULT_HARDWARE_CLOCK_HZ = _default_hardware_clock_hz()  # single source: streamer_config.json (via the dependency-free _clock seam)
# The tick is the SAME fact as the rate; it is derived beside it in the clock seam, not
# re-derived here.  (Imported above, re-exported for this module's readers.)
SUMMARY_DEBOUNCE_MS = 90
PREVIEW_DEBOUNCE_MS = 160
PULSE_FILES_ENV = "ZLC_PULSE_DIR"


def _px(value: int | float, *, minimum: int = 1) -> int:
    return scaled_px(value, minimum=minimum)


def _font_metrics() -> QtGui.QFontMetrics:
    return QtGui.QFontMetrics(QtGui.QFont(FONT, fluent_font_size()))


def _row_height() -> int:
    return _px(ROW_HEIGHT, minimum=22)


def _row_spacing() -> int:
    return _px(CHANNEL_ROW_SPACING, minimum=3)


def _row_region_vmetrics() -> tuple[int, int]:
    """(top margin, inter-row spacing) of the channel-row column -- ONE vertical geometry.

    Port Catalog, Delay/Scan and every Period card stack the SAME row list under the same
    fixed-height header, and the operator reads those rows ACROSS -- so all three layouts
    must advance by the same pitch.  A card that wrote its own (compact) margins/spacing
    literals shaved 2 px per row against the panels and was 47 px off by the last of 22
    rows; deriving all three from here makes that drift structurally impossible.
    """

    return _px(8), _row_spacing()


def _channel_label_width() -> int:
    return _px(CHANNEL_LABEL_WIDTH, minimum=84)


def _channel_name_edit_width() -> int:
    return _px(108, minimum=88)


def _time_unit_width() -> int:
    return _px(TIME_UNIT_WIDTH, minimum=62)


def _hide_button_width() -> int:
    return _px(HIDE_BUTTON_WIDTH, minimum=22)


def _panel_top_height() -> int:
    return _px(PANEL_TOP_HEIGHT, minimum=138)


def _card_gutter() -> int:
    # The gutter (px) around/between flat fluent cards so their 1 px DIVIDER borders sit apart and
    # don't touch or merge.  (Formerly _shadow_pad, which reserved the old drop shadow's outward
    # bleed; flat cards have no bleed, so the SAME value is now simply the inter-card gutter.)
    return _px(5, minimum=4)


def _period_card_width() -> int:
    return _px(PERIOD_CARD_WIDTH, minimum=112)


def _default_pulse_name() -> str:
    return default_pulse_name()


def _pulse_files_dir() -> Path:
    """The folder the Pulse GUI saves/loads programs in: ``$ZLC_PULSE_DIR`` if set, else the
    project's ``pulses/``.  The default comes from :func:`zlc_storage.paths.project_path`, the
    one owner of "where the project root is" -- deriving it again from THIS file's own depth
    would agree only for as long as this file stays exactly where it is, which the migration
    is in the business of changing."""
    configured = os.environ.get(PULSE_FILES_ENV, "").strip()
    directory = Path(configured).expanduser() if configured else project_path("pulses")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_file_stem(name: str) -> str:
    out = []
    for char in str(name or "").strip():
        if char.isalnum() or char in ("-", "_"):
            out.append(char)
        elif char.isspace():
            out.append("_")
    return "".join(out).strip("_") or _default_pulse_name()


def _period_duration_text(period: PulsePeriod) -> str:
    return str(period.duration)


def _period_control_width(card_width: int) -> int:
    return max(_px(76, minimum=68), card_width - 2 * _px(7) - _px(4))


def _is_slot_expr(text: object) -> bool:
    """True when a field value is a bare scan-slot reference like ``s0``."""

    return is_slot_ref(str(text or "").strip())


def _slot_index_of_expr(text: object) -> int | None:
    return slot_ref_index(str(text or "").strip())


def _api_number(name: object) -> int:
    """The 1-based index of API handle ``name`` (``"a1"`` -> 1) for the marker dot."""

    digits = "".join(ch for ch in str(name or "") if ch.isdigit())
    return int(digits) if digits else 1




def _scan_slot_label(state: PulseTableState, index: int) -> str:
    """Human description of scan slot ``index`` for GUI lists/tooltips (single source: slot_label)."""

    slot = state.scan_slots[index]
    return slot_label(slot.kind, slot.target)


def _format_scan_progress(progress: Mapping[str, object] | None) -> str:
    """SINGLE source for the live scan-progress label text (Task #4).

    ``progress`` is the immutable mapping returned by the pulse command port:
    ``{scanning, point, n_points, sweep, n_repeats}`` with
    ``point``/``sweep`` 0-based and ``n_repeats=0`` meaning ∞.  Returns:

    * ``""`` when idle (not scanning, no points, or ``progress is None``) -- the label blanks.
    * ``"Scan: point K / N · sweep r"`` for an INFINITE scan (``n_repeats=0``; no ``/ R``).
    * ``"Scan: point K / N · sweep r / R"`` for a FINITE scan (``n_repeats=R>0``).

    Points/sweeps are shown 1-based (operator-facing) though the dict is 0-based."""

    if not progress:
        return ""
    try:
        scanning = bool(progress.get("scanning"))
        n_points = int(progress.get("n_points", 0))
        point = int(progress.get("point", 0))
        sweep = int(progress.get("sweep", 0))
        n_repeats = int(progress.get("n_repeats", 0))
    except (TypeError, ValueError):
        return ""
    if not scanning or n_points <= 0:
        return ""
    # 1-based display; clamp the point to [1, N] so a malformed/boundary reading can never show
    # "point N+1 / N" (scan_progress_fields keeps point in [0, N-1], so this is belt-and-suspenders).
    point_1 = min(max(point + 1, 1), n_points)
    text = f"Scan: point {point_1} / {n_points} · sweep {sweep + 1}"
    if n_repeats > 0:                      # finite K sweeps -> show "/ R"; ∞ (0) shows just "sweep r"
        text += f" / {n_repeats}"
    return text


def _template_column_stack(state) -> str:
    """``column_stack`` template (one column per slot) -- delegates to the ONE shared scan-table
    template generator with the state's PER-KIND column specs (DAC slots get their signed code
    range, not a duration's ns range).  The task-console Pulse-scan form uses the same source."""
    return scan_table_template("column_stack", state.scan_column_specs())


def _template_grid(state) -> str:
    """Grid (outer-product) template -- delegates to the ONE shared scan-table template generator
    with the state's per-kind column specs."""
    return scan_table_template("grid", state.scan_column_specs())


def _default_scan_code(state) -> str:
    return scan_table_template("column_stack", state.scan_column_specs())


def _format_clock_text(time_step_ns: float) -> str:
    """Read-only ``<freq> MHz · <step> ns`` label for the fixed FPGA clock."""

    step = float(time_step_ns)
    if step <= 0:
        return "—"
    mhz = 1000.0 / step  # (1e9 / step) Hz -> MHz
    return f"{format_compact_number(mhz)} MHz · {format_compact_number(step)} ns"


def _delay_cap_text(time_step_ns: float) -> str:
    """Human description of the per-signal delay magnitude cap (the 32-bit field).  Same
    for TTL channels and DAC buses (both event-scheduled)."""

    max_us = DELAY_MAX_TICKS * float(time_step_ns) / 1000.0
    if max_us >= 1e6:
        return f"±{format_compact_number(max_us / 1e6)} s (event-scheduled; ms-scale delays OK)"
    return f"±{format_compact_number(max_us)} us ({DELAY_MAX_TICKS} ticks)"


def _bus_mode_combo_width() -> int:
    """Width that *just* fits the widest mode word ("Ramp") plus the dropdown arrow.

    Matches the non-text budget FluentComboBox.paintEvent reserves (drop arrow +
    insets ~26 px) PLUS a few px of slack so "Ramp" never elides on the rounding edge,
    measured at the current font so it stays correct under both the real Segoe UI and
    the wider offscreen substitute font used for screenshots.  A real gap (the row
    spacing) then separates it from the value field."""

    return measure_text_width(["Edge", "Ramp", "Hold"], padding=34)


def _set_duration_unit_combo(combo, *, scanned: bool, unit: str) -> None:
    """Populate a period-duration unit combo.  Normal duration: ns/us/ms/s.  A scan-bound
    duration: the internal "str (ns)" expression unit (added + selected automatically, and
    the combo is disabled), so "str (ns)" is never a user-pickable option for a plain
    duration.  Call inside a signals-blocked block when reused on a live combo."""

    items = list(DURATION_UNITS) + (["str (ns)"] if scanned else [])
    if [combo.itemText(i) for i in range(combo.count())] != items:
        combo.clear()
        combo.addItems(items)
    combo.setCurrentText("str (ns)" if scanned else (unit if unit in DURATION_UNITS else "ns"))




def _unit_resolution(step_ns: float, unit: str) -> float:
    factor = UNIT_TO_NS.get(unit or "ns", 1.0)
    if factor <= 0:
        return float(step_ns)
    return float(step_ns) / factor


def _normalize_bus_value_text(text: object, *, lo: int, hi: int) -> str:
    """A typed DAC value, coerced to the bus's SIGNED user range.

    ALL user-layer DAC values are signed LSB counts around true 0 V
    (``bus_signed_range``, the model's one source -- the wire carries
    offset-binary ``signed + bus_zero_code``).  Anything outside ``lo..hi`` is
    clamped rather than rejected: the number shown is then exactly the number
    that will be applied.  Unparseable text reads as 0 (true 0 V) for the same
    reason -- a blank field must not leave the previous value silently in effect.
    """

    raw = str(text or "").strip()
    try:
        value = int(float(raw)) if raw else 0
    except ValueError:
        value = 0
    return str(max(int(lo), min(int(hi), value)))


def _bus_key(name: str) -> str:
    return f"bus:{name}"


def _bus_mode_title(mode: str) -> str:
    mode = str(mode or "hold").strip().lower()
    return {"edge": "Edge", "ramp": "Ramp", "hold": "Hold"}.get(mode, "Hold")


def _bus_mode_value(title: str) -> str:
    title = str(title or "Hold").strip().lower()
    if title.startswith("ram"):
        return "ramp"
    if title.startswith("edg"):
        return "edge"
    return "hold"


def _is_bus_key(key: str) -> bool:
    return str(key).startswith("bus:")


def _display_rows(state: PulseTableState) -> list[dict[str, object]]:
    """Logical, pulse-programmable ports in physical catalog order.

    The editor never rediscovers buses from labels and never iterates raw DAC
    lanes as user channels.  Raw lanes remain in ``PulsePeriod.states`` solely
    because that is the compiler representation.  Clock ports are topology
    owned by the sequencer and are not pulse-programmable rows.
    """

    rows: list[dict[str, object]] = []
    visible = set(state.visible_ports)
    for port in state.port_catalog.ports:
        if port.kind == PORT_CLOCK or port.key not in visible:
            continue
        if port.kind == PORT_DAC:
            rows.append({"kind": "bus", "key": _bus_key(port.key), "name": port.key,
                         "channels": list(port.lanes), "label": port.label})
        else:
            lane = port.lanes[0]
            rows.append({"kind": "channel", "key": lane, "name": port.key,
                         "channels": [lane], "label": port.label})
    return rows


def _port_visibility(state: PulseTableState) -> tuple[int, int]:
    """Visible/total pulse-programmable logical ports (never raw lanes)."""

    programmable = tuple(port for port in state.port_catalog.ports if port.kind != PORT_CLOCK)
    return len(state.visible_ports), len(programmable)


def _hidden_active_ports(state: PulseTableState) -> list[str]:
    visible = set(state.visible_ports)
    active = set(state.period_active_ports())
    return [port.label for port in state.port_catalog.ports
            if port.kind != PORT_CLOCK
            and port.key in active
            and port.key not in visible]


def _display_row_label(row: Mapping[str, object], labels: Mapping[str, str] | None = None) -> str:
    if row.get("kind") == "bus":
        return str((labels or {}).get(str(row["key"])) or row.get("label") or row.get("name"))
    key = str(row["key"])
    return str((labels or {}).get(key) or row.get("label") or key)


def _repeat_summary_text(state: "PulseTableState") -> str:
    """How the program's repeat is phrased everywhere it is shown.

    The reference's exact three-state wording (``pulse_repeat_notation``):
    ``repeat ∞`` for a forever program without an inner bracket, ``repeat
    P2-P3 x2`` when the bracket covers the whole table, and ``repeat ∞ +
    P2-P3 x2`` when a partial inner bracket rides inside the forever loop.
    A one-shot program without a bracket has nothing to say.  The header
    summary and the Preview status both read it here.
    """

    repeat_start, repeat_end = state.repeat_start, state.repeat_end
    if repeat_start is None or repeat_end is None:
        return "repeat ∞" if state.repeat_forever else ""
    inner = f"P{int(repeat_start) + 1}-P{int(repeat_end) + 1} x{int(state.repeat_count)}"
    if state.periods and (int(repeat_start) != 0
                          or int(repeat_end) != len(state.periods) - 1):
        return f"repeat ∞ + {inner}"
    return f"repeat {inner}"


def _preview_repeat_markers(state: "PulseTableState") -> tuple[list[tuple[float, float, str]], float]:
    """All repeat brackets a pulse preview draws, plus the UN-EXPANDED frame length.

    The reference's exact semantics (``pulse_repeat_markers``): the preview
    shows the period table AS AUTHORED -- the inner finite bracket
    ``[repeat_start .. repeat_end] × repeat_count`` reads as a NESTED square
    bracket over its own time span, never as the unrolled copies -- so the
    spans come from the ORIGINAL periods' prefix sum (``period_start_steps``),
    the same single source the scan/DAC annotations already use:

    * no bracket:            ``[×∞ over the whole frame]`` when forever, else none;
    * bracket == whole table: only the inner ``×N`` bracket;
    * partial bracket:        outer ``×∞`` plus the inner ``×N`` when forever,
                              else the inner bracket alone.

    Returns ``(markers, total_seconds)`` -- the total is ``starts[-1]`` of the
    original table, which IS the preview's frame length (the expanded
    ``total_duration_ns`` would stretch the axis over the unrolled copies the
    preview deliberately does not draw).
    """

    slots = state._reference_slots()
    step_ns = float(state.time_step_ns)
    starts_steps = state.period_start_steps(slots=slots, time_step_ns=step_ns)
    starts_s = [steps * step_ns * 1e-9 for steps in starts_steps]
    total = starts_s[-1]
    repeat_start, repeat_end = state.repeat_start, state.repeat_end
    forever = bool(state.repeat_forever)
    if repeat_start is None or repeat_end is None:
        return ([(0.0, total, "×∞")] if forever else []), total
    repeat_start, repeat_end = int(repeat_start), int(repeat_end)
    if repeat_start < 0 or repeat_end < repeat_start or repeat_end + 1 >= len(starts_s):
        return [], total
    inner = (starts_s[repeat_start], starts_s[repeat_end + 1],
             f"×{int(state.repeat_count)}")
    if repeat_start == 0 and repeat_end == len(state.periods) - 1:
        return [inner], total
    return ([(0.0, total, "×∞"), inner] if forever else [inner]), total


def _summary_time_text(value_ns: float) -> str:
    value_ns = float(value_ns)
    # Largest-unit-first table derived from the single source UNITS_TO_NS, skipping the
    # internal "str (ns)" scan-expression alias (it duplicates ns and is never displayed).
    units = tuple(
        (unit, factor)
        for unit, factor in sorted(
            UNITS_TO_NS.items(), key=lambda kv: kv[1], reverse=True
        )
        if unit in DURATION_UNITS
    )
    for unit, factor in units:
        if abs(value_ns) >= factor or unit == "ns":
            return f"{format_compact_number(value_ns / factor, digits=6)} {unit}"
    return f"{format_compact_number(value_ns, digits=6)} ns"


def _set_fixed_height(widget: QtWidgets.QWidget, height: int | None = None) -> QtWidgets.QWidget:
    widget.setFixedHeight(_row_height() if height is None else height)
    return widget


def _set_form_label_geometry(label: FluentLabel) -> FluentLabel:
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.setFixedSize(_channel_label_width(), _row_height())
    return label


def _add_labeled_widget(layout, label_text: str, widget: QtWidgets.QWidget) -> FluentLabel:
    """Add a ``label | control-cell`` form row to ``layout`` -- the ONE shared row builder the
    pulse-editor panels (NamePanel + ChannelPanel) use, so the two never drift on spacing / label
    geometry.  Returns the row's label (callers stash it to restyle later)."""
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(_px(5, minimum=3))
    label = FluentLabel(label_text)
    _set_form_label_geometry(label)
    row.addWidget(label)
    row.addWidget(_form_control_cell(widget), 1)
    layout.addLayout(row)
    return label


def _form_control_cell(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    widget.setFixedHeight(_row_height())
    cell = QtWidgets.QWidget()
    cell.setStyleSheet("background: transparent;")
    layout = QtWidgets.QHBoxLayout(cell)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    fixed_width = widget.minimumWidth() > 0 and widget.maximumWidth() == widget.minimumWidth()
    if fixed_width:
        layout.addWidget(widget, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        layout.addStretch(1)
    else:
        layout.addWidget(widget, 1)
    return cell


def _channel_row_height(channel_count: int) -> int:
    return _px(26 if channel_count > 16 else ROW_HEIGHT, minimum=22)


def _elide_text(text: object, width: int) -> str:
    """Right-elide ``text`` to fit ``width`` px at the current GUI font."""

    return _font_metrics().elidedText(str(text), QtCore.Qt.ElideRight, max(8, int(width)))


class PulseStateUIManager(QtCore.QObject):
    class RunState:
        INIT = "INIT"
        PREPARED = "PREPARED"
        RUNNING = "RUNNING"
        STOP = "STOP"
        SAFE = "SAFE"
        ERROR = "ERROR"
        UNSYNCED = "UNSYNCED"

    class FileState:
        UNTITLED = "UNTITLED"
        SAVE = "SAVE"
        LOAD = "LOAD"
        UNSAVED = "UNSAVED"

    def __init__(
        self,
        *,
        status_dot: FluentStatusDot,
        label: FluentLabel,
        save_button: FluentButton,
        fire_button: FluentButton | None = None,
        title_callback=None,
    ):
        super().__init__()
        self.status_dot = status_dot
        self.label = label
        self.save_button = save_button
        self.fire_button = fire_button
        self.title_callback = title_callback
        self.address_str = ""
        self.pulse_name = "pulse"
        self._runstate = self.RunState.INIT
        self._filestate = self.FileState.UNTITLED
        self._update()

    @property
    def runstate(self):
        return self._runstate

    @runstate.setter
    def runstate(self, value):
        self._runstate = value
        self._update()

    @property
    def filestate(self):
        return self._filestate

    @filestate.setter
    def filestate(self, value):
        self._filestate = value
        self._update()

    def _update(self) -> None:
        # Confocal state semantics: BUTTON BASE COLOURS NEVER CHANGE with run
        # state (a colour-coded state on one button is indistinguishable from
        # the other permanently-coloured buttons).  Run state is shown by the
        # STATUS DOT colour plus a confocal-style ``*`` suffix on On Pulse:
        # the star means "pressing this would apply something new" -- it is
        # present in every state except RUNNING-and-in-sync.
        colors = {
            self.RunState.INIT: GREY,
            self.RunState.PREPARED: YELLOW,
            self.RunState.RUNNING: GREEN,
            self.RunState.STOP: RED,
            self.RunState.SAFE: RED,
            self.RunState.ERROR: RED,
            self.RunState.UNSYNCED: ORANGE,
        }
        self.status_dot.set_color(colors.get(self._runstate, GREY))

        local = self.address_str.replace("\\", "/").split("/")[-1] if self.address_str else ""
        pulse_name = self.pulse_name.strip() or "pulse"
        if self._filestate == self.FileState.SAVE:
            status, star = "saved", ""
        elif self._filestate == self.FileState.LOAD:
            status, star = "loaded", ""
        elif self._filestate == self.FileState.UNSAVED:
            status, star = "unsaved", "*"
        else:
            status, star = "new", "*"
        if local:
            text = f"PulseGUI - {pulse_name} ({status}: {local}){star}"
        else:
            text = f"PulseGUI - {pulse_name} ({status}){star}"
        self.label.setText(text)
        # Save: star + yellow while there are unsaved changes, accent when clean
        # (same rule as confocal's btn_save).  The buttons sit in equal-stretch
        # grid columns, so the text change cannot alter their width.
        self.save_button.setText("Save*" if star else "Save")
        self.save_button.set_color(YELLOW if star else ACCENT)
        if self.fire_button is not None:
            # On Pulse stays GREEN; the star marks every state where pressing
            # it would apply/run something new (UNSYNCED edits, stopped, error,
            # prepared-but-not-fired...).  RUNNING in sync = no star.
            running_clean = self._runstate == self.RunState.RUNNING
            self.fire_button.setText("On Pulse" if running_clean else "On Pulse*")
            self.fire_button.set_color(GREEN)
        if self.title_callback is not None:
            self.title_callback(f"{pulse_name} - PulseGUI{star}")

    def set_pulse_name(self, name: str) -> None:
        self.pulse_name = str(name or "pulse")
        self._update()


class PeriodCard(FluentGroupBox):
    changed = QtCore.pyqtSignal()
    busScanRequested = QtCore.pyqtSignal(str)
    busChanged = QtCore.pyqtSignal()  # a DAC mode/value committed -> refresh hold displays


    def __init__(self, index: int, period: PulsePeriod, *, total_periods: int,
                 channels: Sequence[str], labels: Mapping[str, str],
                 hidden_states: Mapping[str, int], rows: Sequence[Mapping[str, object]],
                 state: PulseTableState, compact: bool, time_step_ns: float,
                 parent=None) -> None:
        """One period of the sequence, as the operator edits it.

        The card owns widgets only; what a period MEANS is read back by
        :meth:`to_period`, which is therefore the specification this constructor
        satisfies -- every dict that method indexes is built here, and nothing
        else is kept.

        ``rows`` are the visible logical ports (:func:`_display_rows`), so the
        card never rediscovers ports from labels.  Raw lanes that no visible row
        covers keep their incoming value in ``hidden_states`` and ride through
        untouched: hiding a port is a display choice and must not rewrite the
        sequence.
        """

        super().__init__("", parent)
        self.time_step_ns = float(time_step_ns)
        self.hidden_states = dict(hidden_states)
        self.checks: dict[str, QtWidgets.QCheckBox] = {}
        self.bus_members: dict[str, list[str]] = {}
        self.bus_mode_combos: dict[str, QtWidgets.QComboBox] = {}
        self.bus_value_edits: dict[str, FluentLineEdit] = {}
        # Each DAC value edit's embedded scan dot, keyed by bus name -- the registry
        # _refresh_bus_displays reads to skip scan-bound fields when refreshing the
        # carried HOLD value (a bound field shows its slot, never a number).
        self.bus_dots: dict[str, QtWidgets.QAbstractButton] = {}
        # SIGNED user range per bus, straight from the model's one source
        # (bus_signed_range): every clamp/validator on a DAC field reads this.
        self.bus_bounds: dict[str, tuple[int, int]] = {}

        # A period is one COLUMN in a row of periods, so the card is exactly as wide as
        # its content needs and never stretches: a card that filled the editor would push
        # every other period off the screen.  Both widths come from the shared helpers, so
        # all cards agree with each other and with the fields inside them.
        card_width = _period_card_width()
        self.setMinimumWidth(card_width)
        self.setMaximumWidth(card_width)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        control_width = _period_control_width(card_width)

        column = QtWidgets.QVBoxLayout(self)
        # The reference's EXACT card geometry: a uniform _px(7) margin on all four
        # sides (no compact variant) with the shared row spacing.  The panels use a
        # _px(8) margin; the missing sliver is added as a spacer right after the
        # fixed-height header (below), so the first channel row starts at the same Y
        # as the Name/Delay rows and every later row advances by the same pitch.
        row_top, row_gap = _row_region_vmetrics()
        card_pad = _px(7)
        column.setContentsMargins(card_pad, card_pad, card_pad, card_pad)
        column.setSpacing(row_gap)
        self.set_period_position(index, total_periods)

        # --- period parameters, in a FIXED-HEIGHT header (Duration label, duration,
        # unit, name), the reference's order.  Every card wraps its header in a top
        # widget of _panel_top_height() so its per-channel rows begin at the SAME Y as
        # every other card's; a period card that added these straight to the column had
        # a shorter header and pushed its checkboxes ~93 px above the matching delay row
        # (the "period / delay / name alignment is wrong" complaint).
        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))

        duration_label = FluentLabel("Duration")
        duration_label.setAlignment(QtCore.Qt.AlignCenter)
        duration_label.setToolTip("Duration")
        top_layout.addWidget(_set_fixed_height(duration_label))

        # The dot binds the field to a scan slot; the HOST owns the slot table, so
        # the card only exposes the button for it to wire.
        self.duration_edit = FluentScanLineEdit(
            _period_duration_text(period),
            tooltip="Duration value; click the dot to cycle scan (sN) -> API (aN) -> off")
        self.duration_edit.setFixedWidth(control_width)
        self.duration_dot = self.duration_edit.dot
        self.duration_edit.setToolTip("How long this period lasts, in the unit below")
        top_layout.addWidget(_set_fixed_height(self.duration_edit))

        self.unit_combo = FluentComboBox()
        _set_duration_unit_combo(
            self.unit_combo,
            scanned=_is_slot_expr(self.duration_edit.text()),
            unit=str(period.unit or "ns"),
        )
        self.unit_combo.setFixedWidth(control_width)
        top_layout.addWidget(_set_fixed_height(self.unit_combo))

        # name: free text, and the only thing telling two identical periods apart.
        self.name_edit = FluentLineEdit(str(period.name or ""))
        self.name_edit.setPlaceholderText("name")
        self.name_edit.setFixedWidth(control_width)
        self.name_edit.setToolTip("This period's name (shown in the preview and the summary)")
        self.name_edit.textChanged.connect(lambda *_: self.changed.emit())
        top_layout.addWidget(_set_fixed_height(self.name_edit))
        top_layout.addStretch()
        column.addWidget(top)
        # The panels carry a _px(8) top margin against the card's _px(7): this spacer
        # makes up exactly the difference (the reference's post-header sliver), so the
        # first channel row lands on the same Y as the Name/Delay rows.
        column.addSpacing(max(0, row_top - card_pad))
        # Seed resolution + validator from the value just loaded, so a freshly built
        # card enforces exactly what an edited one does.
        self._handle_duration_text(self.duration_edit.text())
        # Reflect a scan/API binding the state already carries, so a rebuilt or reopened
        # card shows the SAME orange sN / violet aN marker the operator set.  Without this
        # the dot cycle mutates the model but leaves the field looking untouched -- the
        # whole 3-state effect is invisible (the bug behind "the click does nothing").
        if _is_slot_expr(period.duration):
            slot_index = _slot_index_of_expr(period.duration)
            self.duration_edit.set_scan_bound(
                True, None if slot_index is None else slot_index + 1)
            self.unit_combo.setEnabled(False)
        else:
            api_name = state.api_slot_for("duration", str(index))
            if api_name:
                self.duration_edit.set_api_bound(True, _api_number(api_name))
        self.duration_edit.textChanged.connect(self._handle_duration_text)
        self.duration_edit.textChanged.connect(lambda *_: self.changed.emit())
        self.unit_combo.currentTextChanged.connect(self._handle_unit)
        self.unit_combo.currentTextChanged.connect(lambda *_: self.changed.emit())

        # --- one row per visible logical port, in catalog order.  EVERY row is pinned to
        # the shared _channel_row_height: the row pitch (height + spacing) must equal the
        # Name/Delay panels' exactly, or the columns drift apart row by row.  A natural
        # (unpinned) widget height only happens to match today and would drift with any
        # style change.
        channel_index = {channel: position for position, channel in enumerate(channels)}
        row_height = _channel_row_height(len(rows))
        for row in rows:
            key = str(row["key"])
            label = str(labels.get(key, row.get("label") or row.get("name") or key))
            members = [str(lane) for lane in (row.get("channels") or ())]
            if str(row.get("kind")) == "bus":
                column.addWidget(self._build_bus_row(
                    str(row["name"]), label, members, period, channel_index,
                    state, index, compact, row_height))
                continue
            lane = members[0] if members else key
            check = FluentCheckBox(label)
            check.setToolTip(label if compact else f"{label}: high for this whole period")
            check.setFixedHeight(row_height)
            position = channel_index.get(lane)
            check.setChecked(bool(position is not None and period.states[position]))
            check.toggled.connect(lambda *_: self.changed.emit())
            self.checks[lane] = check
            column.addWidget(check)
        column.addStretch(1)

    def _build_bus_row(self, bus_name: str, label: str, members: Sequence[str],
                       period: PulsePeriod, channel_index: Mapping[str, int],
                       state: PulseTableState, index: int, compact: bool,
                       row_height: int) -> QtWidgets.QWidget:
        """A DAC bus as ONE row: a mode and a value, not its individual lanes.

        The operator sets an integer; the lanes are how the compiler carries it.
        The value shown is decoded from those lanes -- the exact inverse of what
        :meth:`to_period` encodes -- so the field always agrees with the sequence
        it was loaded from.  ``hold`` means "keep whatever the previous period
        left", so it has no value of its own and the field goes away rather than
        showing a number that is not being applied.
        """

        self.bus_members[bus_name] = list(members)
        lo, hi = bus_signed_range(len(members))
        self.bus_bounds[bus_name] = (lo, hi)

        # The lane bits carry the OFFSET-BINARY wire code (signed + bus_zero_code);
        # everything the operator sees is the SIGNED value around true 0 V.
        code = 0
        for bit, lane in enumerate(members):
            position = channel_index.get(lane)
            if position is not None and period.states[position]:
                code |= 1 << bit
        plan = list((state.analog_bus_modes or {}).get(bus_name, ()))
        entry = plan[index] if 0 <= index < len(plan) else {}
        # An untouched bus HOLDS (the model's default): defaulting to "edge" here made
        # the first read_state stamp every visible bus edge@0, poisoning the state.
        mode = str((entry or {}).get("mode", "hold"))
        stored = (entry or {}).get("value")

        # The bus row is pinned to the SAME row height as every channel row (the combo and
        # value edit override their taller natural minimums), so the DAC rows keep the
        # Name/Delay panels' pitch and the columns stay aligned past them.  The row shows
        # NO name label -- like the reference, its identity is read ACROSS from the Port
        # Catalog column; the card only holds the mode combo and the value field, so the
        # widest value (-512) always fits.
        row = QtWidgets.QWidget()
        row.setStyleSheet("background: transparent;")
        row.setFixedHeight(row_height)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        # A clear gap between the mode combo and the value field (they used to sit
        # almost flush, which read as one merged control).
        layout.setSpacing(_px(7, minimum=5))

        combo = FluentComboBox()
        combo.addItems([_bus_mode_title(name) for name in ("edge", "ramp", "hold")])
        combo.setCurrentText(_bus_mode_title(mode))
        combo.setToolTip(f"{bus_name}: output mode")
        combo.setFixedSize(_bus_mode_combo_width(), row_height)
        self.bus_mode_combos[bus_name] = combo
        layout.addWidget(combo, 0)

        # Displayed value, three-state exactly like the reference: a scan-bound value shows
        # its slot expression; a HOLD shows the (read-only) value carried in from the
        # preceding edge/ramp; otherwise the SIGNED stored value (falling back to decoding
        # the lane code minus bus_zero_code -- the lanes carry offset-binary, the user
        # never sees the +2^(B-1) offset).
        if _is_slot_expr(stored):
            text = str(stored)
        elif stored is None and mode == "hold":
            try:
                text = str(int(state.analog_bus_value_at_period_start(index, bus_name)))
            except Exception:
                text = str(code - bus_zero_code(len(members)))
        elif stored is not None:
            text = _normalize_bus_value_text(stored, lo=lo, hi=hi)
        else:
            text = str(code - bus_zero_code(len(members)))
        edit_field = FluentScanLineEdit(
            text,
            tooltip=f"{label}: signed integer {lo}..{hi} (0 = 0 V); "
                    "click the dot to cycle scan (sN) -> API (aN) -> off")
        edit_field.set_numeric_validator("int", bottom=lo, top=hi)
        edit_field.setFixedHeight(row_height)
        edit_field.scanClicked.connect(lambda name=bus_name: self.busScanRequested.emit(name))
        self.bus_value_edits[bus_name] = edit_field
        self.bus_dots[bus_name] = edit_field.dot
        layout.addWidget(edit_field, 1)

        def commit(*_args, name=bus_name, widget=edit_field):
            if not _is_slot_expr(widget.text()):
                self._normalize_bus_value_edit(widget, *self.bus_bounds.get(name, (0, 0)))
            self.busChanged.emit()
            self.changed.emit()

        edit_field.editingFinished.connect(commit)
        combo.currentTextChanged.connect(lambda *_: (self.busChanged.emit(), self.changed.emit()))
        # HOLD keeps the field VISIBLE but read-only, showing the carried value (the
        # reference's behaviour; set_editable, not setEnabled -- disabling would take
        # the scan dot down with it and the binding could never be cleared).
        combo.currentTextChanged.connect(
            lambda title, w=edit_field: w.set_editable(_bus_mode_value(title) != "hold"))
        edit_field.set_editable(_bus_mode_value(combo.currentText()) != "hold")
        # Same scan/API marker the duration field shows: a DAC value bound to a scan slot
        # goes orange + read-only with its sN number; an API-bound value keeps its number
        # with a violet border.  Without this the DAC dot cycle leaves no visible trace.
        if _is_slot_expr(stored):
            slot_index = state.slot_index_for("dac", f"{bus_name}@{index}")
            edit_field.set_scan_bound(True, None if slot_index is None else slot_index + 1)
            combo.setEnabled(False)
        else:
            api_name = state.api_slot_for("dac", f"{bus_name}@{index}")
            if api_name:
                edit_field.set_api_bound(True, _api_number(api_name))
        return row

    def set_period_position(self, index: int, total: int) -> None:
        self.setTitle(f"Period {int(index) + 1}/{max(1, int(total))}")

    def _handle_duration_text(self, text: str) -> None:
        if _is_slot_expr(text):
            was_blocked = self.unit_combo.blockSignals(True)
            _set_duration_unit_combo(self.unit_combo, scanned=True, unit="str (ns)")
            self.unit_combo.blockSignals(was_blocked)
            self.unit_combo.setEnabled(False)
        else:
            self.unit_combo.setEnabled(True)
        self._handle_unit(self.unit_combo.currentText())

    def _handle_unit(self, unit: str) -> None:
        self.duration_edit.set_resolution(_unit_resolution(self.time_step_ns, unit))
        # A period duration must occupy at least one clock tick: snap up to >=1 tick
        # on commit (never to 0), matching the compiler and what the hardware runs.
        self.duration_edit.set_allow_any(False)
        # Numeric unit -> restrict typing to digits/./e (like the confocal float field).
        # "str (ns)" is the affine-expression mode -> allow s0/s1+20 etc (no validator).
        if unit == "str (ns)":
            self.duration_edit.setValidator(None)
        else:
            self.duration_edit.set_numeric_validator("float", bottom=0.0)

    def to_period(self, *, full_channels: Sequence[str], time_step_ns: float, slots: Mapping[str, float] | None = None) -> PulsePeriod:
        states = []
        for channel in full_channels:
            if channel in self.checks:
                states.append(1 if self.checks[channel].isChecked() else 0)
            else:
                states.append(1 if self.hidden_states.get(channel, 0) else 0)
        channel_index = {channel: index for index, channel in enumerate(full_channels)}
        for bus_name in self.bus_value_edits:
            members = self.bus_members.get(bus_name, [])
            mode_combo = self.bus_mode_combos.get(bus_name)
            mode = _bus_mode_value(mode_combo.currentText()) if mode_combo is not None else "hold"
            if mode == "hold":
                continue
            value_edit = self.bus_value_edits[bus_name]
            if _is_slot_expr(value_edit.text()):
                continue  # scanned DAC value; underlying bits stay as previewed
            lo, hi = self.bus_bounds.get(bus_name, (0, 0))
            value_text = _normalize_bus_value_text(value_edit.text(), lo=lo, hi=hi)
            if value_edit.text() != value_text:
                # read_state() must not itself mark the editor dirty: block the textChanged
                # that this normalization setText would otherwise fire (-> changed -> _mark_dirty).
                with _signals_blocked(value_edit):
                    value_edit.setText(value_text)
            # The lanes carry the OFFSET-BINARY wire code; the field holds the SIGNED value.
            code = int(value_text) + bus_zero_code(len(members))
            for bit, channel in enumerate(members):
                if channel in channel_index:
                    states[channel_index[channel]] = 1 if (code >> bit) & 1 else 0
        period = PulsePeriod(
            self.duration_edit.text().strip(),
            tuple(states),
            unit=self.unit_combo.currentText(),
            name=self.name_edit.text().strip(),
        )
        period.duration_ns(slots=slots, time_step_ns=time_step_ns)
        return period


    def bus_modes(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for bus_name, combo in self.bus_mode_combos.items():
            mode = _bus_mode_value(combo.currentText())
            value = None
            if mode != "hold":
                edit = self.bus_value_edits[bus_name]
                if _is_slot_expr(edit.text()):
                    value = edit.text().strip()  # scanned DAC value -> keep slot reference
                else:
                    lo, hi = self.bus_bounds.get(bus_name, (0, 0))
                    value_text = _normalize_bus_value_text(edit.text(), lo=lo, hi=hi)
                    if edit.text() != value_text:
                        edit.setText(value_text)
                    value = int(value_text)
            out[bus_name] = {"mode": mode, "value": value}
        return out

    def _normalize_bus_value_edit(self, edit: FluentLineEdit, lo: int, hi: int) -> None:
        try:
            edit.setText(_normalize_bus_value_text(edit.text(), lo=lo, hi=hi))
        except Exception:
            edit.setText("0")


class _DragItem:
    def __init__(self, widget: QtWidgets.QWidget, item_type: str):
        self.widget = widget
        self.item_type = item_type


class PulseDragContainer(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()
    # click-to-select: a click on a period card / on the gap between cards (no drag)
    cardClicked = QtCore.pyqtSignal(int)   # index in pulse-card (period) space
    gapClicked = QtCore.pyqtSignal(int)    # insert position in period space (0..n)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.items: list[_DragItem] = []
        self.drag_start_pos = None
        self.dragging_index = None
        self.layout_main = QtWidgets.QHBoxLayout(self)
        pad = _card_gutter()
        self.layout_main.setContentsMargins(pad, pad, pad, pad)
        self.layout_main.setSpacing(_px(5, minimum=3))
        self.layout_main.setAlignment(QtCore.Qt.AlignLeft)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        # The insert indicator is an OVERLAY child (absolute geometry, raised above the
        # cards), NOT a layout member: inserting it into the layout on every dragMove
        # re-laid-out every later card (visible jitter/teleporting during a drag).
        self.insert_indicator = QtWidgets.QFrame(self)
        self.insert_indicator.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.insert_indicator.setStyleSheet(f"background: {ACCENT};")
        self.insert_indicator.setFixedWidth(_px(3))
        self.insert_indicator.hide()
        # persistent selection highlight (click-to-select): None or period-space index/pos
        self._selected_card: int | None = None
        self._selected_gap: int | None = None

    def add_item(self, widget: QtWidgets.QWidget, item_type: str) -> None:
        self.items.append(_DragItem(widget, item_type))
        self.layout_main.addWidget(widget)

    def insert_item(self, index: int, widget: QtWidgets.QWidget, item_type: str) -> None:
        self.items.insert(max(0, min(index, len(self.items))), _DragItem(widget, item_type))
        self.refresh_layout()

    def refresh_layout(self) -> None:
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for item in self.items:
            self.layout_main.addWidget(item.widget)
        self.insert_indicator.hide()
        self.update_period_titles()
        self.changed.emit()

    def pulse_cards(self) -> list[PeriodCard]:
        return [item.widget for item in self.items if item.item_type == "pulse"]

    def update_period_titles(self) -> None:
        cards = self.pulse_cards()
        total = len(cards)
        for index, card in enumerate(cards):
            card.set_period_position(index, total)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_start_pos = event.pos()
            self.dragging_index = self._index_at(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_start_pos is None or self.dragging_index is None:
            return super().mouseMoveEvent(event)
        if (event.buttons() & QtCore.Qt.LeftButton) and (event.pos() - self.drag_start_pos).manhattanLength() > QtWidgets.QApplication.startDragDistance():
            self._start_drag(self.dragging_index)
            self.drag_start_pos = None
            self.dragging_index = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # A press that never crossed the drag threshold is a CLICK: select the
        # period card under the cursor, or the GAP between cards ("中缝") --
        # add/remove then act on the selection instead of always on the last.
        if event.button() == QtCore.Qt.LeftButton and self.drag_start_pos is not None:
            index = self._index_at(event.pos())
            if index is not None and self.items[index].item_type == "pulse":
                self.cardClicked.emit(self._period_index_of_item(index))
            else:
                self.gapClicked.emit(self._period_pos_of_insert(self._insert_pos(event.pos())))
        self.drag_start_pos = None
        self.dragging_index = None
        super().mouseReleaseEvent(event)

    def _period_index_of_item(self, items_index: int) -> int:
        """items-space index -> period (pulse-card) index."""
        return sum(1 for item in self.items[:items_index] if item.item_type == "pulse")

    def _period_pos_of_insert(self, items_pos: int) -> int:
        """items-space insert position -> period-space insert position (0..n)."""
        return sum(1 for item in self.items[:items_pos] if item.item_type == "pulse")

    def _start_drag(self, index: int) -> None:
        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setData("application/x-zlc-pulse-card", str(index).encode("utf-8"))
        drag.setMimeData(mime)
        widget = self.items[index].widget
        # Show WHAT is being dragged: a half-transparent snapshot of the card under
        # the cursor (without it the drag was an empty cursor and cards seemed to
        # "vanish" mid-drag).  Scaled down so a tall card doesn't cover the row.
        pixmap = widget.grab()
        if pixmap.height() > _px(180):
            pixmap = pixmap.scaledToHeight(_px(180), QtCore.Qt.SmoothTransformation)
        ghost = QtGui.QPixmap(pixmap.size())
        ghost.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(ghost)
        painter.setOpacity(0.65)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        drag.setPixmap(ghost)
        drag.setHotSpot(QtCore.QPoint(ghost.width() // 2, _px(16)))
        widget.set_outline("#808080")      # whole-card edge marker for the drag source
        drag.exec_(QtCore.Qt.MoveAction)
        widget.set_outline(None)
        self.insert_indicator.hide()
        self.update_period_titles()

    def _ancestor_scroll_area(self) -> QtWidgets.QAbstractScrollArea | None:
        w = self.parentWidget()
        while w is not None and not isinstance(w, QtWidgets.QAbstractScrollArea):
            w = w.parentWidget()
        return w

    def _autoscroll_during_drag(self, pos) -> None:
        """Keep later periods REACHABLE while dragging: out-of-view cards used to
        look like they had disappeared because the view cannot be scrolled by the
        wheel mid-drag.  Nudge the surrounding scroll area when the cursor comes
        near its left/right edge."""
        area = self._ancestor_scroll_area()
        if area is None:
            return
        viewport = area.viewport()
        vpos = viewport.mapFromGlobal(self.mapToGlobal(pos))
        margin = _px(56, minimum=40)
        hbar = area.horizontalScrollBar()
        if vpos.x() > viewport.width() - margin:
            hbar.setValue(hbar.value() + _px(28, minimum=20))
        elif vpos.x() < margin:
            hbar.setValue(hbar.value() - _px(28, minimum=20))

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-zlc-pulse-card"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-zlc-pulse-card"):
            event.acceptProposedAction()
            self._show_insert_indicator(self._insert_pos(event.pos()))
            self._autoscroll_during_drag(event.pos())
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasFormat("application/x-zlc-pulse-card"):
            return super().dropEvent(event)
        old_index = int(bytes(event.mimeData().data("application/x-zlc-pulse-card")).decode("utf-8"))
        insert_pos = self._insert_pos(event.pos())
        new_items = self.items[:]
        dragged = new_items.pop(old_index)
        if insert_pos > old_index:
            insert_pos -= 1
        new_items.insert(insert_pos, dragged)
        if not self._bracket_ok(new_items):
            event.ignore()
            self.insert_indicator.hide()
            return
        self.items = new_items
        self.refresh_layout()
        self.insert_indicator.hide()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.insert_indicator.hide()
        super().dragLeaveEvent(event)

    def _index_at(self, pos) -> int | None:
        for index, item in enumerate(self.items):
            if item.widget.geometry().contains(pos):
                return index
        return None

    def _insert_pos(self, pos) -> int:
        x = pos.x()
        for index, item in enumerate(self.items):
            geo = item.widget.geometry()
            if x < geo.x() + geo.width() // 2:
                return index
        return len(self.items)

    def _indicator_x_for_items_pos(self, items_pos: int) -> int:
        spacing = self.layout_main.spacing()
        if not self.items:
            return _card_gutter()
        if items_pos >= len(self.items):
            geo = self.items[-1].widget.geometry()
            return geo.x() + geo.width() + max(1, spacing // 2) - _px(1)
        geo = self.items[items_pos].widget.geometry()
        return max(0, geo.x() - max(1, spacing // 2) - _px(1))

    def _show_insert_indicator(self, items_pos: int) -> None:
        # OVERLAY positioning: geometry only, no layout mutation (a layout insert
        # per dragMove re-laid-out all later cards -> jitter during the drag).
        pad = _card_gutter()
        self.insert_indicator.setGeometry(
            self._indicator_x_for_items_pos(items_pos), pad,
            self.insert_indicator.width(), max(_px(40), self.height() - 2 * pad))
        self.insert_indicator.show()
        self.insert_indicator.raise_()

    # --- click-to-select visuals (selection highlight mirrors the drag highlight) ---
    def show_selection(self, *, card: int | None = None, gap: int | None = None) -> None:
        """Highlight one period card (border) or one gap (persistent indicator)."""
        self._selected_card = card
        self._selected_gap = gap
        # outline drawn on the card's outer edge (FluentGroupBox.set_outline) -- a
        # stylesheet border here would cascade to every child widget inside the card.
        cards = self.pulse_cards()
        for index, widget in enumerate(cards):
            widget.set_outline(ACCENT if (card is not None and index == card) else None)
        if gap is not None:
            items_pos = self._items_pos_of_period_gap(gap)
            self._show_insert_indicator(items_pos)
        else:
            self.insert_indicator.hide()

    def _items_pos_of_period_gap(self, period_pos: int) -> int:
        """period-space insert position (0..n) -> items-space position."""
        seen = 0
        for index, item in enumerate(self.items):
            if item.item_type == "pulse":
                if seen == period_pos:
                    return index
                seen += 1
        return len(self.items)

    def _bracket_ok(self, items: list[_DragItem]) -> bool:
        start = next((i for i, item in enumerate(items) if item.item_type == "bracket_start"), None)
        end = next((i for i, item in enumerate(items) if item.item_type == "bracket_end"), None)
        if start is None or end is None:
            return True
        return end >= start + 3


class RepeatBracket(FluentGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, kind: str, repeat_count: int = 2, parent=None):
        super().__init__("", parent)
        self.kind = kind
        self.setFixedWidth(_px(78, minimum=60))
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(_px(7), _px(7), _px(7), _px(7))
        layout.setSpacing(_px(6, minimum=4))
        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))
        label = FluentLabel("Repeat")
        label.setAlignment(QtCore.Qt.AlignCenter)
        top_layout.addWidget(_set_fixed_height(label))
        self.repeat_spin = None
        if kind == "end":
            self.repeat_spin = FluentDoubleSpinBox(length=5, allow_minus=False)
            self.repeat_spin.setRange(1, 999)
            self.repeat_spin.setValue(repeat_count)
            self.repeat_spin.setFixedHeight(_row_height())
            self.repeat_spin.valueChanged.connect(self.changed)
            top_layout.addWidget(self.repeat_spin)
        else:
            spacer = QtWidgets.QWidget()
            spacer.setStyleSheet("background: transparent;")
            spacer.setFixedHeight(_row_height())
            top_layout.addWidget(spacer)
        unit_spacer = QtWidgets.QWidget()
        unit_spacer.setStyleSheet("background: transparent;")
        unit_spacer.setFixedHeight(_row_height())
        top_layout.addWidget(unit_spacer)
        top_layout.addStretch()
        layout.addWidget(top)
        layout.addStretch()


class ChannelNamesPanel(FluentGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, state: PulseTableState, *, raw_labels: Mapping[str, str] | None = None, parent=None):
        super().__init__("Port Catalog", parent)
        self.state = state
        self.raw_labels = {str(channel): str(label) for channel, label in dict(raw_labels or {}).items()}
        self.port_labels: dict[str, FluentLineEdit] = {}
        self.raw_label_widgets: dict[str, FluentLabel] = {}
        self.top_labels: dict[str, FluentLabel] = {}
        self.rows = _display_rows(state)
        label_w = _channel_label_width()
        edit_w = _channel_name_edit_width()
        panel_w = label_w + edit_w + _px(5) + _px(20)
        self.setMinimumWidth(panel_w)
        self.setMaximumWidth(panel_w)

        layout = QtWidgets.QVBoxLayout(self)
        row_top, row_gap = _row_region_vmetrics()
        layout.setContentsMargins(_px(8), row_top, _px(8), _px(8))
        layout.setSpacing(row_gap)

        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))

        self.name_edit = FluentLineEdit(state.name)
        self.name_edit.setPlaceholderText("pulse name")
        self.name_edit.textChanged.connect(self.changed)
        self.top_labels["name"] = _add_labeled_widget(top_layout, "Name:", self.name_edit)

        self.total_label = FluentLineEdit("")
        self.total_label.setEnabled(False)
        self.top_labels["total"] = _add_labeled_widget(top_layout, "Total:", self.total_label)
        self.periods_label = FluentLineEdit("")
        self.periods_label.setEnabled(False)
        self.top_labels["periods"] = _add_labeled_widget(top_layout, "Periods:", self.periods_label)
        self.visible_label = FluentLineEdit("")
        self.visible_label.setEnabled(False)
        self.top_labels["visible"] = _add_labeled_widget(top_layout, "Visible:", self.visible_label)
        top_layout.addStretch()
        layout.addWidget(top)

        row_height = _channel_row_height(len(self.rows))
        for row_info in self.rows:
            key = str(row_info["key"])
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(_px(5, minimum=3))
            hardware_text = (str(row_info["name"]) if row_info.get("kind") == "bus"
                             else self.raw_labels.get(key, str(row_info["name"])))
            hardware = FluentLabel(hardware_text)
            if row_info.get("kind") == "bus":
                hardware.setToolTip(", ".join(str(item) for item in row_info.get("channels", [key])))
            else:
                hardware.setToolTip(f"port {row_info['name']} / raw lane {key}")
            hardware.setAlignment(QtCore.Qt.AlignCenter)
            hardware.setFixedSize(label_w, row_height)
            self.raw_label_widgets[key] = hardware
            label = FluentLineEdit(str(row_info.get("label") or row_info["name"]))
            label.setToolTip(
                "The channel's display NAME (editable).  Renaming only changes the human label; the "
                "hardware wiring (raw lane / DAC grouping / ABI fingerprint) is fixed by the sequencer "
                "PortCatalog and is never touched here -- the left column shows that fixed lane/pin.")
            label.setFixedWidth(edit_w)
            label.setFixedHeight(row_height)
            label.textChanged.connect(self.changed)
            # Keyed by the PORT KEY (``row_info['name']``), which is what ``PortCatalog.with_label``
            # renames -- NOT the row key (a raw lane for a channel, ``bus:...`` for a DAC).
            self.port_labels[str(row_info["name"])] = label
            row.addWidget(hardware)
            row.addWidget(label, 1)
            layout.addLayout(row)
        layout.addStretch()

    def read_values(self, state: PulseTableState) -> None:
        """Apply any edited channel display names back onto the state's PortCatalog through the ONE
        rename entry (``with_label``).  Labels are display-only metadata (outside the ABI fingerprint),
        so this never alters topology; the pulse name above is read by the editor separately."""
        catalog = state.port_catalog
        for port_key, widget in self.port_labels.items():
            new_label = widget.text().strip() or port_key
            spec = catalog.by_key.get(port_key)
            if spec is not None and spec.label != new_label:
                catalog = catalog.with_label(port_key, new_label)
        state.port_catalog = catalog


class ChannelPanel(FluentGroupBox):
    changed = QtCore.pyqtSignal()
    clearRequested = QtCore.pyqtSignal(str)
    loadScanRequested = QtCore.pyqtSignal()
    scanSourceToggled = QtCore.pyqtSignal(bool)
    delayApiRequested = QtCore.pyqtSignal(str)  # cycle this channel's delay as an API slot (none->api->none)

    def __init__(self, state: PulseTableState, parent=None):
        super().__init__("Delay / Scan", parent)
        self.state = state
        self.delay_edits: dict[str, FluentLineEdit] = {}
        self.delay_units: dict[str, FluentComboBox] = {}
        self.channel_labels: dict[str, ElidedLabel] = {}
        self.top_labels: dict[str, FluentLabel] = {}
        self.rows = _display_rows(state)
        label_w = _channel_label_width()
        delay_w = _px(70, minimum=60)
        unit_w = _time_unit_width()
        hide_w = _hide_button_width()
        gap = _px(4, minimum=3)
        content_w = label_w + delay_w + unit_w + hide_w + gap * 3 + _px(16)
        self.setMinimumWidth(content_w)
        self.setMaximumWidth(content_w)

        layout = QtWidgets.QVBoxLayout(self)
        row_top, row_gap = _row_region_vmetrics()
        layout.setContentsMargins(_px(8), row_top, _px(8), _px(8))
        layout.setSpacing(row_gap)

        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))

        # The tick step is fixed by the FPGA clock -- it is NOT user-editable.  Show
        # the clock frequency + tick width read-only instead of an editable "Step" box.
        self.step_display = FluentLineEdit(_format_clock_text(state.time_step_ns))
        self.step_display.setEnabled(False)
        self.step_display.setToolTip("FPGA clock (fixed by hardware). One tick = 1 / clock; all times snap to a whole tick.")
        self.top_labels["step"] = _add_labeled_widget(top_layout, "Clock:", self.step_display)

        self.scan_summary = FluentLineEdit("")
        self.scan_summary.setEnabled(False)
        self.scan_summary.setToolTip("Active scan slots and uploaded scan points. Click a dot to bind a field.")
        self.top_labels["scan"] = _add_labeled_widget(top_layout, "Scan:", self.scan_summary)

        # Load Array + the loaded file's tail path beside it (elide from the left so
        # the last dozen-or-so characters of the path stay visible).
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(_px(5, minimum=3))
        self.load_button = FluentButton("Load Array", color=ACCENT)
        self.load_button.setFixedHeight(_row_height())
        self.load_button.setToolTip("Load a scan-table file (.npy/.csv/.txt/.json): one row per scan point, one column per slot.")
        self.load_button.clicked.connect(self.loadScanRequested)
        self.scan_file_label = ElidedLabel("(no file)", mode=QtCore.Qt.ElideLeft)
        self.scan_file_label.setFixedHeight(_row_height())
        self.scan_file_label.setToolTip("The scan-table file currently loaded (used when the toggle below is on).")
        btn_row.addWidget(self.load_button)
        btn_row.addWidget(self.scan_file_label, 1)
        top_layout.addLayout(btn_row)

        # Toggle: off = use the scan table generated in the Scan tab; on = use the
        # array loaded from the file above.
        self.scan_source_toggle = FluentSwitch("Use loaded file")
        self.scan_source_toggle.setFixedHeight(_row_height())
        self.scan_source_toggle.setToolTip(
            "Off: use the scan table generated in the Scan tab (Run).\n"
            "On: use the array loaded with Load Array."
        )
        self.scan_source_toggle.toggled.connect(self.scanSourceToggled)
        top_layout.addWidget(self.scan_source_toggle)
        top_layout.addStretch()
        layout.addWidget(top)

        row_height = _channel_row_height(len(self.rows))
        for row_info in self.rows:
            key = str(row_info["key"])
            members = [str(channel) for channel in row_info.get("channels", [key])]
            is_bus = row_info.get("kind") == "bus"
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(gap)
            label = ElidedLabel(_display_row_label(row_info))
            label.setToolTip(", ".join(members))
            label.setFixedSize(label_w, row_height)
            self.channel_labels[key] = label

            # A delay is the API target itself for a plain channel (key == channel), and the
            # BARE BUS NAME for a bus row (row_info["name"], matching analog_buses) -- the data
            # layer fans a bus-delay slot out to its members.  ONE target token, ONE code path.
            api_target = str(row_info["name"]) if is_bus else key
            mixed_delay = False
            if is_bus:
                member_values = [state.delays.get(channel, 0) for channel in members]
                member_units = [state.delay_units.get(channel, "ns") for channel in members]
                uniform = member_values and all(value == member_values[0] for value in member_values) \
                    and all(unit == member_units[0] for unit in member_units)
                # A notebook can legally give the bus MEMBERS different delays.  The single
                # bus field cannot show them -- displaying 0 and writing it back on the next
                # read_state used to SILENTLY DELETE the per-member values.  Show "(mixed)"
                # instead and leave the state untouched until the user actually types here.
                mixed_delay = bool(member_values) and not uniform
                delay_value = member_values[0] if uniform else ""
                delay_unit = member_units[0] if uniform else "ns"
            else:
                delay_value = state.delays.get(key, 0)
                delay_unit = state.delay_units.get(key, "ns")
            # A delay is a FIXED output delay (a delay line) -- NOT scannable, but it CAN be an
            # API slot the notebook/Task sets by name.  EVERY delay field (plain channel AND DAC
            # bus) carries the same inline dot, API-only: clicking it cycles none -> API (aN,
            # violet, value kept) -> none.  A bus's handle fans out to its members; there is no
            # asymmetry between a TTL channel's delay and the bus's.
            delay_edit = FluentScanLineEdit(str(delay_value))
            delay_edit.scanClicked.connect(lambda t=api_target: self.delayApiRequested.emit(t))
            api_name = self.state.api_slot_for("delay", api_target)
            if api_name:
                delay_edit.set_api_bound(True, _api_number(api_name))
            if mixed_delay:
                delay_edit.setPlaceholderText("(mixed)")
                delay_edit.setToolTip("Members of this bus carry DIFFERENT delays (set via the "
                                      "API).  Typing a value here applies it to ALL members; "
                                      "leaving it untouched keeps the per-member values.")
                delay_edit._zlc_mixed = True
                delay_edit.textEdited.connect(
                    lambda _t, e=delay_edit: setattr(e, "_zlc_mixed", False))
            if is_bus:
                delay_edit.setToolTip(
                    "Physical DAC-bus output delay (may be negative): the whole bus value shifts "
                    "by d, out[t] = in[t-d], first frame correct -- delayed at the SEGMENT level "
                    "(each edge/ramp descriptor is re-played d ticks later; the bus's 10 bits share "
                    f"this one delay), so the range matches TTL: up to {_delay_cap_text(state.time_step_ns)}. "
                    f"The limit is the number of segments in flight per bus (<= {_BUS_EVT_DEPTH}, the DA "
                    "per-bus segment-FIFO depth), not the delay length; only a repeat-forever delay "
                    "spanning many frames can exceed it."
                )
            else:
                delay_edit.setToolTip(
                    "Physical per-channel output delay (may be negative): the whole channel "
                    "waveform shifts by d, out[t] = in[t-d] -- the first frame is already "
                    "correct (no wrap, no modulo). "
                    f"|delay| up to {_delay_cap_text(state.time_step_ns)}. The real limit is the "
                    f"number of edges in flight at once (<= {_EVT_DEPTH}, the event-FIFO depth), "
                    "not the delay length; the compiler reports the worst-case count if exceeded."
                )
            delay_edit.setFixedSize(delay_w, row_height)
            # A delay is a number (digits / decimal point / e-notation / sign only).
            delay_edit.set_numeric_validator("float")
            delay_edit.textChanged.connect(lambda text, ch=key: self._handle_delay_text(ch, text))
            delay_edit.textChanged.connect(self.changed)
            delay_edit.editingFinished.connect(lambda ch=key, edit=delay_edit: self._clamp_delay_edit(ch, edit))
            unit = FluentComboBox()
            unit.addItems(DELAY_UNITS)
            unit.setCurrentText(delay_unit if delay_unit in DELAY_UNITS else "ns")
            unit.setFixedSize(unit_w, row_height)
            unit.currentTextChanged.connect(lambda unit_text, ch=key: self._handle_delay_unit(ch, unit_text))
            unit.currentTextChanged.connect(self.changed)
            clear_btn = FluentButton("X", color=ORANGE)
            clear_btn.setFixedSize(hide_w, row_height)
            clear_btn.setToolTip("Set this row fully off.")
            clear_btn.clicked.connect(lambda _=False, ch=key: self.clearRequested.emit(ch))

            # A delay needs an event FIFO; only real pulse-programmable outputs
            # appear here.  Fixed clock ports were filtered by _display_rows.
            hw_pos = _delay_eligible_position(key)
            not_eligible = (not is_bus) and hw_pos is not None and hw_pos >= NUM_DELAY_CHANNELS
            if not_eligible:
                delay_edit.setEnabled(False)
                unit.setEnabled(False)
                delay_edit.setToolTip(
                    f"This output is past the {NUM_DELAY_CHANNELS} delay-eligible lanes; "
                    "it has no event FIFO and cannot be delayed.")

            self.delay_edits[key] = delay_edit
            self.delay_units[key] = unit
            row.addWidget(label)
            row.addWidget(delay_edit, 1)
            row.addWidget(unit)
            row.addWidget(clear_btn)
            layout.addLayout(row)
            self._handle_delay_text(key, delay_edit.text())
            self._handle_delay_unit(key, unit.currentText())
        layout.addStretch()
        self.set_scan_summary()

    def set_scan_summary(self) -> None:
        n_slots = len(self.state.scan_slots)
        n_points = len(self.state.scan_table)
        if n_slots == 0:
            text = "no scan slots"
        else:
            text = f"{n_slots} slot{'s' if n_slots != 1 else ''} · {n_points} pt{'s' if n_points != 1 else ''}"
        self.scan_summary.setText(text)

    def set_scan_source(self, *, use_loaded: bool, path: str) -> None:
        """Reflect the active scan-table source on the toggle + the file-path label
        (called after a rebuild so the widgets survive a load_state)."""

        with _signals_blocked(self.scan_source_toggle):
            self.scan_source_toggle.setChecked(bool(use_loaded))
        self.scan_file_label.setText(str(path) if path else "(no file)")

    def _handle_delay_text(self, channel: str, text: str) -> None:
        combo = self.delay_units.get(channel)
        edit = self.delay_edits.get(channel)
        if combo is None or edit is None:
            return
        self._handle_delay_unit(channel, combo.currentText())

    def _handle_delay_unit(self, channel: str, unit: str) -> None:
        edit = self.delay_edits.get(channel)
        if edit is not None:
            edit.set_resolution(_unit_resolution(self.state.time_step_ns, unit))

    def _clamp_delay_edit(self, channel: str, edit: FluentLineEdit) -> None:
        """Clamp a finished delay entry to its physical magnitude cap: the 32-bit delay
        field (the same for a real TTL channel and a DAC bus -- both event-scheduled).  A
        larger delay can never be realized.  The compiler still raises a clear error if the
        *span* / in-flight count exceeds the limit after the global shift."""

        text = edit.text().strip()
        if not text:
            return
        try:
            value = float(text)
        except ValueError:
            return
        unit_combo = self.delay_units.get(channel)
        unit_text = unit_combo.currentText() if unit_combo is not None else "ns"
        factor = UNIT_TO_NS.get(unit_text, 1.0) or 1.0
        # TTL and DAC-bus delays are both event-scheduled with the SAME 32-bit range, so the
        # magnitude cap is identical for channels and buses.
        max_ns = DELAY_MAX_TICKS * float(self.state.time_step_ns)
        value_ns = value * factor
        if abs(value_ns) > max_ns + 1e-6:
            clamped = max(-max_ns, min(max_ns, value_ns)) / factor
            edit.setText(format_compact_number(clamped))

    def read_values(self, state: PulseTableState) -> None:
        for row_info in self.rows:
            key = str(row_info["key"])
            if key not in self.delay_edits:
                continue
            # an untouched "(mixed)" bus field keeps the per-member delays exactly as the
            # API set them; only an actual edit overwrites all members uniformly.
            if getattr(self.delay_edits[key], "_zlc_mixed", False):
                continue
            raw = self.delay_edits[key].text().strip()
            unit = self.delay_units[key].currentText()
            try:
                is_zero = float(raw) == 0.0   # numeric 0 / 0.0
            except (TypeError, ValueError):
                is_zero = (raw == "")         # empty -> no delay; an expression like "s0" is kept
            for channel in row_info.get("channels", [key]):
                channel = str(channel)
                # Don't persist a zero/empty delay: it adds noise to the saved JSON and makes a
                # genuinely-stale delay harder to spot.  A real (nonzero / expression) delay stays.
                if is_zero:
                    state.delays.pop(channel, None)
                    state.delay_units.pop(channel, None)
                else:
                    state.delays[channel] = raw
                    state.delay_units[channel] = unit

    def set_channel_display_labels(self, labels: Mapping[str, str]) -> None:
        for channel, label in self.channel_labels.items():
            label.setText(str(labels.get(channel) or channel))


class PulseSequenceEditor(QtWidgets.QWidget):
    """Confocal-style period-card editor for ``PulseTableState``."""

    def __init__(
        self,
        state: PulseTableState | None = None,
        *,
        target_descriptor=None,
        command_port=None,
        channel_pins: Mapping[str, str] | None = None,
        scale: float | None = None,
        window_ratio: float = DEFAULT_WINDOW_RATIO,
        parent=None,
    ):
        app = ensure_qt_app()
        super().__init__(parent)
        self.ui_scale = self._resolve_scale(scale, app=app)
        set_fluent_scale(self.ui_scale)
        self.window_ratio = max(0.45, min(1.0, float(window_ratio)))
        port_target = getattr(command_port, "target", None)
        if target_descriptor is None:
            target_descriptor = port_target
        elif port_target is not None and port_target != target_descriptor:
            raise ValueError("pulse command port belongs to another target descriptor")
        self.target_descriptor = target_descriptor
        self.command_port = command_port
        if state is None:
            port_catalog = getattr(target_descriptor, "port_catalog", None)
            if port_catalog is None:
                if not DEFAULT_CHANNEL_NAMES:
                    raise ValueError(
                        "offline Pulse GUI needs an explicit state or target descriptor")
                port_catalog = PortCatalog.from_channels(DEFAULT_CHANNEL_NAMES)
            visible = [
                port.key for port in port_catalog.ports if port.kind != PORT_CLOCK
            ][:4]
            state = PulseTableState(
                port_catalog=port_catalog,
                visible_ports=visible,
                time_step_ns=self._clock_step_ns(target_descriptor) or DEFAULT_TIME_STEP_NS,
            )
        device_catalog = getattr(target_descriptor, "port_catalog", None)
        if state is not None and device_catalog is not None \
                and state.port_catalog.fingerprint != device_catalog.fingerprint:
            state = state.aligned_to_catalog(device_catalog)
        device_step_ns = self._clock_step_ns(target_descriptor)
        if device_step_ns is not None and state.time_step_ns != device_step_ns:
            state = state.snapped(time_step_ns=device_step_ns)
        self.state = state
        self.channel_pins = {str(channel): str(pin) for channel, pin in dict(channel_pins or {}).items()}
        self._clock_hz = float(
            getattr(target_descriptor, "clock_hz", None)
            or (1e9 / self.state.time_step_ns)
        )
        self._connection_label = str(
            getattr(target_descriptor, "connection_label", "Offline (edit only)")
        )
        self.last_program = None
        # The HELD scan point (Stop ▸ hold point / ◀ step / step ▶): ``(index, raw row, table length)``,
        # or None when not holding.  Set only by _hold_at_point (the single hold seam); cleared by every
        # fresh device upload (_prepare_to_device) and by Stop Pulse (safe_state).
        self._held_scan_point: tuple[int, list[float], int] | None = None
        self.bracket_exists = False
        self.address_str = ""
        # Two scan-table sources, switched by the Delay/Scan panel toggle:
        #   "generated" -> produced by the Scan tab code (Run)
        #   "loaded"    -> read from a file (Load Array)
        self._scan_tables: dict[str, list[list[float]]] = {"generated": [], "loaded": []}
        self._scan_loaded_path = ""
        self._scan_use_loaded = False
        # The scan-SLOT layout the last Run generated its table against (a tuple of (kind, target) per
        # slot).  The generated table has one column per slot in this exact order, so if the slots later
        # change (a scan dot bound/unbound/moved) the generated table is STALE and Run must show its
        # dirty '*' -- a re-Run is needed for the change to take effect.  ``None`` = never Run yet.
        self._scan_generated_slots: tuple | None = None
        self._last_save_state = None
        self._last_load_state = None
        self._building = False
        self._preview_dirty = True
        self._preview_plot = None
        self._preview_canvas = None
        self._left_panels_collapsed = False
        self._summary_timer = QtCore.QTimer(self)
        self._summary_timer.setSingleShot(True)
        self._summary_timer.setInterval(SUMMARY_DEBOUNCE_MS)
        self._summary_timer.timeout.connect(self._update_summary)
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self.refresh_preview)
        self._build_ui()
        self.load_state(self.state)
        self._init_connection_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("PulseGUI@Zou lab")
        self.setFixedSize(self._target_editor_size())
        self.setStyleSheet(fluent_widget_stylesheet())

        root = QtWidgets.QVBoxLayout(self)
        # SAME window-edge inset as every other GUI: ``window_pad(1)`` on all four sides (the ONE
        # WINDOW_PAD unit).  The window title draws at ``scaled_px(TITLE_LEFT_INSET)`` == ``window_pad(1)``,
        # so the body's left edge lines up under the "PulseGUI@Zou lab" title text.  Inter-card gaps are
        # HALF that unit (``window_pad(0.5)``), keeping every spacing a clean multiple of the base.
        root.setContentsMargins(window_pad(1), window_pad(1), window_pad(1), window_pad(1))
        root.setSpacing(window_pad(0.5))

        # A top STRIP, not a boxed card: borderless like the button bar below (bordered=False), so it
        # does not read as an outer frame drawn around the header row (the residual 1 px box).
        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(_px(12), _px(6), _px(12), _px(6))
        header.setSpacing(_px(8, minimum=5))
        self.status_dot = FluentStatusDot(size=16)
        self.label_name = FluentLabel("PulseGUI - Untitled*")
        self.label_name.setMinimumWidth(_px(260, minimum=180))
        self.label_name.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.summary = FluentLineEdit("")
        self.summary.setEnabled(False)
        self.clear_all_button = FluentButton("Clear All", color=ORANGE)
        self.clear_all_button.setToolTip(
            "Reset the schedule: remove every period and every channel delay, leaving one "
            "blank 1 µs period with no channel on.\n"
            "The sequencer-owned PortCatalog and current visibility are kept."
        )
        self.clear_all_button.clicked.connect(self._clear_all)
        header.addWidget(self.status_dot)
        header.addWidget(self.label_name)
        header.addWidget(self.summary, 1)
        header.addWidget(self.clear_all_button)
        root.addWidget(header_frame)

        self.tabs = FluentTabWidget()
        self.edit_tab = QtWidgets.QWidget()
        self.edit_tab.setStyleSheet("background: transparent;")
        edit_layout = QtWidgets.QVBoxLayout(self.edit_tab)
        tab_margin = _px(8, minimum=5)
        edit_layout.setContentsMargins(tab_margin, tab_margin, tab_margin, tab_margin)
        edit_layout.setSpacing(_px(8, minimum=5))

        self.dataset_scroll = FluentScrollArea()
        self.dataset_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.dataset_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.dataset_body = QtWidgets.QWidget()
        dataset = QtWidgets.QHBoxLayout(self.dataset_body)
        gutter = _card_gutter()
        dataset.setContentsMargins(0, 0, 0, 0)
        dataset.setSpacing(0)
        self.names_panel_holder = QtWidgets.QWidget()
        self.names_panel_layout = QtWidgets.QVBoxLayout(self.names_panel_holder)
        self.names_panel_layout.setContentsMargins(gutter, gutter, gutter, gutter)
        self.names_panel_layout.setSpacing(0)
        dataset.addWidget(self.names_panel_holder)

        self.channel_panel_holder = QtWidgets.QWidget()
        self.channel_panel_layout = QtWidgets.QVBoxLayout(self.channel_panel_holder)
        self.channel_panel_layout.setContentsMargins(gutter, gutter, gutter, gutter)
        self.channel_panel_layout.setSpacing(0)
        dataset.addWidget(self.channel_panel_holder)

        # The collapsed stub gets the SAME gutter holder as the panels above so its flat 1 px border
        # is inset consistently and lines up with the panels' -- the left column stays tidy whether a
        # panel or its collapsed stub is shown.
        self.left_panel_stub_holder = QtWidgets.QWidget()
        stub_holder_layout = QtWidgets.QVBoxLayout(self.left_panel_stub_holder)
        stub_holder_layout.setContentsMargins(gutter, gutter, gutter, gutter)
        stub_holder_layout.setSpacing(0)
        self.left_panel_stub = FluentFrame()
        self.left_panel_stub.setFixedWidth(_px(82, minimum=68))
        stub_layout = QtWidgets.QVBoxLayout(self.left_panel_stub)
        stub_layout.setContentsMargins(_px(6), _px(8), _px(6), _px(8))
        stub_layout.setSpacing(_px(6, minimum=4))
        stub_label = FluentLabel("Name\nDelay")
        stub_label.setAlignment(QtCore.Qt.AlignCenter)
        stub_layout.addWidget(stub_label)
        self.stub_show_button = FluentButton("Show", color=ACCENT)
        self.stub_show_button.setFixedHeight(_row_height())
        self.stub_show_button.clicked.connect(self.show_left_panels)
        stub_layout.addWidget(self.stub_show_button)
        stub_layout.addStretch()
        stub_holder_layout.addWidget(self.left_panel_stub)
        self.left_panel_stub_holder.hide()
        dataset.addWidget(self.left_panel_stub_holder)

        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.drag_container = PulseDragContainer()
        self.drag_container.changed.connect(self._mark_dirty)
        # click-to-select a period or a gap; Add/Remove then act on the selection
        # (no selection = the old behaviour: append/remove at the end).
        self._selected_period: int | None = None
        self._selected_gap: int | None = None
        self.drag_container.cardClicked.connect(self._on_period_card_clicked)
        self.drag_container.gapClicked.connect(self._on_period_gap_clicked)
        self.scroll.setWidget(self.drag_container)
        dataset.addWidget(self.scroll, 1)
        self.dataset_scroll.setWidget(self.dataset_body)
        edit_layout.addWidget(self.dataset_scroll, 1)

        self.timeline_hbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.timeline_hbar.setStyleSheet(fluent_scrollbar_stylesheet("QScrollBar"))
        self.timeline_hbar.setFixedHeight(fluent_scrollbar_thickness())   # ONE source (matches the CSS)
        self.timeline_hbar.hide()
        self.timeline_hbar_spacer = QtWidgets.QWidget()
        hbar_row = QtWidgets.QHBoxLayout()
        hbar_row.setContentsMargins(0, 0, 0, 0)
        hbar_row.setSpacing(0)
        hbar_row.addWidget(self.timeline_hbar_spacer)
        hbar_row.addWidget(self.timeline_hbar, 1)
        edit_layout.addLayout(hbar_row)
        inner_hbar = self.scroll.horizontalScrollBar()
        inner_hbar.rangeChanged.connect(self._sync_timeline_scrollbar)
        inner_hbar.valueChanged.connect(self.timeline_hbar.setValue)
        self.timeline_hbar.valueChanged.connect(inner_hbar.setValue)

        # --- Bottom control bar: two titled Fluent cards (Control / Channels),
        # using the same group-box-with-title style as the other panels for
        # visual consistency.  Kept compact (single-line buttons, tight 2x4 grid,
        # small margins) so the name/delay/period area keeps its vertical room. ---
        self.button_frame = FluentFrame(bordered=False)
        # Scope the reset to THIS frame via an ID selector: a bare `QFrame { ... }`
        # cascades `border: none` onto the titled Control/Channels cards nested
        # inside it and strips their group-box borders (root AGENTS §5.1).
        self.button_frame.setObjectName("zlcPulseButtonBar")
        self.button_frame.setStyleSheet("QFrame#zlcPulseButtonBar { background: transparent; border: none; }")
        bar = QtWidgets.QHBoxLayout(self.button_frame)
        # Inset the Control / Channels group boxes by the card gutter on every side so their flat
        # 1 px borders sit a hair inside the button bar instead of flush against its edges.
        _sp = _card_gutter()
        bar.setContentsMargins(_sp, _sp + _px(2), _sp, _sp)
        bar.setSpacing(_px(10, minimum=8))
        cb_h = _px(30, minimum=26)

        control_area = FluentGroupBox("Control")
        control_col = QtWidgets.QVBoxLayout(control_area)
        control_col.setContentsMargins(_px(8), _px(2), _px(8), _px(6))
        control_col.setSpacing(_px(4, minimum=3))
        button_layout = QtWidgets.QGridLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(_px(6, minimum=4))
        self.safe_button = self._control_button("Stop Pulse", self.safe_state, RED)
        self.fire_button = self._control_button("On Pulse", self.fire, GREEN)
        self.fire_button.setToolTip(
            "Apply the editor's pulse to the device and run it.\n"
            "* = pressing would apply something new (edits not yet applied,\n"
            "or the device is not currently running).")
        self.remove_button = self._control_button("Remove", self.remove_period, ORANGE)
        self.add_button = self._control_button("Add Period", self.add_period, ACCENT)
        # ACCENT, not yellow: yellow is reserved for the Save button's dirty
        # state -- a permanently-yellow base button would read as "highlighted".
        self.bracket_button = self._control_button("Add Bracket", self.toggle_bracket, ACCENT)
        self.save_button = self._control_button("Save", self.save_to_file, YELLOW)
        self.load_button = self._control_button("Load", self.load_from_file, ORANGE)
        # Sync pulls the pulse actually APPLIED on the sequencer back into the editor --
        # for when a notebook/raw-API call (PulseController.on_pulse etc.) changed the
        # device behind the GUI's back.  Mirrors the confocal GUI's sync button.
        self.sync_button = self._control_button("Sync", self.sync_from_device, ORANGE)
        self.sync_button.setToolTip(
            "Load the pulse currently applied on the sequencer into the editor\n"
            "(use after changing the device from a notebook / raw API).")
        self.collapse_button = self._control_button("Collapse", self.toggle_left_panels, GREY)
        # 3x3 grid, row-major.  Keep the intuitive PAIRS side-by-side as the first two of
        # each row: On|Off (run/stop), Add|Remove (period), Save|Load (file).  The 3rd
        # column holds the standalone actions (Sync / Add Bracket / Collapse).
        _control_buttons = (
            self.fire_button, self.safe_button, self.sync_button,
            self.add_button, self.remove_button, self.bracket_button,
            self.save_button, self.load_button, self.collapse_button,
        )
        for button in _control_buttons:
            button.setFixedHeight(cb_h)
            button.setMinimumWidth(_px(74, minimum=62))
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        for index, button in enumerate(_control_buttons):
            button_layout.addWidget(button, index // 3, index % 3)
        for col in range(3):
            button_layout.setColumnStretch(col, 1)
        # Centre the buttons in the card so the (taller, height-matched) Control
        # box does not leave the grid floating with an uneven gap.
        control_col.addStretch(1)
        control_col.addLayout(button_layout)
        control_col.addStretch(1)
        bar.addWidget(control_area, 1)

        # --- Connection observation.  Backend selection belongs to the Workbench
        # installation authority; this card reports the bound target and cannot
        # construct or replace an adapter from the GUI. ---
        conn_area = FluentGroupBox("Connection")
        conn_area.setFixedWidth(_px(252, minimum=212))
        conn_col = QtWidgets.QVBoxLayout(conn_area)
        conn_col.setContentsMargins(_px(8), _px(2), _px(8), _px(6))
        conn_col.setSpacing(_px(4, minimum=3))
        self.conn_target_combo = FluentComboBox()
        self.conn_target_combo.setFixedHeight(cb_h)
        if self.command_port is None:
            self.conn_target_combo.addItem("Offline (edit only)", "offline")
        else:
            self.conn_target_combo.addItem("Installation managed", "managed")
        self.conn_target_combo.setToolTip(
            "Hardware binding is owned by the Workbench installation authority.")
        self.conn_target_combo.currentIndexChanged.connect(self._on_conn_target_changed)
        conn_col.addWidget(self.conn_target_combo)
        conn_row = QtWidgets.QHBoxLayout()
        conn_row.setContentsMargins(0, 0, 0, 0)
        conn_row.setSpacing(_px(6, minimum=4))
        self.conn_addr_edit = FluentLineEdit("127.0.0.1:18861")
        self.conn_addr_edit.setFixedHeight(cb_h)
        self.conn_addr_edit.setToolTip("Connection endpoints are installation configuration.")
        self.conn_connect_button = FluentButton("Managed", color=ACCENT)
        self.conn_connect_button.setFixedHeight(cb_h)
        self.conn_connect_button.setMinimumWidth(_px(64, minimum=54))
        self.conn_connect_button.setToolTip("Change devices through the installation manager.")
        self.conn_connect_button.clicked.connect(self._apply_connection)
        conn_row.addWidget(self.conn_addr_edit, 1)
        conn_row.addWidget(self.conn_connect_button)
        conn_col.addLayout(conn_row)
        self.conn_status = FluentLineEdit("")
        self.conn_status.setEnabled(False)
        self.conn_status.setFixedHeight(_row_height())
        conn_col.addWidget(self.conn_status)
        conn_col.addStretch(1)
        bar.addWidget(conn_area)

        view_area = FluentGroupBox("Ports")
        view_area.setFixedWidth(_px(286, minimum=246))
        view_col = QtWidgets.QVBoxLayout(view_area)
        view_col.setContentsMargins(_px(8), _px(2), _px(8), _px(6))
        view_col.setSpacing(_px(4, minimum=3))
        self.add_channel_combo = FluentComboBox()
        self.add_channel_combo.setFixedHeight(cb_h)
        self.add_channel_combo.setToolTip("Pick a hidden sequencer port to show.")
        view_col.addWidget(self.add_channel_combo)
        view_btn_row = QtWidgets.QHBoxLayout()
        view_btn_row.setContentsMargins(0, 0, 0, 0)
        view_btn_row.setSpacing(_px(6, minimum=4))
        self.add_channel_button = FluentButton("Add", color=ACCENT)
        self.add_channel_button.setToolTip("Add the selected hidden port to the table.")
        self.add_channel_button.clicked.connect(self.add_selected_port)
        self.hide_off_button = FluentButton("Hide Off", color=ORANGE)
        self.hide_off_button.setToolTip("Hide ports that are inactive in every period.")
        self.hide_off_button.clicked.connect(self.hide_off_ports)
        self.show_all_button = FluentButton("Show All", color=ACCENT)
        self.show_all_button.setToolTip("Show every programmable sequencer port.")
        self.show_all_button.clicked.connect(self.show_all_ports)
        for button in (self.add_channel_button, self.hide_off_button, self.show_all_button):
            button.setFixedHeight(cb_h)
            button.setMinimumWidth(_px(56, minimum=48))
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            view_btn_row.addWidget(button, 1)
        view_col.addLayout(view_btn_row)
        self.visible_label = FluentLineEdit("")
        self.visible_label.setEnabled(False)
        self.visible_label.setFixedHeight(_row_height())
        view_col.addWidget(self.visible_label)
        view_col.addStretch(1)
        bar.addWidget(view_area)

        self.button_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        edit_layout.addWidget(self.button_frame)
        self.tabs.addTab(self.edit_tab, "Edit")

        self.preview_tab = QtWidgets.QWidget()
        self.preview_tab.setStyleSheet("background: transparent;")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_tab)
        preview_layout.setContentsMargins(tab_margin, tab_margin, tab_margin, tab_margin)
        preview_layout.setSpacing(_px(8, minimum=5))

        preview_controls = FluentFrame()
        preview_controls.setFixedHeight(_px(48, minimum=40))
        preview_row = QtWidgets.QHBoxLayout(preview_controls)
        preview_row.setContentsMargins(_px(12), _px(6), _px(12), _px(6))
        preview_row.setSpacing(_px(10, minimum=6))
        preview_control_h = _px(32, minimum=28)
        self.preview_include_off = FluentSwitch("Show off rows")
        # Wide enough for the toggle plus the full "Show off rows" label even with
        # a wider substitute font (offscreen screenshots), so it never clips.
        self.preview_include_off.setFixedSize(_px(198, minimum=178), preview_control_h)
        self.preview_include_off.setToolTip("Show channels that are always off in the preview.")
        self.preview_include_off.toggled.connect(self._on_include_off_toggled)
        # "Selectors" switch, the SAME control (and default) as the console header:
        # the preview is display-only BY DEFAULT (the wheel scrolls; no misclick
        # zoom) -- flip ON to arm the unified selector layer (wheel zoom / pan,
        # area, cross) on the preview panel in place.
        self.preview_selectors_switch = FluentSwitch("Selectors")
        self.preview_selectors_switch.setFixedSize(
            _px(168, minimum=150), preview_control_h)
        self.preview_selectors_switch.setChecked(False)
        self.preview_selectors_switch.setToolTip(
            "OFF: the preview is display-only (wheel scrolls).\n"
            "ON: zoom / pan / area / cross work on the preview plot.")
        self.preview_selectors_switch.toggled.connect(
            self._on_preview_selectors_toggled)
        # Preview SIZE: one of PANEL_SIZES (the same size presets the console panels use), so the pulse
        # figure's data region scales like every other kind.  The default is optimal_pulse_size for the
        # current channel / period counts (the ONE default source, shared with the loaded panel); once the
        # operator picks a size it stays PINNED (no longer auto-tracks the content) and the chosen size is
        # saved into the figure so a reopen restores it.
        self.preview_size_label = FluentLabel("Size")
        self.preview_size_combo = FluentComboBox()
        self.preview_size_combo.addItems(list(PANEL_SIZES))
        self.preview_size_combo.setCurrentText("2x2")
        self.preview_size_combo.setFixedSize(_px(80, minimum=66), preview_control_h)
        self.preview_size_combo.setToolTip("Plot size preset -- a busy pulse defaults to a bigger size; "
                                           "pick one to pin it (saved with the figure).")
        self._preview_size_pinned = False        # False until the operator picks a size manually
        self.preview_size_combo.activated.connect(self._on_preview_size_picked)
        self.preview_save_figure_button = FluentButton("Save Figure", color=ACCENT)
        self.preview_save_figure_button.setFixedSize(_px(124, minimum=108), preview_control_h)
        self.preview_save_figure_button.clicked.connect(self.save_figure)
        self.preview_status = FluentLineEdit("")
        self.preview_status.setEnabled(False)
        self.preview_status.setFixedHeight(preview_control_h)
        preview_row.addWidget(self.preview_include_off)
        preview_row.addWidget(self.preview_selectors_switch)
        preview_row.addWidget(self.preview_size_label)
        preview_row.addWidget(self.preview_size_combo)
        preview_row.addWidget(self.preview_status, 1)
        preview_row.addWidget(self.preview_save_figure_button)
        preview_layout.addWidget(preview_controls)

        self.preview_scroll = FluentScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # A wheel over the PLOT must zoom the plot ONLY -- never also scroll this page.
        # Accepting on the canvas alone is not enough on every delivery path (Qt can
        # route the wheel to the scroll-area viewport), so filter at the VIEWPORT:
        # any wheel whose position lies on the canvas is forwarded to the canvas and
        # CONSUMED here, regardless of how Qt delivered it.
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_body = QtWidgets.QWidget()
        self.preview_body_layout = QtWidgets.QVBoxLayout(self.preview_body)
        self.preview_body_layout.setContentsMargins(_px(8), _px(8), _px(8), _px(8))
        self.preview_body_layout.setSpacing(0)
        self.preview_placeholder = FluentLabel("Open Preview to render the pulse plot.")
        self.preview_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_body_layout.addWidget(self.preview_placeholder)
        self.preview_scroll.setWidget(self.preview_body)
        preview_layout.addWidget(self.preview_scroll, 1)
        self.tabs.addTab(self.preview_tab, "Preview")
        self.scan_tab = self._build_scan_tab()
        self.tabs.addTab(self.scan_tab, "Scan")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs, 1)

        self.stateui_manager = PulseStateUIManager(
            status_dot=self.status_dot,
            label=self.label_name,
            save_button=self.save_button,
            fire_button=self.fire_button,
            title_callback=self._set_gui_title,
        )
        # Applied-state tracking (confocal UNSYNCED semantics): the canonical JSON of
        # the state at the last successful prepare/fire.  Any edit afterwards flips the
        # run state to UNSYNCED (orange On Pulse); the debounced summary pass compares
        # the keys and restores the previous run state if the edit was undone.
        self._applied_state_key: str | None = None
        self._unsynced_from: str | None = None

    def _control_button(self, text: str, slot, color: str) -> FluentButton:
        button = FluentButton(text, color=color)
        button.clicked.connect(slot)
        return button

    def toggle_left_panels(self) -> None:
        if self._left_panels_collapsed:
            self.show_left_panels()
        else:
            self.hide_left_panels()

    def hide_left_panels(self) -> None:
        self._left_panels_collapsed = True
        self.names_panel_holder.hide()
        self.channel_panel_holder.hide()
        self.left_panel_stub_holder.show()
        self.collapse_button.setText("Show Left")
        self._sync_dataset_geometry()

    def show_left_panels(self) -> None:
        self._left_panels_collapsed = False
        self.left_panel_stub_holder.hide()
        self.names_panel_holder.show()
        self.channel_panel_holder.show()
        self.collapse_button.setText("Collapse")
        self._sync_dataset_geometry()

    def _target_editor_size(self) -> QtCore.QSize:
        # The shared screen-fit rule (qt_widgets.screen_fit_window_size) -- identical
        # to the task console's, so the two GUIs never diverge on window sizing.
        return screen_fit_window_size(self.window_ratio)

    def _set_gui_title(self, title: str) -> None:
        self.setWindowTitle(title)
        window = self.window()
        if window is not self:
            window.setWindowTitle(title)
            title_bar = getattr(window, "titleBar", None)
            if title_bar is not None and hasattr(title_bar, "setTitle"):
                try:
                    title_bar.setTitle(title)
                except Exception:
                    pass

    def load_state(self, state: PulseTableState) -> None:
        self._building = True
        self.state = state
        # Sync the Scan-tab repeats spin to the LOADED whole-sweep count (#3) so a Save round-trips
        # it; signals blocked so this device/file load does not read as a user edit (dirty hint).
        if getattr(self, "scan_repeats_spin", None) is not None:
            with _signals_blocked(self.scan_repeats_spin):
                self.scan_repeats_spin.setValue(int(getattr(state, "scan_repeats", 0)))
        # Restore the Scan-tab editor's SOURCE code from the loaded state so a Save/Load round-trips
        # the editable program (not just the frozen scan_table).  This is IDEMPOTENT for in-session
        # in-session schedule mutations: read_state carried the current editor text, so
        # state.scan_code already equals the editor and the ``!=`` guard skips.  It only fires for a
        # FILE/DEVICE load or Clear All, where it must match the editor to the loaded program and mark
        # Run NOT dirty (code + table came from the same source -> the on-screen table is not stale).
        # Signals blocked so this does not read as a user edit; an empty saved code falls through to
        # _refresh_scan_tab's default-template auto-fill (a brand-new pulse), a non-empty one is left
        # verbatim (it is not the auto template, so the auto-fill guard leaves it).
        if getattr(self, "scan_code", None) is not None:
            saved_code = getattr(state, "scan_code", "") or ""
            if self.scan_code.toPlainText() != saved_code:
                with _signals_blocked(self.scan_code):
                    self.scan_code.setPlainText(saved_code)
                if getattr(self, "scan_run_button", None) is not None:
                    self.scan_run_button.set_dirty(False)
        # Suspend painting while we tear down and rebuild every channel panel and
        # period card (up to 5 periods x 62 channels = hundreds of widgets).  Each
        # addWidget on a *visible* tree would otherwise trigger an immediate
        # relayout + repaint; deferring to a single repaint at the end is the
        # dominant speed-up for "Show All".
        with batched_updates(self):
            # The channel-side panels (names + delay/scan) depend only on the
            # CHANNEL-side state; a period edit (add/remove/reorder/value) leaves
            # them untouched.  Skipping their teardown+rebuild when this key is
            # unchanged cuts a third off every Add/Remove-Period press.
            chan_key = (
                state.port_catalog.fingerprint, tuple(state.visible_ports),
                tuple(sorted((str(k), str(v)) for k, v in (getattr(state, "labels", None) or {}).items())),
                # str(v) not float(v): a delay value is legitimately a string EXPRESSION ("s0", "20+s1")
                # per PulseTableState.delays (float | str); float() would crash load_state on a valid saved
                # pulse.  This key only detects change, so string identity is sufficient and correct.
                tuple(sorted((str(k), str(v)) for k, v in (state.delays or {}).items())),
                tuple(sorted((str(k), str(v)) for k, v in (state.delay_units or {}).items())),
                # A delay API binding lives in api_slots (kind="delay"), NOT in delays, so it must be
                # in this key too: without it, cycling a delay dot to/from API leaves chan_key unchanged,
                # the channel panel is not rebuilt, and the violet aN marker never appears (or clear).
                tuple(sorted((str(s.target), str(s.name))
                             for s in (state.api_slots or ()) if getattr(s, "kind", None) == "delay")),
                len(state.scan_slots), float(state.time_step_ns),
            )
            if chan_key != getattr(self, "_chan_panel_key", None) or not hasattr(self, "channel_panel"):
                self._rebuild_channel_panels()
                self._chan_panel_key = chan_key
            else:
                # same channel-side data: just repoint the panels at the new state
                self.names_panel.state = state
                self.channel_panel.state = state
            self._rebuild_periods()  # ends with _sync_dataset_geometry()
            self._refresh_hidden_combo()
        self._building = False
        # cards were rebuilt: any click-selection now points at dead widgets.
        if hasattr(self, "_selected_period"):
            self._selected_period = None
            self._selected_gap = None
            self.drag_container.insert_indicator.hide()
        # a single period must not be removable (the table needs at least one).
        if hasattr(self, "remove_button"):
            self.remove_button.setEnabled(len(state.periods) > 1)
        self._preview_dirty = True
        # Route EVERY load_state-based mutation (add/remove period, clear channel, clk
        # toggle, show/hide, file load, ...) through the same dirty path as direct
        # widget edits -- so the UNSYNCED (orange On Pulse) hint covers them all.
        # Callers that load DEVICE/FILE state (sync_from_device, load_from_file) set
        # the run/file state explicitly right after, overriding this.
        self._mark_dirty()
        self._update_summary()

    def _clear_all(self) -> None:
        """Header Clear All: reset the SCHEDULE to a single blank 1 us period.

        Cleared: all periods, all channel/bus delays, DA bus plans, scan bindings +
        table, and the repeat bracket (they all describe the schedule being wiped).
        Kept: visibility and the immutable PortCatalog -- those describe the
        hardware hookup, not the pulse."""

        state = self.read_state()
        blank = PulseTableState(
            port_catalog=state.port_catalog,
            periods=[PulsePeriod(
                1.0,
                tuple(0 for _ in state.port_catalog.raw_lanes),
                unit="us",
            )],
            name=state.name,
            time_step_ns=state.time_step_ns,
            visible_ports=list(state.visible_ports),
        )
        self.load_state(blank)

    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _rebuild_channel_panels(self) -> None:
        self._clear_layout(self.names_panel_layout)
        self._clear_layout(self.channel_panel_layout)

        self.names_panel = ChannelNamesPanel(self.state, raw_labels=self.channel_pins)
        self.names_panel.changed.connect(self._handle_names_changed)
        self.names_panel_layout.addWidget(self.names_panel)
        self.name_edit = self.names_panel.name_edit

        self.channel_panel = ChannelPanel(self.state)
        self.channel_panel.changed.connect(self._mark_dirty)
        self.channel_panel.clearRequested.connect(self.clear_channel)
        self.channel_panel.loadScanRequested.connect(self._load_scan_file)
        self.channel_panel.scanSourceToggled.connect(self._on_scan_source_toggled)
        self.channel_panel.delayApiRequested.connect(self._toggle_delay_api)
        self.channel_panel_layout.addWidget(self.channel_panel)
        # Restore the active scan-table source (the panel was just recreated, so its
        # toggle + file label default back to off / empty).
        self.channel_panel.set_scan_source(use_loaded=self._scan_use_loaded, path=self._scan_loaded_path)

    def _rebuild_periods(self) -> None:
        while self.drag_container.layout_main.count():
            item = self.drag_container.layout_main.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self.drag_container.insert_indicator:
                widget.deleteLater()
        self.drag_container.items = []

        labels = self._display_labels_from_name_panel() if hasattr(self, "names_panel") else {
            row["key"]: _display_row_label(row) for row in _display_rows(self.state)
        }
        rows = _display_rows(self.state)
        compact = len(rows) > 16
        total_periods = len(self.state.periods)
        for index, period in enumerate(self.state.periods):
            hidden_states = {
                channel: period.states[self.state.channel_index(channel)]
                for channel in self.state.port_catalog.raw_lanes
            }
            card = PeriodCard(
                index,
                period,
                total_periods=total_periods,
                channels=self.state.port_catalog.raw_lanes,
                labels=labels,
                hidden_states=hidden_states,
                rows=rows,
                state=self.state,
                compact=compact,
                time_step_ns=self.state.time_step_ns,
            )
            card.changed.connect(self._mark_dirty)
            card.busChanged.connect(self._refresh_bus_displays)
            card.duration_dot.clicked.connect(lambda _checked=False, c=card: self._toggle_duration_scan(c))
            card.busScanRequested.connect(lambda bus_name, c=card: self._toggle_dac_scan(c, bus_name))
            self.drag_container.add_item(card, "pulse")
        if self.state.repeat_start is not None and self.state.repeat_end is not None and self.state.repeat_count > 1:
            start = RepeatBracket("start")
            end = RepeatBracket("end", self.state.repeat_count)
            start.changed.connect(self._mark_dirty)
            end.changed.connect(self._mark_dirty)
            self.drag_container.insert_item(self.state.repeat_start, start, "bracket_start")
            self.drag_container.insert_item(self.state.repeat_end + 2, end, "bracket_end")
            self.bracket_exists = True
            self.bracket_button.setText("Del Bracket")
        else:
            self.bracket_exists = False
            self.bracket_button.setText("Add Bracket")
        self.drag_container.refresh_layout()
        self._sync_dataset_geometry()

    def _sync_dataset_geometry(self) -> None:
        if not all(hasattr(self, name) for name in ("names_panel", "channel_panel", "drag_container", "scroll")):
            return
        def vertical_margins(layout: QtWidgets.QLayout | None) -> int:
            if layout is None:
                return 0
            margins = layout.contentsMargins()
            return margins.top() + margins.bottom()

        # sizeHint() below already triggers the layouts' size computation, so the
        # explicit adjustSize() resizes were redundant work (each forces a full
        # relayout of a 62-row panel).  The panels' real height is set via
        # setMinimumHeight / setFixedSize at the end of this method anyway.
        card_hints = [item.widget.sizeHint().height() for item in self.drag_container.items]
        drag_height = (max(card_hints) if card_hints else 0) + vertical_margins(self.drag_container.layout_main)
        names_height = 0 if self.names_panel_holder.isHidden() else self.names_panel.sizeHint().height() + vertical_margins(self.names_panel_layout)
        channel_height = 0 if self.channel_panel_holder.isHidden() else self.channel_panel.sizeHint().height() + vertical_margins(self.channel_panel_layout)
        stub_height = self.left_panel_stub.sizeHint().height() if hasattr(self, "left_panel_stub_holder") and not self.left_panel_stub_holder.isHidden() else 0
        content_height = max(
            names_height,
            channel_height,
            stub_height,
            drag_height,
        )
        content_height += _px(2, minimum=1)
        for widget in (self.names_panel_holder, self.channel_panel_holder, self.left_panel_stub_holder, self.scroll, self.drag_container):
            widget.setMinimumHeight(content_height)
        self.dataset_body.setMinimumHeight(content_height + vertical_margins(self.dataset_body.layout()))
        container_width = self._drag_container_width()
        self.drag_container.setFixedSize(container_width, content_height)
        self._sync_timeline_scrollbar()

    def _display_labels_from_name_panel(self) -> dict[str, str]:
        rows = getattr(getattr(self, "names_panel", None), "rows", _display_rows(self.state))
        return {str(row["key"]): _display_row_label(row) for row in rows}

    def _refresh_visible_display_labels(self) -> None:
        labels = self._display_labels_from_name_panel()
        if hasattr(self, "channel_panel"):
            self.channel_panel.set_channel_display_labels(labels)
        if hasattr(self, "drag_container"):
            for card in self.drag_container.pulse_cards():
                card.set_channel_display_labels(labels)

    def _handle_names_changed(self) -> None:
        if not self._building:
            self._refresh_visible_display_labels()
        self._mark_dirty()
        self._activate_layout_tree()
        QtCore.QTimer.singleShot(0, self._activate_layout_tree)
        QtCore.QTimer.singleShot(0, self._sync_timeline_scrollbar)

    def _sync_timeline_scrollbar(self, *_args) -> None:
        if not hasattr(self, "timeline_hbar"):
            return
        source = self.scroll.horizontalScrollBar()
        self.timeline_hbar.blockSignals(True)
        self.timeline_hbar.setRange(source.minimum(), source.maximum())
        self.timeline_hbar.setPageStep(source.pageStep())
        self.timeline_hbar.setSingleStep(max(1, source.singleStep()))
        self.timeline_hbar.setValue(source.value())
        self.timeline_hbar.blockSignals(False)
        self.timeline_hbar.setVisible(source.maximum() > source.minimum())
        if hasattr(self, "timeline_hbar_spacer"):
            left_width = 0
            for widget in (self.names_panel_holder, self.channel_panel_holder, getattr(self, "left_panel_stub_holder", None)):
                if widget is None or widget.isHidden():
                    continue
                width = widget.width() or widget.sizeHint().width()
                left_width += width
            body_layout = self.dataset_body.layout()
            if body_layout is not None:
                margins = body_layout.contentsMargins()
                left_width += margins.left()
            self.timeline_hbar_spacer.setFixedWidth(max(0, left_width))

    def _drag_container_width(self) -> int:
        widths: list[int] = []
        for item in self.drag_container.items:
            widget = item.widget
            max_width = widget.maximumWidth()
            if 0 < max_width < QtWidgets.QWIDGETSIZE_MAX:
                width = max_width
            else:
                width = widget.sizeHint().width()
            widths.append(max(width, widget.minimumWidth(), widget.width()))
        if not widths:
            return 0
        spacing = max(0, self.drag_container.layout_main.spacing())
        margins = self.drag_container.layout_main.contentsMargins()
        return sum(widths) + spacing * (len(widths) - 1) + margins.left() + margins.right()

    def _activate_layout_tree(self) -> None:
        for widget in (self, self.dataset_body, self.drag_container):
            layout = widget.layout()
            if layout is not None:
                layout.activate()
            widget.updateGeometry()
            widget.update()
        for widget in (self.dataset_scroll, self.scroll, self.scroll.viewport()):
            widget.updateGeometry()
            widget.update()
        if hasattr(self, "timeline_hbar"):
            self._sync_timeline_scrollbar()
        window = self.window()
        if window is not self:
            layout = window.layout()
            if layout is not None:
                layout.activate()
            window.updateGeometry()
            window.update()

    def read_state(self) -> PulseTableState:
        # The tick step is fixed by the FPGA clock (shown read-only in the panel), so
        # it always comes from the current state -- never an editable field.
        time_step_ns = float(self.state.time_step_ns)
        slots = self.state._reference_slots()
        cards = self.drag_container.pulse_cards()
        periods = [
            card.to_period(
                full_channels=self.state.port_catalog.raw_lanes,
                time_step_ns=time_step_ns,
                slots=slots,
            )
            for card in cards
        ]
        scan_slots = self._reconcile_scan_slots(periods)
        api_slots = self._carry_api_slots(periods)
        n_slots = len(scan_slots)
        # Pad a short row (e.g. after binding a NEW slot) with that slot's NOMINAL (reference)
        # value, not 0 -- so a freshly-added scan dimension starts at the field's current value
        # rather than silently forcing a 0 ns duration / 0 DAC code.
        slot_defaults = [float(slot.nominal) for slot in self.state.scan_slots]
        scan_table = [list(row)[:n_slots] + slot_defaults[len(row):n_slots] for row in self.state.scan_table]
        analog_bus_modes: dict[str, list[dict[str, object]]] = {}
        for card in cards:
            for bus_name, entry in card.bus_modes().items():
                analog_bus_modes.setdefault(bus_name, []).append(dict(entry))
        state = PulseTableState(
            port_catalog=self.state.port_catalog,
            visible_ports=self.state.visible_ports,
            periods=periods,
            name=self.name_edit.text().strip() or self.state.name or _default_pulse_name(),
            scan_slots=scan_slots,
            scan_table=scan_table,
            # The Scan-tab editor's SOURCE code rides on the state so Save/Load round-trips the
            # editable program, not just the frozen scan_table numbers.  Persist only GENUINELY
            # USER-EDITED code: while the editor still holds the auto-generated column_stack default
            # (== _scan_auto_code, a DERIVED template) persist "" (codeless) so a reload regenerates
            # it for the THEN-current slot count -- else a stale N-slot default would ride the state
            # and defeat _refresh_scan_tab's slot-count auto-adapt.  Falls back to the current state's
            # value for a headless read_state before _build_scan_tab wires the editor.
            scan_code=(("" if self.scan_code.toPlainText() == getattr(self, "_scan_auto_code", None)
                        else self.scan_code.toPlainText())
                       if getattr(self, "scan_code", None) is not None
                       else getattr(self.state, "scan_code", "")),
            api_slots=api_slots,
            time_step_ns=time_step_ns,
            analog_bus_modes=analog_bus_modes or dict(self.state.analog_bus_modes),
            delays=dict(self.state.delays),
            delay_units=dict(self.state.delay_units),
            repeat_forever=bool(self.state.repeat_forever),
            # Whole-sweep count (#3): 0 = sweep forever (default), K = K sweeps then stop.  Carried
            # on the state so Save round-trips it and prepare(state) hands it to the device (which
            # does the finite stop).  Falls back to the current state's value if the Scan tab's spin
            # was never built (e.g. a headless read_state before _build_scan_tab).
            scan_repeats=(int(self.scan_repeats_spin.value())
                          if getattr(self, "scan_repeats_spin", None) is not None
                          else int(getattr(self.state, "scan_repeats", 0))),
        )
        self.names_panel.read_values(state)
        self.channel_panel.read_values(state)
        state.apply_analog_bus_modes_to_period_states()
        start, end, repeat = self._read_bracket()
        state.repeat_start = start
        state.repeat_end = end
        state.repeat_count = repeat
        state.validate()
        self.state = state
        return state

    def _reconcile_scan_slots(self, periods: Sequence[PulsePeriod]) -> list[ScanSlot]:
        """Carry scan slots through an edit, realigning ``duration`` AND ``dac`` targets.

        Slots are owned by ``self.state`` (created/removed only by the scan dots).
        Drag-reordering moves a period's ``s{i}`` expression to a new index, so we
        re-point each slot's target at the period/card that now holds it:
        ``duration`` slots by the period whose duration is ``s{i}``; ``dac`` slots
        (``bus@period_index``) by the card whose bus value is ``s{i}``.  Without the
        ``dac`` remap a DAC scan + period drag would leave the slot target pointing at
        the old period index (stale compile/unbind/highlight).
        """

        var_to_period: dict[int, int] = {}
        for period_index, period in enumerate(periods):
            slot_index = _slot_index_of_expr(period.duration)
            if slot_index is not None:
                var_to_period[slot_index] = period_index
        var_to_dac_target: dict[int, str] = {}
        for period_index, card in enumerate(self.drag_container.pulse_cards()):
            for bus_name, entry in card.bus_modes().items():
                slot_index = _slot_index_of_expr(entry.get("value"))
                if slot_index is not None:
                    var_to_dac_target[slot_index] = f"{bus_name}@{period_index}"
        out: list[ScanSlot] = []
        for index, slot in enumerate(self.state.scan_slots):
            if slot.kind == "duration":
                period_index = var_to_period.get(index)
                target = str(period_index) if period_index is not None else slot.target
                out.append(ScanSlot("duration", target, slot.label, slot.unit,
                                    slot.nominal, slot.name))
            elif slot.kind == "dac":
                target = var_to_dac_target.get(index, slot.target)
                out.append(ScanSlot("dac", target, slot.label, slot.unit,
                                    slot.nominal, slot.name))
            else:
                out.append(slot)
        return out

    def _carry_api_slots(self, periods: Sequence[PulsePeriod]):
        """Carry API slots (owned by ``self.state``, created/removed only by the dots) across
        an edit.  An API slot does NOT rewrite the field, so -- unlike a scan slot -- there is
        no ``aN`` marker in the cell to re-point it after a period drag; it is carried by index.
        Slots whose target no longer exists (its period was deleted) are dropped so a stale
        handle never trips ``validate``."""
        n = len(periods)
        out = []
        for slot in self.state.api_slots:
            if slot.kind == "duration":
                if str(slot.target).lstrip("-").isdigit() and 0 <= int(slot.target) < n:
                    out.append(slot)
            elif slot.kind == "delay":
                # A delay target is a channel OR a DAC bus (a bus owns one delay fanned out to its
                # members) -- the SAME rule the validator uses.  Checking only ``channels`` dropped
                # every bus-delay api slot on each read_state rebuild, so the dot could never toggle
                # OFF (each click saw "no slot" -> re-bound) and only one bus could ever hold a slot.
                if self.state.is_delay_target(slot.target):
                    out.append(slot)
            elif slot.kind == "dac":
                _, _, period = str(slot.target).partition("@")
                if period.lstrip("-").isdigit() and 0 <= int(period) < n:
                    out.append(slot)
        return out

    def _remember_scan_column(self, state: PulseTableState, kind: str, target: str, slot_index: int) -> None:
        """Stash a field's scan-table column before it is unbound.

        So that toggling a scan dot OFF and back ON restores the values the user
        typed, instead of resetting the column to the field's nominal.
        """

        cache = getattr(self, "_scan_col_cache", None)
        if cache is None:
            cache = self._scan_col_cache = {}
        cache[(kind, str(target))] = [
            float(row[slot_index]) for row in state.scan_table if slot_index < len(row)
        ]

    def _restore_scan_column(self, state: PulseTableState, kind: str, target: str, slot_index: int) -> None:
        cache = getattr(self, "_scan_col_cache", None)
        values = (cache or {}).get((kind, str(target)))
        if not values:
            return
        for row_index, row in enumerate(state.scan_table):
            if slot_index < len(row) and row_index < len(values):
                row[slot_index] = float(values[row_index])

    def _remember_field_state(self, state: PulseTableState, kind: str, target: str) -> None:
        """Snapshot a field's full pre-bind state (mode/value/unit).

        Binding rewrites the field to ``s{i}`` and a plain ``unbind`` only knows
        how to reset it to a hard default (duration -> 1000 ns, DAC -> hold).
        Stashing the original here lets us put the field back EXACTLY as it was
        -- e.g. a DAC that was "edge / 500" returns to "edge / 500", not "hold".
        """

        cache = getattr(self, "_field_state_cache", None)
        if cache is None:
            cache = self._field_state_cache = {}
        key = (kind, str(target))
        try:
            if kind == "duration":
                period = state.periods[int(target)]
                cache[key] = ("duration", period.duration, period.unit)
            elif kind == "dac":
                bus, _, period_str = str(target).rpartition("@")
                period_index = int(period_str)
                plan = state.analog_bus_plan(bus)
                entry = dict(plan[period_index]) if period_index < len(plan) else {"mode": "hold", "value": None}
                cache[key] = ("dac", bus, period_index, entry)
        except Exception:
            cache.pop(key, None)

    def _restore_field_state(self, state: PulseTableState, kind: str, target: str) -> None:
        cache = getattr(self, "_field_state_cache", None)
        saved = (cache or {}).get((kind, str(target)))
        if not saved:
            return
        try:
            if saved[0] == "duration":
                _, duration, unit = saved
                period = state.periods[int(target)]
                state.periods[int(target)] = PulsePeriod(duration, period.states, unit=unit, name=period.name)
            elif saved[0] == "dac":
                _, bus, period_index, entry = saved
                plan = state.analog_bus_plan(bus)
                if period_index < len(plan):
                    plan[period_index] = dict(entry)
                    state.analog_bus_modes[bus] = plan
                    state.apply_analog_bus_modes_to_period_states()
            state.validate()
        except Exception as exc:
            # Restoring is best-effort (the field may have moved/changed since it was cached),
            # but don't SILENTLY swallow it -- surface a short note so a real state-sync error
            # (stale DAC target after a reorder, missing bus, invalid slot) is visible.
            self._message(f"could not restore {kind} field {target!r} after unbind: {exc}")


    def _cycle_field_slot(self, state: PulseTableState, kind: str, target: str, *, unit: str = "ns", label: str = "") -> None:
        """Cycle a field's slot on each dot click: none -> SCAN (sN) -> API (aN) -> none.

        SCAN binds a scan slot (the value moves to the scan table, the field shows ``sN``,
        orange).  API binds a NAMED HANDLE (``aN``) the API/Task sets by name -- the field
        KEEPS its number, stays editable, violet marker.  A delay cell is not scannable, so
        it cycles none -> API -> none (skips the scan step)."""
        scannable = kind in SCAN_SLOT_KINDS
        scan_index = state.slot_index_for(kind, target) if scannable else None
        api_name = state.api_slot_for(kind, target)
        if api_name is not None:
            state.unbind_api_field(kind, target)                       # API -> none
        elif scan_index is not None:
            self._remember_scan_column(state, kind, target, scan_index)
            state.unbind_slot(scan_index)
            self._restore_field_state(state, kind, target)            # SCAN -> API: restore value...
            state.bind_api_field(kind, target, unit=unit)             # ...then tag the handle
        elif scannable:
            self._remember_field_state(state, kind, target)           # none -> SCAN
            # ``sN`` is only the compiler reference.  The public scan axis is a
            # stable semantic identifier (normally the DAC port key, e.g.
            # ``da_x``); make a deterministic suffix only when two fields of
            # the same port are independently scanned.
            raw_name = (str(target).split("@", 1)[0] if kind == "dac"
                        else (str(label).strip() or f"duration_{target}"))
            base_name = re.sub(r"[^0-9A-Za-z_]+", "_", raw_name).strip("_") or "scan_parameter"
            if base_name[0].isdigit():
                base_name = "scan_" + base_name
            used = {slot.name for slot in state.scan_slots}
            name = base_name
            suffix = 2
            while name in used:
                name = f"{base_name}_{suffix}"
                suffix += 1
            new_index = state.bind_field(kind, target, unit=unit, label=label, name=name)
            self._restore_scan_column(state, kind, target, new_index)
        else:
            state.bind_api_field(kind, target, unit=unit)             # delay: none -> API
        # A dot toggle CHANGES the program that On Pulse would upload -- it must get the
        # same confocal UNSYNCED treatment as any other edit (the fast path below bypasses
        # the widget signals that normally call _mark_dirty).
        self._mark_dirty()
        # A scan-slot toggle changes the slot LAYOUT the last generated table was built against, so that
        # table is now stale -> flag Run dirty (a re-Run is needed for the new binding to take effect).
        self._sync_scan_run_dirty(state)
        # Fast path: a scan toggle keeps the structure, so update the existing
        # widgets in place (milliseconds) instead of a full rebuild (~400 ms with
        # all channels shown).  Fall back to load_state if anything looks off.
        self.state = state
        if self._apply_scan_state_in_place(state):
            self._preview_dirty = True
            self._update_summary()
            if hasattr(self, "scan_tab"):
                self._refresh_scan_tab()
        else:
            self.load_state(state)

    def _toggle_duration_scan(self, card: "PeriodCard") -> None:
        try:
            state = self.read_state()
            index = self.drag_container.pulse_cards().index(card)
            unit = card.unit_combo.currentText()
            self._cycle_field_slot(
                state, "duration", str(index),
                unit="ns" if unit == "str (ns)" else unit,
            )
        except Exception as exc:
            self._message(str(exc))

    def _refresh_bus_displays(self) -> None:
        """A DAC mode/value was committed: recompute every HOLD period's shown value (the
        value carried in from the preceding edge/ramp) in place, so it tracks upstream
        edits.  Cheap, no full rebuild; edge/ramp fields keep the typed target."""

        if getattr(self, "_building", False):
            return
        try:
            state = self.read_state()
            state.apply_analog_bus_modes_to_period_states()
        except Exception:
            return
        for period_index, card in enumerate(self.drag_container.pulse_cards()):
            for bus_name, edit in getattr(card, "bus_value_edits", {}).items():
                combo = card.bus_mode_combos.get(bus_name)
                mode = _bus_mode_value(combo.currentText()) if combo is not None else "hold"
                dot = card.bus_dots.get(bus_name)
                if mode != "hold" or (dot is not None and dot.isChecked()):
                    continue  # only hold (and non-scanned) fields show a carried value
                try:
                    value = int(state.analog_bus_value_at_period_start(period_index, bus_name))
                except Exception:
                    continue
                with _signals_blocked(edit):
                    edit.setText(str(value))

    def _toggle_dac_scan(self, card: "PeriodCard", bus_name: str) -> None:
        try:
            state = self.read_state()
            index = self.drag_container.pulse_cards().index(card)
            target = f"{bus_name}@{index}"
            self._cycle_field_slot(state, "dac", target, unit="value", label=bus_name)
        except Exception as exc:
            self._message(str(exc))

    def _toggle_delay_api(self, target: str) -> None:
        """A delay dot was clicked -> cycle that delay as an API slot (none -> api -> none).
        ``target`` is a channel name OR a bare DAC-bus name (both are valid delay targets --
        a bus fans out to its members).  A delay is not scannable, so ``_cycle_field_slot``
        skips the scan step; the unit is the field's own (a bus reads its first member's)."""
        try:
            state = self.read_state()
            members = state.bus_channels(min_width=1).get(str(target))
            source = members[0] if members else str(target)   # bus -> first member's unit
            self._cycle_field_slot(state, "delay", str(target),
                                   unit=state.delay_units.get(source, "ns"))
        except Exception as exc:
            self._message(str(exc))

    def _load_scan_file(self) -> None:
        try:
            state = self.read_state()
            if not state.scan_slots:
                self._message("Bind at least one field to a scan slot (click a dot) before loading an array.")
                return
            start = str(Path(self.address_str).parent if self.address_str else _pulse_files_dir())
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load scan array", start, "Scan array (*.npy *.csv *.txt *.json)")
            if not path:
                return
            if Path(path).suffix.lower() == ".json":
                # JSON is a typed pulse/program artifact, not a numeric text
                # matrix.  Reuse the Scan-tab importer so both buttons apply
                # the same schema validation and wire-to-user conversion.
                self._ingest_scan_program_file(Path(path))
                return
            loaded = snap_scan_table(
                # pass the slot count so a 1-D array is read as N points x n_slots
                # (n_slots=1 -> a column of points), not 1 point x N slots.
                load_scan_table(path, n_slots=len(state.scan_slots) or None),
                state.scan_slots,
                time_step_ns=state.time_step_ns,
                dac_ranges=state.scan_slot_dac_ranges(),
            )
            self._scan_tables["loaded"] = loaded
            self._scan_loaded_path = path
            self._scan_use_loaded = True
            self._apply_scan_source()
            if hasattr(self, "preview_status"):
                self.preview_status.setText(f"Loaded {len(loaded)} scan points from {Path(path).name}")
        except Exception as exc:
            self._message(str(exc))

    def _open_scan_tab(self) -> None:
        if hasattr(self, "tabs") and hasattr(self, "scan_tab"):
            self.tabs.setCurrentWidget(self.scan_tab)
            self._refresh_scan_tab()

    def _build_scan_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        tab.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(tab)
        margin = _px(8, minimum=5)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(_px(8, minimum=5))

        info = FluentFrame()
        info.setMinimumHeight(_px(64, minimum=52))
        info_layout = QtWidgets.QVBoxLayout(info)
        info_layout.setContentsMargins(_px(12), _px(8), _px(12), _px(8))
        self.scan_slots_label = FluentLabel("")
        self.scan_slots_label.setWordWrap(True)
        info_layout.addWidget(self.scan_slots_label)
        # Run controls row: how many whole sweeps to play (#3) + a live "where is the scan" readout
        # (#4).  ``scan_repeats`` (0 = sweep forever, the default) writes ``state.scan_repeats`` in
        # read_state, so it round-trips through Save and reaches the device via prepare(state) -- the
        # DEVICE does the finite stop (no GUI counting loop).  Reuses the in-file numeric idiom
        # (FluentDoubleSpinBox(length=5, allow_minus=False), as the bracket "Repeat" spin), with 0
        # decimals so it reads as an integer count.
        run_row = QtWidgets.QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(_px(6, minimum=4))
        run_row.addWidget(FluentLabel("Scan repeats (0 = ∞)"))
        self.scan_repeats_spin = FluentDoubleSpinBox(length=5, allow_minus=False)
        self.scan_repeats_spin.setDecimals(0)
        self.scan_repeats_spin.setRange(0, 999)
        self.scan_repeats_spin.setValue(int(getattr(self.state, "scan_repeats", 0)))
        self.scan_repeats_spin.setFixedHeight(_row_height())
        self.scan_repeats_spin.setToolTip(
            "How many WHOLE scan sweeps to play before stopping.  0 = sweep forever (the default); "
            "K ≥ 1 = K full sweeps then stop.  The device performs the finite stop.")
        # A repeats edit is a STATE change (it rides on read_state -> prepare(state)); mark the
        # editor unsynced, the same dirty path the scan-code editor uses.
        self.scan_repeats_spin.valueChanged.connect(self._mark_dirty)
        run_row.addWidget(self.scan_repeats_spin)
        # Stop the running scan and HOLD the current point (#1): reload a single-point program built from
        # exactly the scan point the device is on RIGHT NOW and loop it forever (NOT seamless -- a fresh
        # load of that one point), so the experiment parks on the current setpoint.  Re-Run to resume.
        self.scan_hold_button = FluentButton("Stop ▸ hold point", color=RED)
        self.scan_hold_button.setFixedHeight(_row_height())
        self.scan_hold_button.setToolTip(
            "Stop the running scan and HOLD the CURRENT scan point: reloads a single-point pulse built "
            "from that point's values and loops it forever.  Re-run the Scan to resume sweeping.")
        self.scan_hold_button.clicked.connect(self._stop_scan_to_current_point)
        run_row.addWidget(self.scan_hold_button)
        # Debug stepping through the scan table: walk the HELD point one row back / forward
        # (clamped at the table ends, no wrap).  Pressing step while the sweep is still RUNNING
        # first stops+holds like Stop ▸ hold point, stepping FROM the live point in ONE reload;
        # with no scan table they fall into _hold_at_point's harmless message (same as the hold
        # button).  ORANGE = the in-file utility-action colour (Remove / Load / Sync).
        self.scan_step_back_button = FluentButton("◀ step", color=ORANGE)
        self.scan_step_back_button.setToolTip(
            "Step the HELD scan point one row BACK in the scan table (clamped at point 1, no wrap).  "
            "If the scan is still running, stops and holds first -- like Stop ▸ hold point.")
        self.scan_step_forward_button = FluentButton("step ▶", color=ORANGE)
        self.scan_step_forward_button.setToolTip(
            "Step the HELD scan point one row FORWARD in the scan table (clamped at the last point, no "
            "wrap).  If the scan is still running, stops and holds first -- like Stop ▸ hold point.")
        for step_button, step_delta in ((self.scan_step_back_button, -1), (self.scan_step_forward_button, +1)):
            step_button.setFixedHeight(_row_height())
            step_button.clicked.connect(lambda _checked=False, d=step_delta: self._step_held_scan_point(d))
            run_row.addWidget(step_button)
        # Live scan-position readout (#4): a QTimer polls the managed command port
        # while a scan runs and writes this label (single-source text via _format_scan_progress) PLUS the
        # current point's VALUES (#1 "show the current scan points"); blank when offline/idle.
        self.scan_progress_label = FluentLabel("")
        run_row.addWidget(self.scan_progress_label, 1)
        info_layout.addLayout(run_row)
        layout.addWidget(info)
        # Poll the device for the live scan position; defensive (no sequencer / no scan_progress
        # method -> silently blank, never throws).  Started here, ticking for the tab's lifetime.
        self._scan_progress_timer = QtCore.QTimer(self)
        self._scan_progress_timer.setInterval(200)
        self._scan_progress_timer.timeout.connect(self._poll_scan_progress)
        self._scan_progress_timer.start()

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(_px(8, minimum=5))

        editor_box = FluentGroupBox("Generate the scan table (Python)")
        editor_layout = QtWidgets.QVBoxLayout(editor_box)
        editor_layout.setContentsMargins(_px(8), _px(28, minimum=24), _px(8), _px(8))
        editor_layout.setSpacing(_px(6, minimum=4))
        self.scan_code = FluentCodeEdit()         # shared monospace code editor (Fluent chrome)
        # Top row: load a saved program (.py) or drop in a template that adapts to the
        # currently bound slot count (column_stack is the default starting point).
        template_buttons = QtWidgets.QHBoxLayout()
        template_buttons.setSpacing(_px(6, minimum=4))
        load_prog_btn = FluentButton("Load Program", color=ACCENT)
        load_prog_btn.setToolTip("Load a Python program (.py) into the editor.")
        load_prog_btn.clicked.connect(self._load_scan_program)
        tmpl_cs_btn = FluentButton("Template: column_stack", color=GREY)
        tmpl_cs_btn.setToolTip("Insert the column_stack template (one column per slot), adapted to the bound slots.")
        tmpl_cs_btn.clicked.connect(lambda: self._insert_scan_template("column_stack"))
        tmpl_grid_btn = FluentButton("Template: grid", color=GREY)
        tmpl_grid_btn.setToolTip("Insert the grid template (outer product of two arrays), adapted to the bound slots.")
        tmpl_grid_btn.clicked.connect(lambda: self._insert_scan_template("grid"))
        for button in (load_prog_btn, tmpl_cs_btn, tmpl_grid_btn):
            button.setFixedHeight(_row_height())
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            template_buttons.addWidget(button, 1)
        editor_layout.addLayout(template_buttons)
        editor_layout.addWidget(self.scan_code, 1)
        # Run the code -> generated scan table; Save the resulting array.  Loading an
        # ARRAY file lives in the Delay/Scan panel ("Load Array" + source toggle); a
        # second "Load File" here would just duplicate it.  (Load PROGRAM, above, loads
        # .py code into the editor -- a different thing.)
        code_buttons = QtWidgets.QHBoxLayout()
        code_buttons.setSpacing(_px(6, minimum=4))
        run_btn = FluentButton("Run", color=GREEN)
        run_btn.setFixedHeight(_row_height())
        run_btn.setToolTip("Run the code; assign an N_points x N_slots array to 'scan_table'.")
        run_btn.clicked.connect(self._run_scan_code)
        # Confocal dirty semantics (same convention as 'On Pulse*'): a trailing '*'
        # means the editor code changed but has NOT been re-Run, so the scan table
        # on screen is stale.  Editing the code (typing, template insert, load) marks
        # it dirty; a successful Run clears it.  Reuses FluentButton.set_dirty.
        self.scan_run_button = run_btn
        self.scan_code.textChanged.connect(lambda: self.scan_run_button.set_dirty(True))
        save_btn = FluentButton("Save Array", color=YELLOW)
        save_btn.setFixedHeight(_row_height())
        save_btn.setToolTip("Save the generated scan table to a .npy/.csv file.")
        save_btn.clicked.connect(self._save_scan_array)
        for button in (run_btn, save_btn):
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            code_buttons.addWidget(button, 1)
        editor_layout.addLayout(code_buttons)
        body.addWidget(editor_box, 3)

        preview_box = FluentGroupBox("Scan table")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(_px(8), _px(28, minimum=24), _px(8), _px(8))
        # read-only twin of the code editor on the left (same Fluent chrome, no grey panel)
        self.scan_table_view = FluentCodeEdit(read_only=True)
        preview_layout.setSpacing(_px(6, minimum=4))
        preview_layout.addWidget(self.scan_table_view, 1)
        # Mirror the left column's button-row footprint so the two boxes share an
        # identical bottom edge.  Without this the table view hangs ~one row
        # lower than the code editor, and its grey border reads as a stray
        # "extra grey edge" protruding past the left box.
        scan_table_footer = QtWidgets.QWidget()
        scan_table_footer.setFixedHeight(_row_height())
        scan_table_footer.setStyleSheet("background: transparent;")
        preview_layout.addWidget(scan_table_footer)
        body.addWidget(preview_box, 2)
        layout.addLayout(body, 1)

        # Tracks the last auto-written column_stack default.  While the editor still
        # holds exactly that text (the user has not edited / loaded / picked grid), the
        # default is regenerated to match the current slot count.
        self._scan_auto_code = ""
        return tab

    def _poll_scan_progress(self) -> None:
        """Update the live scan-position label from the managed target (#4).

        Defensive by contract: no command port, or a port without ``scan_progress``
        method, or any error reading it -> the label blanks and the poll never throws (a transient
        device/network blip must not crash the GUI timer).  ``_format_scan_progress`` is the SINGLE
        source of the text (blank when idle).  Only POLLS when the Scan tab is the current view: the
        label lives there, and ``scan_progress`` is a real RPC round-trip on a remote sequencer, so a
        background tab (Edit/Preview) must not keep hitting the network 5x/s for an unseen label."""
        label = getattr(self, "scan_progress_label", None)
        if label is None:
            return
        if not self.isVisible():
            return                                       # hidden (e.g. hide_on_close): no RPC for an unseen window
        if getattr(self, "tabs", None) is not None and self.tabs.currentWidget() is not getattr(self, "scan_tab", None):
            return                                       # Scan tab not visible -> skip the RPC poll
        progress = None
        reader = getattr(self.command_port, "scan_progress", None)
        if callable(reader):
            try:
                progress = reader()
            except Exception:
                progress = None
        # When HELD (Stop ▸ hold point), show the frozen point; otherwise the live position PLUS the
        # current point's VALUES so the running scan point is visible, not just its index (#1).
        held = getattr(self, "_held_scan_point", None)
        if held is not None:
            label.setText(self._held_scan_point_text(held))
        else:
            label.setText(_format_scan_progress(progress) + self._current_scan_point_text(progress))

    @staticmethod
    def _freeze_state_to_scan_point(state, point_index: int):
        """Return a STATIC copy of ``state`` holding exactly scan point ``point_index`` -- the program the
        Stop button reloads to HOLD the current point's pulse, looped forever.  BAKES that point's slot
        values into their targets (period duration / DAC level, waveform mode preserved) and CLEARS the
        scan via ``with_slots_resolved``.  Resolves the held point through the ONE public per-point
        contract ``with_slots_resolved(slot_point(k))`` -- byte-for-byte the call the virtual sequencer
        renders each streamed scan point with (``devices/virtual.py``), so the held preview and the
        streamed hardware point can never diverge.  Testable without the GUI."""
        table = list(getattr(state, "scan_table", None) or [])
        if not table:
            frozen = state.__class__.from_dict(state.to_dict())
            frozen.scan_repeats = 0
            return frozen
        k = max(0, min(int(point_index), len(table) - 1))
        frozen = state.with_slots_resolved(state.slot_point(k))
        frozen.scan_repeats = 0
        return frozen

    def _scan_point_values_text(self, row) -> str:
        # Operator-facing identity is the stable semantic ScanSlot.name (``da_x``, ``t_off``), never
        # the positional compiler token ``sN``.  Strict zip catches a corrupt row/catalog mismatch
        # instead of silently attaching a value to the wrong physical parameter.
        return ", ".join(
            f"{name} = {float(value):g}"
            for name, value in zip(self.state.scan_names, row, strict=True)
        )

    def _current_scan_point_text(self, progress) -> str:
        """The CURRENT scan point's VALUES (#1) at the live position -- looked up in the streaming table."""
        table = list(getattr(self.state, "scan_table", None) or [])
        if not progress or not table:
            return ""
        try:
            k = int(progress.get("point", 0) or 0)
        except Exception:
            return ""
        return ("  " + self._scan_point_values_text(table[k])) if 0 <= k < len(table) else ""

    def _held_scan_point_text(self, held) -> str:
        k, row, total = held
        return f"held at point {int(k) + 1}/{int(total)}: {self._scan_point_values_text(row)}"

    def _live_scan_point_index(self) -> int:
        """The device's CURRENT scan-point index (0-based) from ``scan_progress()`` -- 0 when no
        sequencer / idle / unreadable, the same defensive read the Stop handler always did."""
        try:
            progress = self.command_port.scan_progress() if self.command_port is not None else {}
        except Exception:
            progress = {}
        try:
            return int((progress or {}).get("point", 0) or 0)
        except Exception:
            return 0

    def _hold_at_point(self, index: int) -> None:
        """HOLD scan point ``index`` (clamped into the table): reload a single-point program built from
        exactly that point's values and loop it forever (not seamless).  The SINGLE source of the hold
        mechanics -- Stop ▸ hold point and the ◀ step / step ▶ buttons all land here.  Re-Run to resume."""
        if self.command_port is None:
            self._message("No installation command port is attached.")
            return
        state = self.read_state()
        table = list(getattr(state, "scan_table", None) or [])
        if not table:
            self._message("No scan table to hold -- run a scan first.")
            return
        k = max(0, min(int(index), len(table) - 1))
        # Plain-dict read, NOT getattr: on a test's ``Editor.__new__(Editor)`` instance (no Qt
        # __init__) a missing attribute falls through to sip's wrapper check -> RuntimeError.
        held = self.__dict__.get("_held_scan_point")
        if held is not None and int(held[0]) == k:
            return                               # already holding exactly this point (a step clamped at an end)
        frozen_row = list(table[k])              # the raw row (for the readout); device gets the baked state
        frozen = self._freeze_state_to_scan_point(state, k)
        try:
            # Hold is a semantic run request through the installation command port; the
            # frozen state has no scan table, so it continuously plays exactly one point.
            self.command_port.run(frozen)
            self.stateui_manager.runstate = PulseStateUIManager.RunState.RUNNING
            self._held_scan_point = (k, frozen_row, len(table))
            # No success popup: the held point is shown persistently in the scan-progress label
            # (_held_scan_point_text), so a modal dialog here is just noise the user has to dismiss.
        except Exception as exc:
            self.stateui_manager.runstate = PulseStateUIManager.RunState.ERROR
            self._message(str(exc))

    def _stop_scan_to_current_point(self) -> None:
        """#1 thin shell: stop the running scan and HOLD the point the device is on RIGHT NOW.  The
        hold mechanics live in :meth:`_hold_at_point` (single source, shared with ◀ step / step ▶)."""
        self._hold_at_point(self._live_scan_point_index())

    def _step_held_scan_point(self, delta: int) -> None:
        """◀ step / step ▶ (debug): move the HELD scan point by ``delta`` rows, clamped to the table
        ends (no wrap-around).  Not held yet (the sweep is still running) -> stop+hold like Stop ▸ hold
        point and step FROM the live point, in ONE reload.  No sequencer / no scan table -> the same
        harmless message path as the hold button (via :meth:`_hold_at_point`)."""
        held = self.__dict__.get("_held_scan_point")     # dict read: see _hold_at_point's note
        base = int(held[0]) if held is not None else self._live_scan_point_index()
        self._hold_at_point(base + int(delta))

    def _refresh_scan_tab(self) -> None:
        if not hasattr(self, "scan_slots_label"):
            return
        try:
            state = self.read_state()
        except Exception:
            state = self.state
        if state.scan_slots:
            lines = ["Columns of the scan table (one row = one scan point):"]
            ranges = state.scan_slot_dac_ranges()
            step = float(state.time_step_ns)
            for index, slot in enumerate(state.scan_slots):
                if slot.kind == "dac":
                    rng = ranges[index] if index < len(ranges) and ranges[index] else (-512, 511)
                    allowed = f"signed integer {rng[0]}..{rng[1]} (0 = 0 V)"
                else:
                    allowed = f"snapped to a whole {format_compact_number(step)} ns tick (≥ 1 tick)"
                lines.append(
                    f"  {slot.name} (compiler {slot_var(index)}): "
                    f"{_scan_slot_label(state, index)}  [{slot.unit}]  "
                    f"(nominal {format_compact_number(slot.nominal)}) → {allowed}"
                )
            lines.append("")
            lines.append(
                "Every point is snapped automatically before it runs (durations → whole ticks, "
                "DAC → integer codes in range), so this table is exactly what the hardware plays."
            )
            sections = ["\n".join(lines)]
        else:
            sections = []
        # API slots are listed alongside scan slots: a named handle (a1/a2...) the notebook/API
        # or the Calibrate task sets BY NAME (state.aN = value), with the field's value kept.
        if state.api_names():
            api_lines = ["API slots — set by name from a notebook/API or the Calibrate task:"]
            for name in state.api_names():
                where = []
                for slot in (s for s in state.api_slots if s.name == name):
                    if slot.kind == "duration":
                        where.append(f"period {slot.target} duration")
                    elif slot.kind == "delay":
                        where.append(f"{slot.target} delay")
                    else:
                        bus, _, p = str(slot.target).partition("@")
                        where.append(f"{bus} @ period {p}")
                api_lines.append(f"  {name}: {', '.join(where)}   e.g.  pulse.{name} = <value>")
            sections.append("\n".join(api_lines))
        if sections:
            self.scan_slots_label.setText("\n\n".join(sections))
        else:
            self.scan_slots_label.setText(
                "No scan or API slots bound yet. In the Edit tab, click the dot next to any duration "
                "or DAC value: 1st click = semantic scan parameter (orange; sN is only its compiler "
                "column), 2nd = API handle (aN, violet), 3rd = off. "
                "(A channel delay can be an API slot but is not scannable.)"
            )
        rows = state.scan_table
        if rows:
            header = "   ".join(state.scan_names)
            shown = ["   ".join(format_compact_number(value) for value in row) for row in rows[:40]]
            footer = f"\n... {len(rows)} points total" if len(rows) > 40 else f"\n{len(rows)} point(s)"
            self.scan_table_view.setPlainText(header + "\n" + "\n".join(shown) + footer)
        else:
            self.scan_table_view.setPlainText("(empty — Run code, or Load Array in the Delay/Scan panel)")
        # Default = column_stack template, auto-adapting to the current slot count, as
        # long as the user has not edited it / loaded a program / picked the grid template.
        current = self.scan_code.toPlainText()
        if not current.strip() or current == getattr(self, "_scan_auto_code", ""):
            fresh = _default_scan_code(state)
            was_dirty = self.scan_run_button.is_dirty()
            self.scan_code.setPlainText(fresh)
            self._scan_auto_code = fresh
            # This branch only runs when the user has NEVER edited the code (it is
            # empty or still the previous auto template).  The setPlainText above
            # fires textChanged -> set_dirty(True), which put a spurious star on
            # Run every time the tab was opened -- restore the pre-refresh state.
            self.scan_run_button.set_dirty(was_dirty)

    def _generate_scan_rows(self, state: PulseTableState) -> list[list[float]]:
        """Execute the current scan CODE against ``state``'s slots and return the snapped rows.

        The ONE source of the generated scan table: it runs the editor's scan snippet with
        ``n_slots`` = the state's CURRENT scan-slot count, so the table is always shaped for the
        slots as they are right now.  Pure (no widget mutation)."""
        import numpy as np
        import math as _math

        namespace = {"np": np, "numpy": np, "math": _math, "n_slots": len(state.scan_slots)}
        # SECURITY: this runs the user-entered scan snippet as arbitrary Python. It is a
        # LOCAL experiment tool -- only run code you wrote or trust (a loaded scan .py can do
        # anything Python can). Do not paste/load untrusted scan programs.
        exec(self.scan_code.toPlainText(), namespace)  # noqa: S102 - local experiment tool, trusted input only
        table = namespace.get("scan_table")
        if table is None:
            raise ValueError("Assign an N_points x N_slots array to a 'scan_table' variable.")
        array = np.atleast_2d(np.asarray(table, dtype=float))
        # Snap behind the scenes so what runs is always hardware-legal: durations
        # -> whole ticks (>= 1), DAC -> integer codes clamped to each bus width.
        return snap_scan_table(
            [[float(value) for value in row] for row in array],
            state.scan_slots,
            time_step_ns=state.time_step_ns,
            dac_ranges=state.scan_slot_dac_ranges(),
        )

    def _current_scan_table(self, state: PulseTableState) -> list[list[float]]:
        """The scan table the CURRENT UI defines for ``state``, reconciled to its slots as they are NOW.

        On Pulse must upload the table for the current binding, never a table corrupted by an
        intermediate edit.  The AUTHORITATIVE values live in the active SOURCE cache -- the last
        generated code output (``_scan_tables['generated']``) or the loaded array -- not in
        ``state.scan_table``, which a mid-edit (e.g. a slot briefly unbound to zero columns during a
        move) can pad down to the new slot's nominal and so LOSE the user's values.  Re-snap the
        source rows to the current slots and reconcile each row to the slot count (short rows padded
        with the slot nominal, extra columns dropped) -- the exact rule ``_apply_scan_source`` uses,
        so On Pulse and the Scan-tab preview always agree.  Falls back to whatever ``state`` already
        carries if the source cache is empty (a directly-loaded pulse's embedded table)."""
        n = len(state.scan_slots)
        source = self._scan_tables.get("loaded" if self._scan_use_loaded else "generated") or []
        rows = source if source else [list(row) for row in state.scan_table]
        # Reconcile FIRST, snap SECOND: ``snap_scan_table`` is strict on row width (#C3 -- the
        # experiment-input seams fail loud on a mismatched table), so the documented GUI tolerance
        # for a mid-edit slot move (short cached rows pad with the slot NOMINAL, extra columns
        # drop -- the same rule ``_apply_scan_source`` uses) must run BEFORE the snap sees the
        # rows.  This UI-cache reconcile is the one deliberate tolerance point.
        slot_defaults = [float(slot.nominal) for slot in state.scan_slots]
        rows = [list(row)[:n] + slot_defaults[len(row):n] for row in rows]
        if rows and n:
            rows = snap_scan_table(
                rows, state.scan_slots,
                time_step_ns=state.time_step_ns, dac_ranges=state.scan_slot_dac_ranges(),
            )
        return [list(row) for row in rows]

    @staticmethod
    def _scan_slot_signature(state: PulseTableState) -> tuple:
        """The scan-SLOT layout as an ordered ``((kind, target), ...)`` tuple -- the identity of the slots
        the generated scan table is shaped for (one column per slot, in this order).  Run's dirty star
        keys off this: bind / unbind / move a scan slot and it changes, so a Run generated against the old
        layout is stale (see :meth:`_sync_scan_run_dirty`)."""
        return tuple((slot.kind, slot.target) for slot in state.scan_slots)

    def _sync_scan_run_dirty(self, state: PulseTableState) -> None:
        """Mark the Scan-tab **Run** button dirty (``*``) when the GENERATED scan table no longer matches
        the current scan slots -- so a scan-slot change (a dot bound / unbound / moved) visibly tells the
        user a re-Run is needed for it to take effect.  Only meaningful while the GENERATED source is
        active (the loaded-array source applies immediately, it is not code that needs re-running); and
        only once a table WAS generated (``_scan_generated_slots`` set) -- before the first Run there is
        nothing stale to flag (the empty default template is handled by the code editor's own dirty)."""
        if not hasattr(self, "scan_run_button") or self._scan_use_loaded:
            return
        if self._scan_generated_slots is None:
            return                                        # never generated a table yet -> nothing stale
        if self._scan_slot_signature(state) != self._scan_generated_slots:
            self.scan_run_button.set_dirty(True)

    def _run_scan_code(self) -> None:
        try:
            state = self.read_state()
            if not state.scan_slots:
                self._message("Bind at least one field to a scan slot first (click a dot in the Edit tab).")
                return
            self._scan_tables["generated"] = self._generate_scan_rows(state)
            # Remember the slot layout this table was generated for, so a later slot change re-dirties
            # Run (the table has one column per slot in THIS order).
            self._scan_generated_slots = self._scan_slot_signature(state)
            self._scan_use_loaded = False
            self._apply_scan_source()
            self._open_scan_tab()
            # Successful run -> the on-screen table matches the code AND the current slots: clear the star.
            # (_apply_scan_source -> load_state may re-touch scan_code; clear AFTER it.)
            self.scan_run_button.set_dirty(False)
        except Exception as exc:
            self._message(f"Scan code error: {exc}")

    def _apply_scan_source(self) -> None:
        """Make the active scan table = the selected source (generated or loaded),
        reconciled to the current slot count, then rebuild so every view agrees."""

        state = self.read_state()
        rows = self._scan_tables.get("loaded" if self._scan_use_loaded else "generated", [])
        # An EMPTY source cache (the generator was never Run this session; no file loaded) must NOT
        # wipe the active table -- a freshly LOADED pulse carries its own ``scan_table``, and clobbering
        # it to [] is exactly the "load a pulse and its scan program vanishes" bug (#7). Fall back to the
        # current table, the SAME rule ``_refresh_scan_tab`` uses (the display side), so the apply side
        # and the refresh side can never drift on what "no source yet" means.
        if not rows:
            rows = [list(row) for row in state.scan_table]
        n = len(state.scan_slots)
        # Pad a short row with the slot's NOMINAL (reference) value, not 0 -- same rule as
        # read_state, so a newly-bound slot starts at the field's current value instead of
        # silently forcing a 0 ns duration / 0 DAC code.
        slot_defaults = [float(slot.nominal) for slot in state.scan_slots]
        rows = [list(row)[:n] + slot_defaults[len(row):n] for row in rows]
        state.set_scan_table(rows)
        self.load_state(state)

    def _on_scan_source_toggled(self, use_loaded: bool) -> None:
        self._scan_use_loaded = bool(use_loaded)
        if self._scan_use_loaded and not self._scan_tables.get("loaded"):
            self._message("No file loaded yet — use Load Array first. Using the generated table.")
            self._scan_use_loaded = False
        self._apply_scan_source()
        if hasattr(self, "preview_status"):
            which = "loaded file" if self._scan_use_loaded else "generated code"
            self.preview_status.setText(f"Scan-table source: {which}")
        self._mark_dirty()

    def _load_scan_program(self) -> None:
        try:
            start = str(Path(self.address_str).parent if self.address_str else _pulse_files_dir())
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load scan program / table", start,
                "Scan program or saved table (*.py *.txt *.npy *.csv *.json);;"
                "Python program (*.py *.txt);;"
                "Scan array (*.npy *.csv);;"
                "Saved pulse / program (*.json)",
            )
            if not path:
                return
            self._ingest_scan_program_file(Path(path))
        except Exception as exc:
            self._message(f"Load program error: {exc}")

    def _ingest_scan_program_file(self, path: Path) -> None:
        """Load a scan source in ANY of the formats Save/notebooks produce.

        * ``.py`` / ``.txt`` -- Python that builds ``scan_table`` (goes into the editor);
        * ``.npy`` / ``.csv`` -- an explicitly exported scan array;
        * ``.json``          -- a saved PULSE (its embedded scan table) or a saved compiled
          PROGRAM (``<stem>_program.json``; its wire-domain ``scan_points`` are converted
          back to user units: duration ticks -> ns, DAC offset-binary codes -> signed).

        Arrays/tables land in the loaded-file source (exactly like Load Array) and are
        snapped to the bound slots; Python lands in the code editor."""

        suffix = path.suffix.lower()
        if suffix in ("", ".py", ".txt"):
            self.scan_code.setPlainText(path.read_text(encoding="utf-8"))
            # A loaded program is user content -> stop auto-regenerating the default.
            self._scan_auto_code = ""
            if hasattr(self, "preview_status"):
                self.preview_status.setText(f"Loaded scan program: {path.name}")
            return

        state = self.read_state()
        if not state.scan_slots:
            raise ValueError("bind at least one field to a scan slot (click a dot) before loading a table.")
        if suffix in (".npy", ".csv"):
            table = load_scan_table(path, n_slots=len(state.scan_slots) or None)
        elif suffix == ".json":
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "periods" in payload:        # a saved pulse
                table = [list(row) for row in (payload.get("scan_table") or [])]
                if not table:
                    raise ValueError(f"{path.name} is a saved pulse with no scan table.")
            elif isinstance(payload, dict) and "ticks" in payload:        # a saved compiled program
                points = [list(p) for p in (payload.get("scan_points") or [])]
                if not points:
                    raise ValueError(f"{path.name} is a compiled program with no scan points.")
                kinds = [str(k) for k in (payload.get("slot_kinds") or [])]
                step = float(state.time_step_ns)
                zero_code = 1 << (10 - 1)   # 10-bit DAC wire code, 512 == signed 0 (true 0 V)
                table = []
                for row in points:
                    user_row = []
                    for j, raw in enumerate(row):
                        kind = kinds[j] if j < len(kinds) else "duration"
                        user_row.append(float(raw) * step if kind == "duration" else float(raw) - zero_code)
                    table.append(user_row)
            else:
                raise ValueError(f"{path.name} is neither a saved pulse nor a compiled program.")
        else:
            raise ValueError(f"unsupported scan source {path.suffix!r} (use .py, .npy, .csv or .json).")

        loaded = snap_scan_table(
            table, state.scan_slots,
            time_step_ns=state.time_step_ns,
            dac_ranges=state.scan_slot_dac_ranges(),
        )
        self._scan_tables["loaded"] = loaded
        self._scan_loaded_path = str(path)
        self._scan_use_loaded = True
        self._apply_scan_source()
        if hasattr(self, "preview_status"):
            self.preview_status.setText(f"Loaded {len(loaded)} scan points from {path.name}")

    def _insert_scan_template(self, kind: str) -> None:
        try:
            state = self.read_state()
        except Exception:
            state = self.state
        code = _template_grid(state) if kind == "grid" else _template_column_stack(state)
        self.scan_code.setPlainText(code)
        self._scan_code_initialized = True

    def _save_scan_array(self) -> None:
        try:
            state = self.read_state()
            if not state.scan_table:
                self._message("No scan table to save yet. Run code or load a file first.")
                return
            import numpy as np

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save scan array", str(self._default_scan_path(state)), "Scan array (*.npy *.csv)"
            )
            if not path:
                return
            target = Path(path)
            if target.suffix == "":
                target = target.with_suffix(".npy")
            array = np.asarray(state.scan_table, dtype=float)
            if target.suffix.lower() == ".csv":
                np.savetxt(target, array, delimiter=",")
            else:
                np.save(target, array)
            if hasattr(self, "preview_status"):
                self.preview_status.setText(f"Saved scan array: {target.name}")
        except Exception as exc:
            self._message(str(exc))

    def _default_scan_path(self, state: PulseTableState) -> Path:
        directory = Path(self.address_str).parent if self.address_str else _pulse_files_dir()
        return directory / f"{_safe_file_stem(state.name)}_scan.npy"

    @staticmethod
    def _resize_analog_bus_modes(state: PulseTableState) -> None:
        target = len(state.periods)
        for bus_name in list(state.analog_bus_modes):
            entries = [dict(entry) for entry in state.analog_bus_modes.get(bus_name, [])]
            if len(entries) < target:
                entries.extend({"mode": "hold", "value": None} for _ in range(target - len(entries)))
            elif len(entries) > target:
                entries = entries[:target]
            state.analog_bus_modes[bus_name] = entries

    def _read_bracket(self):
        start = None
        end = None
        repeat = 1
        pulse_seen = 0
        for item in self.drag_container.items:
            if item.item_type == "pulse":
                pulse_seen += 1
            elif item.item_type == "bracket_start":
                start = pulse_seen
            elif item.item_type == "bracket_end":
                end = pulse_seen - 1
                spin = getattr(item.widget, "repeat_spin", None)
                repeat = int(spin.value()) if spin is not None else 2
        if start is None or end is None:
            return None, None, 1
        return start, end, repeat

    # --- period selection (click a card or a gap; Add/Remove act on it) ---------
    def _on_period_card_clicked(self, index: int) -> None:
        if self._selected_period == index:        # click again -> deselect
            self._selected_period = None
        else:
            self._selected_period = index
        self._selected_gap = None
        self.drag_container.show_selection(card=self._selected_period, gap=None)

    def _on_period_gap_clicked(self, pos: int) -> None:
        if self._selected_gap == pos:             # click again -> deselect
            self._selected_gap = None
        else:
            self._selected_gap = pos
        self._selected_period = None
        self.drag_container.show_selection(card=None, gap=self._selected_gap)

    def _clear_period_selection(self) -> None:
        self._selected_period = None
        self._selected_gap = None
        if hasattr(self, "drag_container"):
            self.drag_container.show_selection(card=None, gap=None)

    @staticmethod
    def _shift_slot_targets(state: PulseTableState, *, inserted: int | None = None,
                            removed: int | None = None) -> None:
        """Re-aim scan slots after a period insert/remove at an arbitrary index.

        duration slots target a period index (``target=str(i)``); dac slots target
        ``"<bus>@<i>"``.  A slot bound to a REMOVED period is unbound; slots bound
        past the edit point shift by one.  (ScanSlot is frozen -> rebuild.)"""
        import dataclasses

        for slot_index in reversed(range(len(state.scan_slots))):
            slot = state.scan_slots[slot_index]
            if slot.kind == "duration":
                period = int(slot.target)
            elif slot.kind == "dac":
                period = slot.dac_period
            else:
                continue
            if removed is not None:
                if period == removed:
                    state.unbind_slot(slot_index)
                    continue
                if period > removed:
                    period -= 1
            if inserted is not None and period >= inserted:
                period += 1
            new_target = str(period) if slot.kind == "duration" else f"{slot.dac_bus}@{period}"
            if new_target != slot.target:
                state.scan_slots[slot_index] = dataclasses.replace(slot, target=new_target)

        # API slots target a period the SAME way (duration index / dac "bus@i"; a delay slot
        # targets a CHANNEL, never a period -> untouched).  Shift / drop them identically so a
        # period remove/insert never leaves an api slot pointing at a missing period -- the
        # regression where removing a period of the long-short-long template raised in validate
        # (an a1 slot still bound to the now-deleted image_2).
        kept: list = []
        for slot in state.api_slots:
            if slot.kind == "duration":
                period = int(slot.target) if str(slot.target).lstrip("-").isdigit() else -1
            elif slot.kind == "dac":
                bus, _, p = str(slot.target).partition("@")
                period = int(p) if p.lstrip("-").isdigit() else -1
            else:
                kept.append(slot)            # delay slot: targets a channel, no period shift
                continue
            if removed is not None:
                if period == removed:
                    continue                 # bound to the removed period -> drop the handle
                if period > removed:
                    period -= 1
            if inserted is not None and period >= inserted:
                period += 1
            new_target = str(period) if slot.kind == "duration" else f"{bus}@{period}"
            kept.append(slot if new_target == slot.target else dataclasses.replace(slot, target=new_target))
        state.api_slots = kept

    @staticmethod
    def _edit_analog_bus_modes(state: PulseTableState, *, inserted: int | None = None,
                               removed: int | None = None) -> None:
        """Keep the per-period analog bus plans aligned with a mid-list insert/remove."""
        for bus_name in list(state.analog_bus_modes):
            entries = [dict(entry) for entry in state.analog_bus_modes.get(bus_name, [])]
            if removed is not None and removed < len(entries):
                entries.pop(removed)
            if inserted is not None:
                entries.insert(min(inserted, len(entries)), {"mode": "hold", "value": None})
            state.analog_bus_modes[bus_name] = entries

    def add_period(self) -> None:
        state = self.read_state()
        # insert position: at the selected gap, after the selected card, else append.
        if self._selected_gap is not None:
            insert_at = max(0, min(self._selected_gap, len(state.periods)))
        elif self._selected_period is not None:
            insert_at = min(self._selected_period + 1, len(state.periods))
        else:
            insert_at = len(state.periods)
        state.periods.insert(insert_at, PulsePeriod(
            1_000,
            tuple(0 for _ in state.port_catalog.raw_lanes),
            unit="ns",
            name="",
        ))
        if insert_at < len(state.periods) - 1:    # mid-list: re-aim slots/plans/bracket
            self._shift_slot_targets(state, inserted=insert_at)
            self._edit_analog_bus_modes(state, inserted=insert_at)
            if state.repeat_start is not None and state.repeat_start >= insert_at:
                state.repeat_start += 1
            if state.repeat_end is not None and state.repeat_end >= insert_at:
                state.repeat_end += 1
        self._resize_analog_bus_modes(state)
        state.apply_analog_bus_modes_to_period_states()
        state.validate()
        self._clear_period_selection()
        self.load_state(state)

    def remove_period(self) -> None:
        state = self.read_state()
        if len(state.periods) <= 1:
            # the table must always keep at least one period -- removing the last
            # one would leave nothing to edit or compile.
            self._message("Cannot remove the only period: a pulse needs at least one.")
            return
        # target: the selected card, the card AFTER a selected gap, else the last.
        if self._selected_period is not None:
            target = min(self._selected_period, len(state.periods) - 1)
        elif self._selected_gap is not None:
            target = min(self._selected_gap, len(state.periods) - 1)
        else:
            target = len(state.periods) - 1
        self._shift_slot_targets(state, removed=target)
        state.periods.pop(target)
        self._edit_analog_bus_modes(state, removed=target)
        self._resize_analog_bus_modes(state)
        state.apply_analog_bus_modes_to_period_states()
        if state.repeat_start is not None and state.repeat_end is not None:
            if state.repeat_start > target:
                state.repeat_start -= 1
            if state.repeat_end >= target:
                state.repeat_end -= 1
            state.repeat_end = min(state.repeat_end, len(state.periods) - 1)
            if state.repeat_end <= state.repeat_start:
                state.repeat_start = state.repeat_end = None
                state.repeat_count = 1
        state.validate()
        self._clear_period_selection()
        self.load_state(state)

    def toggle_bracket(self) -> None:
        if self.bracket_exists:
            self.state = self.read_state()
            self.state.repeat_start = self.state.repeat_end = None
            self.state.repeat_count = 1
            self.load_state(self.state)
            return
        if len(self.drag_container.pulse_cards()) < 2:
            self._message("Repeat needs at least two periods.")
            return
        start = RepeatBracket("start")
        end = RepeatBracket("end", 2)
        start.changed.connect(self._mark_dirty)
        end.changed.connect(self._mark_dirty)
        self.drag_container.insert_item(0, start, "bracket_start")
        self.drag_container.insert_item(len(self.drag_container.items), end, "bracket_end")
        self.bracket_exists = True
        self.bracket_button.setText("Del Bracket")
        self._sync_dataset_geometry()
        self._mark_dirty()

    def clear_channel(self, channel: str) -> None:
        try:
            state = self.read_state()
            if _is_bus_key(channel):
                bus = channel.split(":", 1)[1]
                # Clear the LOGICAL bus first: reset its per-period plan to all-hold (value 0),
                # else apply_analog_bus_modes_to_period_states would re-project a stale
                # edge/ramp value back onto the member bits and the "clear" would not stick.
                if bus in state.analog_bus_modes:
                    state.analog_bus_modes[bus] = [{"mode": "hold", "value": None} for _ in state.periods]
                    state.apply_analog_bus_modes_to_period_states()
                for member in state.bus_channels().get(bus, []):
                    state.clear_channel(member)
            else:
                state.clear_channel(channel)
        except Exception as exc:
            self._message(str(exc))
            return
        self.load_state(state)

    def hide_off_ports(self) -> None:
        state = self.read_state()
        active = set(state.period_active_ports())
        programmable = [
            port.key for port in state.port_catalog.ports if port.kind != PORT_CLOCK
        ]
        keepers = [key for key in state.visible_ports if key in active]
        min_visible = min(4, len(programmable))
        for key in programmable:
            if len(keepers) >= min_visible:
                break
            if key not in keepers:
                keepers.append(key)
        if not keepers:
            keepers = programmable[:min_visible]
        keep_set = set(keepers)
        state.visible_ports = [key for key in programmable if key in keep_set]
        state.validate()
        self.load_state(state)

    def show_all_ports(self) -> None:
        state = self.read_state()
        state.visible_ports = [
            port.key for port in state.port_catalog.ports if port.kind != PORT_CLOCK
        ]
        state.validate()
        self.load_state(state)

    def add_selected_port(self) -> None:
        data = self.add_channel_combo.currentData()
        key = str(data if data is not None else "")
        if not key:
            return
        state = self.read_state()
        state.show_port(key)
        self.load_state(state)

    def _refresh_hidden_combo(self) -> None:
        self.add_channel_combo.clear()
        visible = set(self.state.visible_ports)
        for port in self.state.port_catalog.ports:
            if port.kind == PORT_CLOCK or port.key in visible:
                continue
            if port.kind == PORT_DAC:
                display = f"{port.label}  ({port.width} pins)"
            else:
                lane = port.lanes[0]
                raw = self.channel_pins.get(lane, lane)
                display = f"{raw}  ({port.label})" if port.label != lane else raw
            self.add_channel_combo.addItem(display, port.key)
        has_hidden = self.add_channel_combo.count() > 0
        self.add_channel_combo.setEnabled(has_hidden)
        self.add_channel_button.setEnabled(has_hidden)

    @staticmethod
    def _state_key(state: PulseTableState) -> str:
        import json

        return json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))

    def _prepare_to_device(self, *, fire: bool = False):
        # ANY fresh device upload (On Pulse / Prepare / re-running the Scan) supersedes a held scan
        # point -> clear the HELD readout so the label returns to the live position (#1).  The hold
        # path itself does NOT come through here (it uploads via _hold_at_point, which re-marks it).
        self._held_scan_point = None
        state = self.read_state()
        # RE-DERIVE the scan table for the CURRENT slots before every upload, from the active SOURCE
        # (the generated-code output cache / the loaded array) rather than the table read_state carried
        # forward -- which a mid-edit slot move can have padded down to the new slot's nominal.  This is
        # what makes On Pulse order-independent: move a scan slot then On Pulse and the uploaded table
        # still matches the moved slot with the real values, no manual "re-run scan points" step.
        if state.scan_slots:
            state.set_scan_table(self._current_scan_table(state))
        clock_step_ns = self._clock_step_ns(self.target_descriptor)
        if self.command_port is None:
            if clock_step_ns is not None:
                state.to_sequence(time_step_ns=clock_step_ns)
            else:
                state.to_sequence()
            self._applied_state_key = self._state_key(state)
            return None
        if fire:
            program = self.command_port.run(state)
        else:
            program = self.command_port.prepare(state)
        # record what is now APPLIED on the device (the UNSYNCED baseline)
        self._applied_state_key = self._state_key(state)
        return program

    # --- S0.6 ownership bridge.  prepare/run/stop cross one managed semantic
    # command port; the frontend never receives the sequencer or a raw fire verb.
    def prepare(self) -> None:
        RunState = PulseStateUIManager.RunState
        try:
            self.last_program = self._prepare_to_device()
            if self.last_program is None:
                self._message("Offline: sequence validated only.")
            self.stateui_manager.runstate = RunState.PREPARED
        except Exception as exc:
            self.stateui_manager.runstate = RunState.ERROR
            self._message(str(exc))

    def fire(self) -> None:
        RunState = PulseStateUIManager.RunState
        # (the HELD-point clear rides on _prepare_to_device -- the single "fresh upload" seam)
        try:
            # On Pulse = prepare + fire through the PulseController seam in ONE call (controller.on_pulse
            # prepares then fires); the GUI never holds a separate raw fire capability.
            self.last_program = self._prepare_to_device(fire=True)
            if self.command_port is None:
                self._message("Offline: sequence validated only.")
                self.stateui_manager.runstate = RunState.PREPARED
                return
            self.stateui_manager.runstate = RunState.RUNNING
        except Exception as exc:
            self.stateui_manager.runstate = RunState.ERROR
            self._message(str(exc))

    def safe_state(self) -> None:
        RunState = PulseStateUIManager.RunState
        self._held_scan_point = None          # Stop Pulse ends the held single-point loop -> no longer HELD (#1)
        try:
            if self.command_port is not None:
                self.command_port.stop()
            self.stateui_manager.runstate = RunState.SAFE
        except Exception as exc:
            self.stateui_manager.runstate = RunState.ERROR
            self._message(str(exc))

    def sync_from_device(self) -> None:
        """Pull the pulse actually applied to the managed target into the editor.

        The command backend records the SOURCE payload (the PulseTableState
        JSON) of every successful prepare -- whether it came from this GUI or a
        notebook/raw-API call (``PulseController.on_pulse`` etc.).  Sync loads
        that state back into the editor so the GUI again reflects the device."""
        import json

        RunState = PulseStateUIManager.RunState
        if self.command_port is None:
            self._message("Offline: nothing to sync from.")
            return
        try:
            snap = self.command_port.snapshot()
            payload = (snap or {}).get("last_payload_json")
            if not payload:
                self._message("The sequencer has no applied pulse yet (nothing was prepared).")
                return
            data = json.loads(str(payload))
            if not isinstance(data, dict) or "periods" not in data:
                self._message("The applied payload is a raw sequence (not a pulse table); cannot sync it into the editor.")
                return
            state = PulseTableState.from_dict(data)
            self.load_state(state)
            self._applied_state_key = self._state_key(self.read_state())
            remote_state = str((snap or {}).get("state") or "")
            if remote_state == "running":
                self.stateui_manager.runstate = RunState.RUNNING
            elif remote_state == "prepared":
                self.stateui_manager.runstate = RunState.PREPARED
            else:
                self.stateui_manager.runstate = RunState.STOP
            # synced from the device, not from a file: mark unsaved so Save hints.
            self.stateui_manager.filestate = PulseStateUIManager.FileState.UNSAVED
            if hasattr(self, "preview_status"):
                self.preview_status.setText("Synced from device.")
        except Exception as exc:
            self.stateui_manager.runstate = RunState.ERROR
            self._message(str(exc))

    # --- Installation-owned connection observation -----------------------------
    def _on_conn_target_changed(self) -> None:
        self.conn_addr_edit.setEnabled(False)

    def _apply_connection(self) -> None:
        if self.command_port is None:
            self._message("Offline: editing only, no backend calls.")
        else:
            self._message(
                "This target is owned by the installation authority; "
                "change devices through the installation manager."
            )

    def _refresh_connection_label(self) -> None:
        if hasattr(self, "conn_status"):
            self.conn_status.setText(self._connection_label or "Offline (edit only)")

    def _init_connection_ui(self) -> None:
        """Reflect the immutable installation target; never construct a backend."""
        self.conn_target_combo.setEnabled(False)
        self.conn_addr_edit.setEnabled(False)
        self.conn_connect_button.setEnabled(False)
        self.conn_addr_edit.setText("")
        self._refresh_connection_label()

    def save_to_file(self) -> None:
        try:
            state = self.read_state()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save pulse",
                str(self._default_save_path(state)),
                "ZLC pulse (*.json)",
            )
            if path:
                path_obj = Path(path)
                if path_obj.suffix == "":
                    path_obj = path_obj.with_suffix(".json")
                # The editable pulse JSON is the sole persisted source of the scan
                # table.  Preview and compiled program are derived artifacts.
                # Per-artifact failures are reported so a partial save is visible.
                saved: list[str] = []
                failed: list[str] = []
                state.save(path_obj)
                saved.append(path_obj.name)
                try:
                    figure_path = path_obj.with_suffix(".png")
                    self._save_preview_image(state, figure_path)
                    saved.append(figure_path.name)
                except Exception as exc:
                    failed.append(f"preview ({exc})")
                # The compiled, FPGA-ready program is ALWAYS part of the bundle (scan or
                # not): the payload dispatcher picks the scan path when slots + a table
                # are bound and the plain runtime program otherwise -- exactly what On
                # Pulse uploads.
                try:
                    import json

                    from zlc_neutral_atom.timing.runtime_compiler import (
                        compile_runtime_program_for_payload,
                    )

                    program = compile_runtime_program_for_payload(
                        state, port_catalog=state.port_catalog,
                        clock_hz=1e9 / float(state.time_step_ns),
                    )
                    program_path = path_obj.with_name(path_obj.stem + "_program.json")
                    program_path.write_text(json.dumps(program.to_dict(), indent=2), encoding="utf-8")
                    saved.append(program_path.name)
                except Exception as exc:
                    failed.append(f"program ({exc})")
                self.address_str = str(path_obj)
                self._last_save_state = state.to_dict()
                self._last_load_state = None
                self.stateui_manager.address_str = str(path_obj)
                self.stateui_manager.filestate = PulseStateUIManager.FileState.SAVE
                if hasattr(self, "preview_status"):
                    message = f"Saved: {', '.join(saved)}"
                    if failed:
                        message += "  |  skipped: " + "; ".join(failed)
                    self.preview_status.setText(message)
        except Exception as exc:
            self._message(str(exc))

    def load_from_file(self) -> None:
        try:
            start = str(Path(self.address_str).parent if self.address_str else _pulse_files_dir())
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load pulse", start, "ZLC pulse (*.json)")
            if path:
                # Picking the compiled sibling is an easy mistake -- redirect to
                # the editable pulse itself.
                path_obj = Path(path)
                suffix = "_program.json"
                if path_obj.name.endswith(suffix):
                    candidate = path_obj.with_name(path_obj.name[: -len(suffix)] + ".json")
                    if candidate.exists():
                        path_obj = candidate
                        self._message(f"That file is a compiled artifact; loading the pulse {candidate.name} instead.")
                path = str(path_obj)
                state = PulseTableState.load(path)
                # A saved document is aligned topology-to-topology.  Raw lane
                # equality alone cannot prove DAC/clock semantics.
                if state.port_catalog.fingerprint != self.state.port_catalog.fingerprint:
                    try:
                        state = state.aligned_to_catalog(self.state.port_catalog)
                    except ValueError as exc:
                        self._message(str(exc))
                        return
                self.address_str = path
                self._last_load_state = state.to_dict()
                self._last_save_state = None
                # A freshly opened pulse's embedded scan table is authoritative.
                self._scan_tables = {"generated": [list(row) for row in state.scan_table], "loaded": []}
                self._scan_loaded_path = ""
                self._scan_use_loaded = False
                self.stateui_manager.address_str = path
                self.stateui_manager.filestate = PulseStateUIManager.FileState.LOAD
                self.load_state(state)
        except Exception as exc:
            self._message(str(exc))

    def _mark_dirty(self) -> None:
        if self._building:
            return
        self._preview_dirty = True
        # Confocal-style UNSYNCED: an edit while the device is running/prepared means
        # the device no longer matches the editor -> orange status dot + "On Pulse*"
        # star.  Cheap here (no state serialisation); the debounced _update_summary
        # does the accurate compare and restores the run state if the edit brought
        # the state back.
        RunState = PulseStateUIManager.RunState
        if self.stateui_manager.runstate in (RunState.RUNNING, RunState.PREPARED):
            self._unsynced_from = self.stateui_manager.runstate
            self.stateui_manager.runstate = RunState.UNSYNCED
        self._summary_timer.start()
        if hasattr(self, "tabs") and self.tabs.currentWidget() is getattr(self, "preview_tab", None):
            self._preview_timer.start()

    def _update_summary(self) -> None:
        try:
            state = (
                self.read_state()
                if hasattr(self, "channel_panel") and hasattr(self, "drag_container")
                else self.state
            )
            # The pulse COUNT is the only thing we need from the (expensive) full expansion;
            # cache it keyed on the state so a burst of debounced refreshes for the SAME state
            # doesn't rebuild the sequence each time.  Identical display, less compute.
            state_key = self._state_key(state)
            if state_key == getattr(self, "_summary_state_key", None):
                pulse_count = self._summary_pulse_count
            else:
                pulse_count = len(state.to_sequence().pulses)
                self._summary_state_key = state_key
                self._summary_pulse_count = pulse_count
            # accurate UNSYNCED resolution (the cheap _mark_dirty flip was pessimistic):
            # if the editor state matches what is applied on the device again, restore.
            RunState = PulseStateUIManager.RunState
            if (self.stateui_manager.runstate == RunState.UNSYNCED
                    and self._applied_state_key is not None
                    and state_key == self._applied_state_key):
                self.stateui_manager.runstate = self._unsynced_from or RunState.PREPARED
            hidden = _hidden_active_ports(state)
            visible_ports, total_ports = _port_visibility(state)
            total_ns = state.total_duration_ns()
            parts = [
                f"{visible_ports}/{total_ports} ports visible",
                f"{len(state.periods)} periods",
                f"step {state.time_step_ns:g} ns",
                f"{total_ns:.3g} ns",
                f"{pulse_count} pulses",
                _repeat_summary_text(state),
            ]
            if state.scan_slots:
                parts.append(f"scan {len(state.scan_slots)} slots × {len(state.scan_table)} pts")
            if hasattr(self, "channel_panel"):
                self.channel_panel.state = state
                self.channel_panel.set_scan_summary()
            if hidden:
                parts.append(f"hidden active: {', '.join(hidden)}")
            boundary_active = state.repeat_forever_boundary_active_channels()
            if boundary_active:
                labels = [state.label_for(channel) for channel in boundary_active[:4]]
                suffix = "" if len(boundary_active) <= 4 else f", +{len(boundary_active) - 4}"
                parts.append(f"table restart high every {_summary_time_text(total_ns)}: {', '.join(labels)}{suffix}")
            self.summary.setText(" | ".join(parts))
            if hasattr(self, "names_panel"):
                # Clean, auto-scaled units (1e9 ns -> "1 s", 1.5e6 ns -> "1.5 ms", ...) instead
                # of a long raw-ns number.  Tooltip keeps the exact ns for when it matters.
                self.names_panel.total_label.setText(_summary_time_text(total_ns))
                self.names_panel.total_label.setToolTip(f"{total_ns:.9g} ns total (one frame)")
                self.names_panel.periods_label.setText(f"{len(state.periods)}")
                self.names_panel.visible_label.setText(f"{visible_ports}/{total_ports}")
            if hasattr(self, "visible_label"):
                self.visible_label.setText(
                    f"Visible {visible_ports}/{total_ports} ports | Hidden {total_ports - visible_ports}")
            self._update_file_state(state)
        except Exception as exc:
            self.summary.setText(str(exc))
            if hasattr(self, "stateui_manager"):
                self.stateui_manager.filestate = PulseStateUIManager.FileState.UNSAVED

    def _update_file_state(self, state: PulseTableState) -> None:
        if not hasattr(self, "stateui_manager"):
            return
        current = state.to_dict()
        self.stateui_manager.pulse_name = state.name
        self.stateui_manager.address_str = self.address_str
        if not self.address_str:
            self.stateui_manager.filestate = PulseStateUIManager.FileState.UNTITLED
        elif self._last_save_state == current:
            self.stateui_manager.filestate = PulseStateUIManager.FileState.SAVE
        elif self._last_load_state == current:
            self.stateui_manager.filestate = PulseStateUIManager.FileState.LOAD
        else:
            self.stateui_manager.filestate = PulseStateUIManager.FileState.UNSAVED

    def _default_save_path(self, state: PulseTableState) -> Path:
        directory = Path(self.address_str).parent if self.address_str else _pulse_files_dir()
        return directory / f"{_safe_file_stem(state.name)}.json"

    def _default_figure_path(self, state: PulseTableState) -> Path:
        directory = Path(self.address_str).parent if self.address_str else _pulse_files_dir()
        return directory / f"{_safe_file_stem(state.name)}.png"


    def _on_preview_size_picked(self, *_args) -> None:
        """The operator picked a size: PIN it (stop auto-tracking the content) and re-render."""
        self._preview_size_pinned = True
        self._request_preview_refresh()


    def save_figure(self) -> None:
        try:
            state = self.read_state()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save pulse figure",
                str(self._default_figure_path(state)),
                "Pulse figure (*.png)",
            )
            if not path:
                return
            image_path = Path(path)
            if image_path.suffix == "":
                image_path = image_path.with_suffix(".png")
            # Save Figure writes the picture, and only the picture.  The old
            # twin .npz came from a DataFigure.save that no longer exists: data
            # persistence belongs to the run repository, and a preview is not a
            # run -- it is a drawing of the table currently being edited.  The
            # export is the SAME drawing saved at the style's savefig.dpi (600).
            image_path.write_bytes(self.preview_png_bytes(state, export=True))
            self.preview_status.setText(f"Saved figure: {image_path.name}")
        except Exception as exc:
            self._message(str(exc))


    # ------------------------------------------------------------------ preview
    def _preview_channels(self, state: PulseTableState, *, include_always_off: bool,
                          sequence=None) -> list[str]:
        """The digital channels the preview DRAWS -- the ONE source both the snapshot (status line)
        and the timeline render count, so "how many rows" can never drift between the two.

        The universe is the catalog's DIGITAL ports (one lane each), never ``sequence.channels`` --
        the compiled sequence lists only lanes that carry a pulse, so an always-off channel is absent
        there and "Show off rows" could never reveal it.  Clock and DAC lanes are excluded (a clk lane
        is not engine-driven; a DAC owns its own analog row).  ``include_always_off`` shows the whole
        universe; otherwise only channels with a real ON pulse (falling back to the first lane so the
        plot is never empty)."""
        universe = [port.lanes[0] for port in state.port_catalog.digital_ports]
        if include_always_off:
            return universe
        if sequence is None:
            sequence = state.to_sequence(expand_repeat=False)
        active = {
            str(pulse.channel)
            for pulse in (getattr(sequence, "pulses", ()) or ())
            if getattr(pulse, "value", 0) and float(getattr(pulse, "duration", 0.0) or 0.0) > 0.0
        }
        visible = [channel for channel in universe if channel in active]
        return visible or (universe[:1] if universe else [])

    def _preview_snapshot(self, state: PulseTableState, *, include_always_off: bool):
        """The compiled sequence as one dataset: time across, one component per channel.

        The compiler is the single source for what the hardware will do, so the
        preview draws the SAME ``to_sequence`` result the run path uploads rather
        than re-deriving levels from the table.  Samples sit ON the pulse
        boundaries, twice each: a digital line is piecewise constant, and the two
        samples at one boundary are what make the transition draw vertical
        instead of as a ramp.

        Returns ``(snapshot, channels)``.
        """

        import numpy as np
        from zlc_data import (
            AxisId, AxisSpec, BlockId, COMPONENT, ComponentValidity, DataBlock,
            DatasetRevision, DatasetSchema, PointLayout, REPEAT, SCAN_POINT,
            StreamGenerationId, ValidityContract, ValueSchema,
        )
        from zlc_data.value import OwnedSnapshot

        sequence = state.to_sequence(expand_repeat=False)
        pulses = list(getattr(sequence, "pulses", ()) or ())
        channels = self._preview_channels(
            state, include_always_off=include_always_off, sequence=sequence)
        if not channels:
            return None, []

        total = float(getattr(sequence, "duration", 0.0) or 0.0)
        boundaries = {0.0, total}
        for pulse in pulses:
            boundaries.add(float(pulse.start))
            boundaries.add(float(pulse.start) + float(pulse.duration))
        times = sorted(moment for moment in boundaries if 0.0 <= moment <= max(total, 0.0))
        if len(times) < 2:
            times = [0.0, max(total, 1e-9)]
        samples: list[float] = []
        for index, moment in enumerate(times):
            samples.append(moment)
            if index + 1 < len(times):
                samples.append(times[index + 1])

        # The compiler addresses lanes by their raw key (``ch00``); the operator reads them
        # by the board name (``cooling``).  Translate for DISPLAY only -- the dataset the run
        # path uploads still speaks raw keys, so the axis labels are the one place this maps.
        labels = dict(getattr(state.port_catalog, "channel_labels", {}) or {})
        display_names = [labels.get(name, name) for name in channels]

        index_of = {name: position for position, name in enumerate(channels)}
        values = np.zeros((1, len(samples), len(channels)), dtype=np.float64)
        for pulse in pulses:
            position = index_of.get(str(pulse.channel))
            if position is None:
                continue
            start = float(pulse.start)
            stop = start + float(pulse.duration)
            level = float(getattr(pulse, "value", 1) or 0)
            for sample_index, moment in enumerate(samples):
                if start <= moment < stop:
                    values[0, sample_index, position] = level
        # Offset the rows so lines never overlap: the operator has to read WHICH
        # channel, and a shared baseline makes identical lines indistinguishable.
        for position in range(len(channels)):
            values[0, :, position] += float(len(channels) - 1 - position) * 1.5

        repeat = AxisSpec(AxisId("pulse.preview.repeat"), "Repeat", REPEAT, 1, (0,))
        time_axis = AxisSpec(
            AxisId("pulse.preview.time"), "Time", SCAN_POINT, len(samples),
            tuple(value * 1e6 for value in samples), "us")
        channel_axis = AxisSpec(
            AxisId("pulse.preview.channel"), "Channel", COMPONENT,
            len(channels), tuple(display_names))
        schema = DatasetSchema(
            repeat,
            (time_axis,),
            PointLayout.rect_c((len(samples),)),
            ValueSchema(
                (channel_axis,),
                ValidityContract.components(channel_axis.axis_id),
                values.dtype,
                # Rows are stacked for legibility, so the number on the axis is a
                # display level, not a voltage the hardware would produce.
                value_unit="level",
            ),
        )
        block = DataBlock(
            BlockId("pulse-preview-block"),
            DatasetRevision(max(1, int(getattr(self, "_preview_revision", 1)))),
            values,
            ComponentValidity((channel_axis.axis_id,), np.ones(values.shape, dtype=np.bool_)),
            schema,
        )
        # The generation names where this data came from -- the editor's own
        # table, not a run.  A preview has no acquisition lineage and must not
        # claim one; saying so plainly is what keeps the provenance honest.
        return OwnedSnapshot(block.ref(StreamGenerationId("pulse-editor-preview")), block), channels

    def preview_png_bytes(self, state: PulseTableState | None = None, *,
                          include_always_off: bool | None = None,
                          pixel_ratio: float = 1.0,
                          export: bool = False) -> bytes:
        """The preview as PNG bytes -- the ONE place pixels are produced.

        Display, Save Figure and Save Image all go through here, so what is
        written to disk is the same PICTURE that was on screen.  The dpi split
        is the reference's: the on-screen raster carries the caller's screen
        ``pixel_ratio`` (device pixels, blitted 1:1 -- never a soft
        logical-pixel image stretched by Qt), while ``export`` saves the same
        drawing at the style's ``savefig.dpi`` (600), independent of any screen.
        """

        from zlc_frontend.matplotlib_render import render_pulse_timeline_png

        if state is None:
            state = self.read_state()
        if include_always_off is None:
            include_always_off = bool(
                getattr(self, "preview_include_off", None)
                and self.preview_include_off.isChecked())
        return render_pulse_timeline_png(
            **self._preview_render_kwargs(state, include_always_off=include_always_off),
            pixel_ratio=pixel_ratio,
            export=export,
        )

    def _preview_render_kwargs(self, state: PulseTableState, *,
                               include_always_off: bool) -> dict:
        """ONE extraction feeding BOTH preview exits (PNG bytes and the
        interactive raster front), so the picture a person selects on can never
        differ from the picture that is saved.

        The plot layer owns the faithful pulse-timeline render (filled step blocks +
        coloured baselines + repeat brackets); this window is a PURE data source, so
        the frontend never imports the pulse/neutral packages.  The table is drawn AS
        AUTHORED (expand_repeat=False, the reference's preview call): an inner
        bracket reads as a nested square bracket over its own span, never as the
        unrolled copies the hardware plays.
        """

        sequence = state.to_sequence(expand_repeat=False)
        raw_pulses = list(getattr(sequence, "pulses", ()) or ())
        # Rows come from the ONE channel source (the digital-port universe, filtered by "Show off
        # rows"), NOT sequence.channels -- an always-off channel is absent from the compiled sequence,
        # so counting rows off it would make "Show off rows" a no-op.
        channels = self._preview_channels(
            state, include_always_off=include_always_off, sequence=sequence)
        pulses = [
            {"channel": str(pulse.channel), "start": float(pulse.start),
             "stop": float(pulse.stop), "duration": float(pulse.duration),
             "value": bool(pulse.value), "name": str(getattr(pulse, "name", "") or "")}
            for pulse in raw_pulses
        ]
        # The frame length and every bracket span come from ONE prefix sum over the
        # AUTHORED period table (period_start_steps inside _preview_repeat_markers),
        # NEVER from sequence.duration (a trailing all-off period would vanish) and
        # never from the expanded total (the unrolled copies are not drawn).
        markers, total = _preview_repeat_markers(state)
        labels = {channel: state.label_for(channel) for channel in channels}
        # A delayed channel can push its last edge past the frame; the reference
        # stretches the ∞ bracket over that tail so the loop reads as enclosing it.
        seq_end = max([0.0] + [float(pulse.stop) for pulse in raw_pulses])
        if seq_end > 0.0:
            markers = [
                (start, max(stop, seq_end), label) if "∞" in str(label)
                else (start, stop, label)
                for (start, stop, label) in markers
            ]
        traces = self._preview_analog_traces(state, include_always_off=include_always_off)
        regions, segments = self._preview_scan_annotations(state)
        return dict(
            pulses=pulses,
            channels=channels,
            channel_labels=labels,
            total_duration=total,
            title=str(getattr(state, "name", "") or ""),
            repeat_markers=markers,
            repeat_notation=_repeat_summary_text(state),
            size=self._preview_size_for(state, include_always_off=include_always_off),
            analog_traces=traces,
            scan_regions=regions,
            scan_dac_segments=segments,
        )

    def _preview_analog_traces(self, state: PulseTableState, *,
                               include_always_off: bool) -> list[dict]:
        """Each DAC bus folded into ONE plain-data preview row (the reference's flow).

        The plan is resolved through the compiler's own helpers -- scan slots to their
        reference values, ramps to the Bresenham staircase, ``looping=True`` for the
        steady state the endlessly repeating preview converges to -- so the trace shows
        exactly what the hardware would drive.  With "Show off rows" OFF a bus with no
        signal (all HOLD, no scanned value) hides like an always-off TTL channel.
        """

        slots = state._reference_slots()
        step_ns = float(state.time_step_ns)
        starts_steps = state.period_start_steps(slots=slots, time_step_ns=step_ns)
        scanned_buses = {
            str(slot.target).split("@", 1)[0]
            for slot in (state.scan_slots or ()) if slot.kind == "dac"}
        port_labels = {port.key: port.label for port in state.port_catalog.ports}
        traces: list[dict] = []
        for bus_name, members in state.bus_channels().items():
            plan = state._resolved_bus_plan(bus_name, slots)
            has_signal = bus_name in scanned_buses or any(
                str((entry or {}).get("mode", "hold")) != "hold"
                for entry in ((state.analog_bus_modes or {}).get(bus_name) or ()))
            if not include_always_off and not has_signal:
                continue
            lo, hi = bus_signed_range(len(members))
            ticks = _analog_bus_ticks(plan, starts_steps)
            traces.append({
                "name": bus_name,
                "label": port_labels.get(bus_name, bus_name),
                "min": lo,
                "max": hi,
                "starts": [tick * step_ns * 1e-9 for tick in ticks],
                "values": [
                    _analog_bus_value_at_tick(plan, starts_steps, tick, looping=True)
                    for tick in ticks[:-1]
                ],
            })
        return traces

    def _preview_scan_annotations(self, state: PulseTableState) -> tuple[list[dict], list[dict]]:
        """The time spans each scan slot affects, as plain render data.

        A scanned DURATION spans its whole period (every channel feels the boundary
        move); a scanned DAC value spans its period on its own bus row at the slot's
        reference level.  Each slot carries its 1-based number exactly once.
        """

        slots = state._reference_slots()
        step_ns = float(state.time_step_ns)
        starts_steps = state.period_start_steps(slots=slots, time_step_ns=step_ns)
        starts_s = [tick * step_ns * 1e-9 for tick in starts_steps]
        regions: list[dict] = []
        segments: list[dict] = []
        for number, slot in enumerate(state.scan_slots or (), start=1):
            if slot.kind == "duration":
                try:
                    period = int(str(slot.target))
                except (TypeError, ValueError):
                    continue
                if 0 <= period < len(starts_s) - 1:
                    regions.append({"start": starts_s[period],
                                    "stop": starts_s[period + 1],
                                    "number": number})
            elif slot.kind == "dac":
                target = str(slot.target)
                bus_name, _, period_text = target.partition("@")
                try:
                    period = int(period_text)
                except (TypeError, ValueError):
                    continue
                if 0 <= period < len(starts_s) - 1:
                    segments.append({"trace_name": bus_name,
                                     "start": starts_s[period],
                                     "stop": starts_s[period + 1],
                                     "value": float(slot.nominal),
                                     "number": number})
        return regions, segments

    def _preview_size_for(self, state: PulseTableState, *, include_always_off: bool) -> str:
        """The size preset the preview renders at -- the operator's PINNED pick (the Size dropdown once
        they choose one) else the content-derived default (:func:`optimal_pulse_size` over the drawn row
        + period counts).  The ONE size source shared by the on-screen preview, Save Figure and Save
        Image, so all three draw the same picture at the same size."""
        from zlc_frontend.render_style import optimal_pulse_size

        if getattr(self, "_preview_size_pinned", False) and getattr(self, "preview_size_combo", None):
            return self.preview_size_combo.currentText()
        channels = self._preview_channels(state, include_always_off=include_always_off)
        return optimal_pulse_size(len(channels), len(getattr(state, "periods", ()) or ()))

    def refresh_preview(self) -> None:
        """Redraw the Preview tab from the CURRENT table.

        Pixels arrive as bytes, rasterised clear of the Qt objects, so this side
        never owns a canvas whose lifetime it would have to manage.  A program
        that cannot be previewed says why in the status line rather than leaving
        the previous image up -- a stale picture reads as "this is your pulse".
        """

        if not hasattr(self, "preview_body_layout"):
            return
        self._preview_revision = int(getattr(self, "_preview_revision", 0)) + 1
        try:
            state = self.read_state()
            # When the operator has NOT pinned a size, the dropdown TRACKS the content-derived default
            # so it always shows the size the picture is actually drawn at (setCurrentText fires
            # currentTextChanged, not `activated`, so it does not spuriously pin the size).
            include_off_now = bool(getattr(self, "preview_include_off", None)
                                   and self.preview_include_off.isChecked())
            if not getattr(self, "_preview_size_pinned", False) and getattr(self, "preview_size_combo", None):
                effective = self._preview_size_for(state, include_always_off=include_off_now)
                if self.preview_size_combo.currentText() != effective:
                    self.preview_size_combo.setCurrentText(effective)
            logical = self._present_preview(state, include_always_off=include_off_now)
        except Exception as exc:
            self.preview_status.setText(
                f"Preview unavailable: {str(exc).splitlines()[0][:120]}")
            return
        self.preview_placeholder.hide()
        # The scroll area holds the body at its OWN size hint (setWidgetResizable(False)),
        # so a layout stretch cannot grow it: without this the board keeps its minimum
        # and the plot shows as a sliver.  Size the body to the board's LOGICAL size
        # (plus the layout margins) so the whole plot is visible and the scroll bars
        # only appear when it genuinely overflows the viewport.
        margins = self.preview_body_layout.contentsMargins()
        self.preview_body.resize(
            logical[0] + margins.left() + margins.right(),
            logical[1] + margins.top() + margins.bottom())
        self._preview_dirty = False
        # Match the reference wording exactly (C22): "N/M plotted (active channels) | repeat …".
        # N = channels the render drew, M = the DIGITAL-port universe those rows are drawn from (the
        # SAME set _preview_channels counts, so "all channels" reads M/M), and the mode names whether
        # off rows were included -- so the operator reads the same status main shows.
        include_off = bool(getattr(self, "preview_include_off", None)
                           and self.preview_include_off.isChecked())
        _snapshot, drawn = self._preview_snapshot(state, include_always_off=include_off)
        total = len(state.port_catalog.digital_ports)
        mode = "all channels" if include_off else "active channels"
        notation = _repeat_summary_text(state)
        self.preview_status.setText(
            f"{len(drawn)}/{total} plotted ({mode})"
            + (f" | {notation}" if notation else ""))

    def _ensure_preview_host(self):
        """The ONE interactive preview surface: the frontend's reusable
        :class:`SinglePanelHost` (board + gesture binding + identity), built
        once and wired to this editor's two answers -- nothing pulse-specific
        lives below this seam."""

        host = getattr(self, "preview_host", None)
        if host is None:
            from zlc_frontend.qt_widgets import SinglePanelHost

            host = SinglePanelHost("pulse", group="pulse-preview")
            self.preview_host = host
            self.preview_board = host.board
            self.preview_body_layout.addWidget(
                host, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
            host.viewCommitted.connect(self._on_preview_view_committed)
            # A host built after the operator flipped the switch inherits it,
            # exactly like a console panel added while Selectors is ON.
            host.set_selectors_enabled(
                bool(getattr(self, "preview_selectors_switch", None)
                     and self.preview_selectors_switch.isChecked()))
        return host

    def _on_preview_selectors_toggled(self, on: bool) -> None:
        """Preview "Selectors" switch: arm (ON) or park (OFF) the preview
        panel's selector layer in place -- a pure display gate, same semantics
        as the console header's switch."""

        host = getattr(self, "preview_host", None)
        if host is not None:
            host.set_selectors_enabled(bool(on))

    def _present_preview(self, state: PulseTableState, *,
                         include_always_off: bool,
                         display_revision: int | None = None) -> tuple[int, int]:
        """Render the CURRENT table and hand it to the reusable panel host.

        The editor owns only its OWN facts: what to draw
        (:meth:`_preview_render_kwargs`), when the content changed (the state
        fingerprint drives the provenance revision), and the pinned time view.
        The host owns everything generic -- frame identity, present, gesture
        binding, logical-size pinning.  Returns the host's LOGICAL pixel size.
        """

        import hashlib

        from zlc_data import (
            BlockId, DatasetRevision, DatasetRevisionRef, StreamGenerationId)
        from zlc_frontend.figure import DatasetId, EvaluatedInput
        from zlc_frontend.matplotlib_render import render_pulse_timeline_panel

        kwargs = self._preview_render_kwargs(
            state, include_always_off=include_always_off)
        ratio = float(self.devicePixelRatioF() or 1.0)
        fingerprint = hashlib.sha256(
            str(self._state_key(state)).encode("utf-8")).hexdigest()
        if fingerprint != getattr(self, "_preview_content_fp", None):
            self._preview_content_fp = fingerprint
            self._preview_content_rev = int(
                getattr(self, "_preview_content_rev", 0)) + 1
        if display_revision is None:
            display_revision = int(getattr(self, "_preview_display_rev", 0)) + 1
        self._preview_display_rev = int(display_revision)
        provenance = EvaluatedInput(
            DatasetId("pulse.preview"),
            DatasetRevisionRef(
                BlockId("pulse-preview"),
                StreamGenerationId("pulse-editor"),
                fingerprint,
                DatasetRevision(self._preview_content_rev),
            ),
        )
        raster, payload = render_pulse_timeline_panel(
            **kwargs,
            pixel_ratio=ratio,
            evaluated_input=provenance,
            display_revision=self._preview_display_rev,
            x_limits=getattr(self, "_preview_view_x_limits", None),
        )
        host = self._ensure_preview_host()
        return host.present_panel(raster, payload, pixel_ratio=ratio)

    def _on_preview_view_committed(self, candidate) -> None:
        """Answer a wheel-zoom/pan commit: re-render the SAME table at the
        candidate's x limits and present the accepted front under the
        candidate's display revision.  Landing back on the home span clears
        the pin so later refreshes track the full frame again."""

        limits = tuple(float(value) for value in candidate.x_limits)
        home = tuple(float(value) for value in candidate.home_x_limits)
        self._preview_view_x_limits = None if limits == home else limits
        include_off = bool(getattr(self, "preview_include_off", None)
                           and self.preview_include_off.isChecked())
        self._present_preview(
            self.read_state(),
            include_always_off=include_off,
            display_revision=int(candidate.display_revision),
        )

    def _apply_scan_state_in_place(self, state: PulseTableState) -> bool:
        """Whether a scan toggle can be absorbed without rebuilding the cards.

        Always False.  A scan binding changes which fields are slot-bound, and a
        card renders that at build time; the caller already falls back to
        ``load_state``, which is correct.  A fast path that refreshed the wrong
        subset of widgets would show a binding the table does not have, and a
        wrong picture of the sequence costs more than the rebuild does.
        """

        return False

    def _save_preview_image(self, state: PulseTableState, image_path: Path) -> Path:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(self.preview_png_bytes(state, export=True))
        return image_path

    def _on_tab_changed(self, _index: int) -> None:
        current = self.tabs.currentWidget()
        if current is self.preview_tab:
            # ENTERING Preview from another tab is a big context switch: drop any temporary size PIN so the
            # plot re-derives the optimal size for the CURRENT channel / period counts (the pin is only a
            # transient pick that lives WHILE staying on Preview, not across leaving-and-returning).
            self._preview_size_pinned = False
            self.refresh_preview()
        elif current is getattr(self, "scan_tab", None):
            self._refresh_scan_tab()

    def _on_include_off_toggled(self, *_args) -> None:
        """Toggling "show all / off channels" changes how many rows the render draws, so the OPTIMAL size
        changes too.  Drop any temporary size PIN first so the plot re-derives the best size for the new
        visible-channel count, then refresh (the pin is a transient in-Preview pick, not something a
        show-all toggle should keep honouring)."""
        self._preview_size_pinned = False
        self._request_preview_refresh()

    def _request_preview_refresh(self, *_args) -> None:
        self._preview_dirty = True
        if hasattr(self, "tabs") and self.tabs.currentWidget() is getattr(self, "preview_tab", None):
            self._preview_timer.start()



    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        # Preview viewport wheel isolation: a wheel whose position is on the pulse-plot
        # canvas zooms the plot and is CONSUMED -- the preview page must never scroll
        # underneath it.  Wheels off the canvas (margins, placeholder) scroll normally.
        if (event.type() == QtCore.QEvent.Wheel
                and hasattr(self, "preview_scroll")
                and obj is self.preview_scroll.viewport()):
            canvas = getattr(self, "_preview_canvas", None)
            if canvas is not None:
                global_pos = event.globalPos() if hasattr(event, "globalPos") else event.globalPosition().toPoint()
                local = canvas.mapFromGlobal(global_pos)
                if canvas.rect().contains(local):
                    # Re-issue the wheel in CANVAS coordinates: the original event
                    # is viewport-local, and forwarding it unchanged would zoom
                    # about the wrong point (or miss the axes entirely).
                    mapped = QtGui.QWheelEvent(
                        QtCore.QPointF(local), QtCore.QPointF(global_pos),
                        event.pixelDelta(), event.angleDelta(),
                        event.buttons(), event.modifiers(),
                        event.phase(), event.inverted())
                    QtWidgets.QApplication.sendEvent(canvas, mapped)
                    return True          # consumed: the page does not scroll
        return super().eventFilter(obj, event)

    def _message(self, text: str) -> None:
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            if hasattr(self, "summary"):
                self.summary.setText(str(text))
            if hasattr(self, "preview_status"):
                self.preview_status.setText(str(text))
            return
        fluent_message(self, "Pulse", text, kind="warning")

    @staticmethod
    def _settle_qt_events(ms: int = 1000) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        app.processEvents()
        if int(ms) > 0:
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(int(ms), loop.quit)
            loop.exec_()
        app.processEvents()

    def grab_screenshot(self, path: str | Path, *, settle_ms: int = 1000) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._settle_qt_events(settle_ms)
        self.grab().save(str(path))
        return path

    def to_sequence(self):
        """Return the current editor state as a ``PulseSequence``."""

        return self.read_state().to_sequence()

    @staticmethod
    def _clock_step_ns(target_descriptor) -> float | None:
        clock_hz = getattr(target_descriptor, "clock_hz", None)
        if clock_hz is None:
            return None
        clock_hz = float(clock_hz)
        if clock_hz <= 0:
            return None
        return 1e9 / clock_hz

    @staticmethod
    def _resolve_scale(scale: float | None, *, app: QtWidgets.QApplication) -> float:
        # None -> the SHARED automatic rule in qt_widgets (resolve_fluent_auto_scale):
        # every GUI window must agree on the control size for a given screen.
        return set_fluent_scale(scale)


def show_pulse_gui(
    *,
    state: PulseTableState | None = None,
    target_descriptor=None,
    command_port=None,
    channel_pins: Mapping[str, str] | None = None,
    scale: float | None = None,
    window_ratio: float = DEFAULT_WINDOW_RATIO,
    hide_on_close: bool = False,
) -> PulseSequenceEditor:
    ensure_qt_app()          # the editor is a QWidget: the app must exist BEFORE its ctor
    editor = PulseSequenceEditor(
        state=state,
        target_descriptor=target_descriptor,
        command_port=command_port,
        channel_pins=channel_pins,
        scale=scale,
        window_ratio=window_ratio,
    )
    # hide_on_close=True (the session-bound notebook editor): the X HIDES the window so a later
    # exp.pulse_gui() restores the SAME editor (its loaded program + edits) instead of a blank
    # new one.  The standalone .bat keeps the default destroy-on-close.
    # ONE launcher sequence (launch_fluent_window: wrap -> wire -> size -> centre -> show ->
    # retain), shared with every other show_* GUI so the steps cannot drift per-launcher.
    window = launch_fluent_window(
        editor, title="PulseGUI@Zou lab", hide_on_close=hide_on_close,
        # FluentWindow owns the title rendering; propagate it into the editor's own title row.
        wire=lambda _w: editor._set_gui_title(editor.windowTitle()))
    editor._zlc_window = window
    return editor


__all__ = ["PulseSequenceEditor", "show_pulse_gui", "ensure_qt_app"]
