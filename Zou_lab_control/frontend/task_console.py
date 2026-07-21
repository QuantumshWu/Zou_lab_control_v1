"""Task console: a configurable dashboard with a clean VIEW / LOGIC split.

Two permanent tabs:

* **Monitor** -- the drag-and-snap board of PLOT panels.  A plot is a PURE VIEW,
  fully decoupled from acquisition: it shows nothing until you pick a hub signal in
  its Setting, and it NEVER builds, owns or starts a measurement.  Each panel card
  is JUST the plot frame (the frontend canvas + a thin border that doubles as the
  DRAG HANDLE -- grab the border, drop the card, it snaps to the layout grid).
  Everything else (title, size, signal source, fit, limits, save) lives behind the
  gear button / the panel's own Edit tab.

* **Logic** -- the list of LOGIC NODES (measurement / processor / task).  A logic
  node is the thing that PRODUCES data: added STOPPED, it shows as a row with a
  status colour dot (grey=stopped, green=running, red=error) + its name + an Edit
  button.  Its Edit tab carries its auto-generated parameter form + Start / Stop
  (no curve fit -- fitting is a plotter concern).  Start builds the node from
  the form values with display SUPPRESSED (it only publishes signals to the hub --
  never opens a matplotlib plot) and runs it; Stop cancels it.  You then Add plot
  panels on the Monitor board pointed at the signals it publishes.

A plot panel's data source is a one-line expression evaluated against the named
signals of a :class:`~Zou_lab_control.neutral_atom.core.signals.SignalHub`
(the same trusted-local-code posture as the pulse GUI's Scan tab):

    value = frame_0                     # show emCCD event 0 of the camera cycle
    value = occupied - b_occupied       # arbitrary math across signals (two detectors)
    value = history('counts', 200).ravel()

Layouts (plot panels + logic nodes + positions + sizes + expressions + params)
save/load as ONE JSON and are machine-portable: all plot geometry is owned by
frontend.panel_plot and never part of the layout.
"""

from __future__ import annotations

import hashlib
import inspect
import math as _math
import os
import re
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from .live import (
    panel_plot,
    recommended_grid_size,
    region_binding,
)
from zlc_frontend import board_layout as _layout
from zlc_data.console_records import (
    ADDABLE_PANEL_KINDS,
    PANEL_INPUT_FORMAT,
    panel_allows_multi_slot,
    panel_input_slots,
    CONSOLE_STATE_SCHEMA,
    BLANK_SOURCE as _BLANK_SOURCE,
    DEFAULT_UPDATE_MS,
    LOGIC_KINDS,
    LOGIC_NODE_CONFIG_FIELDS as _LOGIC_NODE_CONFIG_FIELDS,
    LogicNodeConfig as _LogicNodeConfig,
    PANEL_INPUT_SLOTS,
    PANEL_KINDS,
    PANEL_SINGLE_SLOT_KINDS,
    PanelConfig as _PanelConfig,
    TASK_CONSOLE_STATE_FIELDS as _TASK_CONSOLE_STATE_FIELDS,
    UPDATE_INTERVALS,
    layout_record,
)
from zlc_frontend.qt_widgets import LogicNodeRow as _LogicNodeRow
from zlc_data.panel_size import PANEL_SIZES, panel_size_cells
from zlc_data.shape_text import slot_label   # the ONE human slot-label formatter (period/channel)
from zlc_frontend.console_state import (
    TASK_FILES_ENV as _TASK_FILES_ENV,
    TaskConsoleState as _TaskConsoleState,
    default_console_state as _default_console_state,
    resolve_task_state as _resolve_task_state,
    task_files_dir as _task_files_dir,
)
# Submodules of the package are reached as ATTRIBUTES of this ONE binding: they are
# deliberately absent from the facade's __all__, and the package guard forbids anyone
# outside importing ``zlc_frontend.qt_widgets.<sub>`` as a path.
import zlc_frontend.qt_widgets as _qt_widgets
from zlc_frontend.qt_widgets import PulseSlotsWidget, SignalExprWidget
from zlc_frontend.qt_widgets import (
    ACCENT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    TEXT,
    WINDOW_SCREEN_FRACTION,
    YELLOW,
    fluent_message,
    FluentButton,
    FluentCodeEdit,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentFrame,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    setting_label_width,
    FluentStatusDot,
    FluentStatusStrip,
    FluentSwitch,
    FluentTabWidget,
    ensure_qt_app,
    launch_fluent_window,
    fluent_widget_stylesheet,
    scaled_px,
    window_pad,
    screen_fit_window_size,
    set_fluent_scale,
    signals_blocked as _signals_blocked,
)

try:  # guarded like pulse_gui: the console degrades without matplotlib-qt
    import matplotlib.pyplot as plt
    from .qt_canvas import panel_canvas
except Exception:  # pragma: no cover - depends on the local matplotlib install
    plt = None
    panel_canvas = None

# Single source for turning a path param into an unambiguous, project-anchored display
# string (an absolute path, never a bare CWD-relative name) -- the same seam the analysis
# layer uses to resolve paths, so the field shows exactly the file/folder that is used.
from zlc_storage.paths import display_path

# The ONE param-kind -> widget registry every form in this module dispatches through: a
# ParamDecl of kind K is built / read / seeded / validated / refreshed by PARAM_WIDGETS[K].
# Adding a kind is one handler there + one whitelist entry on ParamDecl, not 5-7 ladders here.
from .param_widgets import PARAM_WIDGETS, SPAN_KINDS, ParamWidgetContext, RefreshProviders
# The grouped-signal-picker cluster lives in param_widgets (the leaf; see the note above
# strip_node_prefix) -- forward imports only, so the leaf never back-imports this module.
from .param_widgets import (
    coerce_short_labels,
    read_editable_combo,
)

# ParamDecl is the ONE declarative param record both the measurement form and the plot
# panels use (PANEL_PARAMS).  Importing it here (frontend -> neutral_atom is allowed; the
# reverse is not) lets the panel params be real ParamDecls validated by the kind whitelist,
# instead of a parallel ParamSpec class with its own smaller ladder.

# The default mid-run buffer key -- the spec layer's ONE spelling (TaskSpec.mid_run_key's
# default), imported so the console's spec-less fallbacks can never drift from it.
from zlc_data.vocabulary import DEFAULT_MID_RUN_KEY
from zlc_storage import exact_mapping


#: Now :data:`zlc_frontend.console_state.TASK_FILES_ENV`; the old name stays an alias.
TASK_FILES_ENV = _TASK_FILES_ENV

# The console window itself now lives in ``zlc_workbench.task_console.plot_bridge_console``.
# With this move the shell holds ZERO definitions of its own: everything below this file's
# imports is aliases, and every name IS the moved object.
import zlc_workbench.task_console.plot_bridge_console as _plot_bridge_console

TASK_FRAME_KEY = _plot_bridge_console.TASK_FRAME_KEY
MID_RUN_TAG = _plot_bridge_console.MID_RUN_TAG
_StopAttempt = _plot_bridge_console._StopAttempt
TaskConsole = _plot_bridge_console.TaskConsole
show_task_console = _plot_bridge_console.show_task_console
# The panel card, its board, the card-geometry bridge and their constants now live in
# ``zlc_workbench.task_console.plot_bridge`` -- the sanctioned Qt x matplotlib zone (C20):
# a card HOLDS its matplotlib plotter/canvas inside a Qt group box, which neither
# qt_widgets (no matplotlib) nor zlc_data (no toolkit) may hold.  These names ARE the
# moved objects (not copies), so every caller, isinstance check and test keyed on the
# shell path keeps working while the shell is taken apart.
import zlc_workbench.task_console.plot_bridge as _plot_bridge

SIG_VERSIONS_KEY = _plot_bridge.SIG_VERSIONS_KEY
SIG_VALID_KEY = _plot_bridge.SIG_VALID_KEY
COORD_FRAMES_KEY = _plot_bridge.COORD_FRAMES_KEY
_BLANK_SOURCE = _plot_bridge._BLANK_SOURCE
GRID_UNIT = _plot_bridge.GRID_UNIT
GAP = _plot_bridge.GAP
_RELIM_MODES = _plot_bridge._RELIM_MODES
_RELIM_PARAM = _plot_bridge._RELIM_PARAM
REBUILD_DEBOUNCE_MS = _plot_bridge.REBUILD_DEBOUNCE_MS
_MONITOR_UNSET = _plot_bridge._MONITOR_UNSET
_card_y_is_view_axis = _plot_bridge._card_y_is_view_axis
_opaque_white_composite = _plot_bridge._opaque_white_composite
_board_metrics = _plot_bridge._board_metrics
_cell_size = _plot_bridge._cell_size
_card_size = _plot_bridge._card_size
_aabb = _plot_bridge._aabb
_overlaps_with_gap = _plot_bridge._overlaps_with_gap
_first_free_slot = _plot_bridge._first_free_slot
_board_width = _plot_bridge._board_width
pack = _plot_bridge.pack
drop_index = _plot_bridge.drop_index
_unit_df_for = _plot_bridge._unit_df_for
_GridFocus = _plot_bridge._GridFocus
PanelCard = _plot_bridge.PanelCard
_PanelBoard = _plot_bridge._PanelBoard


# WHICH hardware a logic node references is the NODE's own ``referenced_devices()`` declaration.
# Every such node needs runtime authority; ``occupied_devices()`` is only the narrower subset the
# authority turns into EXCLUSIVE claims while observe-only references receive OBSERVE claims.
# There is no second frontend kind-string table and no global "stop everything" rule.

#: Every panel + logic-node name is "<base> #N" with N counting from 1 (G1), so two panels /
#: nodes of the same kind are always told apart -- in the card title, the Edit tab, the frame
#: title, and the signal-flow grouping.  ONE source of that scheme for panels and nodes alike.
# The six pure value/text helpers this module used to define now live with their owners
# (S5-shell(z)); the names below are the shell's plain aliases for them.  ``_GridFocus`` and
# ``_StopAttempt`` deliberately did NOT go: the census calls them render-free because it reads
# IMPORTS, but they HOLD live objects (a grid plotter + its canvas + mpl callback ids; a running
# daemon thread), so their home is wherever those objects live.
from zlc_data.shape_text import indexed_unique_name, strip_node_prefix  # noqa: E402
from zlc_data.param_decl import acquisition_param_decls as _acquisition_param_decls  # noqa: E402
from zlc_frontend.form import (  # noqa: E402
    lenient_float as _safe_float,
    python_to_text as _py_to_text,
    text_to_python as _text_to_py,
)
#: The expression-namespace help text, read straight from its owner.  The shell used to wrap it in
#: a function guarding a module-level cache -- an indirection whose only job was to memoise an
#: imported CONSTANT, which is not a thing that needs memoising.


# The grouped-signal-picker helper cluster (signal_state / grouped_signal_items /
# signal_tree_groups / fill_grouped_signal_combo / read_editable_combo / coerce_short_labels,
# #combo-parity) moved DOWN to param_widgets.py -- the leaf this module already imports -- so the
# leaf's old lazy back-imports of it are gone (no frontend cycle).  strip_node_prefix stays here:
# it is the Logic tab's rule (shared vocabulary, not a picker widget helper).


#: Now :class:`zlc_frontend.qt_widgets.PulseSlotsWidget`; the old name stays a plain
#: alias so nothing that referenced it has to learn a new one.
_PulseSlotsWidget = PulseSlotsWidget

#: Now :class:`zlc_frontend.qt_widgets.SignalExprWidget`; the old name stays a
#: plain alias so nothing that referenced it has to learn a new one.
_SignalExprWidget = SignalExprWidget


# Every Monitor-board panel is a PURE VIEW: it is fully decoupled
# from acquisition -- it shows a hub signal and carries the full plotter Edit (the
# whole DataFigure fit set + manual limits + the display knobs), but it NEVER
# builds / owns / starts a node.  The things that PRODUCE data
# (measurement / processor / task) are LOGIC NODES on the separate Logic tab, not
# board panels (see LogicNodeConfig / LogicNodeRow).

# The KINDS of LOGIC NODE the Logic tab hosts -- the node layers (measurement /
# processor / task).  "camera" is the continuous camera Measurement (a live frame
# stream); "measurement" a swept measurement; "processor" a reactive transform;
# "task" a one-shot orchestration.  A node is added STOPPED and Start/Stop'd from
# its own Edit -- it only ever publishes to the hub (display suppressed).

# The panel-param CATALOG and its four resolvers now live in ``zlc_frontend.panel_params``
# (S5-shell(y)); the names below are the shell's plain aliases for them.
from zlc_frontend.panel_params import (  # noqa: E402
    CMAPS,
    PANEL_PARAMS,
    GRID_TITLE_PARAMS,
    panel_display_decls as _panel_display_decls,
    panel_param_default as _panel_param_default,
    resolved_cmap as _resolved_cmap,
    resolved_param as _resolved_param,
)






# The per-panel Analysis controls now live in ``zlc_frontend.qt_widgets.analysis_controls``.
# These names ARE the moved objects (not copies), so every caller, isinstance check and test
# keyed on the shell path keeps working while the shell is taken apart.
_general_fit_models_for_kind = _qt_widgets.analysis_controls._general_fit_models_for_kind
_FitFixSeedEditor = _qt_widgets.analysis_controls._FitFixSeedEditor
_apply_analysis_state_to_widgets = _qt_widgets.analysis_controls._apply_analysis_state_to_widgets
AnalysisControls = _qt_widgets.analysis_controls.AnalysisControls















#: The placement-only stand-in now lives with the packer it serves.
_GeomProxy = _layout.GeomProxy














#: Image containers the Edit-tab Save offers, in menu order (first = default).  A DATA-layer choice of
#: the output FILE FORMAT (not an art knob): the figure's geometry / dpi / typography are unchanged; only
#: the container the ``DataFigure.save(image_ext=...)`` writes changes.  Lowercase to match matplotlib's
#: ``savefig`` extensions and confocal's ``save_type`` naming (jpg / png).  The matching ``.npz`` data
#: file is format-independent -- identical arrays + ``info`` for every choice.
SAVE_IMAGE_FORMATS: tuple[str, ...] = ("png", "pdf", "jpg")





# ====================================================================== state


#: The ONE console-record validator now lives in :mod:`zlc_data.console_records`;
#: panel, logic-node and console-state records all read it from there.
_layout_record = layout_record


#: Both console records now live in :mod:`zlc_data.console_records`; the old names
#: stay as plain aliases, so every existing caller and export keeps working.
PanelConfig = _PanelConfig
LogicNodeConfig = _LogicNodeConfig


#: The saved-layout record, its JSON file and the name->layout resolver now live in
#: :mod:`zlc_frontend.console_state` (L315 keeps a domain-schema reader out of storage;
#: L327 lets the frontend depend on storage).  The old names stay as plain aliases.
TaskConsoleState = _TaskConsoleState
default_console_state = _default_console_state
resolve_task_state = _resolve_task_state









# The three editors left the shell: the pure-Qt ones (MeasurementPanel, LogicNodeEditor)
# into zlc_frontend.qt_widgets one-per-file, the figure-holding Edit tab (PanelEditor) into
# the plot_bridge zone.  These names ARE the moved objects.
import zlc_workbench.task_console.plot_bridge_editor as _plot_bridge_editor

MeasurementPanel = _qt_widgets.measurement_panel.MeasurementPanel
LogicNodeEditor = _qt_widgets.logic_node_editor.LogicNodeEditor
PanelEditor = _plot_bridge_editor.PanelEditor






# ====================================================================== logic tab
#: Now :class:`zlc_frontend.qt_widgets.LogicNodeRow`; old name kept as a plain alias.
LogicNodeRow = _LogicNodeRow










__all__ = [
    "LogicNodeConfig",
    "PanelConfig",
    "TaskConsole",
    "TaskConsoleState",
    "default_console_state",
    "show_task_console",
]
