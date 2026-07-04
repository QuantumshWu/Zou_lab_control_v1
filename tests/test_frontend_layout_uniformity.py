"""Contract: the WHOLE frontend uses ONE row/label logic (no three competing styles).

Every settings / Edit / param form is built from exactly two primitives:
  * ``FluentSectionLabel``  -- bold, dark, own-line GROUP header,
  * ``FluentSettingRow``    -- grey fixed-width label | control,
with the section/row hierarchy shown by weight+colour ONLY (never indentation, never a third
colour) and the column width from the shared ``setting_label_width``.  This guards against the
regressions the user kept hitting: a ``QFormLayout`` (dark auto-width labels), a bare bold
``FluentLabel`` masquerading as a section, or an ad-hoc indented sub-form -- i.e. a fourth/fifth
style creeping back in.  A design rule that CAN be a test MUST be one (AGENTS rule 7 lesson).
"""

import pytest


def _editors(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from Zou_lab_control.frontend import devtools as dt
    from Zou_lab_control.frontend.task_console import LogicNodeConfig

    console = dt.demo_console(shots=3)
    # a measurement (pulse-scan) node Edit -- the form with signal_expr + pulse_slots composites
    row = console._add_logic_node(LogicNodeConfig(kind="measurement", name="Pulse scan"))
    console._edit_logic_node(row)
    meas_editor = console._logic_editors[id(row)]
    # a plot panel Edit -- the Display/Panel/Processing sections
    card = next(c for c in console.cards if c.config.kind == "1d")
    console._edit_card(card)
    plot_editor = console._panel_editors[id(card)]
    return console, meas_editor, plot_editor, QtWidgets


def test_param_forms_use_one_row_label_system(monkeypatch):
    pytest.importorskip("PyQt5")
    from Zou_lab_control.frontend.qt_fluent import FluentSectionLabel, FluentSettingRow, FluentLabel

    console, meas_editor, plot_editor, QtWidgets = _editors(monkeypatch)
    try:
        for name, editor in (("measurement", meas_editor), ("plot", plot_editor)):
            # 1) NO QFormLayout anywhere (its labels render dark + auto-width = a different style)
            assert not editor.findChildren(QtWidgets.QFormLayout), \
                f"{name} Edit uses a QFormLayout -- forms must be VBox of FluentSettingRow rows"
            # 2) the two canonical primitives ARE present (sections + rows)
            assert editor.findChildren(FluentSectionLabel), f"{name} Edit has no FluentSectionLabel header"
            assert editor.findChildren(FluentSettingRow), f"{name} Edit has no FluentSettingRow row"
            # 3) NO bare FluentLabel used as a bold section header (the old grey-bold section() style).
            #    Prose captions are allowed (normal weight); a BOLD FluentLabel = a rogue section style.
            for lab in editor.findChildren(FluentLabel):
                if not isinstance(lab, FluentSectionLabel) and lab.font().bold():
                    raise AssertionError(
                        f"{name} Edit has a bold FluentLabel {lab.text()!r} acting as a section -- "
                        "use FluentSectionLabel (the ONE section style)")
    finally:
        console.shutdown()


def test_section_labels_are_bold_rows_are_grey(monkeypatch):
    """The hierarchy is weight+colour: every FluentSectionLabel is bold; every FluentSettingRow's
    label is the grey non-bold column.  (Confirms the two roles are visually distinct + uniform.)"""
    pytest.importorskip("PyQt5")
    from Zou_lab_control.frontend.qt_fluent import FluentSectionLabel, FluentSettingRow

    console, meas_editor, plot_editor, _Qt = _editors(monkeypatch)
    try:
        for editor in (meas_editor, plot_editor):
            for sec in editor.findChildren(FluentSectionLabel):
                assert sec.font().bold(), f"section header {sec.text()!r} must be bold"
            for roww in editor.findChildren(FluentSettingRow):
                lbl = getattr(roww, "_label", None)
                assert lbl is not None and not lbl.font().bold(), \
                    "a FluentSettingRow label must be the non-bold grey column (not bold like a section)"
    finally:
        console.shutdown()
