"""One owner for every Workbench Setting popup toggle.

`main` gives every Setting button a true toggle: the second click closes the
popup instead of letting the auto-dismiss-then-reopen race turn the button into
a no-op.  Before this owner existed each window hand-copied that debounce, one
window silently lacked it, and one window called the placement helper with a
missing argument so its Setting button raised on click.
"""

from __future__ import annotations

import pytest

from PyQt5 import QtWidgets

from zlc_frontend.qt_widgets import (
    FluentPopup,
    FluentSettingsPopupAnchor,
    ensure_qt_app,
)


@pytest.fixture(scope="module")
def application():
    return ensure_qt_app()


def _anchor(application, *, reopen_debounce_s: float = 0.25):
    host = QtWidgets.QWidget()
    button = QtWidgets.QPushButton("Setting…", host)
    popup = FluentPopup(host)
    content = QtWidgets.QLabel("display", popup)
    layout = QtWidgets.QVBoxLayout(popup)
    layout.addWidget(content)
    owner = FluentSettingsPopupAnchor(
        popup,
        button,
        reopen_debounce_s=reopen_debounce_s,
    )
    return host, button, popup, content, owner


def test_toggle_shows_then_hides_and_prepares_only_before_showing(application):
    host, _button, popup, content, owner = _anchor(application, reopen_debounce_s=0.0)
    prepared = []
    try:
        owner.toggle(content, prepare=lambda: prepared.append("seed"))
        application.processEvents()
        assert popup.isVisible()
        assert prepared == ["seed"]

        owner.toggle(content, prepare=lambda: prepared.append("seed"))
        application.processEvents()
        assert not popup.isVisible()
        # The hide branch must not reseed the editors behind the popup.
        assert prepared == ["seed"]
    finally:
        popup.hide()
        host.deleteLater()


def test_reopen_debounce_swallows_the_click_that_auto_dismissed_the_popup(
    application,
):
    host, _button, popup, content, owner = _anchor(application)
    prepared = []
    try:
        owner.toggle(content, prepare=lambda: prepared.append("seed"))
        application.processEvents()
        assert popup.isVisible()

        # A press outside the popup auto-hides it before the button release
        # arrives; the release must not reopen what the press just dismissed.
        popup.hide()
        application.processEvents()
        owner.toggle(content, prepare=lambda: prepared.append("seed"))
        application.processEvents()
        assert not popup.isVisible()
        assert prepared == ["seed"]
    finally:
        popup.hide()
        host.deleteLater()


def test_disabled_anchor_never_opens(application):
    host, button, popup, content, owner = _anchor(application, reopen_debounce_s=0.0)
    try:
        button.setEnabled(False)
        owner.toggle(content)
        application.processEvents()
        assert not popup.isVisible()
    finally:
        popup.hide()
        host.deleteLater()


def test_custom_present_keeps_shared_toggle_ownership(application):
    host, _button, popup, content, owner = _anchor(
        application,
        reopen_debounce_s=0.0,
    )
    presented = []
    try:
        owner.toggle(
            content,
            prepare=lambda: presented.append("prepared"),
            present=lambda: (presented.append("presented"), popup.show()),
        )
        application.processEvents()
        assert popup.isVisible()
        assert presented == ["prepared", "presented"]
    finally:
        popup.hide()
        host.deleteLater()


def test_rejects_a_non_callable_prepare_and_a_foreign_popup(application):
    host, button, popup, content, owner = _anchor(application, reopen_debounce_s=0.0)
    try:
        with pytest.raises(TypeError):
            owner.toggle(content, prepare=object())
        with pytest.raises(TypeError):
            owner.toggle(content, present=object())
        with pytest.raises(TypeError):
            FluentSettingsPopupAnchor(QtWidgets.QWidget(host), button)
        with pytest.raises(ValueError):
            FluentSettingsPopupAnchor(popup, button, reopen_debounce_s=-1.0)
    finally:
        popup.hide()
        host.deleteLater()


def test_every_workbench_setting_button_opens_through_the_shared_owner():
    """No window may hand-roll the toggle or call the placement helper directly."""

    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    workbenches = (
        root / "Zou_lab_control" / "workbench",
        root / "zlc_workbench",
    )
    offenders = {}
    paths = [
        path
        for workbench in workbenches
        for path in workbench.rglob("*.py")
    ]
    for path in sorted(paths):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        attrs = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }
        found = []
        if "show_fluent_popup_for_anchor" in names:
            found.append("show_fluent_popup_for_anchor")
        if "_settings_dismissed_at" in attrs:
            found.append("_settings_dismissed_at")
        if found:
            offenders[path.relative_to(root).as_posix()] = found
    assert offenders == {}, offenders
