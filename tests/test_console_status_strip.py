"""TaskConsole status-strip behavior."""

from __future__ import annotations

from zlc_frontend.qt_widgets import FluentStatusStrip, ensure_qt_app


def test_status_strip_is_fixed_height_and_rejects_unknown_severity() -> None:
    import pytest

    ensure_qt_app()
    strip = FluentStatusStrip()
    before = strip.height()
    strip.show_message("x" * 4000, severity="warning")

    assert strip.height() == before
    assert strip.text() == "x" * 4000
    assert strip.severity == "warning"
    with pytest.raises(ValueError, match="unknown severity"):
        strip.show_message("bad", severity="foreign")
