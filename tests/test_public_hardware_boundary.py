"""Mechanical public-object boundary for installation-owned hardware.

The legacy device classes this file used to enumerate died with the legacy backend
(directive 2026-07-21); what remains is the live claim: a public session hands out
catalog VALUES, never raw device objects.
"""

from __future__ import annotations

from Zou_lab_control.api import connect



def test_public_session_exposes_catalog_values_not_raw_devices(tmp_path):
    exp = connect("virtual", repository=tmp_path / "workspace")
    try:
        assert not hasattr(exp, "devices")
        assert exp.device_catalog.roles()
        for forbidden in ("camera", "sequencer", "trap_array", "devices"):
            assert not hasattr(exp.device_catalog, forbidden)
        assert not hasattr(exp, "camera")
        assert not hasattr(exp, "sequencer")
    finally:
        exp.close()
