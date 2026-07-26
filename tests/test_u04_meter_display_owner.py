"""METER's display state belongs to zlc_frontend, beside the other three.

The four dataset-evaluated intents need display state.  IMAGE, CURVE and
HISTOGRAM each owned a module under `zlc_frontend/`; METER's state was a private class inside one Qt
window (`Zou_lab_control/workbench/_figure.py`).  A board could therefore render
a METER panel but had nowhere to keep its display state, and any second consumer
would have had to import a GUI module to get one - the exact reverse of the
package direction this migration exists to establish.

There is deliberately no `meter_display_form_spec`: a METER has no authored
display parameter yet, and an empty form would put an empty tab in every Setting
popup.  This test pins that absence as a decision rather than an oversight.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import zlc_frontend
from zlc_frontend import MeterDisplayState
from zlc_frontend.figure import (
    DATASET_VIEW_INTENTS,
    DocumentViewContract,
    PULSE_CONTRACT,
    VIEW_CONTRACTS,
    ViewContract,
    contract_for,
)
from zlc_frontend.figure.model import ViewIntent


ROOT = Path(__file__).resolve().parents[1]


def test_the_state_is_owned_by_the_frontend_package():
    assert inspect.getmodule(MeterDisplayState).__name__ == "zlc_frontend.meter_display"
    assert "MeterDisplayState" in zlc_frontend.__all__


def test_every_dataset_evaluated_intent_has_display_state_in_one_package():
    """No dataset-evaluated intent may live in a GUI module.

    PULSE is document-fed, so this dataset-state table must not absorb it.
    """

    owners = {
        ViewIntent.IMAGE: "zlc_frontend.image_display",
        ViewIntent.CURVE: "zlc_frontend.curve_display",
        ViewIntent.HISTOGRAM: "zlc_frontend.histogram_display",
        ViewIntent.METER: "zlc_frontend.meter_display",
    }
    assert set(owners) == DATASET_VIEW_INTENTS
    import importlib

    for intent, module_name in owners.items():
        module = importlib.import_module(module_name)
        assert any(
            name.endswith("DisplayState") for name in vars(module)
        ), f"{intent} owner {module_name} has no display state"


def test_view_contract_catalog_separates_dataset_and_document_sources():
    assert set(VIEW_CONTRACTS) == set(ViewIntent)
    assert {
        intent
        for intent, contract in VIEW_CONTRACTS.items()
        if isinstance(contract, ViewContract)
    } == DATASET_VIEW_INTENTS
    assert {
        intent
        for intent, contract in VIEW_CONTRACTS.items()
        if isinstance(contract, DocumentViewContract)
    } == {ViewIntent.PULSE}
    assert contract_for(ViewIntent.PULSE) is PULSE_CONTRACT
    assert isinstance(PULSE_CONTRACT, DocumentViewContract)
    assert PULSE_CONTRACT.source_schema == "zlc_pulse.PulseTimelineDocument"

    from zlc_frontend.panel_render import PanelComposer, PanelRenderError

    with pytest.raises(PanelRenderError, match="document-fed"):
        PanelComposer("pulse", intent=ViewIntent.PULSE)


def test_the_state_validates_its_panel_and_selection():
    state = MeterDisplayState(2, None)
    assert state.panel_index == 2
    assert state.expected_selection is None
    assert state.revision == 0

    with pytest.raises(TypeError, match="panel_index"):
        MeterDisplayState(True, None)
    with pytest.raises(ValueError, match="panel_index"):
        MeterDisplayState(-1, None)
    with pytest.raises(TypeError, match="expected_selection must be Selection"):
        MeterDisplayState(0, object())
    with pytest.raises(TypeError, match="meter display revision"):
        MeterDisplayState(0, None, revision="1")


def test_the_state_is_immutable():
    state = MeterDisplayState(0, None)
    with pytest.raises(Exception):
        state.panel_index = 1  # frozen dataclass


def test_no_empty_form_spec_was_invented_to_match_the_siblings():
    source = (ROOT / "zlc_frontend" / "meter_display.py").read_text(encoding="utf-8")
    assert "def meter_display_form_spec" not in source
    assert "FormSpec" not in source
    # ...and the reason is written down where the next editor will read it.
    assert "no authored display parameter" in source


def test_the_workbench_no_longer_owns_a_private_copy():
    workbench_sources = {
        path: path.read_text(encoding="utf-8")
        for path in (ROOT / "zlc_workbench").rglob("*.py")
    }
    assert workbench_sources
    assert all("_MeterDisplayState" not in source for source in workbench_sources.values())
    assert all(
        "class MeterDisplayState" not in source
        for source in workbench_sources.values()
    )
    assert any(
        "MeterDisplayState" in source for source in workbench_sources.values()
    ), "the workbench must consume the frontend owner"
