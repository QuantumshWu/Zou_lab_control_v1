"""METER uses the same headless frontend display contract as every Figure host."""

from __future__ import annotations

import inspect

import pytest

import zlc_frontend
from zlc_data import AxisId, AxisSourceRef
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


def test_the_state_validates_its_panel_and_exact_address():
    address = ((AxisSourceRef.tensor(AxisId("meter.value")), 3),)
    state = MeterDisplayState(2, address)
    assert state.panel_index == 2
    assert state.expected_address == address
    assert state.revision == 0

    with pytest.raises(TypeError, match="panel_index"):
        MeterDisplayState(True, None)
    with pytest.raises(ValueError, match="panel_index"):
        MeterDisplayState(-1, None)
    with pytest.raises(TypeError):
        MeterDisplayState(0, object())
    with pytest.raises(TypeError, match="meter display revision"):
        MeterDisplayState(0, None, revision="1")


def test_the_state_is_immutable():
    state = MeterDisplayState(0, None)
    with pytest.raises(Exception):
        state.panel_index = 1  # frozen dataclass
