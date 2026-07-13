"""Strict canonical codecs for persistent headless figure values."""

from __future__ import annotations

from typing import Any

from zlc_data import AxisId, selection_from_tree, selection_to_tree
from zlc_storage.canonical import decode, encode

from .model import (
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    DisplayReduction,
    DisplayReductionMethod,
    FigureDocument,
    FigureLayer,
    FigureSelection,
    FixedIndex,
    LatestNonempty,
    ViewIntent,
    ViewSpec,
)


VIEW_SPEC_SCHEMA = "zlc_frontend.ViewSpec/v1"
FIGURE_DOCUMENT_SCHEMA = "zlc_frontend.FigureDocument/v1"


def _exact(tree: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError(f"{context} has an invalid field set")
    return tree


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be non-empty text without surrounding whitespace")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _selector_to_tree(selector) -> dict[str, Any] | None:
    if selector is None:
        return None
    if isinstance(selector, FixedIndex):
        return {"kind": "FIXED_INDEX", "index": selector.index}
    if isinstance(selector, LatestNonempty):
        return {"kind": "LATEST_NONEMPTY"}
    raise TypeError(f"unsupported selector {type(selector).__name__}")


def _selector_from_tree(tree: Any):
    if tree is None:
        return None
    if not isinstance(tree, dict) or not isinstance(tree.get("kind"), str):
        raise ValueError("display selector must be a tagged map or null")
    if tree["kind"] == "FIXED_INDEX" and set(tree) == {"kind", "index"}:
        return FixedIndex(_integer(tree["index"], "selector index"))
    if tree["kind"] == "LATEST_NONEMPTY" and set(tree) == {"kind"}:
        return LatestNonempty()
    raise ValueError(f"invalid display selector {tree.get('kind')!r}")


def _reduction_to_tree(reduction: DisplayReduction | None) -> dict[str, Any] | None:
    if reduction is None:
        return None
    if not isinstance(reduction, DisplayReduction):
        raise TypeError("reduction must be DisplayReduction or None")
    return {"method": reduction.method.value}


def _reduction_from_tree(tree: Any) -> DisplayReduction | None:
    if tree is None:
        return None
    data = _exact(tree, {"method"}, "DisplayReduction")
    return DisplayReduction(DisplayReductionMethod(_text(data["method"], "reduction method")))


def view_spec_to_tree(spec: ViewSpec) -> dict[str, Any]:
    if not isinstance(spec, ViewSpec):
        raise TypeError("spec must be ViewSpec")
    return {
        "schema": VIEW_SPEC_SCHEMA,
        "schema_fingerprint": spec.schema_fingerprint,
        "intent": spec.intent.value,
        "axis_bindings": [
            {
                "axis_id": binding.axis_id.value,
                "role": binding.role.value,
                "selector": _selector_to_tree(binding.selector),
                "reduction": _reduction_to_tree(binding.reduction),
            }
            for binding in spec.axis_bindings
        ],
        "display_selections": [
            selection_to_tree(selection) for selection in spec.display_selections
        ],
    }


def view_spec_from_tree(tree: Any) -> ViewSpec:
    data = _exact(
        tree,
        {
            "schema",
            "schema_fingerprint",
            "intent",
            "axis_bindings",
            "display_selections",
        },
        "ViewSpec",
    )
    if data["schema"] != VIEW_SPEC_SCHEMA:
        raise ValueError(f"expected schema {VIEW_SPEC_SCHEMA!r}")
    raw_bindings = data["axis_bindings"]
    if not isinstance(raw_bindings, list):
        raise ValueError("ViewSpec axis_bindings must be a list")
    raw_selections = data["display_selections"]
    if not isinstance(raw_selections, list):
        raise ValueError("ViewSpec display_selections must be a list")
    bindings = []
    for raw in raw_bindings:
        item = _exact(
            raw,
            {"axis_id", "role", "selector", "reduction"},
            "AxisViewBinding",
        )
        bindings.append(
            AxisViewBinding(
                AxisId(_text(item["axis_id"], "axis_id")),
                AxisViewRole(_text(item["role"], "axis view role")),
                _selector_from_tree(item["selector"]),
                _reduction_from_tree(item["reduction"]),
            )
        )
    return ViewSpec(
        _text(data["schema_fingerprint"], "schema_fingerprint"),
        ViewIntent(_text(data["intent"], "intent")),
        tuple(bindings),
        tuple(selection_from_tree(item) for item in raw_selections),
    )


def encode_view_spec(spec: ViewSpec) -> bytes:
    return encode(view_spec_to_tree(spec))


def decode_view_spec(payload: bytes) -> ViewSpec:
    spec = view_spec_from_tree(decode(payload))
    if bytes(payload) != encode_view_spec(spec):
        raise ValueError("ViewSpec payload uses a non-canonical typed representation")
    return spec


def figure_document_to_tree(document: FigureDocument) -> dict[str, Any]:
    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    return {
        "schema": FIGURE_DOCUMENT_SCHEMA,
        "document_id": document.document_id,
        "revision": document.revision,
        "datasets": [
            {
                "dataset_id": descriptor.dataset_id.value,
                "label": descriptor.label,
                "schema_fingerprint": descriptor.schema_fingerprint,
            }
            for descriptor in document.datasets
        ],
        "layers": [
            {
                "layer_id": layer.layer_id,
                "dataset_id": layer.dataset_id.value,
                "view": view_spec_to_tree(layer.view),
            }
            for layer in document.layers
        ],
        "selections": [
            {
                "selection_id": item.selection_id,
                "dataset_id": item.dataset_id.value,
                # Delegate the embedded value to the zlc_data owner codec.
                "selection": selection_to_tree(item.selection),
            }
            for item in document.selections
        ],
    }


def figure_document_from_tree(tree: Any) -> FigureDocument:
    data = _exact(
        tree,
        {"schema", "document_id", "revision", "datasets", "layers", "selections"},
        "FigureDocument",
    )
    if data["schema"] != FIGURE_DOCUMENT_SCHEMA:
        raise ValueError(f"expected schema {FIGURE_DOCUMENT_SCHEMA!r}")
    if not isinstance(data["datasets"], list):
        raise ValueError("FigureDocument datasets must be a list")
    if not isinstance(data["layers"], list):
        raise ValueError("FigureDocument layers must be a list")
    if not isinstance(data["selections"], list):
        raise ValueError("FigureDocument selections must be a list")
    datasets = []
    for raw in data["datasets"]:
        item = _exact(
            raw,
            {"dataset_id", "label", "schema_fingerprint"},
            "DatasetDescriptor",
        )
        datasets.append(
            DatasetDescriptor(
                DatasetId(_text(item["dataset_id"], "dataset_id")),
                _text(item["label"], "dataset label"),
                _text(item["schema_fingerprint"], "schema_fingerprint"),
            )
        )
    layers = []
    for raw in data["layers"]:
        item = _exact(raw, {"layer_id", "dataset_id", "view"}, "FigureLayer")
        layers.append(
            FigureLayer(
                _text(item["layer_id"], "layer_id"),
                DatasetId(_text(item["dataset_id"], "dataset_id")),
                view_spec_from_tree(item["view"]),
            )
        )
    selections = []
    for raw in data["selections"]:
        item = _exact(
            raw,
            {"selection_id", "dataset_id", "selection"},
            "FigureSelection",
        )
        selections.append(
            FigureSelection(
                _text(item["selection_id"], "selection_id"),
                DatasetId(_text(item["dataset_id"], "dataset_id")),
                selection_from_tree(item["selection"]),
            )
        )
    return FigureDocument(
        _text(data["document_id"], "document_id"),
        _integer(data["revision"], "revision"),
        tuple(datasets),
        tuple(layers),
        tuple(selections),
    )


def encode_figure_document(document: FigureDocument) -> bytes:
    return encode(figure_document_to_tree(document))


def decode_figure_document(payload: bytes) -> FigureDocument:
    document = figure_document_from_tree(decode(payload))
    if bytes(payload) != encode_figure_document(document):
        raise ValueError("FigureDocument payload uses a non-canonical typed representation")
    return document


__all__ = [
    "FIGURE_DOCUMENT_SCHEMA",
    "VIEW_SPEC_SCHEMA",
    "decode_figure_document",
    "decode_view_spec",
    "encode_figure_document",
    "encode_view_spec",
    "figure_document_from_tree",
    "figure_document_to_tree",
    "view_spec_from_tree",
    "view_spec_to_tree",
]
