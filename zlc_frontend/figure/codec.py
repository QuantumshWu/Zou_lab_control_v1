"""Strict canonical codecs for persistent headless figure values."""

from __future__ import annotations

from typing import Any

from zlc_data import AxisId, selection_from_tree, selection_to_tree
from zlc_storage.canonical import (
    decode,
    encode,
    exact_mapping,
)

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


VIEW_SPEC_SCHEMA = "zlc_frontend.ViewSpec"
FIGURE_DOCUMENT_SCHEMA = "zlc_frontend.FigureDocument"


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
    if not isinstance(tree, dict):
        raise ValueError("display selector must be a tagged map or null")
    kind = tree.get("kind")
    if kind == "FIXED_INDEX" and set(tree) == {"kind", "index"}:
        return FixedIndex(tree["index"])
    if kind == "LATEST_NONEMPTY" and set(tree) == {"kind"}:
        return LatestNonempty()
    raise ValueError(f"invalid display selector {kind!r}")


def _reduction_to_tree(reduction: DisplayReduction | None) -> dict[str, Any] | None:
    if reduction is None:
        return None
    if not isinstance(reduction, DisplayReduction):
        raise TypeError("reduction must be DisplayReduction or None")
    return {"method": reduction.method.value}


def _reduction_from_tree(tree: Any) -> DisplayReduction | None:
    if tree is None:
        return None
    data = exact_mapping(
        tree,
        {"method"},
        "DisplayReduction",
        discriminator=None,
    )
    return DisplayReduction(DisplayReductionMethod(data["method"]))


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
    data = exact_mapping(
        tree,
        {
            "schema",
            "schema_fingerprint",
            "intent",
            "axis_bindings",
            "display_selections",
        },
        VIEW_SPEC_SCHEMA,
    )
    raw_bindings = data["axis_bindings"]
    if not isinstance(raw_bindings, list):
        raise ValueError("ViewSpec axis_bindings must be a list")
    raw_selections = data["display_selections"]
    if not isinstance(raw_selections, list):
        raise ValueError("ViewSpec display_selections must be a list")
    bindings = []
    for raw in raw_bindings:
        item = exact_mapping(
            raw,
            {"axis_id", "role", "selector", "reduction"},
            "AxisViewBinding",
            discriminator=None,
        )
        bindings.append(
            AxisViewBinding(
                AxisId(item["axis_id"]),
                AxisViewRole(item["role"]),
                _selector_from_tree(item["selector"]),
                _reduction_from_tree(item["reduction"]),
            )
        )
    return ViewSpec(
        data["schema_fingerprint"],
        ViewIntent(data["intent"]),
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
    data = exact_mapping(
        tree,
        {"schema", "document_id", "revision", "datasets", "layers", "selections"},
        FIGURE_DOCUMENT_SCHEMA,
    )
    if not isinstance(data["datasets"], list):
        raise ValueError("FigureDocument datasets must be a list")
    if not isinstance(data["layers"], list):
        raise ValueError("FigureDocument layers must be a list")
    if not isinstance(data["selections"], list):
        raise ValueError("FigureDocument selections must be a list")
    datasets = []
    for raw in data["datasets"]:
        item = exact_mapping(
            raw,
            {"dataset_id", "label", "schema_fingerprint"},
            "DatasetDescriptor",
            discriminator=None,
        )
        datasets.append(
            DatasetDescriptor(
                DatasetId(item["dataset_id"]),
                item["label"],
                item["schema_fingerprint"],
            )
        )
    layers = []
    for raw in data["layers"]:
        item = exact_mapping(
            raw,
            {"layer_id", "dataset_id", "view"},
            "FigureLayer",
            discriminator=None,
        )
        layers.append(
            FigureLayer(
                item["layer_id"],
                DatasetId(item["dataset_id"]),
                view_spec_from_tree(item["view"]),
            )
        )
    selections = []
    for raw in data["selections"]:
        item = exact_mapping(
            raw,
            {"selection_id", "dataset_id", "selection"},
            "FigureSelection",
            discriminator=None,
        )
        selections.append(
            FigureSelection(
                item["selection_id"],
                DatasetId(item["dataset_id"]),
                selection_from_tree(item["selection"]),
            )
        )
    return FigureDocument(
        data["document_id"],
        data["revision"],
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
