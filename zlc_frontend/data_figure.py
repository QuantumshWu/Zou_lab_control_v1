"""Static notebook rendering facade over one frozen frontend evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from io import BytesIO
import math
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

from zlc_data import FitResultBatch, Selection, validate_fit_result_source_binding
from zlc_storage import canonical_text

from .figure import (
    AxisViewRole,
    FigureDocument,
    FigureEvaluator,
    FigureEvaluationPolicy,
    ResolvedDatasetMap,
    ViewIntent,
)
from .figure.contract import _validate_selection_fit_view

if TYPE_CHECKING:
    from .fit_image_projection import RadialGaussianImageFitPanel


@dataclass(frozen=True, slots=True)
class FigurePanelRegion:
    """One display-only panel hit target in normalized raster coordinates."""

    key: str
    selection: Selection | None
    fit_storage_index: int | None
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        canonical_text(self.key, "figure panel key")
        if self.selection is not None and not isinstance(self.selection, Selection):
            raise TypeError("panel selection must be Selection or None")
        if self.fit_storage_index is not None and (
            isinstance(self.fit_storage_index, bool)
            or not isinstance(self.fit_storage_index, Integral)
            or self.fit_storage_index < 0
        ):
            raise ValueError("fit_storage_index must be non-negative or None")
        if self.fit_storage_index is not None:
            object.__setattr__(self, "fit_storage_index", int(self.fit_storage_index))
        bounds = tuple(float(value) for value in (
            self.left,
            self.top,
            self.right,
            self.bottom,
        ))
        if any(not math.isfinite(value) for value in bounds):
            raise ValueError("panel bounds must be finite")
        left, top, right, bottom = bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError("panel bounds must be an ordered normalized rectangle")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "bottom", bottom)

    def contains(self, x: float, y: float) -> bool:
        x_value, y_value = float(x), float(y)
        return (
            self.left <= x_value <= self.right
            and self.top <= y_value <= self.bottom
        )


class DataFigure:
    """Own one immutable, already-resolved notebook figure.

    ``DataFigure`` never resolves repositories, sessions, devices, or live
    streams.  Construction evaluates the supplied frozen snapshots once and
    releases them; later renders consume only immutable presentation DTOs.
    """

    __slots__ = (
        "_document",
        "_evaluated",
        "_fit_results",
        "_render_memory_limit_bytes",
    )

    def __init__(
        self,
        document: FigureDocument,
        datasets: ResolvedDatasetMap,
        *,
        fit_results: Mapping[str, FitResultBatch] | None = None,
        evaluation_memory_limit_bytes: int | None = None,
        render_memory_limit_bytes: int | None = None,
    ) -> None:
        if not isinstance(document, FigureDocument):
            raise TypeError("document must be FigureDocument")
        if not isinstance(datasets, ResolvedDatasetMap):
            raise TypeError("datasets must be ResolvedDatasetMap")
        supplied = {} if fit_results is None else dict(fit_results)
        if any(not isinstance(key, str) or not key for key in supplied):
            raise TypeError("fit_results keys must be non-empty layer ids")
        if any(not isinstance(value, FitResultBatch) for value in supplied.values()):
            raise TypeError("fit_results values must be FitResultBatch")
        render_limit = self._validated_memory_limit(
            render_memory_limit_bytes,
            "render_memory_limit_bytes",
        )

        layers = {layer.layer_id: layer for layer in document.layers}
        fit_layers = {}
        for layer_id, result in supplied.items():
            try:
                layer = layers[layer_id]
            except KeyError as exc:
                raise ValueError(
                    f"fit overlay references unknown layer {layer_id!r}"
                ) from exc
            snapshot = datasets.resolve(layer.dataset_id)
            if result.spec.committed_transform is not None:
                try:
                    _validate_selection_fit_view(
                        snapshot.block.schema,
                        result,
                        layer.view,
                    )
                except ValueError as exc:
                    raise ValueError(
                        "transformed fit overlay is not faithfully displayable: "
                        f"{exc}"
                    ) from exc
            validate_fit_result_source_binding(result, snapshot.ref, snapshot.block.schema)
            fit_layers[layer_id] = (layer, result)

        if evaluation_memory_limit_bytes is None:
            policy = FigureEvaluationPolicy()
        else:
            if (
                isinstance(evaluation_memory_limit_bytes, bool)
                or not isinstance(evaluation_memory_limit_bytes, Integral)
                or evaluation_memory_limit_bytes <= 0
            ):
                raise ValueError(
                    "evaluation_memory_limit_bytes must be a positive integer or None"
                )
            policy = replace(
                FigureEvaluationPolicy(),
                max_live_nbytes=int(evaluation_memory_limit_bytes),
            )
        evaluated = FigureEvaluator(policy).evaluate(document, datasets)
        allowed_batch_roles = {
            AxisViewRole.BATCH,
            AxisViewRole.FACET,
            AxisViewRole.SELECTED,
            AxisViewRole.SLIDER,
        }
        for layer, result in fit_layers.values():
            fit_axes = result.fit_axis_specs
            if len(fit_axes) == 1:
                if (
                    layer.view.intent is not ViewIntent.CURVE
                    or layer.view.binding(fit_axes[0].axis_id).role is not AxisViewRole.X
                ):
                    raise ValueError("one-axis fit overlay requires its fitted axis as curve x")
            elif len(fit_axes) == 2:
                if (
                    layer.view.intent is not ViewIntent.IMAGE
                    or layer.view.binding(fit_axes[0].axis_id).role
                    is not AxisViewRole.IMAGE_X
                    or layer.view.binding(fit_axes[1].axis_id).role
                    is not AxisViewRole.IMAGE_Y
                ):
                    raise ValueError(
                        "two-axis fit overlay requires its fitted axes as image x/y"
                    )
            else:
                raise ValueError("only one- and two-axis fit overlays are supported")
            for axis in result.batch_axis_specs:
                if layer.view.binding(axis.axis_id).role not in allowed_batch_roles:
                    raise ValueError(
                        f"fit batch axis {axis.axis_id} is not uniquely displayed or selected"
                    )

        self._document = document
        self._evaluated = evaluated
        self._fit_results = tuple(sorted(supplied.items()))
        self._render_memory_limit_bytes = render_limit

    @property
    def document(self) -> FigureDocument:
        return self._document

    @property
    def evaluated(self):
        return self._evaluated

    @property
    def render_memory_limit_bytes(self) -> int | None:
        """Frozen default admission limit for every later render/export."""
        return self._render_memory_limit_bytes

    @property
    def has_fit_overlays(self) -> bool:
        """Whether this immutable figure carries authoritative fit overlays."""
        return bool(self._fit_results)

    def render(
        self,
        *,
        dpi: float = 100.0,
        memory_limit_bytes: int | None = None,
    ):
        """Create a caller-owned Figure with canonical artist styles frozen in.

        The caller owns later Matplotlib mutations and draws.  Product-controlled PNG/export
        paths use the render owner's serialized compose API instead.
        """

        from .matplotlib_render import (
            render_evaluated_figure,
        )

        self._check_render_budget(dpi, memory_limit_bytes)

        return render_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            dpi=dpi,
        )

    def to_png_bytes(
        self,
        *,
        dpi: float = 100.0,
        memory_limit_bytes: int | None = None,
    ) -> bytes:
        from .matplotlib_render import save_evaluated_figure

        effective_limit = self._check_render_budget(dpi, memory_limit_bytes)
        output = BytesIO()
        save_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            output,
            image_format="png",
            dpi=dpi,
        )
        payload = output.getvalue()
        if effective_limit is not None and len(payload) > effective_limit:
            raise MemoryError("PNG payload exceeds figure render memory limit")
        return payload

    def to_png_bytes_with_panel_regions(
        self,
        *,
        dpi: float = 100.0,
        memory_limit_bytes: int | None = None,
    ) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
        """Encode the same frozen figure plus exact display-panel hit regions."""

        from .matplotlib_render import encode_evaluated_figure_with_panel_regions

        effective_limit = self._check_render_budget(dpi, memory_limit_bytes)
        payload, regions = encode_evaluated_figure_with_panel_regions(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            dpi=dpi,
        )
        if effective_limit is not None and len(payload) > effective_limit:
            raise MemoryError("PNG payload exceeds figure render memory limit")
        return payload, regions

    def radial_gaussian_image_fit_panels(
        self,
        layer_id: str,
        *,
        artifact_identity: str,
    ) -> tuple[RadialGaussianImageFitPanel, ...]:
        """Return typed saved-fit IMAGE panels without exposing fit authority.

        The immutable projections retain exact source/artifact identity, sparse
        logical holes, authoritative axis metadata, focus summaries, and only
        the published centre/radius annotation.  No solver or predicted image
        is evaluated on this path.
        """

        from .fit_image_projection import radial_gaussian_image_fit_panels

        return radial_gaussian_image_fit_panels(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            layer_id,
            artifact_identity=artifact_identity,
        )

    def _repr_png_(self) -> bytes:
        return self.to_png_bytes()

    def export(
        self,
        path: str | Path,
        *,
        image_format: str | None = None,
        dpi: float = 100.0,
        memory_limit_bytes: int | None = None,
    ) -> Path:
        target = Path(path)
        if image_format is None:
            image_format = target.suffix.lstrip(".") or "png"
        if not target.suffix:
            target = target.with_suffix(f".{image_format}")
        from .matplotlib_render import save_evaluated_figure

        self._check_render_budget(dpi, memory_limit_bytes)
        save_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            target,
            image_format=image_format,
            dpi=dpi,
        )
        return target

    def _check_render_budget(
        self,
        dpi: float,
        memory_limit_bytes: int | None,
    ) -> int | None:
        requested = self._validated_memory_limit(
            memory_limit_bytes,
            "memory_limit_bytes",
        )
        frozen = self._render_memory_limit_bytes
        if frozen is not None and requested is not None and requested > frozen:
            raise ValueError(
                "memory_limit_bytes cannot weaken the DataFigure render limit"
            )
        effective = frozen if requested is None else requested
        if effective is None:
            return None
        from .matplotlib_render import estimate_render_peak_nbytes

        required = estimate_render_peak_nbytes(self._evaluated, dpi=dpi)
        if required > effective:
            raise MemoryError(
                f"figure render peak {required} exceeds limit {effective}"
            )
        return effective

    @staticmethod
    def _validated_memory_limit(value: int | None, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer or None")
        return int(value)


__all__ = ["DataFigure", "FigurePanelRegion"]
