"""Local Fit composition for one already-frozen :class:`DataFigure`.

This module does not own a solver or a persistence format.  It binds the exact
snapshot already carried by ``DataFigure`` through ``zlc_data``'s sole
``suggest_fit_draft -> bind_fit -> BoundFit.run`` path, then publishes an
explicitly saved result by rebuilding that same immutable figure with
``DataFigure.with_fit_results`` and using the current archive owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from zlc_data import (
    FitNumericPolicy,
    FitResultBatch,
    FitSpec,
    Selection,
    bind_fit,
    encode_fit_result_batch,
    fit_model_catalog,
    suggest_fit_draft,
)
from zlc_frontend import (
    DataFigure,
    FitAuthoringOption,
    LoadedFigureArchive,
    fit_authoring_option,
    load_figure_archive,
)

from .projection import (
    _DEFAULT_FIT_TIMEOUT_SECONDS,
    _FitSaveReceipt,
    _FitWorkbenchBindings,
)


def _saved_seed(
    figure: DataFigure,
    layer_id: str,
) -> tuple[FitSpec | None, Selection | None]:
    result = figure.fit_results.get(layer_id)
    if result is None:
        return None, None
    spec = result.spec
    transform = spec.committed_transform
    if transform is None:
        return spec, None
    operations = tuple(transform.spec.operations)
    if len(operations) == 1 and isinstance(operations[0], Selection):
        return spec, operations[0]
    return spec, None


def local_fit_bindings(
    figure: DataFigure,
    *,
    initial_selection: Selection | None = None,
    open_fit: bool = False,
    archive_path: str | Path | None = None,
    archive_metadata: Mapping[str, object] | None = None,
    timeout_seconds: float = _DEFAULT_FIT_TIMEOUT_SECONDS,
) -> _FitWorkbenchBindings:
    """Bind one exact single-layer figure to the shared Fit host."""

    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    if initial_selection is not None and not isinstance(initial_selection, Selection):
        raise TypeError("initial_selection must be Selection or None")
    if archive_metadata is not None and not isinstance(archive_metadata, Mapping):
        raise TypeError("archive_metadata must be a mapping or None")
    if len(figure.document.layers) != 1:
        raise ValueError("local Fit requires exactly one Figure layer")
    layer = figure.document.layers[0]
    snapshot = figure.datasets.resolve(layer.dataset_id)
    schema = snapshot.block.schema
    seed_spec, saved_selection = _saved_seed(figure, layer.layer_id)
    selected_model = None if seed_spec is None else seed_spec.model_id
    authored_initial_selection = (
        saved_selection if initial_selection is None else initial_selection
    )
    frozen_metadata = (
        {}
        if archive_metadata is None
        else dict(archive_metadata)
    )

    def prepare(
        fit_axis_ids,
        authority_selection: Selection | None,
    ) -> tuple[FitAuthoringOption, ...]:
        axes = tuple(fit_axis_ids)
        if not axes:
            raise ValueError("local Figure Fit requires exact named display axes")
        if authority_selection is not None and not isinstance(
            authority_selection,
            Selection,
        ):
            raise TypeError("local Figure Fit selection must be Selection or None")
        retains_saved_authority = (
            seed_spec is not None
            and (
                (
                    saved_selection is None
                    and authority_selection is None
                )
                or (
                    saved_selection is not None
                    and authority_selection == saved_selection
                )
            )
        )

        options = []
        for definition in fit_model_catalog():
            try:
                if (
                    retains_saved_authority
                    and seed_spec is not None
                    and definition.model_id == seed_spec.model_id
                ):
                    bound = bind_fit(seed_spec, schema)
                else:
                    bound = suggest_fit_draft(
                        schema,
                        definition.model_id,
                        fit_axis_ids=axes,
                        selection=authority_selection,
                        constraints=(
                            seed_spec.constraints
                            if seed_spec is not None
                            and definition.model_id == seed_spec.model_id
                            else ()
                        ),
                        numeric_policy=(
                            seed_spec.numeric_policy
                            if seed_spec is not None
                            and definition.model_id == seed_spec.model_id
                            else FitNumericPolicy()
                        ),
                    )
            except ValueError:
                continue
            options.append(fit_authoring_option(bound))
        if not options:
            raise ValueError(
                "the displayed named axes and selection have no compatible Fit model"
            )
        return tuple(options)

    def execute(spec: FitSpec, cancel_check, deadline_monotonic: float):
        if not isinstance(spec, FitSpec):
            raise TypeError("local Figure Fit execution requires FitSpec")
        return bind_fit(spec, schema).run(
            snapshot,
            cancel_check=cancel_check,
            deadline_monotonic=deadline_monotonic,
        )

    def result(execution) -> FitResultBatch:
        if not isinstance(execution, FitResultBatch):
            raise TypeError("local Figure Fit must return FitResultBatch")
        return execution

    def save(execution, destination, display) -> _FitSaveReceipt:
        if not isinstance(execution, FitResultBatch):
            raise TypeError("local Figure Fit save requires FitResultBatch")
        if destination is None:
            raise ValueError("local Figure Fit save requires an archive path")
        merged = dict(figure.fit_results)
        merged[layer.layer_id] = execution
        fitted = figure.with_fit_results(merged)
        saved_path = fitted.save_archive(
            Path(destination),
            display=display,
            metadata=frozen_metadata,
        )
        reopened = load_figure_archive(saved_path)
        reopened_result = reopened.figure.fit_results.get(layer.layer_id)
        if reopened_result is None or (
            encode_fit_result_batch(reopened_result)
            != encode_fit_result_batch(execution)
        ):
            raise RuntimeError("saved Figure archive did not reopen the exact Fit result")
        return _FitSaveReceipt(
            reopened,
            f"figure-archive:{reopened.payload_digest}",
            str(reopened.path),
            reopened_result,
        )

    def reload_saved(handle) -> FitResultBatch:
        if not isinstance(handle, LoadedFigureArchive):
            raise TypeError("local Fit reload requires LoadedFigureArchive")
        reopened_result = handle.figure.fit_results.get(layer.layer_id)
        if reopened_result is None:
            raise ValueError("saved Figure archive lost its fitted layer")
        return reopened_result

    return _FitWorkbenchBindings(
        prepare,
        execute,
        result,
        save,
        reload_saved,
        selected_model=selected_model,
        initial_selection=authored_initial_selection,
        open_fit=bool(open_fit),
        timeout_seconds=timeout_seconds,
        save_requires_path=True,
        initial_save_path=archive_path,
        allow_prepared_transform=True,
    )


__all__ = ["local_fit_bindings"]
