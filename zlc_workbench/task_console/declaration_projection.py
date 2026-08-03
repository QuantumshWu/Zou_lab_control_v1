"""Mechanical projection of ordinary Logic-node declarations into TaskConsole."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from zlc_frontend.form import project_authoring_form
from zlc_neutral_atom.input_spec import ArtifactInputSpec
from zlc_neutral_atom.logic_node_declaration import (
    DynamicChoicePresentation,
    LogicNodeDeclaration,
    PathPresentationHint,
)
from .input_binding import ResolvedArtifactInput, project_input_fields
from zlc_workbench.task_console.attachment_builders import (
    processor_attachment,
    run_attachment,
)
from zlc_workbench.task_console.artifact_resolution import (
    resolve_producer_final_artifact,
)
from zlc_workbench.task_console.catalog_bridge import ConsoleNodeSpec


def _artifact_resolver(
    declaration: LogicNodeDeclaration,
    *,
    resolve_artifact_reference: Callable[[ResolvedArtifactInput], object]
    | None = None,
) -> Callable[[ResolvedArtifactInput], object] | None:
    artifact_specs = tuple(
        spec
        for spec in declaration.input_specs
        if isinstance(spec, ArtifactInputSpec)
    )
    if resolve_artifact_reference is None:
        if any(spec.allow_saved_reference for spec in artifact_specs):
            raise ValueError(
                "saved Artifact inputs require an explicit composition resolver"
            )
        return resolve_producer_final_artifact if artifact_specs else None
    else:
        if not callable(resolve_artifact_reference):
            raise TypeError("resolve_artifact_reference must be callable or None")
        return resolve_artifact_reference


def _dynamic_choice_projection(
    declaration: LogicNodeDeclaration,
    values: tuple[DynamicChoicePresentation, ...],
) -> dict[str, DynamicChoicePresentation]:
    resolved = tuple(values)
    resolver = declaration.resolve_dynamic_choices
    if resolver is None:
        if resolved:
            raise ValueError("this Logic node declares no dynamic choices")
        return {}
    if any(not isinstance(value, DynamicChoicePresentation) for value in resolved):
        raise TypeError("dynamic choices contain another value type")
    keys = tuple(value.field_key for value in resolved)
    expected = tuple(
        field.key
        for field in declaration.authoring_schema.fields
        if field.dynamic_choices
    )
    if keys != expected:
        raise ValueError("dynamic choice resolver changed its declared field order")
    return {value.field_key: value for value in resolved}


def _path_projection(values, path_roots=None) -> dict[str, PathPresentationHint]:
    hints = tuple(values)
    if any(not isinstance(value, PathPresentationHint) for value in hints):
        raise TypeError("path presentation owner returned another value type")
    roots = {} if path_roots is None else dict(path_roots)
    projected = {}
    for value in hints:
        base = value.base_dir
        if base:
            parts = Path(base).parts
            if parts and parts[0] in roots:
                root = Path(roots[parts[0]]).expanduser()
                if not root.is_absolute():
                    raise ValueError("TaskConsole path roots must be absolute")
                value = replace(
                    value,
                    base_dir=str(root.joinpath(*parts[1:]).resolve()),
                )
            elif not Path(base).is_absolute():
                raise ValueError(
                    f"unbound TaskConsole path root {parts[0] if parts else base!r}"
                )
        projected[value.field_key] = value
    return projected


def project_declaration_spec(
    declaration: LogicNodeDeclaration,
    *,
    dynamic_choices: tuple[DynamicChoicePresentation, ...] = (),
    form: object | None = None,
    editor_factory: Callable[..., object] | None = None,
    editor_builder: Callable[[object], tuple[object, Callable[..., object]]]
    | None = None,
    path_roots=None,
) -> ConsoleNodeSpec:
    """Mechanically project one owner declaration into the generic host DTO."""

    if not isinstance(declaration, LogicNodeDeclaration):
        raise TypeError("declaration must be LogicNodeDeclaration")
    projected_form = (
        project_authoring_form(
            declaration.authoring_schema,
            dynamic_choices=_dynamic_choice_projection(
                declaration,
                dynamic_choices,
            ),
            path_presentations=_path_projection(
                declaration.path_presentations,
                path_roots,
            ),
        )
        if form is None
        else form
    )
    if editor_builder is not None:
        if form is not None or editor_factory is not None:
            raise ValueError(
                "editor_builder cannot be combined with a prebuilt form/editor"
            )
        if not callable(editor_builder):
            raise TypeError("editor_builder must be callable or None")
        built = editor_builder(projected_form)
        if not isinstance(built, tuple) or len(built) != 2:
            raise TypeError("editor_builder must return (form, editor_factory)")
        projected_form, editor_factory = built

    return ConsoleNodeSpec(
        declaration=declaration,
        form=projected_form,
        input_fields=project_input_fields(
            declaration.input_specs,
            path_presentations=_path_projection(
                declaration.input_path_presentations,
                path_roots,
            ),
        ),
        editor_factory=editor_factory,
    )


def project_run_declaration(
    declaration: LogicNodeDeclaration,
    *,
    prepare: Callable[[object, object | None], object],
    bind_request: Callable[[object, object], object] | None = None,
    dynamic_choices: tuple[DynamicChoicePresentation, ...] = (),
    resolve_artifact_reference: Callable[[ResolvedArtifactInput], object]
    | None = None,
    start_prepared: Callable[[object, object, object], object]
    | None = None,
    editor_builder: Callable[[object], tuple[object, Callable[..., object]]]
    | None = None,
    path_roots=None,
):
    """Build the common finite-run attachment for one declaration."""

    if not callable(prepare):
        raise TypeError("prepare must be callable")
    request_binder = declaration.bind_request if bind_request is None else bind_request
    if not callable(request_binder):
        raise TypeError("run declaration requires one request binder")
    spec = project_declaration_spec(
        declaration,
        dynamic_choices=dynamic_choices,
        editor_builder=editor_builder,
        path_roots=path_roots,
    )
    return run_attachment(
        spec,
        bind_request=request_binder,
        prepare=prepare,
        start_prepared=start_prepared,
        resolve_artifact_reference=_artifact_resolver(
            declaration,
            resolve_artifact_reference=resolve_artifact_reference,
        ),
    )


def project_processor_declaration(
    declaration: LogicNodeDeclaration,
    *,
    prepare: Callable[[object], object],
    bind_request: Callable[[object, object], object] | None = None,
    dynamic_choices: tuple[DynamicChoicePresentation, ...] = (),
    resolve_artifact_reference: Callable[[ResolvedArtifactInput], object]
    | None = None,
    path_roots=None,
):
    """Build the common reactive Processor attachment for one declaration."""

    if not callable(prepare):
        raise TypeError("Processor prepare must be callable")
    request_binder = declaration.bind_request if bind_request is None else bind_request
    if not callable(request_binder):
        raise TypeError("Processor declaration requires one request binder")
    spec = project_declaration_spec(
        declaration,
        dynamic_choices=dynamic_choices,
        path_roots=path_roots,
    )
    return processor_attachment(
        spec,
        bind_request=request_binder,
        prepare=prepare,
        resolve_artifact_reference=_artifact_resolver(
            declaration,
            resolve_artifact_reference=resolve_artifact_reference,
        ),
    )


__all__ = [
    "project_declaration_spec",
    "project_processor_declaration",
    "project_run_declaration",
]
