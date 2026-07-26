"""Mechanical projection of ordinary Logic-node declarations into TaskConsole."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zlc_neutral_atom.input_spec import ArtifactInputSpec
from zlc_neutral_atom.logic_node_declaration import (
    DynamicChoicePresentation,
    LogicNodeDeclaration,
)
from zlc_workbench.form_projection import (
    DynamicChoiceProjection,
    PathPresentation,
    PresentedChoice,
    project_authoring_form,
)
from zlc_workbench.input_binding import ResolvedArtifactInput, project_input_fields
from zlc_workbench.task_console.attachment_builders import (
    processor_attachment,
    run_attachment,
)
from zlc_workbench.task_console.artifact_resolution import (
    resolve_producer_final_artifact,
)
from zlc_workbench.task_console.catalog_bridge import (
    ConsoleDefaultPanel,
    ConsoleNodeSpec,
    ConsoleSignalDecl,
)


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
    context: object | None,
) -> dict[str, DynamicChoiceProjection]:
    resolver = declaration.resolve_dynamic_choices
    if resolver is None:
        if context is not None:
            raise ValueError("this Logic node declares no dynamic choice context")
        return {}
    if context is None:
        raise ValueError("this Logic node requires its installation choice context")
    resolved = tuple(resolver(context))
    if any(not isinstance(value, DynamicChoicePresentation) for value in resolved):
        raise TypeError("dynamic choice resolver returned another value type")
    keys = tuple(value.field_key for value in resolved)
    expected = tuple(
        field.key
        for field in declaration.authoring_schema.fields
        if field.dynamic_choices
    )
    if keys != expected:
        raise ValueError("dynamic choice resolver changed its declared field order")
    return {
        value.field_key: DynamicChoiceProjection(
            tuple(
                PresentedChoice(option.value, option.label)
                for option in value.choices
            ),
            value.default,
            value.unavailable_reason,
        )
        for value in resolved
    }


def _path_projection(values) -> dict[str, PathPresentation]:
    return {
        value.field_key: PathPresentation(
            mode=value.mode,
            file_filter=value.file_filter,
            base_dir=value.base_dir,
        )
        for value in values
    }


def project_declaration_spec(
    declaration: LogicNodeDeclaration,
    *,
    dynamic_choice_context: object | None = None,
    form: object | None = None,
    editor_factory: Callable[..., object] | None = None,
) -> ConsoleNodeSpec:
    """Mechanically project one owner declaration into the generic host DTO."""

    if not isinstance(declaration, LogicNodeDeclaration):
        raise TypeError("declaration must be LogicNodeDeclaration")
    projected_form = (
        project_authoring_form(
            declaration.authoring_schema,
            dynamic_choices=_dynamic_choice_projection(
                declaration,
                dynamic_choice_context,
            ),
            path_presentations=_path_projection(
                declaration.path_presentations
            ),
        )
        if form is None
        else form
    )

    return ConsoleNodeSpec(
        definition=declaration.definition,
        title=declaration.definition.title,
        description=declaration.description,
        form=projected_form,
        declared_outputs=tuple(
            ConsoleSignalDecl(
                output.declaration,
                output.short,
                output.axis_label,
                output.description,
            )
            for output in declaration.outputs
        ),
        build_request=declaration.build_request,
        input_specs=declaration.input_specs,
        input_fields=project_input_fields(
            declaration.input_specs,
            path_presentations=_path_projection(
                declaration.input_path_presentations
            ),
        ),
        default_panels=tuple(
            ConsoleDefaultPanel(view.output_name, view.kind, view.params)
            for view in declaration.default_views
        ),
        request_output_declarations=declaration.request_output_declarations,
        request_output_axis_label=declaration.request_output_axis_label,
        request_output_description=declaration.request_output_description,
        editor_factory=editor_factory,
    )


def project_run_declaration(
    declaration: LogicNodeDeclaration,
    *,
    prepare: Callable[[object], object],
    bind_request: Callable[[object, object], object] | None = None,
    dynamic_choice_context: object | None = None,
    resolve_artifact_reference: Callable[[ResolvedArtifactInput], object]
    | None = None,
    start: Callable[[object, object, object], object] | None = None,
    start_with_live_output: Callable[[object, object], object] | None = None,
    materialize_final_presentations: Callable[[object, object, object], object]
    | None = None,
):
    """Build the common finite-run attachment for one declaration."""

    if not callable(prepare):
        raise TypeError("prepare must be callable")
    if bind_request is not None and not callable(bind_request):
        raise TypeError("bind_request must be callable or None")
    spec = project_declaration_spec(
        declaration,
        dynamic_choice_context=dynamic_choice_context,
    )
    return run_attachment(
        spec,
        bind_request=(
            declaration.bind_request if bind_request is None else bind_request
        ),
        prepare=prepare,
        start=start,
        start_with_live_output=start_with_live_output,
        materialize_final_presentations=materialize_final_presentations,
        resolve_artifact_reference=_artifact_resolver(
            declaration,
            resolve_artifact_reference=resolve_artifact_reference,
        ),
    )


def project_processor_declaration(
    declaration: LogicNodeDeclaration,
    *,
    prepare: Callable[[object], object],
    project_presentations: Callable[..., Mapping[str, object]] | None = None,
    dynamic_choice_context: object | None = None,
    resolve_artifact_reference: Callable[[ResolvedArtifactInput], object]
    | None = None,
):
    """Build the common reactive Processor attachment for one declaration."""

    if not callable(prepare):
        raise TypeError("Processor prepare must be callable")
    if project_presentations is not None and not callable(project_presentations):
        raise TypeError("project_presentations must be callable or None")
    spec = project_declaration_spec(
        declaration,
        dynamic_choice_context=dynamic_choice_context,
    )
    return processor_attachment(
        spec,
        bind_request=declaration.bind_request,
        prepare=prepare,
        project_presentations=project_presentations,
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
