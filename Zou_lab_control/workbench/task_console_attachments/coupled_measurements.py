"""Attachments for the three calibrated coupled Measurements."""

from __future__ import annotations

from zlc_neutral_atom.logic_nodes.grey_molasses_detuning import (
    GREY_MOLASSES_CAPABILITY_GAP,
    GREY_MOLASSES_DETUNING_DEFINITION,
    GREY_MOLASSES_DETUNING_OUTPUT_DECLARATIONS,
    CalibratedGreyMolassesDetuningIntent,
    bind_grey_molasses_detuning_inputs,
    build_grey_molasses_intent_from_authoring,
    grey_molasses_default_rf_role,
    grey_molasses_detuning_authoring_schema,
)
from zlc_neutral_atom.logic_nodes.readout_common.calibration_input import (
    calibration_input_specs,
)
from zlc_neutral_atom.logic_nodes.readout_duration_fidelity import (
    READOUT_DURATION_FIDELITY_DEFINITION,
    READOUT_DURATION_FIDELITY_OUTPUT_DECLARATIONS,
    CalibratedReadoutDurationFidelityIntent,
    bind_readout_duration_fidelity_inputs,
    build_readout_duration_intent_from_authoring,
    readout_duration_fidelity_authoring_schema,
)
from zlc_neutral_atom.logic_nodes.temperature_release_recapture import (
    TEMPERATURE_RELEASE_RECAPTURE_DEFINITION,
    TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATIONS,
    CalibratedTemperatureReleaseRecaptureIntent,
    bind_temperature_release_recapture_inputs,
    build_temperature_intent_from_authoring,
    temperature_release_recapture_authoring_schema,
)
from zlc_workbench.form_projection import (
    DynamicChoiceProjection,
    PathPresentation,
    PresentedChoice,
    project_authoring_form,
)
from zlc_workbench.input_binding import project_input_fields
from zlc_workbench.task_console.catalog_bridge import (
    ConsoleNodeSpec,
    ConsoleSignalDecl,
)

from ._common import run_attachment


_PULSE_PATH = PathPresentation(
    mode="file",
    file_filter="Pulse program (*.json);;All files (*)",
    base_dir="pulses",
)
def _attachment(spec, *, expected_bound, bind_request, prepare):
    def prepare_bound(request):
        if not isinstance(request, expected_bound):
            raise TypeError("Measurement owner returned another bound intent")
        return prepare(request.intent, request.calibration_ref)

    return run_attachment(
        spec,
        bind_request=bind_request,
        prepare=prepare_bound,
    )


def coupled_measurement_attachments(
    *,
    installed_rf_roles: tuple[str, ...],
    prepare_temperature,
    prepare_readout_duration,
    prepare_grey_molasses,
):
    """Return the closed three-capability tuple in catalog order."""

    inputs = calibration_input_specs()
    input_fields = project_input_fields(inputs)

    temperature = ConsoleNodeSpec(
        definition=TEMPERATURE_RELEASE_RECAPTURE_DEFINITION,
        title="Temperature",
        description=(
            "Autonomous hardware scan with two exact camera events per cell; "
            "publishes calibrated survival without dropping repeat/scan axes"
        ),
        form=project_authoring_form(
            temperature_release_recapture_authoring_schema(),
            path_presentations={"pulse": _PULSE_PATH},
        ),
        declared_outputs=(
            ConsoleSignalDecl(
                TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATIONS[0],
                "survival",
                "Survival",
                "release-recapture survival",
            ),
        ),
        build_request=build_temperature_intent_from_authoring,
        input_specs=inputs,
        input_fields=input_fields,
    )

    readout_duration = ConsoleNodeSpec(
        definition=READOUT_DURATION_FIDELITY_DEFINITION,
        title="Fidelity vs duration",
        description=(
            "Apply and read back camera integration time at each point, then "
            "publish calibrated readout fidelity"
        ),
        form=project_authoring_form(
            readout_duration_fidelity_authoring_schema(),
            path_presentations={"pulse": _PULSE_PATH},
        ),
        declared_outputs=(
            ConsoleSignalDecl(
                READOUT_DURATION_FIDELITY_OUTPUT_DECLARATIONS[0],
                "fidelity",
                "Fidelity",
                "readout fidelity",
            ),
        ),
        build_request=build_readout_duration_intent_from_authoring,
        input_specs=inputs,
        input_fields=input_fields,
    )

    roles = tuple(installed_rf_roles)
    grey_molasses = ConsoleNodeSpec(
        definition=GREY_MOLASSES_DETUNING_DEFINITION,
        title="Grey molasses detuning",
        description=(
            "Autonomous release-recapture whose synchronized RF table advances "
            "from the hardware scan clock"
        ),
        form=project_authoring_form(
            grey_molasses_detuning_authoring_schema(),
            dynamic_choices={
                "rf_role": DynamicChoiceProjection(
                    tuple(PresentedChoice(role, role) for role in roles),
                    grey_molasses_default_rf_role(roles),
                    GREY_MOLASSES_CAPABILITY_GAP if not roles else "",
                )
            },
            path_presentations={"pulse": _PULSE_PATH},
        ),
        declared_outputs=(
            ConsoleSignalDecl(
                GREY_MOLASSES_DETUNING_OUTPUT_DECLARATIONS[0],
                "recapture",
                "Recapture rate",
                "grey-molasses recapture rate",
            ),
        ),
        build_request=build_grey_molasses_intent_from_authoring,
        input_specs=inputs,
        input_fields=input_fields,
    )

    return (
        _attachment(
            temperature,
            expected_bound=CalibratedTemperatureReleaseRecaptureIntent,
            bind_request=bind_temperature_release_recapture_inputs,
            prepare=prepare_temperature,
        ),
        _attachment(
            readout_duration,
            expected_bound=CalibratedReadoutDurationFidelityIntent,
            bind_request=bind_readout_duration_fidelity_inputs,
            prepare=prepare_readout_duration,
        ),
        _attachment(
            grey_molasses,
            expected_bound=CalibratedGreyMolassesDetuningIntent,
            bind_request=bind_grey_molasses_detuning_inputs,
            prepare=prepare_grey_molasses,
        ),
    )


__all__ = ["coupled_measurement_attachments"]
