"""Thin Calibration-output adapter for the sole :mod:`zlc_plot` stack."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np

from zlc_data.value import expand_dataset_validity
from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImageFrame,
    ImagePlot,
    ImagePointOverlay,
    PlotKind,
    PlotLabels,
    PlotSession,
    PointStatus,
    default_plot_spec,
)
from zlc_storage.durability import atomic_write_bytes, durable_mkdir

from ..outputs import CALIBRATION_FINAL_OUTPUT_DECLARATIONS


def _calibration_outputs(
    outputs: Mapping[str, FinalDatasetOutput],
) -> dict[str, FinalDatasetOutput]:
    values = dict(outputs)
    declarations = {
        declaration.name: declaration
        for declaration in CALIBRATION_FINAL_OUTPUT_DECLARATIONS
    }
    if set(values) != set(declarations):
        raise ValueError("Calibration report requires every declared FINAL output")
    for name, value in values.items():
        if not isinstance(value, FinalDatasetOutput):
            raise TypeError("Calibration outputs must be FinalDatasetOutput values")
        if value.declaration != declarations[name]:
            raise ValueError("Calibration output declaration does not match its name")
    return values


def _site_overlay(output: FinalDatasetOutput) -> ImagePointOverlay:
    snapshot = output.snapshot
    schema = snapshot.block.schema
    axes = schema.cell_schema.data_axes
    if (
        schema.repeat_axis.size != 1
        or schema.point_table.row_count != 1
        or len(axes) != 2
        or axes[1].size != 2
    ):
        raise ValueError("Calibration site centres must have shape (1,1,SITE,2)")
    site_axis = axes[0]
    coordinates = np.asarray(snapshot.block.values).reshape((site_axis.size, 2))
    component_validity = np.asarray(
        expand_dataset_validity(snapshot.block.validity, schema),
        dtype=np.bool_,
    ).reshape((site_axis.size, 2))
    retained = np.flatnonzero(np.all(np.isfinite(coordinates), axis=1))
    valid = np.all(component_validity[retained], axis=1)
    return ImagePointOverlay(
        revision=snapshot.ref.revision.value,
        coordinates=coordinates[retained],
        point_ids=tuple(f"site-{index}" for index in retained),
        labels=tuple(str(site_axis.coordinate_at(index)) for index in retained),
        statuses=tuple(
            PointStatus.UNKNOWN if accepted else PointStatus.INVALID
            for accepted in valid
        ),
    )


def _site_thresholds(
    output: FinalDatasetOutput,
    expected_site_axis,
) -> tuple[float | None, ...]:
    snapshot = output.snapshot
    schema = snapshot.block.schema
    axes = schema.cell_schema.data_axes
    if (
        schema.repeat_axis.size != 1
        or schema.point_table.row_count != 1
        or len(axes) != 1
    ):
        raise ValueError("Calibration thresholds must have shape (1,1,SITE)")
    if axes[0] != expected_site_axis:
        raise ValueError("Calibration thresholds and samples use different SITE axes")
    values = np.asarray(snapshot.block.values).reshape((axes[0].size,))
    valid = np.asarray(
        expand_dataset_validity(snapshot.block.validity, schema),
        dtype=np.bool_,
    ).reshape((axes[0].size,))
    return tuple(
        float(value) if accepted and np.isfinite(value) else None
        for value, accepted in zip(values, valid, strict=True)
    )


def _calibration_pages(outputs: Mapping[str, FinalDatasetOutput]):
    values = _calibration_outputs(outputs)

    site_map = values["site_map"].snapshot
    site_map_spec = default_plot_spec(site_map.block.schema, PlotKind.IMAGE)
    if not isinstance(site_map_spec, ImagePlot):
        raise RuntimeError("Calibration SiteMap did not resolve to ImagePlot")
    site_map_spec = replace(
        site_map_spec,
        labels=PlotLabels(
            title="Reference average | calibrated sites",
            value="Counts",
        ),
    )

    fidelity = values["fidelity_site"].snapshot
    fidelity_spec = default_plot_spec(fidelity.block.schema, PlotKind.CURVE)
    if not isinstance(fidelity_spec, CurvePlot):
        raise RuntimeError("Calibration fidelity did not resolve to CurvePlot")
    fidelity_spec = replace(
        fidelity_spec,
        labels=PlotLabels(
            title="Per-site held-out fidelity",
            x="Site",
            y="Fidelity",
        ),
    )

    samples = values["readout_samples"].snapshot
    sample_axes = samples.block.schema.cell_schema.data_axes
    if len(sample_axes) != 1:
        raise ValueError("Calibration readout samples require one SITE data axis")
    distribution_spec = FacetGridPlot(
        AxisRef.data(sample_axes[0].axis_id.value),
        HistogramPlot(
            samples=(AxisRef.repeat(), AxisRef.point_rows()),
            labels=PlotLabels(x="Readout signal", y="Count"),
        ),
        labels=PlotLabels(title="Per-site readout distributions"),
    )

    return (
        (
            "site_map",
            "Site map",
            ImageFrame(site_map, _site_overlay(values["fidelity_centers"])),
            site_map_spec,
        ),
        ("fidelity", "Fidelity", fidelity, fidelity_spec),
        ("distribution", "Distribution", samples, distribution_spec),
    )


def _configure_distribution(session, outputs) -> None:
    sample_axis = (
        outputs["readout_samples"].snapshot.block.schema.cell_schema.data_axes[0]
    )
    session.set_facet_thresholds(
        _site_thresholds(outputs["fidelity_threshold"], sample_axis),
        display=False,
    )
    session.fit("bimodal_gaussian", live=False, fit_all_facets=True)


def export_calibration_plot_pages(
    destination: str | Path,
    outputs: Mapping[str, FinalDatasetOutput],
) -> tuple[Path, ...]:
    """Export the same three pages through headless ``PlotSession`` instances."""

    values = _calibration_outputs(outputs)
    root = durable_mkdir(Path(destination).expanduser().resolve())
    written: list[Path] = []
    for key, _title, source, spec in _calibration_pages(values):
        session = PlotSession(source, spec)
        try:
            if key == "distribution":
                _configure_distribution(session, values)
            stream = BytesIO()
            session.save(stream, format="png")
            path = root / f"{key}.png"
            atomic_write_bytes(path, stream.getvalue())
            written.append(path)
        finally:
            session.close()
    return tuple(written)


__all__ = ["export_calibration_plot_pages"]
