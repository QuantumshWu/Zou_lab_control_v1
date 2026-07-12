# Pulse documents

The checked-in JSON files are immutable `zlc_pulse.PulseDocument/v2`
authoring documents. They are source data, not FPGA projects or runtime
snapshots. A document owns one explicit `PulseTarget`, stable period and
parameter identities, literal nominal values, and optional frozen scan rows.

## Physical target

`deployed_target.json` is the standalone canonical target used by the server.
Every checked-in hardware document embeds that same complete 62-lane target;
`visible_ports` alone selects the small authoring view appropriate to a task.
Raw lanes remain in bitstream order while logical ports own their physical
lanes, DAC encoding, safe code, bus index, and latch-clock topology.

Physical identity and display labels are deliberately distinct. For example,
the camera trigger has physical `PortKey`/raw lane `ch11` and human label
`emCCD`; trap is `ch09` with label `trap`. Formal trigger schedules therefore
name `ch11`, while an editor may render the friendlier label.

Binding a document to a live target is a verified physical re-key operation.
It cannot guess a port by name or shape, reclassify an active lane, or inject a
new hardware-clock output. The compiler also verifies the live ABI before
producing a wire image.

The deployed server accepts only the approved full-board target ABI. It also
checks the frozen layout explicitly: 18 leading TTL outputs, four 10-bit DAC
lane groups, their four fixed latch-clock positions, low engine safe states,
and the offset-binary midpoint DAC safe code. Each AXI/UART hardware session is
bound to the full target and, before any artifact I/O, validates the TargetIR
against that topology, the clock and geometry, and a deterministic repack of
the wire image. The running FPGA continues to use the existing
geometry-fingerprint handshake. That handshake does not attest bitstream
contents or the XDC pin map; the approved frozen `.bit` remains a deployment-SOP
responsibility, and this software does not synthesize or program a replacement.

## Typed scan and API bindings

Each adjustable field has exactly one typed `PulseFieldRef`:

- period duration: `(duration, PeriodId)`;
- DAC action: `(dac, PeriodId, PortKey)`;
- fixed output delay: `(delay, PortKey)`.

A `ScanParameter` or `ApiParameter` binds its semantic `ParameterId` directly
to that field. There are no authored positional variables or expression
strings. `FrozenScanTable.columns` stores ParameterIds, so reordering parameter
objects cannot move a numeric column onto another physical field.

Scan generation is explicit and reproducible:

```python
from dataclasses import replace

from zlc_pulse import attach_scan_recipe, freeze_scan_table, load_pulse_document

document = load_pulse_document("pulses/release_recapture.json")
table, report = freeze_scan_table(
    document,
    columns=("t_off",),
    raw_rows=((20,), (40,), (60,)),
)
document = replace(document, scan_table=table)
document = attach_scan_recipe(
    document,
    source="scan_columns = {'t_off': [20, 40, 60]}\n",
    generated_columns={"t_off": (20, 40, 60)},
)
```

The normalizer reports adjustments per ParameterId. Recipe provenance is
accepted only when its named output regenerates the current frozen table; its
digest also binds the FieldRefs, units, target clock grid, and relevant DAC
topology. A manual or imported frozen table may intentionally omit a recipe.

## Exact timing and analog actions

Authoritative period, delay, and API values must already lie exactly on the
document clock grid. They are rejected rather than silently rounded. Scan-table
construction is the one explicit normalization boundary and returns the exact
played values plus a per-parameter report.

Each period embeds zero or more `AnalogStep` values. `edge` jumps to a signed
DAC code at the period start, `ramp` reaches the code over the period, and an
absent step means hold. Raw DAC member lanes cannot also appear as TTL controls.
The compiler alone converts signed values into target wire codes and emits the
separate hardware bus-segment table.

## Compilation

```python
from zlc_pulse import (
    PulseExecutionForm,
    compile_pulse_artifact,
    load_pulse_document,
)

document = load_pulse_document("pulses/mot_field_template.json")
artifact = compile_pulse_artifact(
    document,
    clock_hz=50_000_000,
    execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
)
```

Compilation lowers stable field identities to dense wire slots privately. The
result contains the immutable target IR, packed current-bitstream image, and—
for finite formal runs—deterministic trigger schedules. Execution policy such
as continuous monitoring or repeated scan sweeps belongs to the runtime
request, never to the saved authoring document.

Before packing, one pulse-owned capability gate rejects every integer field the
current serializer would otherwise mask or truncate. Delay FIFO capacity is
checked against the physical instruction stream with the frozen RTL's exact
push/pop boundary (`event distance < delay`). Compact loop bodies stay
compressed, point-dependent affine loop membership is evaluated per point, a
finite DAC run includes its terminal SAFE descriptor, and cyclic digital
capacity distinguishes the one-time FIRE transition from the steady sweep.
The algorithm is bounded by the stored edge/point data, not by `loop_count`.

Trigger schedules and playback objects are optional materialized projections.
They return immediately when no channel is requested and reject projections
above their explicit cardinality limit; the authoritative compact TargetIR and
the hardware execution path remain valid without expanding billions of loop
iterations in host memory.
