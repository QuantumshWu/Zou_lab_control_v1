# Pulse Presets

This directory stores checked-in `PulseTableState` JSON presets.  They are
frontend/API data, not hardware projects.

## `camera_imaging_40ch.json`

Default qCMOS camera-imaging preset for the 40-channel FPGA pulse-streamer.

- Hardware channels are `ch00..ch39` in FPGA bit order.
- The GUI shows only `ch00..ch03` by default to keep simple imaging pulses
  readable.
- Display labels are:
  - `ch00`: `trap`
  - `ch01`: `cooling`
  - `ch02`: `probe`
  - `ch03`: `qcm_trigger`
- Hidden or unconfigured channels stay off when uploaded to the 40-bit FPGA
  mask.
- `camera_exposure` uses `duration="x"` and `unit="str (ns)"`.
- The default `x_ns` is `19980000`, so the preset starts as the original
  19.98 ms exposure segment.  Changing `pulse.x` in a notebook changes this
  readout/probe exposure.
- The checked-in preset has no inner finite repeat bracket.  Its table head
  keeps `ch03` low, so it does not create the slow table-boundary qCMOS spike
  that can happen when a long finite bracket is nested inside
  `repeat_forever=True`.

Typical control-computer workflow:

```python
import Zou_lab_control.neutral_atom as na

exp = na.connect("remote_template", sequencer={"host": "192.168.0.20"})
pulse = exp.timing.bind_pulse("pulses/camera_imaging_40ch.json")

pulse.x = 2_000_000  # ns
pulse.on_pulse(wait=True, repeat_forever=False)

scan = exp.readout.detection_time(
    [2e-3, 4e-3, 8e-3],
    shots=30,
    pulse=pulse,
    live=False,
    display=True,
)
```

For repeated oscilloscope output from the API use:

```python
pulse.on_pulse(wait=False, repeat_forever=True)
```

In that mode the whole table repeats, so the intended `ch03` trigger appears
once per table period.  The Pulse GUI does not expose a separate whole-table
repeat switch; its preview still shows the whole table as `∞`, and its editable
repeat control is the period bracket.  For camera acquisition or finite
debugging, keep `repeat_forever=False` in API calls so the sequencer can report
done.  The qCMOS readout helpers convert this preset to a finite trigger sequence
for the requested frame count before waiting for the camera.
