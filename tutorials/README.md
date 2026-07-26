# Neutral-atom tutorial

There is one user tutorial: [`neutral_atom_tutorial.ipynb`](neutral_atom_tutorial.ipynb).
It runs the current virtual installation through inspect/capture, provenance,
site-map/PSF calibration, per-model thresholds and fidelity, one-event
occupancy, a named-axis autonomous MOT scan, release-recapture survival, and
the current GUIs. The current release-recapture product publishes the survival
curve; a public capture-radius-to-temperature fit/artifact owner is not yet
available, so the tutorial does not invent a µK result.

Start it with `..\start_tutorials_jupyter_lab.bat`. For real-hardware setup use
[`../docs/REAL_HARDWARE_BRINGUP_zh.md`](../docs/REAL_HARDWARE_BRINGUP_zh.md) and
`../fpga/run_server.bat`. The published real composition is pulse-only; complete
qCMOS + remote-sequencer readout remains unavailable and is not implied by this
virtual tutorial.

Unless `ZLC_TUTORIAL_WORKSPACE` is set explicitly, runtime artifacts go to the
project-owned `_output/tutorials/neutral-atom/` tree; the tutorial directory
itself remains source-only.
