"""Device contracts, virtual devices, and hardware adapters."""

from .base import BaseDevice, CameraDevice, SequencerDevice, TrapArrayDevice, validate_device_contract
from .qcmos import DEFAULT_DCAM_MODULE, QCMOSCamera, QCMOSConfig
from .sequencer import (
    ManualSequencer,
    RemoteSequencer,
    RuntimeSequenceProgram,
    RuntimeSequencer,
    SequencerService,
    VerilogSequencer,
    compile_pulse_table_runtime_program,
    compile_runtime_program,
    compile_runtime_program_for_payload,
    serve_runtime_sequencer,
)
from .virtual import DEFAULT_CHANNELS, VirtualCamera, VirtualSequencer, VirtualTrapArray, virtual_config

_REGISTRY_EXPORTS = {
    "DEVICE_CLASSES",
    "DeviceSet",
    "apply_device_overrides",
    "available_device_configs",
    "device_class_registry",
    "device_config_dir",
    "load_devices",
    "read_config",
    "register_device_class",
    "resolve_class",
}

_PULSE_STREAMER_EXPORTS = {
    "PulseStreamerHDLFiles",
    "PulseStreamerProbeNames",
    "VivadoPulseStreamerSession",
    "generate_pulse_streamer_core",
    "generate_pulse_streamer_top_example",
    "validate_pulse_streamer_program",
    "write_pulse_streamer_hdl_bundle",
    "write_vivado_pulse_streamer_tcl",
}


def __getattr__(name: str):
    if name == "CommandSequencerBackend":
        from .sequencer_server import CommandSequencerBackend

        return CommandSequencerBackend
    if name == "run_sequencer_server":
        from .sequencer_server import run_server

        return run_server
    if name == "build_sequencer_server_arg_parser":
        from .sequencer_server import build_arg_parser

        return build_arg_parser
    if name in _REGISTRY_EXPORTS:
        from . import registry

        return getattr(registry, name)
    if name in _PULSE_STREAMER_EXPORTS:
        from . import fpga_pulse_streamer

        return getattr(fpga_pulse_streamer, name)
    raise AttributeError(name)

__all__ = [
    "BaseDevice",
    "CameraDevice",
    "CommandSequencerBackend",
    "DEFAULT_CHANNELS",
    "DEFAULT_DCAM_MODULE",
    "DEVICE_CLASSES",
    "DeviceSet",
    "ManualSequencer",
    "PulseStreamerHDLFiles",
    "PulseStreamerProbeNames",
    "VivadoPulseStreamerSession",
    "QCMOSCamera",
    "QCMOSConfig",
    "RemoteSequencer",
    "RuntimeSequenceProgram",
    "RuntimeSequencer",
    "SequencerDevice",
    "SequencerService",
    "TrapArrayDevice",
    "VerilogSequencer",
    "VirtualCamera",
    "VirtualSequencer",
    "VirtualTrapArray",
    "apply_device_overrides",
    "available_device_configs",
    "build_sequencer_server_arg_parser",
    "device_class_registry",
    "device_config_dir",
    "generate_pulse_streamer_core",
    "generate_pulse_streamer_top_example",
    "load_devices",
    "read_config",
    "register_device_class",
    "resolve_class",
    "run_sequencer_server",
    "validate_pulse_streamer_program",
    "validate_device_contract",
    "virtual_config",
    "compile_runtime_program",
    "compile_pulse_table_runtime_program",
    "compile_runtime_program_for_payload",
    "serve_runtime_sequencer",
    "write_pulse_streamer_hdl_bundle",
    "write_vivado_pulse_streamer_tcl",
]
