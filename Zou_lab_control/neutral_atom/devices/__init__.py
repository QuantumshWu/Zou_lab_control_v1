"""Device contracts, virtual devices, and hardware adapters."""

from .base import BaseDevice, CameraDevice, SequencerDevice, TrapArrayDevice, validate_device_contract
from .qcmos import QCMOSCamera, QCMOSConfig
from .sequencer import (
    ManualSequencer,
    RemoteSequencer,
    RuntimeSequenceProgram,
    RuntimeSequencer,
    SequencerService,
    VerilogSequencer,
    compile_runtime_program,
    serve_runtime_sequencer,
)
from .virtual import DEFAULT_CHANNELS, VirtualCamera, VirtualSequencer, VirtualTrapArray, virtual_config

_REGISTRY_EXPORTS = {
    "DEVICE_CLASSES",
    "DeviceSet",
    "apply_device_overrides",
    "available_device_configs",
    "device_config_dir",
    "load_devices",
    "read_config",
    "resolve_class",
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
    raise AttributeError(name)

__all__ = [
    "BaseDevice",
    "CameraDevice",
    "CommandSequencerBackend",
    "DEFAULT_CHANNELS",
    "DEVICE_CLASSES",
    "DeviceSet",
    "ManualSequencer",
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
    "device_config_dir",
    "load_devices",
    "read_config",
    "resolve_class",
    "run_sequencer_server",
    "validate_device_contract",
    "virtual_config",
    "compile_runtime_program",
    "serve_runtime_sequencer",
]
