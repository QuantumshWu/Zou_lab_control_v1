"""Desktop composition contracts with no eager frontend side effects.

Notebook hardware composition still imports a small transitional workbench
submodule.  Keeping this package facade lazy prevents that import from loading
the renderer or workspace graph; the remaining ownership move is a separate
migration slice rather than a hidden runtime dependency.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "InstallationAsset": ("asset_map", "InstallationAsset"),
    "InstallationAssetMap": ("asset_map", "InstallationAssetMap"),
    "CatalogEntry": ("legacy", "CatalogEntry"),
    "CatalogRoute": ("legacy", "CatalogRoute"),
    "CatalogRouter": ("legacy", "CatalogRouter"),
    "LegacyHandoffTimeout": ("legacy", "LegacyHandoffTimeout"),
    "SerializedLegacyAggBridge": ("legacy", "SerializedLegacyAggBridge"),
    "LegacyDeviceNotRegistered": (
        "legacy_runtime",
        "LegacyDeviceNotRegistered",
    ),
    "LegacyDeviceRegistration": (
        "legacy_runtime",
        "LegacyDeviceRegistration",
    ),
    "LegacyDeviceRegistry": ("legacy_runtime", "LegacyDeviceRegistry"),
    "LegacyNodeAlreadyManaged": (
        "legacy_runtime",
        "LegacyNodeAlreadyManaged",
    ),
    "LegacyNodeStartFailed": ("legacy_runtime", "LegacyNodeStartFailed"),
    "LegacyNodeStarted": ("legacy_runtime", "LegacyNodeStarted"),
    "LegacyRuntimeTransition": (
        "legacy_runtime",
        "LegacyRuntimeTransition",
    ),
    "LegacyRunHandle": ("legacy_runtime", "LegacyRunHandle"),
    "LegacyRuntimeFence": ("legacy_runtime", "LegacyRuntimeFence"),
    "LegacyStopReceipt": ("legacy_runtime", "LegacyStopReceipt"),
    "LegacyStopStatus": ("legacy_runtime", "LegacyStopStatus"),
    "PulseCommandPort": ("pulse_control", "PulseCommandPort"),
    "PulseTargetDescriptor": ("pulse_control", "PulseTargetDescriptor"),
    "BoardController": ("workspace", "BoardController"),
    "BoardModel": ("workspace", "BoardModel"),
    "BoardPublishPort": ("workspace", "BoardPublishPort"),
    "CoherenceSourceBinding": ("workspace", "CoherenceSourceBinding"),
    "PanelHost": ("workspace", "PanelHost"),
    "PanelSlot": ("workspace", "PanelSlot"),
    "RunHandleStatusBinding": ("workspace", "RunHandleStatusBinding"),
    "RunStatusView": ("workspace", "RunStatusView"),
    "WorkspaceModel": ("workspace", "WorkspaceModel"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = sorted(_LAZY_EXPORTS)
