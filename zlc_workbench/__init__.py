"""Desktop composition contracts; importing this package does not import Qt."""

from .asset_map import InstallationAsset, InstallationAssetMap
from .legacy import (
    CatalogEntry,
    CatalogRoute,
    CatalogRouter,
    LegacyHandoffTimeout,
    SerializedLegacyAggBridge,
)
from .legacy_runtime import (
    LegacyDeviceNotRegistered,
    LegacyDeviceRegistration,
    LegacyDeviceRegistry,
    LegacyNodeAlreadyManaged,
    LegacyNodeStartFailed,
    LegacyNodeStarted,
    LegacyRuntimeTransition,
    LegacyRunHandle,
    LegacyRuntimeFence,
    LegacyStopReceipt,
    LegacyStopStatus,
)
from .workspace import (
    BoardController,
    BoardModel,
    BoardPublishPort,
    CoherenceSourceBinding,
    PanelHost,
    PanelSlot,
    RunHandleStatusBinding,
    RunStatusView,
    WorkspaceModel,
)

__all__ = [
    "BoardController",
    "BoardModel",
    "BoardPublishPort",
    "CatalogEntry",
    "CatalogRoute",
    "CatalogRouter",
    "CoherenceSourceBinding",
    "InstallationAsset",
    "InstallationAssetMap",
    "LegacyHandoffTimeout",
    "LegacyDeviceNotRegistered",
    "LegacyDeviceRegistration",
    "LegacyDeviceRegistry",
    "LegacyNodeAlreadyManaged",
    "LegacyNodeStartFailed",
    "LegacyNodeStarted",
    "LegacyRuntimeTransition",
    "LegacyRunHandle",
    "LegacyRuntimeFence",
    "LegacyStopReceipt",
    "LegacyStopStatus",
    "PanelHost",
    "PanelSlot",
    "RunHandleStatusBinding",
    "RunStatusView",
    "SerializedLegacyAggBridge",
    "WorkspaceModel",
]
