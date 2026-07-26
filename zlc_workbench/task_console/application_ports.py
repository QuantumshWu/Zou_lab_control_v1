"""Closed application capabilities consumed by the generic TaskConsole."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from zlc_neutral_atom.catalog import DefinitionKey

from .capability import ConsoleCapabilityAttachment


@dataclass(frozen=True, slots=True)
class TaskConsoleApplicationPorts:
    """One immutable explicit attachment tuple.

    Concrete preparers and presenters are closed inside the attachments by the
    outer composition root.  The shell therefore cannot locate an Experiment,
    discover a package, or ask for a service by name.
    """

    attachments: tuple[ConsoleCapabilityAttachment, ...]
    _by_key: Mapping[DefinitionKey, ConsoleCapabilityAttachment] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        attachments = tuple(self.attachments)
        if not attachments:
            raise ValueError("TaskConsole requires at least one attachment")
        if any(
            not isinstance(value, ConsoleCapabilityAttachment)
            for value in attachments
        ):
            raise TypeError(
                "attachments must contain ConsoleCapabilityAttachment values"
            )
        by_key: dict[DefinitionKey, ConsoleCapabilityAttachment] = {}
        for attachment in attachments:
            if attachment.key in by_key:
                raise ValueError(
                    f"duplicate TaskConsole attachment {attachment.key}"
                )
            by_key[attachment.key] = attachment
        object.__setattr__(self, "attachments", attachments)
        object.__setattr__(self, "_by_key", MappingProxyType(by_key))

    def attachment_for(
        self,
        key: DefinitionKey,
    ) -> ConsoleCapabilityAttachment | None:
        if not isinstance(key, DefinitionKey):
            raise TypeError("key must be DefinitionKey")
        return self._by_key.get(key)


__all__ = ["TaskConsoleApplicationPorts"]
