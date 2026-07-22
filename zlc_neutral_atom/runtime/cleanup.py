"""Plan cleanup diagnostics from domain-owned session closure."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CleanupReport:
    """Errors raised while a plan closes its device-specific sessions."""

    errors: tuple[BaseException, ...] = ()

    def __post_init__(self) -> None:
        errors = tuple(self.errors)
        if any(not isinstance(error, BaseException) for error in errors):
            raise TypeError("cleanup errors must contain exceptions")
        object.__setattr__(self, "errors", errors)

    @classmethod
    def complete(
        cls,
        *,
        errors: tuple[BaseException, ...] = (),
    ) -> "CleanupReport":
        return cls(tuple(errors))
