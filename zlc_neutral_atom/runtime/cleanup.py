"""Plan cleanup diagnostics from domain-owned session closure."""

from __future__ import annotations

from collections.abc import Callable
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


def run_cleanup_steps(
    *steps: Callable[[], CleanupReport],
) -> CleanupReport:
    """Run every physical cleanup step and aggregate all failures."""

    errors: list[BaseException] = []
    for step in steps:
        try:
            report = step()
            if not isinstance(report, CleanupReport):
                raise TypeError("hardware cleanup step must return CleanupReport")
        except BaseException as error:
            errors.append(error)
            continue
        errors.extend(report.errors)
    return CleanupReport.complete(errors=tuple(errors))
