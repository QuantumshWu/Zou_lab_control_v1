"""Shared runtime claims for concrete neutral-atom readout workloads."""

from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_storage import RepositoryRootLease, RepositoryRootLeaseBorrow


def acquire_repository_borrows(
    *leases: RepositoryRootLease,
) -> tuple[RepositoryRootLeaseBorrow, ...]:
    """Atomically-or-rollback the repository holds for one flat analysis."""

    if any(type(lease) is not RepositoryRootLease for lease in leases):
        raise TypeError("leases must contain exact RepositoryRootLease values")
    held: list[RepositoryRootLeaseBorrow] = []
    try:
        for lease in leases:
            held.append(lease.borrow())
        return tuple(held)
    except BaseException as primary:
        try:
            release_repository_borrows(tuple(held))
        except BaseException as close_error:
            record_secondary_failure(
                primary,
                "repository borrow rollback also failed",
                close_error,
            )
        raise


def release_repository_borrows(
    borrows: tuple[RepositoryRootLeaseBorrow, ...],
) -> None:
    """Release every hold in reverse order without hiding later failures."""

    first: BaseException | None = None
    for borrow in reversed(tuple(borrows)):
        if type(borrow) is not RepositoryRootLeaseBorrow:
            error: BaseException = TypeError(
                "borrows must contain exact RepositoryRootLeaseBorrow values"
            )
        else:
            try:
                borrow.close()
                continue
            except BaseException as caught:
                error = caught
        if first is None:
            first = error
        else:
            record_secondary_failure(
                first,
                "another repository borrow also failed to close",
                error,
            )
    if first is not None:
        raise first


__all__ = [
    "acquire_repository_borrows",
    "release_repository_borrows",
]
