"""Private identity check shared by composition-owned device endpoints."""

from zlc_neutral_atom.runtime import BoundDevice


def require_current_endpoint_binding(
    binding: object,
    endpoint: str,
    binding_id: str | None,
    connection_generation: str | None,
) -> None:
    if not isinstance(binding, BoundDevice):
        raise TypeError(f"{endpoint} endpoint requires BoundDevice")
    if binding_id is None:
        return
    if (
        binding.binding_id != binding_id
        or binding.connection_generation != connection_generation
    ):
        raise RuntimeError(f"{endpoint} endpoint binding generation changed")
