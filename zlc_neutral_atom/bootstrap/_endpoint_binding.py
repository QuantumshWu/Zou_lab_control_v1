"""Private identity check shared by neutral-atom composition endpoints."""

from zlc_neutral_atom.runtime.ports import BoundDevice


def require_current_endpoint_binding(
    binding: object,
    endpoint: str,
    binding_instance_id: str | None,
) -> None:
    if not isinstance(binding, BoundDevice):
        raise TypeError(f"{endpoint} endpoint requires BoundDevice")
    if binding_instance_id is None:
        return
    if binding.binding_instance_id != binding_instance_id:
        raise RuntimeError(f"{endpoint} endpoint binding instance changed")
