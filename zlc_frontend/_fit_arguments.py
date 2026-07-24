"""Safe, reversible text grammar for Figure Fit parameter constraints.

The editor deliberately accepts only comma-separated keyword assignments with
finite numeric literals.  It is parsed as syntax, never evaluated.  A bare
parameter name fixes that parameter (``center=50``); the ``_initial``,
``_lower`` and ``_upper`` suffixes author the corresponding solver constraint.
An empty string means the model's automatic initializer and domains.
"""

from __future__ import annotations

import ast
import math

from zlc_data import FitParameterConstraint


_FIELDS = ("initial", "lower", "upper")


def _finite_number(node: ast.AST, *, keyword: str) -> float:
    sign = 1.0
    value_node = node
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op,
        (ast.UAdd, ast.USub),
    ):
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        value_node = node.operand
    if (
        not isinstance(value_node, ast.Constant)
        or isinstance(value_node.value, bool)
        or not isinstance(value_node.value, (int, float))
    ):
        raise ValueError(
            f"{keyword} must be a numeric literal, not an expression"
        )
    try:
        value = sign * float(value_node.value)
    except OverflowError as error:
        raise ValueError(f"{keyword} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{keyword} must be finite")
    return 0.0 if value == 0.0 else value


def _constraint_target(
    keyword: str,
    parameter_names: frozenset[str],
) -> tuple[str, str]:
    if keyword in parameter_names:
        return keyword, "fixed"
    for field in _FIELDS:
        suffix = f"_{field}"
        if keyword.endswith(suffix):
            parameter_name = keyword[: -len(suffix)]
            if parameter_name in parameter_names:
                return parameter_name, field
    choices = ", ".join(sorted(parameter_names))
    raise ValueError(
        f"unknown Fit argument {keyword!r}; model parameters: {choices}"
    )


def parse_fit_arguments(
    text: str,
    parameter_names: tuple[str, ...],
) -> tuple[FitParameterConstraint, ...]:
    """Parse the Fit line into typed constraints without executing input."""

    if not isinstance(text, str):
        raise TypeError("Fit arguments must be text")
    names = tuple(parameter_names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name.isidentifier() for name in names)
    ):
        raise ValueError("Fit model parameter names must be unique identifiers")
    stripped = text.strip()
    if not stripped:
        return ()
    try:
        expression = ast.parse(f"__fit__({stripped})", mode="eval").body
    except SyntaxError as error:
        detail = error.msg or "invalid syntax"
        raise ValueError(
            "Fit arguments require comma-separated assignments such as "
            f"center=50, sigma_lower=0 ({detail})"
        ) from None
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.func.id != "__fit__"
        or expression.args
    ):
        raise ValueError("Fit arguments accept keyword assignments only")

    known = frozenset(names)
    fields_by_parameter: dict[str, dict[str, float]] = {}
    seen_keywords: set[str] = set()
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("Fit arguments do not accept ** expansion")
        if keyword.arg in seen_keywords:
            raise ValueError(f"duplicate Fit argument {keyword.arg!r}")
        seen_keywords.add(keyword.arg)
        parameter_name, field = _constraint_target(keyword.arg, known)
        fields = fields_by_parameter.setdefault(parameter_name, {})
        if field in fields:
            raise ValueError(
                f"Fit parameter {parameter_name!r} assigns {field} twice"
            )
        fields[field] = _finite_number(keyword.value, keyword=keyword.arg)

    constraints = []
    for parameter_name in names:
        fields = fields_by_parameter.get(parameter_name)
        if fields:
            constraints.append(FitParameterConstraint(parameter_name, **fields))
    return tuple(constraints)


def _number_text(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)


def format_fit_arguments(
    constraints: tuple[FitParameterConstraint, ...],
    parameter_names: tuple[str, ...],
) -> str:
    """Return a parser-round-trippable line ordered by model parameters."""

    prepared = tuple(constraints)
    if any(not isinstance(value, FitParameterConstraint) for value in prepared):
        raise TypeError("constraints must contain FitParameterConstraint values")
    by_name = {value.parameter_name: value for value in prepared}
    names = tuple(parameter_names)
    if set(by_name) - set(names):
        raise ValueError("constraints name parameters outside the model")
    assignments = []
    for parameter_name in names:
        constraint = by_name.get(parameter_name)
        if constraint is None:
            continue
        if constraint.fixed is not None:
            assignments.append(
                f"{parameter_name}={_number_text(constraint.fixed)}"
            )
        if constraint.initial is not None:
            assignments.append(
                f"{parameter_name}_initial={_number_text(constraint.initial)}"
            )
        if constraint.lower is not None:
            assignments.append(
                f"{parameter_name}_lower={_number_text(constraint.lower)}"
            )
        if constraint.upper is not None:
            assignments.append(
                f"{parameter_name}_upper={_number_text(constraint.upper)}"
            )
    return ", ".join(assignments)


__all__ = ["format_fit_arguments", "parse_fit_arguments"]
