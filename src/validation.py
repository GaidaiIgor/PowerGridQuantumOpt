"""Generic validation helpers shared across the package."""

from __future__ import annotations

from collections.abc import Sequence

def validate_bounds(
    name: str,
    value: float | Sequence[float],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    include_min: bool = True,
    include_max: bool = False,
    require_ordered: bool = True,
) -> None:
    """Validate scalar values or numeric sequences against configurable bounds."""
    lower_cmp = ">=" if include_min else ">"
    upper_cmp = "<=" if include_max else "<"

    def _check_one(label: str, x: float) -> None:
        if min_value is not None:
            lower_ok = x >= min_value if include_min else x > min_value
            if not lower_ok:
                raise ValueError(f"{label} must satisfy {lower_cmp} {min_value}, got {x}.")
        if max_value is not None:
            upper_ok = x <= max_value if include_max else x < max_value
            if not upper_ok:
                raise ValueError(f"{label} must satisfy {upper_cmp} {max_value}, got {x}.")

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 0:
            raise ValueError(f"{name} must not be empty.")
        if require_ordered:
            for idx in range(1, len(value)):
                if value[idx - 1] > value[idx]:
                    raise ValueError(f"{name} must be ordered (non-decreasing), got {value}.")
        for idx, item in enumerate(value):
            _check_one(f"{name}[{idx}]", item)
        return

    _check_one(name, value)
