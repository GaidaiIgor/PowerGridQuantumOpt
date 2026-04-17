"""Evaluation-result data structure and incumbent-comparison logic for continuous solver candidates."""

from dataclasses import dataclass, field
from typing import Any, Self


@dataclass
class EvaluationResult:
    """Stores one evaluated continuous-parameter assignment and its metrics.
    :var generator_statuses: Binary generator on/off bitstring associated with this result.
    :var params: Full continuous optimization vector.
    :var fun: Objective value without scaled constraint violation.
    :var violation: Raw constraint-violation value.
    :var total: Objective value plus scaled violation.
    :var extra: Additional solver-specific metadata for this result.
    """
    generator_statuses: str
    params: list[float]
    fun: float
    violation: float
    total: float
    extra: dict[str, Any] = field(default_factory=dict)

    def is_better_than(self, other: Self | None, violation_tolerance: float) -> bool:
        """Returns whether this result should replace another incumbent.
        :param other: Current incumbent result, or ``None`` when no incumbent exists yet.
        :param violation_tolerance: Maximum violation still treated as feasible.
        :return: Whether this result is better than ``other``.
        """
        if other is None:
            return True
        self_is_feasible = self.violation <= violation_tolerance
        other_is_feasible = other.violation <= violation_tolerance
        if self_is_feasible:
            if other_is_feasible:
                return self.fun < other.fun
            return True
        if other_is_feasible:
            return False
        return (self.violation, self.fun) < (other.violation, other.fun)
