"""Evaluation-result data structure and incumbent-comparison logic for continuous solver candidates."""

from dataclasses import dataclass
from typing import Self


@dataclass
class EvaluationResult:
    """Stores one evaluated continuous-parameter assignment and its metrics.
    :var inner_optimization_time: Elapsed time in seconds from the start of the inner optimization when this result was found.
    :var generator_statuses: Binary generator on/off bitstring associated with this result.
    :var params: Full continuous optimization vector.
    :var fun: Objective value without constraint penalty.
    :var penalty: Penalty from violated constraints.
    :var total: Penalized objective value.
    :var final: Whether this result is considered final by the solver (will not be further improved).
    :var success: Whether the solver reported success for this result when ``final`` is true.
    :var message: Solver-status message for this result when ``final`` is true.
    """
    inner_optimization_time: float
    generator_statuses: str
    params: list[float]
    fun: float
    penalty: float
    total: float
    final: bool = False
    success: bool | None = None
    message: str | None = None

    def is_better_than(self, other: Self | None, feasibility_tolerance: float) -> bool:
        """Returns whether this result should replace another incumbent.
        :param other: Current incumbent result, or ``None`` when no incumbent exists yet.
        :param feasibility_tolerance: Maximum penalty still treated as feasible.
        :return: Whether this result is better than ``other``.
        """
        if other is None:
            return True
        self_is_feasible = self.penalty <= feasibility_tolerance
        other_is_feasible = other.penalty <= feasibility_tolerance
        if self_is_feasible:
            if other_is_feasible:
                return self.fun < other.fun
            return True
        if other_is_feasible:
            return False
        return (self.penalty, self.fun) < (other.penalty, other.fun)
