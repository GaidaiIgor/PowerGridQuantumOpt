"""History-entry data structure for solver progress tracking."""

from dataclasses import dataclass

from .EvaluationResult import EvaluationResult


@dataclass
class HistoryEntry:
    """Stores one history record for solver progress tracking.
    :var time: Elapsed classical optimization time in seconds.
    :var expectation_jobs: Number of expectation-evaluation quantum jobs completed at this history point, or ``None`` when unavailable.
    :var fidelity_jobs: Number of QNSPSA fidelity-evaluation quantum jobs completed at this history point, or ``None`` when unavailable.
    :var optimizer_stats: Optimizer statistics available when this history point was recorded.
    :var result: Objective, violation, and parameter data for this history point."""
    time: float
    expectation_jobs: int | None
    fidelity_jobs: int | None
    optimizer_stats: dict[str, int | float]
    result: EvaluationResult
