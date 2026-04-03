"""History-entry data structure for solver progress tracking."""

from dataclasses import dataclass

from .EvaluationResult import EvaluationResult


@dataclass
class HistoryEntry:
    """Stores one history record for solver progress tracking.
    :var time: Elapsed classical optimization time in seconds.
    :var job_ind: Job index where this history point was found when that concept exists, or ``None`` otherwise.
    :var optimizer_stats: Optimizer statistics available when this history point was recorded.
    :var result: Objective, penalty, and parameter data for this history point.
    """
    time: float
    job_ind: int | None
    optimizer_stats: dict[str, int | float]
    result: EvaluationResult
