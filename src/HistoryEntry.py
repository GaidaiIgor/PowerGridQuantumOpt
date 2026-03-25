"""History-entry data structure for solver progress tracking."""

from dataclasses import dataclass

from .EvaluationResult import EvaluationResult


@dataclass
class HistoryEntry:
    """Stores one history record for solver progress tracking.
    :var time: Elapsed classical optimization time in seconds.
    :var job_ind: Job index where this history point was found when that concept exists, or ``None`` otherwise.
    :var result: Objective, penalty, and parameter data for this history point.
    """
    time: float
    job_ind: int | None
    result: EvaluationResult
