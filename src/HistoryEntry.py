"""History-entry data structure for solver progress tracking."""

from dataclasses import dataclass

from .EvaluationResult import EvaluationResult


@dataclass
class HistoryEntry:
    """Stores one history record for solver progress tracking.
    :var time: Elapsed classical optimization time in seconds.
    :var num_jobs: Number of quantum jobs used up to this history point, or ``None`` for classical runs.
    :var evaluation_result: Objective, penalty, and parameter data for this history point.
    """
    time: float
    num_jobs: int | None
    evaluation_result: EvaluationResult
