"""Plotting helpers for power-grid optimization outputs."""
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cattrs.preconf.json import make_converter

from plots.general import Line, plot_general, save_figure
from src.HistoryEntry import HistoryEntry


def plot_probability_distribution(probs: dict[str, float], y_max: float = 1):
    """Plots a probability distribution as a bar chart.
    :param probs: Mapping of bitstring labels to probabilities.
    :param y_max: Maximum value shown on the y-axis.
    """
    xs, ys = list(zip(*sorted(probs.items())))
    plt.bar(xs, ys)
    plt.ylim((0, y_max))
    plt.ylabel("Probability")
    # save_figure()


def plot_instance_objective_histories():
    """Plots objective-over-time history lines for each instance from classical and hybrid runs."""
    data_path = Path(__file__).resolve().parent.parent / "data/5"
    classical_df = pd.read_csv(data_path / "scip" / ".solutions.csv")
    hybrid_df = pd.read_csv(data_path / "hybrid" / ".solutions.csv")
    converter = make_converter()

    lines = []
    first_classical = True
    for history_text in classical_df["history"].dropna():
        history = converter.loads(history_text, list[HistoryEntry])
        if len(history) == 0:
            continue
        xs = [entry.time for entry in history]
        ys = [entry.result.fun for entry in history]
        lines.append(Line(xs, ys, color="blue", label="Classical" if first_classical else "_nolabel_"))
        first_classical = False

    first_hybrid = True
    for history_text in hybrid_df["history"].dropna():
        history = converter.loads(history_text, list[HistoryEntry])
        if len(history) == 0:
            continue
        xs = [entry.time for entry in history]
        ys = [entry.result.fun for entry in history]
        lines.append(Line(xs, ys, color="red", label="Hybrid" if first_hybrid else "_nolabel_"))
        first_hybrid = False

    plot_general(lines, axis_labels=("Time [s]", "Objective"), boundaries=(None, None, 1, 17))
    save_figure()


def plot_polar_vs_rectangular():
    """Plots normalized objective histories for polar and rectangular CasADi results on the 5-generator dataset."""
    plot_histories([5], np.linspace(0, 1800, 50).tolist(), ["casadi", "casadi_rectangular"])


def plot_average_normalized_objective_histories():
    """Plots average normalized objective histories for the default solver comparison."""
    plot_histories([13], np.linspace(0, 3600, 50).tolist(), ["scip", "smac", "uniform", "hybrid"], "hybrid")


def plot_histories(num_generators: Sequence[int], grid_times: Sequence[float], solver_ids: Sequence[str], ref_solver: str | None = None):
    """Plots average normalized objective histories for configured solvers.
    :param num_generators: Generator counts whose datasets should be plotted together.
    :param grid_times: Uniform time grid used to align objective histories.
    :param solver_ids: Solver ids whose CSV files should be loaded from subfolders inside each dataset folder.
    :param ref_solver: Solver whose per-instance last history time limits the compared solvers, or ``None`` to disable trimming.
    """
    solver_names = {"scip": "SCIP", "smac": "SMAC", "uniform": "Uniform", "hybrid": "Hybrid"}
    infeasible_tolerance = 1e-10
    instance_ids = list(range(120))
    lines = []
    labeled_solvers = set()
    for num_gens_ind, num_gens in enumerate(num_generators):
        data_path = Path(__file__).resolve().parent.parent / f"data/{num_gens}"
        solver_histories = {}
        for solver_id in solver_ids:
            csv_path = data_path / solver_id / ".solutions.csv"
            if csv_path.exists():
                solver_histories[solver_id] = _load_solver_histories(csv_path, infeasible_tolerance)
        _trim_histories_to_ref_solver(solver_histories, ref_solver)
        best_objectives = _get_best_objectives(instance_ids, solver_histories)
        for solver_index, solver_id in enumerate(solver_ids):
            if solver_id not in solver_histories:
                continue
            curve = _get_average_normalized_curve(grid_times, instance_ids, solver_histories[solver_id], best_objectives)
            label = solver_names.get(solver_id, solver_id) if solver_id not in labeled_solvers else "_nolabel_"
            lines.append(Line(grid_times, curve, color=solver_index, marker=num_gens_ind, label=label))
            labeled_solvers.add(solver_id)
    plot_general(lines, axis_labels=("Time [s]", "Normalized Objective"), boundaries=(None, None, 0.9, 1.01))
    save_figure()


def _load_solver_histories(csv_path: Path, infeasible_tolerance: float) -> dict[int, list[HistoryEntry] | None]:
    """Loads solver histories grouped by instance from a CSV file.
    :param csv_path: Path to the solver CSV file.
    :param infeasible_tolerance: Maximum penalty still treated as feasible.
    :return: Mapping from instance id to sorted history entries, or ``None`` when the CSV history is null.
    """
    df = pd.read_csv(csv_path)
    converter = make_converter()
    histories = {}
    for instance, history_text in zip(df["instance"], df["history"]):
        if pd.isna(history_text):
            histories[instance] = None
            continue
        histories[instance] = [entry for entry in converter.loads(history_text, list[HistoryEntry]) if entry.result.penalty <= infeasible_tolerance]
    return histories


def _trim_histories_to_ref_solver(solver_histories: dict[str, dict[int, list[HistoryEntry] | None]], ref_solver: str | None) -> None:
    """Trims non-reference histories to the reference solver's last per-instance timestamp.
    :param solver_histories: Histories grouped by solver name and then by instance.
    :param ref_solver: Solver whose last per-instance timestamp defines the cutoff, or ``None`` to disable trimming.
    """
    if ref_solver is None or ref_solver not in solver_histories:
        return
    ref_cutoff_times = {instance: history[-1].time for instance, history in solver_histories[ref_solver].items() if history}
    for solver_id, histories in solver_histories.items():
        if solver_id == ref_solver:
            continue
        for instance, history in histories.items():
            if history and instance in ref_cutoff_times:
                histories[instance] = [entry for entry in history if entry.time <= ref_cutoff_times[instance]]


def _get_best_objectives(instance_ids: list[int], solver_histories: dict[str, dict[int, list[HistoryEntry] | None]]) -> dict[int, float | None]:
    """Finds the best known feasible objective per instance.
    :param instance_ids: Instance ids to include in aggregation.
    :param solver_histories: Histories grouped by solver name and then by instance.
    :return: Mapping from instance id to the lowest objective found by any loaded solver.
    """
    best_objectives = {}
    for instance in instance_ids:
        solver_objectives = [history[-1].result.fun for histories in solver_histories.values() if (history := histories.get(instance))]
        best_objectives[instance] = min(solver_objectives) if len(solver_objectives) > 0 else None
    return best_objectives


def _get_average_normalized_curve(grid_times: Sequence[float], instance_ids: list[int], solver_histories: dict[int, list[HistoryEntry] | None],
                                  best_objectives: dict[int, float]) -> np.ndarray:
    """Computes average normalized objective curve on a uniform time grid.
    :param grid_times: Uniform time grid used for alignment.
    :param instance_ids: Instance ids included in averaging denominator.
    :param solver_histories: Histories for one solver grouped by instance.
    :param best_objectives: Best known objective per instance across the loaded solvers.
    :return: Average normalized objective values for each grid time.
    """
    totals = np.zeros(len(grid_times))
    feasible_count = 0
    for instance in instance_ids:
        if (best_objective := best_objectives[instance]) is None:
            continue
        feasible_count += 1
        history = solver_histories.get(instance)
        if not history:
            continue
        history_times = np.array([entry.time for entry in history])
        normalized_objectives = np.array([best_objective / entry.result.fun for entry in history])
        indices = np.searchsorted(history_times, grid_times, side="right") - 1
        totals[indices >= 0] += normalized_objectives[indices[indices >= 0]]
    return totals / feasible_count


if __name__ == "__main__":
    # plot_instance_objective_histories()
    # plot_polar_vs_rectangular()
    plot_average_normalized_objective_histories()
    plt.show()
