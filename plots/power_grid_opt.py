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


def plot_average_histories():
    """Plots average normalized objective histories for the default solver comparison."""
    num_generators = [10, 11, 12, 13]
    solver_ids = ["hybrid/nl_1", "uniform"]
    ref_solver = "hybrid/nl_1"
    solver_names = {"scip": "SCIP", "smac": "SMAC", "uniform": "Uniform", "hybrid/nl_1": "Hybrid", "hybrid/nl_2": "Hybrid L2"}

    history_data = get_history_data_2(num_generators, solver_ids, ref_solver)
    lines = []
    labeled_solvers = set()
    for num_gens_ind, num_gens in enumerate(num_generators):
        for solver_ind, solver_id in enumerate(solver_ids):
            xs, ys = history_data[num_gens][solver_id]
            label = solver_names.get(solver_id, solver_id) if solver_id not in labeled_solvers else "_nolabel_"
            lines.append(Line(xs, ys, color=solver_ind, marker=num_gens_ind, label=label))
            labeled_solvers.add(solver_id)
    plot_general(lines, axis_labels=("Time [s]", "Normalized Objective"), boundaries=(None, None, 0.9, 1.01))
    save_figure()


def plot_history_diff():
    """Plots differences between the two configured solver curves."""
    num_generators = [13]
    solver_ids = ["hybrid/nl_1", "uniform"]
    ref_solver = "hybrid/nl_1"

    history_data = get_history_data_2(num_generators, solver_ids, ref_solver)
    lines = []
    for num_gens_ind, num_gens in enumerate(num_generators):
        xs, first_ys = history_data[num_gens][solver_ids[0]]
        _, second_ys = history_data[num_gens][solver_ids[1]]
        lines.append(Line(xs, first_ys - second_ys, color=num_gens_ind, marker=0, label=str(num_gens)))
    lines.append(Line([0, 10000], [0, 0], color="black", marker="none", style="--"))
    plot_general(lines, axis_labels=("Time [s]", "Normalized Objective Difference"), boundaries=(0, xs[-1], -0.1, 0.1))
    save_figure()


def get_history_data_2(num_generators: list[int], solver_ids: list[str], ref_solver: str) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Collects average normalized objective histories across datasets.
    :param num_generators: Generator counts whose datasets should be loaded.
    :param solver_ids: Solver ids whose CSV files should be loaded from subfolders inside each dataset folder.
    :param ref_solver: Solver id used to select the fastest instances and time grid in each dataset.
    :return: History data keyed by generator count and then solver id.
    """
    histories = {}
    for num_gens in num_generators:
        full_df = pd.read_csv(Path(__file__).resolve().parent.parent / f"data/{num_gens}/{ref_solver}/.solutions.csv")
        fastest_100_df = full_df.loc[pd.to_numeric(full_df["classical_opt_time"], errors="coerce").nsmallest(100).index]
        time_grid = np.linspace(0, max(fastest_100_df["classical_opt_time"]), 50)
        instance_ids = fastest_100_df["instance"].astype(int).tolist()
        data = get_history_data(num_gens, time_grid, instance_ids, solver_ids)
        histories[num_gens] = {solver_id: (time_grid, data[solver_id]) for solver_id in solver_ids}
    return histories


def get_history_data(num_generators: int, grid_times: Sequence[float], instance_ids: Sequence[int], solver_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Computes average normalized objective curves for one dataset.
    :param num_generators: Generator count whose dataset folder should be loaded.
    :param grid_times: Uniform time grid used to align objective histories.
    :param instance_ids: Instance ids included in the average normalized objective curves.
    :param solver_ids: Solver ids whose CSV files should be loaded from subfolders inside the dataset folder.
    :return: Average normalized objective curve for each loaded solver keyed by solver id.
    """
    violation_tolerance = 1e-10
    data_path = Path(__file__).resolve().parent.parent / f"data/{num_generators}"
    solver_histories = {}
    for solver_id in solver_ids:
        csv_path = data_path / solver_id / ".solutions.csv"
        if csv_path.exists():
            solver_histories[solver_id] = _load_solver_histories(csv_path, violation_tolerance, grid_times[-1])
    best_objectives = _get_best_objectives(instance_ids, solver_histories)
    return {solver_id: _get_average_normalized_curve(grid_times, instance_ids, histories, best_objectives) for solver_id, histories in solver_histories.items()}


def _load_solver_histories(csv_path: Path, violation_tolerance: float, max_time: float) -> dict[int, list[HistoryEntry] | None]:
    """Loads solver histories grouped by instance from a CSV file.
    :param csv_path: Path to the solver CSV file.
    :param violation_tolerance: Maximum violation still treated as feasible.
    :param max_time: Maximum time of loaded history entries.
    :return: Mapping from instance id to history entries, or ``None`` when the CSV history is null.
    """
    df = pd.read_csv(csv_path)
    converter = make_converter()
    histories = {}
    for instance, history_text in zip(df["instance"], df["history"]):
        if pd.isna(history_text):
            histories[instance] = None
            continue
        histories[instance] = [entry for entry in converter.loads(history_text, list[HistoryEntry])
                               if entry.result.violation <= violation_tolerance and entry.time <= max_time]
    return histories


def _get_best_objectives(instance_ids: Sequence[int], solver_histories: dict[str, dict[int, list[HistoryEntry] | None]]) -> dict[int, float | None]:
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


def _get_average_normalized_curve(grid_times: Sequence[float], instance_ids: Sequence[int], solver_histories: dict[int, list[HistoryEntry] | None],
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
    # plot_average_histories()
    plot_history_diff()
    plt.show()
