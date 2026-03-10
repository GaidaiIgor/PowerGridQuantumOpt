"""Plotting helpers for power-grid optimization outputs."""
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.general import Line, plot_general, save_figure


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


def plot_instance_objective_histories() -> None:
    """Plots objective-over-time history lines for each instance from classical and hybrid runs."""
    data_path = Path(__file__).resolve().parent.parent / "data/5"
    classical_df = pd.read_csv(data_path / ".solutions_classical.csv")
    hybrid_df = pd.read_csv(data_path / ".solutions_hybrid.csv")

    lines = []
    first_classical = True
    for history_text in classical_df["history"].dropna():
        history = ast.literal_eval(history_text)
        if len(history) == 0:
            continue
        xs = [entry["time"] for entry in history]
        ys = [entry["objective"] for entry in history]
        lines.append(Line(xs, ys, color="blue", label="Classical" if first_classical else "_nolabel_"))
        first_classical = False

    first_hybrid = True
    for history_text in hybrid_df["history"].dropna():
        history = ast.literal_eval(history_text)
        if len(history) == 0:
            continue
        xs = [entry["time"] for entry in history]
        ys = [entry["objective"] for entry in history]
        lines.append(Line(xs, ys, color="red", label="Hybrid" if first_hybrid else "_nolabel_"))
        first_hybrid = False

    plot_general(lines, axis_labels=("Time [s]", "Objective"), boundaries=(None, None, 1, 17))
    save_figure()


def plot_average_normalized_objective_histories() -> None:
    """Plots average normalized objective histories for classical and hybrid solvers."""
    num_generators_list = [5, 10, 11]
    grid_times = np.arange(0, 1800, 5)
    lines = []
    for i, num_generators in enumerate(num_generators_list):
        data_path = Path(__file__).resolve().parent.parent / f"data/{num_generators}"
        classical_histories = _load_solver_histories(data_path / ".solutions_classical.csv")
        hybrid_histories = _load_solver_histories(data_path / ".solutions_hybrid.csv")
        instance_ids = sorted(set(classical_histories) | set(hybrid_histories))
        best_objectives = _get_best_objectives(instance_ids, classical_histories, hybrid_histories)
        line_style = 0
        classical_curve = _get_average_normalized_curve(grid_times, instance_ids, classical_histories, best_objectives)
        hybrid_curve = _get_average_normalized_curve(grid_times, instance_ids, hybrid_histories, best_objectives)
        lines.append(Line(grid_times, classical_curve, color="blue", style=line_style, label="Classical" if i == 0 else "_nolabel_"))
        lines.append(Line(grid_times, hybrid_curve, color="red", style=line_style, label="Hybrid" if i == 0 else "_nolabel_"))
    plot_general(lines, axis_labels=("Time [s]", "Normalized Objective"), boundaries=(None, None, 0, 1))
    save_figure()


def _load_solver_histories(csv_path: Path) -> dict[int, list[dict[str, float | int]] | None]:
    """Loads solver histories grouped by instance from a CSV file.
    :param csv_path: Path to the solver CSV file.
    :return: Mapping from instance id to sorted history entries, or ``None`` when the CSV history is null.
    """
    df = pd.read_csv(csv_path)
    histories = {}
    for instance, history_text in zip(df["instance"], df["history"]):
        if pd.isna(history_text):
            histories[instance] = None
            continue
        history = ast.literal_eval(history_text)
        histories[instance] = history
    return histories


def _get_best_objectives(instance_ids: list[int], classical_histories: dict[int, list[dict[str, float | int]] | None],
                         hybrid_histories: dict[int, list[dict[str, float | int]] | None]) -> dict[int, float]:
    """Finds the best known feasible objective per instance.
    :param instance_ids: Instance ids to include in aggregation.
    :param classical_histories: Classical histories by instance.
    :param hybrid_histories: Hybrid histories by instance.
    :return: Mapping from instance id to the lowest objective found by either solver.
    """
    best_objectives = {}
    for instance in instance_ids:
        objectives = ([entry["objective"] for entry in classical_histories.get(instance) or []] +
                      [entry["objective"] for entry in hybrid_histories.get(instance) or []])
        best_objectives[instance] = min(objectives) if len(objectives) > 0 else None
    return best_objectives


def _get_average_normalized_curve(grid_times: np.ndarray, instance_ids: list[int], solver_histories: dict[int, list[dict[str, float | int]] | None],
                                  best_objectives: dict[int, float]) -> np.ndarray:
    """Computes average normalized objective curve on a uniform time grid.
    :param grid_times: Uniform time grid used for alignment.
    :param instance_ids: Instance ids included in averaging denominator.
    :param solver_histories: Histories for one solver grouped by instance.
    :param best_objectives: Best known objective per instance across both solvers.
    :return: Average normalized objective values for each grid time.
    """
    totals = np.zeros(len(grid_times))
    for instance in instance_ids:
        history = solver_histories.get(instance)
        if not history:
            continue
        history_times = np.array([entry["time"] for entry in history])
        normalized_objectives = np.array([best_objectives[instance] / entry["objective"] for entry in history])
        indices = np.searchsorted(history_times, grid_times, side="right") - 1
        totals[indices >= 0] += normalized_objectives[indices[indices >= 0]]
    return totals / len(instance_ids)


if __name__ == "__main__":
    # plot_instance_objective_histories()
    plot_average_normalized_objective_histories()
    plt.show()
