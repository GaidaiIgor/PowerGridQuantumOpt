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
    ref_ind = 0
    violation_tolerance = 1e-10
    solver_names = {"scip": "SCIP", "smac": "SMAC", "uniform": "Uniform", "hybrid/nl_1": "Hybrid", "hybrid/nl_2": "Hybrid L2"}

    history_data = load_histories(num_generators, solver_ids, ref_ind, violation_tolerance)
    lines = []
    labeled_solvers = set()
    for num_gens_ind, num_gens in enumerate(num_generators):
        xs, solver_data = history_data[num_gens_ind]
        for solver_ind, solver_id in enumerate(solver_ids):
            ys = solver_data[solver_ind]
            label = solver_names.get(solver_id, solver_id) if solver_id not in labeled_solvers else "_nolabel_"
            lines.append(Line(xs, ys, color=solver_ind, marker=num_gens_ind, label=label))
            labeled_solvers.add(solver_id)
    plot_general(lines, axis_labels=("Time [s]", "Normalized Objective"), boundaries=(None, None, 0.9, 1.01))
    save_figure()


def plot_history_diff():
    """Plots differences between the two configured solver curves."""
    num_generators = [13]
    solver_ids = ["hybrid/nl_1", "uniform"]
    ref_ind = 0
    violation_tolerance = 1e-10

    history_data = load_histories(num_generators, solver_ids, ref_ind, violation_tolerance)
    lines = []
    for num_gens_ind, num_gens in enumerate(num_generators):
        xs, solver_data = history_data[num_gens_ind]
        first_ys, second_ys = solver_data
        lines.append(Line(xs, first_ys - second_ys, color=num_gens_ind, marker=0, label=str(num_gens)))
    lines.append(Line([0, 10000], [0, 0], color="black", marker="none", style="--"))
    plot_general(lines, axis_labels=("Time [s]", "Normalized Objective Difference"), boundaries=(0, xs[-1], -0.1, 0.1))
    save_figure()


def plot_ar_vs_instance():
    """Plots `ar_uniform_fun` and `ar_opt_fun` against row index for the configured generator count."""
    num_generators = 10
    df = load_dfs(num_generators, ["hybrid/nl_1"], 0)[0]
    lines = [Line(np.arange(len(df)), df["ar_uniform_fun"], color=0, style="none", label="AR Uniform"),
             Line(np.arange(len(df)), df["ar_opt_fun"], color=1, style="none", label="AR Opt")]
    plot_general(lines, axis_labels=("Instance index", "Approximation Ratio"), boundaries=(0, 99, 0, 1))
    save_figure()


def plot_ar_diff_vs_instance():
    """Plots `ar_opt_fun - ar_uniform_fun` against row index for the configured generator count."""
    num_generators = 10
    df = load_dfs(num_generators, ["hybrid/nl_1"], 0)[0]
    lines = [Line(np.arange(len(df)), df["ar_opt_fun"] - df["ar_uniform_fun"], style="none", label="AR Opt - AR Uniform"),
             Line([0, len(df) - 1], [0, 0], color="black", marker="none", style="--")]
    plot_general(lines, axis_labels=("Instance index", "Opt - Uniform AR Diff"), boundaries=(0, 99, -0.4, 0.4))
    save_figure()


def plot_average_ar_vs_generators():
    """Plots average `ar_uniform_fun` and `ar_opt_fun` against generator count for the configured datasets."""
    generator_counts = [10, 11, 12, 13]
    dfs = [load_dfs(num_generators, ["hybrid/nl_1"], 0)[0] for num_generators in generator_counts]
    lines = [Line(generator_counts, [df["ar_uniform_fun"].mean() for df in dfs], color=0, label="AR Uniform"),
             Line(generator_counts, [df["ar_opt_fun"].mean() for df in dfs], color=1, label="AR Opt")]
    plot_general(lines, axis_labels=("Number of Generators", "Average Approximation Ratio"), boundaries=(min(generator_counts), max(generator_counts), 0, 1))
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    save_figure()


def load_histories(num_generators: list[int], solver_ids: list[str], ref_ind: int, violation_tolerance: float) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Collects average normalized objective histories across datasets.
    :param num_generators: Generator counts whose datasets should be loaded.
    :param solver_ids: Solver ids whose CSV files should be loaded from subfolders inside each dataset folder.
    :param ref_ind: Index of the solver used to select the fastest instances and time grid in each dataset.
    :param violation_tolerance: Maximum violation still treated as feasible in extracted histories.
    :return: History data in `num_generators` order, each entry storing one time grid and solver curves in `solver_ids` order.
    """
    histories = []
    for num_gens in num_generators:
        dfs = load_dfs(num_gens, solver_ids, ref_ind)
        time_grid = np.linspace(0, max(dfs[ref_ind]["classical_opt_time"]), 50)
        histories.append((time_grid, extract_normalized_solver_histories(dfs, time_grid, violation_tolerance)))
    return histories


def load_dfs(num_generators: int, solver_ids: list[str], ref_ind: int | None = None) -> list[pd.DataFrame]:
    """Loads solver data and optionally restricts all data frames to the fastest reference instances.
    :param num_generators: Generator count whose solver CSV should be loaded.
    :param solver_ids: Solver ids whose CSV files should be loaded.
    :param ref_ind: Index of the reference solver inside `solver_ids`, or ``None`` to skip trimming.
    :return: Solver data frames in `solver_ids` order after optional trimming.
    """
    dfs = [pd.read_csv(Path(__file__).resolve().parent.parent / f"data/{num_generators}/{solver_id}/.solutions.csv") for solver_id in solver_ids]
    if ref_ind is None:
        return dfs
    ref_df = dfs[ref_ind]
    fastest_100_inds = pd.to_numeric(ref_df["classical_opt_time"], errors="coerce").nsmallest(100).index
    return [df.loc[fastest_100_inds] for df in dfs]


def extract_normalized_solver_histories(solver_dfs: list[pd.DataFrame], time_grid: Sequence[float], violation_tolerance: float) -> list[np.ndarray]:
    """Computes average normalized objective curves for aligned solver data frames.
    :param solver_dfs: Solver data frames aligned by row index.
    :param time_grid: Uniform time grid used to align objective histories.
    :param violation_tolerance: Maximum violation still treated as feasible in extracted histories.
    :return: Average normalized objective curve for each loaded solver in input order.
    """
    assert all(df.index.equals(solver_dfs[0].index) for df in solver_dfs[1:]), "Solver data frames must share row indices."
    all_solver_histories = [extract_solver_histories(df, violation_tolerance, time_grid[-1]) for df in solver_dfs]
    best_objectives = extract_best_objectives(all_solver_histories)
    return [get_average_normalized_history(time_grid, solver_histories, best_objectives) for solver_histories in all_solver_histories]


def extract_solver_histories(df: pd.DataFrame, violation_tolerance: float, max_time: float) -> list[list[HistoryEntry] | None]:
    """Extracts solver histories from one aligned data frame.
    :param df: Solver data frame whose rows define the aligned history order.
    :param violation_tolerance: Maximum violation still treated as feasible.
    :param max_time: Maximum time of loaded history entries.
    :return: History entries for each row, or ``None`` when the row history is null.
    """
    converter = make_converter()
    histories = []
    for history_text in df["history"]:
        if pd.isna(history_text):
            histories.append(None)
            continue
        histories.append([entry for entry in converter.loads(history_text, list[HistoryEntry])
                          if entry.result.violation <= violation_tolerance and entry.time <= max_time])
    return histories


def extract_best_objectives(all_solver_histories: list[list[list[HistoryEntry] | None]]) -> list[float | None]:
    """Finds the best known feasible objective for each instance.
    :param all_solver_histories: Histories indexed by: 0) solver ind; 1) instance ind; 2) HistoryEntry index.
    :return: Lowest objective found by any loaded solver for each instance.
    """
    if len(all_solver_histories) == 0:
        return []
    best_instance_objectives = []
    for instance_histories in zip(*all_solver_histories, strict=True):
        best_solver_objectives = [solver_history[-1].result.fun for solver_history in instance_histories if solver_history]
        best_instance_objectives.append(min(best_solver_objectives) if len(best_solver_objectives) > 0 else None)
    return best_instance_objectives


def get_average_normalized_history(time_grid: Sequence[float], solver_histories: list[list[HistoryEntry] | None], best_objectives: list[float | None]) \
    -> np.ndarray:
    """Computes average normalized objective curve on a uniform time grid.
    :param time_grid: Uniform time grid used for alignment.
    :param solver_histories: Histories for one solver aligned by row index.
    :param best_objectives: Best known objective per aligned row across the loaded solvers.
    :return: Average normalized objective values for each grid time.
    """
    totals = np.zeros(len(time_grid))
    feasible_count = 0
    for history, best_objective in zip(solver_histories, best_objectives, strict=True):
        if best_objective is None:
            continue
        feasible_count += 1
        if not history:
            continue
        history_times = np.array([entry.time for entry in history])
        indices = np.searchsorted(history_times, time_grid, side="right") - 1
        normalized_history = np.array([best_objective / entry.result.fun for entry in history])
        totals[indices >= 0] += normalized_history[indices[indices >= 0]]
    return totals / feasible_count


if __name__ == "__main__":
    # plot_instance_objective_histories()
    # plot_average_histories()
    # plot_history_diff()
    # plot_ar_vs_instance()
    # plot_ar_diff_vs_instance()
    plot_average_ar_vs_generators()
    plt.show()
