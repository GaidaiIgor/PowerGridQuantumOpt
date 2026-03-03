"""Plotting helpers for power-grid optimization outputs."""
import ast
from pathlib import Path

import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    plot_instance_objective_histories()
    plt.show()
