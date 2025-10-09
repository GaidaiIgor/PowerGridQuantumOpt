import matplotlib.pyplot as plt

from plots.general import save_figure


def plot_probability_distribution(probs: dict[str, float], y_max: float = 1):
    xs, ys = list(zip(*sorted(probs.items())))
    plt.bar(xs, ys)
    plt.ylim((0, y_max))
    plt.ylabel("Probability")
    # save_figure()
