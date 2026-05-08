"""Runs custom analysis modes that are intentionally kept outside solver classes."""

import pickle
from math import floor
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
from numpy import random
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from common.utils import get_variational_quantum_program
from plots.general import apply_plot_settings, save_figure
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.PowerFlowProblem import PowerFlowProblem
from src.Sampler import ExactSampler


def plot_sampling_distribution(instance: int = 0, seed: int = 0, target_ci_length: float = 0.1, num_repetitions: int = 1000):
    """Plots the repeated-sampling distribution of the AR sample mean.
    :param instance: Stored instance index.
    :param seed: Random seed used to generate one circuit angle vector and repeated samples.
    :param target_ci_length: Maximum allowed full 90 percent confidence interval length for the sampled mean.
    :param num_repetitions: Number of repeated sample sets used to form the histogram.
    """
    data = collect_sampling_distribution(instance, seed, target_ci_length, num_repetitions)
    print(f"Success probability: {data["success_probability"]}")
    print(f"99% CI for success probability: {data["success_probability_ci"]}")
    apply_plot_settings(plt.gcf())
    plt.hist(data["sampled_means"], bins=30, edgecolor="black")
    plt.axvline(data["exact_expectation"], color="black", linestyle="--", label="exact expectation")
    plt.axvline(data["target_ci_left"], color="red", linestyle="--", label="target CI")
    plt.axvline(data["target_ci_right"], color="red", linestyle="--")
    plt.xlabel("sample mean")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    save_figure("plots/out/sampling_distribution.jpg")
    plt.show()


def collect_sampling_distribution(instance: int = 0, seed: int = 0, target_ci_length: float = 0.1, num_repetitions: int = 1000) -> dict[str, object]:
    """Collects repeated sample means from the selected angle-vector probability distribution.
    :param instance: Stored instance index.
    :param seed: Random seed used to generate one circuit angle vector and repeated samples.
    :param target_ci_length: Maximum allowed full 90 percent confidence interval length for the sampled mean.
    :param num_repetitions: Number of repeated sample sets used to form the histogram.
    :return: Sample means, target interval bounds, observed success summary, and exact expectation.
    """
    analysis = analyze_distributions(instance, seed, target_ci_length)
    sampled_values = random.default_rng(seed).choice(analysis["ar_values"], size=(num_repetitions, analysis["required_num_samples"]), p=analysis["probs_list"])
    sampled_means = sampled_values.mean(axis=1)
    target_ci_left = analysis["expectation"] - target_ci_length / 2
    target_ci_right = analysis["expectation"] + target_ci_length / 2
    success_mask = (sampled_means >= target_ci_left) & (sampled_means <= target_ci_right)
    success_count = np.count_nonzero(success_mask)
    success_probability = success_count / num_repetitions
    return {"sampled_means": sampled_means,
            "exact_expectation": analysis["expectation"],
            "target_ci_left": target_ci_left,
            "target_ci_right": target_ci_right,
            "success_probability": success_probability,
            "success_probability_ci": proportion_confint(success_count, num_repetitions, alpha=0.01, method="beta")}


def analyze_distributions(instance: int = 0, seed: int = 0, target_ci_length: float = 0.1) -> dict[str, object]:
    """Computes AR distribution values and moment summaries for one seeded angle vector.
    :param instance: Stored instance index.
    :param seed: Random seed used to generate one circuit angle vector.
    :param target_ci_length: Maximum allowed full 90 percent confidence interval length for the sampled mean.
    :return: AR values, probabilities, exact moments, and required sample count.
    """
    data_folder = Path("data/5")
    num_layers = 1
    violation_mult = 10 ** 7
    max_inner_time_s = 30
    voltage_deviation_mult = 10
    silent = True

    with (data_folder / f"{instance}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file), voltage_deviation_mult)
    inner_optimizer = CasadiOptimizer(problem, max_inner_time_s, violation_mult, silent=silent)

    bitstrings = [format(i, f"0{len(problem.generators)}b") for i in range(2 ** len(problem.generators))]
    totals = np.array([inner_optimizer.optimize(bitstring).total for bitstring in bitstrings])
    ar_values = np.min(totals) / totals
    vqp = get_variational_quantum_program(len(problem.generators), num_layers, "exact", seed=seed)
    angles = random.default_rng(seed).uniform(-np.pi, np.pi, len(vqp.circuit.parameters))
    probs_dict = ExactSampler().get_sample_probabilities(vqp.circuit, angles)
    probs_list = np.array([probs_dict.get(bitstring, 0) for bitstring in bitstrings])
    expectation = np.dot(probs_list, ar_values)
    std = np.sqrt(np.dot(probs_list, (ar_values - expectation) ** 2))
    ar_3rd_moment = np.dot(probs_list, np.abs(ar_values - expectation) ** 3) / std ** 3
    z_score = norm.ppf(0.95)
    required_num_samples = floor((2 * z_score * std / target_ci_length) ** 2) + 1
    return {"ar_values": ar_values, "probs_list": probs_list, "expectation": expectation, "std": std, "ar_3rd_moment": ar_3rd_moment,
            "required_num_samples": required_num_samples}


if __name__ == "__main__":
    instance = 0
    seed = 0
    target_ci_length = 0.1
    num_repetitions = 10000

    plot_sampling_distribution(instance, seed, target_ci_length, num_repetitions)
    # print(analyze_distributions())
