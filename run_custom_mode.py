"""Runs custom analysis modes that are intentionally kept outside solver classes."""

import pickle
from math import floor
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
from numpy import ndarray, random
from pandas import DataFrame
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from common.utils import get_variational_quantum_program
from plots.general import apply_plot_settings, save_figure
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.PowerFlowProblem import PowerFlowProblem
from src.Sampler import ExactSampler


def plot_sampling_distribution(instance: int, target_ci_length: float, target_ci_confidence: float, num_repetitions: int, sample_ci_confidence: float,
                               seed: int | None = None):
    """Plots the repeated-sampling distribution of the AR sample mean.
    :param instance: Stored instance index.
    :param target_ci_length: Maximum allowed full confidence interval length for the sampled mean.
    :param target_ci_confidence: Target success probability that sampled mean falls inside the target confidence interval.
    :param num_repetitions: Number of repeated sample sets used to form the histogram.
    :param sample_ci_confidence: Confidence level for the sampled success probability confidence interval.
    :param seed: Random seed used to generate one circuit angle vector and repeated samples.
    """
    num_layers = 1
    problem = read_instance(instance)
    vqp = get_variational_quantum_program(len(problem.generators), num_layers, "exact", seed=seed)
    angles = random.default_rng(seed).uniform(-np.pi, np.pi, len(vqp.circuit.parameters))
    analysis = analyze_distribution(problem, [angles], target_ci_length, target_ci_confidence, seed=seed)
    sampled_values = random.default_rng(seed).choice(analysis["ar_values"], size=(num_repetitions, analysis["required_num_samples"][0]),
                                                    p=analysis["probs_list"][0])
    sampled_means = sampled_values.mean(axis=1)
    target_ci_left = analysis["expectation"][0] - target_ci_length / 2
    target_ci_right = analysis["expectation"][0] + target_ci_length / 2
    success_mask = (sampled_means >= target_ci_left) & (sampled_means <= target_ci_right)
    success_count = np.count_nonzero(success_mask)
    success_probability_ci = proportion_confint(success_count, num_repetitions, alpha=1 - sample_ci_confidence, method="beta")
    print(f"Success probability: {success_count / num_repetitions}")
    print(f"{sample_ci_confidence:.0%} CI for success probability: {success_probability_ci}")

    apply_plot_settings(plt.gcf())
    plt.hist(sampled_means, bins=30, edgecolor="black")
    plt.axvline(analysis["expectation"][0], color="black", linestyle="--", label="exact expectation")
    plt.axvline(target_ci_left, color="red", linestyle="--", label="target CI")
    plt.axvline(target_ci_right, color="red", linestyle="--")
    plt.xlabel("sample mean")
    plt.ylabel("frequency")
    plt.legend()
    plt.tight_layout()
    save_figure("plots/out/sampling_distribution.jpg")
    plt.show()


def analyze_distribution_range(instances: range, num_angles: int, target_ci_length: float, target_ci_confidence: float, num_repetitions: int = 10000,
                               test_samples: bool = False, sample_ci_confidence: float | None = None, seed: int | None = None):
    """Computes required sample count summaries for a range of stored instances.
    :param instances: Stored instance indexes.
    :param num_angles: Number of random angle vectors to analyze for each instance.
    :param target_ci_length: Maximum allowed full 90 percent confidence interval length for the sampled mean.
    :param target_ci_confidence: Target confidence level for the sampled mean confidence interval.
    :param num_repetitions: Number of repeated sample sets used when testing predicted sample counts.
    :param test_samples: Whether predicted sample counts should be tested empirically.
    :param sample_ci_confidence: Confidence level for the sampled success probability confidence interval.
    :param seed: Random seed used to generate angle vectors and test samples.
    """
    num_layers = 1
    rand_gen = random.default_rng(seed)
    first_instance = next(iter(instances))
    problem = read_instance(first_instance)
    vqp = get_variational_quantum_program(len(problem.generators), num_layers, "exact", seed=seed)
    records = []
    for instance in instances:
        angle_vectors = rand_gen.uniform(-np.pi, np.pi, (num_angles, len(vqp.circuit.parameters)))
        analysis = analyze_distribution(instance, angle_vectors, target_ci_length, target_ci_confidence, num_repetitions, test_samples, sample_ci_confidence,
                                        seed)
        records += [{"instance": instance, "required_num_samples": required_num_samples} for required_num_samples in analysis["required_num_samples"]]

    data = DataFrame(records)
    summary = data.groupby("instance")["required_num_samples"].agg(["mean", "max"])
    for instance, row in summary.iterrows():
        print(f"Instance {instance}: mean required_num_samples = {row["mean"]}, max required_num_samples = {row["max"]}")
    print(f"Global: mean required_num_samples = {data["required_num_samples"].mean()}, max required_num_samples = {data["required_num_samples"].max()}")


def analyze_distribution(problem: PowerFlowProblem | int,
                         angles_list: list[ndarray],
                         target_ci_length: float,
                         target_ci_confidence: float,
                         num_repetitions: int = 10000,
                         test_samples: bool = False,
                         sample_ci_confidence: float | None = None,
                         seed: int | None = None) -> dict[str, object]:
    """Computes AR distribution values and moment summaries for multiple angle vectors.
    :param problem: Power-flow instance or stored instance index to analyze.
    :param angles_list: Circuit angle vectors whose probability distributions should be analyzed.
    :param target_ci_length: Maximum allowed full 90 percent confidence interval length for the sampled mean.
    :param target_ci_confidence: Target confidence level for the sampled mean confidence interval.
    :param num_repetitions: Number of repeated sample sets used when testing predicted sample counts.
    :param test_samples: Whether predicted sample counts should be tested empirically.
    :param sample_ci_confidence: Confidence level for the sampled success probability confidence interval.
    :param seed: Random seed used to build the variational quantum program.
    :return: AR values plus per-angle probabilities, exact moments, and required sample counts.
    """
    num_layers = 1
    violation_mult = 10 ** 7
    max_inner_time_s = 30
    silent = True

    problem_ind = problem if isinstance(problem, int) else None
    if isinstance(problem, int):
        problem = read_instance(problem)
    inner_optimizer = CasadiOptimizer(problem, max_inner_time_s, violation_mult, silent=silent)
    bitstrings = [format(i, f"0{len(problem.generators)}b") for i in range(2 ** len(problem.generators))]
    totals = np.array([inner_optimizer.optimize(bitstring).total for bitstring in bitstrings])
    ar_values = np.min(totals) / totals
    vqp = get_variational_quantum_program(len(problem.generators), num_layers, "exact", seed=seed)
    z_score = norm.ppf((1 + target_ci_confidence) / 2)
    probs_list = []
    expectations = []
    stds = []
    ar_3rd_moments = []
    required_num_samples = []
    for i, angles in enumerate(angles_list):
        probs_dict = ExactSampler().get_sample_probabilities(vqp.circuit, angles)
        probs = np.array([probs_dict.get(bitstring, 0) for bitstring in bitstrings])
        expectation = np.dot(probs, ar_values)
        std = np.sqrt(np.dot(probs, (ar_values - expectation) ** 2))
        required_samples = floor((2 * z_score * std / target_ci_length) ** 2) + 1

        probs_list.append(probs)
        expectations.append(expectation)
        stds.append(std)
        ar_3rd_moments.append(np.dot(probs, np.abs(ar_values - expectation) ** 3) / std ** 3)
        required_num_samples.append(required_samples)
        if test_samples:
            assert sample_ci_confidence is not None, "sample_ci_confidence is required when test_samples is True"
            sample_seed = None if seed is None else seed + i
            if not test_num_samples(ar_values, probs, expectation, required_samples, target_ci_length, target_ci_confidence, num_repetitions,
                                    sample_ci_confidence, sample_seed):
                print(f"Sample count test failed: index={problem_ind}, angles={angles}, seed={sample_seed}, required_num_samples={required_samples}")

    return {"ar_values": ar_values, "probs_list": probs_list, "expectation": expectations, "std": stds, "ar_3rd_moment": ar_3rd_moments,
            "required_num_samples": required_num_samples}


def read_instance(instance: int) -> PowerFlowProblem:
    """Reads a stored power-flow instance.
    :param instance: Stored instance index.
    :return: Power-flow problem for the stored instance.
    """
    data_folder = Path("data/5")
    voltage_deviation_mult = 10
    with (data_folder / f"{instance}.pkl").open("rb") as file:
        return PowerFlowProblem(pickle.load(file), voltage_deviation_mult)


def test_num_samples(ar_values: ndarray, probs: ndarray, expectation: float, required_num_samples: int, target_ci_length: float,
                     target_ci_confidence: float, num_repetitions: int, sample_ci_confidence: float, seed: int | None = None) -> bool:
    """Tests whether the required number of samples passes its confidence interval criterion.
    :param ar_values: Approximation-ratio values sampled by the selected angle distribution.
    :param probs: Sampling probabilities for the selected angle distribution.
    :param expectation: Exact expectation for the selected angle distribution.
    :param required_num_samples: Number of samples predicted by the confidence interval criterion.
    :param target_ci_length: Maximum allowed full confidence interval length for the sampled mean.
    :param target_ci_confidence: Target success probability that sampled mean falls inside the target confidence interval.
    :param num_repetitions: Number of repeated sample sets used to estimate success probability.
    :param sample_ci_confidence: Confidence level for the sampled success probability confidence interval.
    :param seed: Random seed used to generate repeated samples.
    :return: Whether the required number of samples passes its confidence interval criterion.
    """
    sampled_values = random.default_rng(seed).choice(ar_values, size=(num_repetitions, required_num_samples), p=probs)
    sampled_means = sampled_values.mean(axis=1)
    target_ci_left = expectation - target_ci_length / 2
    target_ci_right = expectation + target_ci_length / 2
    success_mask = (sampled_means >= target_ci_left) & (sampled_means <= target_ci_right)
    success_count = np.count_nonzero(success_mask)
    success_probability_ci = proportion_confint(success_count, num_repetitions, alpha=1 - sample_ci_confidence, method="beta")
    return success_probability_ci[0] <= target_ci_confidence <= success_probability_ci[1]


if __name__ == "__main__":
    instance = 0
    num_angles = 100
    target_ci_length = 0.1
    target_ci_confidence = 0.9
    num_repetitions = 10000
    test_samples = True
    sample_ci_confidence = 0.999
    seed = 0
    angles = np.array([1.8799063, -1.11660544, 1.86383895, -1.7258123, -0.86514466, -0.51868881, 0.2601866, -2.43402012, -0.58466421, -3.13970336,
                       1.53548939, 2.21090156, -2.26865917, 1.28042375, 2.01755021, 3.02741664, 2.16009981, -0.47685302, 3.01397305, 2.97813185])

    analyze_distribution_range(range(100), num_angles, target_ci_length, target_ci_confidence, num_repetitions, test_samples, sample_ci_confidence, seed)
    # plot_sampling_distribution(instance, target_ci_length, target_ci_confidence, num_repetitions, sample_ci_confidence, seed)
    # test_sampled_distributions(range(instance, instance + 1), num_angles, target_ci_length, target_ci_confidence, num_repetitions, seed)
