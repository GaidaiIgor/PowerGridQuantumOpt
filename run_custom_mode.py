"""Runs custom analysis modes that are intentionally kept outside solver classes."""

import pickle
from math import ceil, log
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
from numpy import ndarray, random
from pandas import DataFrame
from scipy.stats import binom, norm
from statsmodels.stats.proportion import proportion_confint

from common.utils import get_variational_quantum_program
from plots.general import apply_plot_settings, save_figure
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.PowerFlowProblem import PowerFlowProblem
from src.Sampler import ExactSampler


def plot_sampling_distribution(instance: int, target_ci_length: float, target_ci_confidence: float, num_repetitions: int, sample_ci_confidence: float,
                               shots_estimation_method: str = "bernstein", seed: int | None = None):
    """Plots the repeated-sampling distribution of the AR sample mean.
    :param instance: Stored instance index.
    :param target_ci_length: Maximum allowed full confidence interval length for the sampled mean.
    :param target_ci_confidence: Target success probability that sampled mean falls inside the target confidence interval.
    :param num_repetitions: Number of repeated sample sets used to form the histogram.
    :param sample_ci_confidence: Confidence level for the sampled success probability confidence interval.
    :param shots_estimation_method: Method used to estimate the required number of shots.
    :param seed: Random seed used to generate one circuit angle vector and repeated samples.
    """
    num_layers = 1
    problem = read_instance(instance)
    vqp = get_variational_quantum_program(len(problem.generators), num_layers, "exact", seed=seed)
    angles = random.default_rng(seed).uniform(-np.pi, np.pi, len(vqp.circuit.parameters))
    analysis = analyze_distribution(problem, [angles], target_ci_length, target_ci_confidence, shots_estimation_method=shots_estimation_method, seed=seed)
    sampled_values = random.default_rng(seed).choice(analysis["ar_values"], size=(num_repetitions, analysis["required_num_shots"][0]),
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


def analyze_distribution_range(instances: range,
                               num_angles: int,
                               target_ci_length: float,
                               target_ci_confidence: float,
                               shots_estimation_method: str = "bernstein",
                               test_samples: bool = False,
                               num_repetitions: int = 10000,
                               fail_prob: float | None = None,
                               seed: int | None = None):
    """Computes required shot count summaries for a range of stored instances.
    :param instances: Stored instance indexes.
    :param num_angles: Number of random angle vectors to analyze for each instance.
    :param target_ci_length: Maximum allowed full 90 percent confidence interval length for the sampled mean.
    :param target_ci_confidence: Target confidence level for the sampled mean confidence interval.
    :param shots_estimation_method: Method used to estimate the required number of shots.
    :param test_samples: Whether predicted shot counts should be tested empirically.
    :param num_repetitions: Number of repeated sample sets used when testing predicted shot counts.
    :param fail_prob: Confidence level for the sampled success probability confidence interval.
    :param seed: Random seed used to generate angle vectors and test samples.
    """
    num_layers = 1
    rand_gen = random.default_rng(seed)
    first_instance = next(iter(instances))
    problem = read_instance(first_instance)
    vqp = get_variational_quantum_program(len(problem.generators), num_layers, "exact", seed=seed)
    records = []
    for instance in instances:
        print(f"Analyzing instance: {instance}")
        angle_vectors = rand_gen.uniform(-np.pi, np.pi, (num_angles, len(vqp.circuit.parameters)))
        analysis = analyze_distribution(instance, angle_vectors, target_ci_length, target_ci_confidence, shots_estimation_method, test_samples,
                                        num_repetitions, fail_prob, seed)
        records += [{"instance": instance, "required_num_shots": required_num_shots} for required_num_shots in analysis["required_num_shots"]]

    data = DataFrame(records)
    summary = data.groupby("instance")["required_num_shots"].agg(["mean", "max"])
    for instance, row in summary.iterrows():
        print(f"Instance {instance}: mean required_num_shots = {row["mean"]}, max required_num_shots = {row["max"]}")
    print(f"Global: mean required_num_shots = {data["required_num_shots"].mean()}, max required_num_shots = {data["required_num_shots"].max()}")


def analyze_distribution(problem: PowerFlowProblem | int,
                         angles_list: list[ndarray],
                         target_ci_length: float,
                         target_ci_confidence: float,
                         shots_estimation_method: str = "bernstein",
                         test_samples: bool = False,
                         num_repetitions: int = 10000,
                         fail_prob: float | None = None,
                         seed: int | None = None) -> dict[str, object]:
    """Computes AR distribution values and statistics for multiple angle vectors.
    :param problem: Power-flow instance or stored instance index to analyze.
    :param angles_list: Circuit angle vectors whose probability distributions should be analyzed.
    :param target_ci_length: Maximum allowed full 90 percent confidence interval length for the sampled mean.
    :param target_ci_confidence: Target confidence level for the sampled mean confidence interval.
    :param shots_estimation_method: Method used to estimate the required number of shots.
    :param test_samples: Whether predicted shot counts should be tested empirically.
    :param num_repetitions: Number of repeated sample sets used when testing predicted shot counts.
    :param fail_prob: Probability at or below which sample test is considered to be failed.
    :param seed: Random seed used to build the variational quantum program.
    :return: AR values plus per-angle probabilities, exact statistics, and required shot counts.
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

    probs_list = []
    expectations = []
    stds = []
    required_num_shots = []
    for i, angles in enumerate(angles_list):
        probs_dict = ExactSampler().get_sample_probabilities(vqp.circuit, angles)
        probs = np.array([probs_dict.get(bitstring, 0) for bitstring in bitstrings])
        expectation = np.dot(probs, ar_values)
        std = np.sqrt(np.dot(probs, (ar_values - expectation) ** 2))
        required_shots = estimate_num_shots(target_ci_length, target_ci_confidence, std, shots_estimation_method)

        probs_list.append(probs)
        expectations.append(expectation)
        stds.append(std)
        required_num_shots.append(required_shots)
        if test_samples:
            assert fail_prob is not None, "fail_prob is required when test_samples is True"
            sample_seed = None if seed is None else seed + i
            target_range = (expectation - target_ci_length / 2, expectation + target_ci_length / 2)
            sample_success_prob = get_sample_success_probability(ar_values, probs, required_shots, num_repetitions, target_range, target_ci_confidence,
                                                                 sample_seed)
            if sample_success_prob <= fail_prob:
                angles_string = np.array2string(angles, separator=", ", max_line_width=9999)
                print(f"Shot count sample test failed: index={problem_ind}, angles=np.array({angles_string}), seed={sample_seed}, "
                      f"required_num_shots={required_shots}, sample_success_prob={sample_success_prob}")

    return {"ar_values": ar_values, "probs_list": probs_list, "expectation": expectations, "std": stds, "required_num_shots": required_num_shots}


def read_instance(instance: int) -> PowerFlowProblem:
    """Reads a stored power-flow instance.
    :param instance: Stored instance index.
    :return: Power-flow problem for the stored instance.
    """
    data_folder = Path("data/5")
    voltage_deviation_mult = 10
    with (data_folder / f"{instance}.pkl").open("rb") as file:
        return PowerFlowProblem(pickle.load(file), voltage_deviation_mult)


def estimate_num_shots(target_ci_length: float, target_ci_confidence: float, std: float, estimation_method: str) -> int:
    """Estimates the number of shots required for the target confidence interval length.
    :param target_ci_length: Maximum allowed full confidence interval length for the sampled mean.
    :param target_ci_confidence: Target confidence level for the sampled mean confidence interval.
    :param std: Standard deviation of the approximation-ratio distribution.
    :param estimation_method: Method used to estimate the required number of shots.
    :return: Estimated number of shots.
    """
    match estimation_method:
        case "bernstein":
            radius = target_ci_length / 2
            return ceil((2 * std ** 2 + 2 * radius / 3) * log(2 / (1 - target_ci_confidence)) / radius ** 2)
        case "normal":
            z_score = norm.ppf((1 + target_ci_confidence) / 2)
            return ceil((2 * z_score * std / target_ci_length) ** 2)
        case _:
            raise ValueError(f"Unsupported sample estimation method: {estimation_method}")


def get_sample_success_probability(ar_values: ndarray, probs: ndarray, num_shots: int, num_repetitions: int, target_range: tuple[float, float],
                                   success_prob: float, seed: int | None = None) -> float:
    """Computes the binomial lower-tail probability for sampled mean successes.
    :param ar_values: Approximation-ratio values sampled by the selected angle distribution.
    :param probs: Sampling probabilities for the selected angle distribution.
    :param num_shots: Number of shots used for each sample mean.
    :param num_repetitions: Number of repeated sample sets used to estimate success probability.
    :param target_range: Inclusive target interval for sampled means.
    :param success_prob: Success probability that sampled mean falls inside the target confidence interval.
    :param seed: Random seed used to generate repeated samples.
    :return: Probability of observing the sampled success count or fewer under the target success probability.
    """
    sampled_values = random.default_rng(seed).choice(ar_values, size=(num_repetitions, num_shots), p=probs)
    sampled_means = sampled_values.mean(axis=1)
    success_mask = (sampled_means >= target_range[0]) & (sampled_means <= target_range[1])
    success_count = np.count_nonzero(success_mask)
    return binom.cdf(success_count, num_repetitions, success_prob)


if __name__ == "__main__":
    instance = 0
    num_angles = 100
    target_ci_length = 0.1
    target_ci_confidence = 0.9
    num_repetitions = 10000
    test_samples = True
    fail_prob = 0.001
    shots_estimation_method = "bernstein"
    seed = 49
    angles = np.array(
        [-2.14244289, -0.6079359, -0.8316442, -2.24066045, -2.20993552, 2.92493802, -2.7095807, -2.36323528, 2.93287444, 2.82948614, -1.03415279, -2.49574782,
         0.68273255, -0.38894665, 2.84035811, -2.33523773, 1.74808776, -2.98555302, -1.8368707, -1.25331558])

    # analyze_distribution_range(range(100), num_angles, target_ci_length, target_ci_confidence, shots_estimation_method, test_samples, \
    #                            num_repetitions, fail_prob, seed)
    analyze_distribution(1, [angles], target_ci_length, target_ci_confidence, shots_estimation_method, test_samples, num_repetitions, fail_prob, seed)
    # plot_sampling_distribution(instance, target_ci_length, target_ci_confidence, num_repetitions, sample_ci_confidence, shots_estimation_method, seed)
    # test_sampled_distributions(range(instance, instance + 1), num_angles, target_ci_length, target_ci_confidence, num_repetitions, seed)
