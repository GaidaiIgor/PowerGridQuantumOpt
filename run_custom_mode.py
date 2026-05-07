"""Runs custom analysis modes that are intentionally kept outside solver classes."""

import pickle
from math import floor
from pathlib import Path

import numpy as np
from numpy import random
from scipy.stats import norm

from common.utils import get_variational_quantum_program
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.PowerFlowProblem import PowerFlowProblem
from src.Sampler import ExactSampler


def run_distribution_analysis(instance: int = 0, seed: int = 0, target_ci_length: float = 0.1) -> dict[str, object]:
    """Computes random-angle AR distribution moments for one stored power-flow instance.
    :param instance: Stored instance index.
    :param seed: Random seed used to generate one circuit angle vector.
    :param target_ci_length: Maximum allowed full 90 percent confidence interval length for the sampled mean.
    :return: Distribution-analysis values for the sampled angle vector.
    """
    assert target_ci_length > 0, "Confidence interval length threshold must be positive."
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
    return {"instance": instance, "ar_expectation": expectation, "ar_std": std, "ar_3rd_moment": ar_3rd_moment, "required_num_samples": required_num_samples}


if __name__ == "__main__":
    print(run_distribution_analysis())
