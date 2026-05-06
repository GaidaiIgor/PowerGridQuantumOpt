"""Runs custom analysis modes that are intentionally kept outside solver classes."""

import pickle
from pathlib import Path

import numpy as np
from numpy import random

from common.utils import get_variational_quantum_program
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.PowerFlowProblem import PowerFlowProblem
from src.Sampler import ExactSampler


def run_distribution_analysis(instance: int = 0, seed: int = 0) -> dict[str, object]:
    """Computes random-angle AR distribution moments for one stored power-flow instance.
    :param instance: Stored instance index.
    :param seed: Random seed used to generate one circuit angle vector.
    :return: Distribution-analysis values for the sampled angle vector.
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
    return {"instance": instance, "angles": angles.tolist(), "ar_expectation": expectation, "ar_std": std, "ar_3rd_moment": ar_3rd_moment}


if __name__ == "__main__":
    print(run_distribution_analysis())
