from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import numpy as np
from numpy import random

from src import utils
from src.PowerGridProblem import PowerGridProblem, PowerGridSolution
from src.Sampler import ExactSampler
from src.VariationalQuantumProgram import VariationalQuantumProgram


class PowerGridSolver(ABC):
    """ Base class for power grid problem solvers. """

    @abstractmethod
    def solve(self, problem: PowerGridProblem, *args, **kwargs) -> PowerGridSolution:
        """ Solves a given power grid optimization problem and returns its solution. """
        pass


@dataclass
class HybridSolver(PowerGridSolver):
    vqp: VariationalQuantumProgram

    def solve(self, problem: PowerGridProblem, penalty_mult: float = 10) -> PowerGridSolution:
        seed = 0
        rng = random.default_rng(seed)

        initial_angles = rng.uniform(-np.pi, np.pi, len(self.vqp.circuit.parameters))
        cost_function = partial(problem.evaluate, penalty_mult=penalty_mult)
        result = self.vqp.optimize_parameters(cost_function, initial_angles)
        assert result.success, f"Angle optimization failed: {result.message}"

        best_sample = min(problem.optimize_power.cache.items(), key=lambda pair: pair[1].total)
        solution = PowerGridSolution(best_sample[0], best_sample[1].x, best_sample[1].fun)
        solution.extra["opt_result"] = best_sample[1]

        exact_sampler = ExactSampler()
        solution.extra["final_probs"] = exact_sampler.get_sample_probabilities(self.vqp.circuit, result.x)
        solution.extra["cost_expectation"] = utils.get_cost_expectation(cost_function, solution.extra["final_probs"])
        solution.extra["num_jobs"] = result.nfev
        return solution
