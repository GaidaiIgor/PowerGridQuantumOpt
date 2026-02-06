from dataclasses import dataclass, field
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import OptimizeResult, NonlinearConstraint, LinearConstraint

from PowerFlowProblem import PowerFlowProblem


@dataclass
class ContinuousPowerOptimizer:
    """ Optimizes continuous part of the given power flow problem for a fixed state of binary variables. """
    problem: PowerFlowProblem
    penalty_mult: float
    cache: dict[str, OptimizeResult] = field(default_factory=dict)

    @staticmethod
    def get_initial_point(bounds: list[NDArray[float]]) -> list[float]:
        """ Returns initial point for the optimization. """
        initial_point = [np.average(bound) for bound in bounds]
        return initial_point

    @staticmethod
    def convert_bounds_to_constraints(bounds: list[NDArray[float]]) -> LinearConstraint:
        """ Converts bounds to a linear constraint object. """
        eye = np.eye(len(bounds))
        bounds_matrix = np.array(bounds)
        constraint = LinearConstraint(eye, bounds_matrix[:, 0], bounds_matrix[:, 1])
        return constraint

    def split_params(self, params: list[float]) -> tuple[NDArray[float], NDArray[float], NDArray[float], NDArray[float]]:
        """ Splits overall parameter list into list of active powers, reactive power, voltages and angles. """
        active_powers = np.array(params[:len(self.problem.generators)])
        reactive_powers = np.array(params[len(self.problem.generators):2 * len(self.problem.generators)])
        voltage_magnitudes = np.array(params[2 * len(self.problem.generators):2 * len(self.problem.generators) + len(self.problem.graph)])
        phase_angles = np.array(params[2 * len(self.problem.generators) + len(self.problem.graph):])
        return active_powers, reactive_powers, voltage_magnitudes, phase_angles

    def evaluate_constraints(self, params: list[float]) -> list[float]:
        """ Evaluates all constraints. Feasible constraints are >= 0. """
        active_powers, reactive_powers, voltage_magnitudes, phase_angles = self.split_params(params)
        return self.problem.evaluate_constraints(active_powers, reactive_powers, voltage_magnitudes, phase_angles)

    def get_generation_cost(self, generator_statuses: str, params: list[float]) -> float:
        """ Returns the total cost of generation for a given set of enabled generators at given optimization parameters. """
        active_powers = self.split_params(params)[0]
        return self.problem.get_generation_cost(generator_statuses, active_powers)

    def get_penalty(self, params: list[float], constraints: list[LinearConstraint | NonlinearConstraint]) -> float:
        """ Evaluates penalty term for a given optimization parameter vector and list of constraints. """
        penalty = 0
        for constraint in constraints:
            if isinstance(constraint, LinearConstraint):
                residuals = constraint.residual(params)
                violations = np.concatenate(residuals)
            elif isinstance(constraint, NonlinearConstraint):
                vals = np.array(constraint.fun(params))
                violations = np.concatenate((vals - constraint.lb, constraint.ub - vals))
            penalty += self.penalty_mult * np.sum(np.minimum(violations, 0) ** 2)
        return penalty

    def _optimize(self, generator_statuses: str) -> OptimizeResult:
        """ Finds optimal continuous variables for a given set of enabled generators.
        Returns optimization result with extra fields: penalty and total (fun + penalty). """
        bounds = self.problem.get_bounds(generator_statuses)
        initial_point = self.get_initial_point(bounds)
        upper_bound = [0] * (2 * len(self.problem.graph) + 1) + [np.inf] * self.problem.graph.size()
        constraints = [self.convert_bounds_to_constraints(bounds), NonlinearConstraint(self.evaluate_constraints, 0, upper_bound)]
        cost_function = partial(self.get_generation_cost, generator_statuses)
        options = {"maxiter": 2 ** 31 - 1}
        result = optimize.minimize(cost_function, initial_point, method="SLSQP", constraints=constraints, options=options)
        result.penalty = self.get_penalty(result.x, constraints)
        result.total = result.fun + result.penalty
        return result

    def optimize(self, generator_statuses: str) -> OptimizeResult:
        """ Cache wrapper around _optimize. """
        if generator_statuses not in self.cache:
            self.cache[generator_statuses] = self._optimize(generator_statuses)
        return self.cache[generator_statuses]

    def get_optimized_cost(self, generator_statuses: str) -> float:
        """ Returns the value of the cost function (with penalties) after optimization. """
        return self.optimize(generator_statuses).total
