"""Continuous-variable optimizers for fixed generator commitment states."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Sequence

import casadi as ca
import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import NonlinearConstraint, OptimizeResult

from .PowerFlowProblem import PowerFlowProblem


@dataclass
class Constraint:
    """Stores one symbolic constraint together with its valid range.
    :var expression: Symbolic constraint expression evaluated by the solver.
    :var lb: Inclusive lower bound for the expression value.
    :var ub: Inclusive upper bound for the expression value.
    """
    expression: ca.SX
    lb: float
    ub: float


@dataclass
class ContinuousPowerOptimizer(ABC):
    """Base class for continuous parameter optimization for fixed generator assignments.
    :var problem: Power-flow instance that provides constraints, bounds, and generation costs.
    :var penalty_mult: Multiplier applied to summed squared constraint violations.
    :var cache: Map from generator-status bitstring to cached optimization result.
    """
    problem: PowerFlowProblem
    penalty_mult: float
    cache: dict[str, OptimizeResult] = field(default_factory=dict)

    def optimize(self, generator_statuses: str) -> OptimizeResult:
        """Returns cached result or runs the solver for the given generator statuses.
        :param generator_statuses: Binary generator on/off bitstring.
        :return: Cached or newly computed optimizer result for the status pattern.
        """
        if generator_statuses not in self.cache:
            self.cache[generator_statuses] = self._optimize(generator_statuses)
        return self.cache[generator_statuses]

    @abstractmethod
    def _optimize(self, generator_statuses: str) -> OptimizeResult:
        """Finds optimal continuous variables for a given set of enabled generators.
        :param generator_statuses: Binary generator on/off bitstring.
        :return: Optimizer result with normalized ``fun``, ``penalty``, and ``total`` attributes.
        """

    @staticmethod
    def get_initial_point(bounds: Sequence[NDArray[float]]) -> list[float]:
        """Returns initial point for the optimization.
        :param bounds: Per-variable lower and upper bounds.
        :return: Midpoint of each bound interval as initial parameter values.
        """
        return [np.average(bound) for bound in bounds]

    def evaluate_equality_constraints(self, params: Sequence[float]) -> list[float]:
        """Returns equality constraints only.
        :param params: Full continuous optimization vector.
        :return: Equality-constraint values.
        """
        return self.problem.evaluate_constraints_split(*self.problem.split_params(params))[0]

    def evaluate_inequality_constraints(self, params: Sequence[float]) -> list[float]:
        """Returns inequality constraints only.
        :param params: Full continuous optimization vector.
        :return: Inequality-constraint values that are feasible when nonpositive.
        """
        return self.problem.evaluate_constraints_split(*self.problem.split_params(params))[1]

    def get_generation_cost(self, generator_statuses: str, params: Sequence[float]) -> float:
        """Returns total generation cost for the given statuses and continuous parameters.
        :param generator_statuses: Binary generator on/off bitstring.
        :param params: Full continuous optimization vector.
        :return: Total generation cost for active generators.
        """
        return self.problem.get_generation_cost(generator_statuses, self.problem.split_params(params)[0])

    def get_penalty(self, params: Sequence[float]) -> float:
        """Evaluates penalty term for a parameter vector from the full constraint list.
        :param params: Full continuous optimization vector.
        :return: Penalty value for violated constraints.
        """
        equality_constraints, inequality_constraints = self.problem.evaluate_constraints_split(*self.problem.split_params(params))
        return self.penalty_mult * (np.sum(np.square(equality_constraints)) + np.sum(np.square(np.maximum(inequality_constraints, 0))))


@dataclass
class SLSQPOptimizer(ContinuousPowerOptimizer):
    """Optimizes continuous variables for fixed generator commitments with SLSQP."""

    def _optimize(self, generator_statuses: str) -> OptimizeResult:
        """Runs SLSQP for a given set of enabled generators.
        :param generator_statuses: Binary generator on/off bitstring.
        :return: SciPy optimizer result with normalized ``fun``, ``penalty``, and ``total`` attributes.
        """
        bounds = self.problem.get_bounds(generator_statuses)
        initial_point = self.get_initial_point(bounds)
        constraints = [NonlinearConstraint(self.evaluate_equality_constraints, 0, 0), NonlinearConstraint(self.evaluate_inequality_constraints, -np.inf, 0)]
        cost_function = partial(self.get_generation_cost, generator_statuses)
        result = optimize.minimize(cost_function, initial_point, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 2 ** 31 - 1})
        result.penalty = self.get_penalty(result.x)
        result.total = result.fun + result.penalty
        return result


@dataclass
class CasadiOptimizer(ContinuousPowerOptimizer):
    """Optimizes continuous variables for fixed generator commitments with CasADi and IPOPT.
    :var solver: Reusable IPOPT-backed nonlinear solver for the problem structure.
    :var constraints: Symbolic constraints together with their valid ranges.
    :var silent: Whether to suppress IPOPT and CasADi solver output.
    :var max_iter: Optional IPOPT iteration limit. ``None`` leaves IPOPT at its own default.
    """
    silent: bool = False
    max_iter: int | None = None
    solver: ca.Function = field(init=False, repr=False)
    constraints: list[Constraint] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Builds reusable symbolic problem representation after initialization."""
        self.solver, self.constraints = self._build_solver()

    def _build_solver(self) -> tuple[ca.Function, list[Constraint]]:
        """Builds reusable CasADi symbolic model and IPOPT solver.
        :return: IPOPT solver together with symbolic constraints and their valid ranges.
        """
        params = ca.SX.sym("params", 2 * len(self.problem.generators) + 2 * len(self.problem.graph))
        active_powers, reactive_powers, voltages, angles = self.problem.split_params(params)
        constraints = self._build_constraints(active_powers, reactive_powers, voltages, angles)
        options = {"error_on_fail": False}
        if self.max_iter is not None:
            options["ipopt.max_iter"] = self.max_iter
        if self.silent:
            options |= {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        solver = ca.nlpsol(
            "continuous_power_optimizer",
            "ipopt",
            {"x": params, "f": self._build_objective(active_powers), "g": ca.vertcat(*(constraint.expression for constraint in constraints))},
            options
        )
        return solver, constraints

    def _build_constraints(self, active_powers: ca.SX, reactive_powers: ca.SX, voltages: ca.SX, angles: ca.SX) -> list[Constraint]:
        """Builds CasADi constraints that mirror the numeric power-flow model.
        :param active_powers: Symbolic active generation values.
        :param reactive_powers: Symbolic reactive generation values.
        :param voltages: Symbolic voltage magnitudes.
        :param angles: Symbolic phase angles.
        :return: Symbolic constraints paired with matching lower and upper bounds.
        """
        constraints = [Constraint(angles[0], 0, 0)]
        for node_label, node_data in self.problem.graph.nodes(data=True):
            real_flows = []
            imag_flows = []
            for _, neighbor_label, line_data in self.problem.graph.edges(node_label, data=True):
                neighbor_data = self.problem.graph.nodes[neighbor_label]
                delta = angles[node_data["node_ind"]] - angles[neighbor_data["node_ind"]]
                v_i = voltages[node_data["node_ind"]]
                v_j = voltages[neighbor_data["node_ind"]]
                alpha = line_data["admittance"].real
                beta = line_data["admittance"].imag
                real_flows.append(alpha * v_i ** 2 - v_i * v_j * (alpha * ca.cos(delta) + beta * ca.sin(delta)))
                imag_flows.append(-beta * v_i ** 2 + v_i * v_j * (beta * ca.cos(delta) - alpha * ca.sin(delta)))
                if node_data["node_ind"] < neighbor_data["node_ind"]:
                    current_magnitude_2 = abs(line_data["admittance"]) ** 2 * (v_i ** 2 + v_j ** 2 - 2 * v_i * v_j * ca.cos(delta))
                    constraints.append(Constraint(current_magnitude_2 - line_data["capacity"] ** 2, -np.inf, 0))

            real_gen_power = sum((active_powers[gen_ind] for gen_ind in node_data["gen_inds"]), ca.SX(0))
            imag_gen_power = sum((reactive_powers[gen_ind] for gen_ind in node_data["gen_inds"]), ca.SX(0))
            constraints.extend(
                [Constraint(real_gen_power - node_data["load"].real - sum(real_flows, ca.SX(0)), 0, 0),
                Constraint(imag_gen_power - node_data["load"].imag - sum(imag_flows, ca.SX(0)), 0, 0)])
        return constraints

    def _build_objective(self, active_powers: ca.SX) -> ca.SX:
        """Builds symbolic generation-cost objective without commitment constants.
        :param active_powers: Symbolic active generation values.
        :return: Symbolic objective expression.
        """
        return sum((gen.cost_terms[0] * active_powers[i] ** 2 + gen.cost_terms[1] * active_powers[i] for i, gen in enumerate(self.problem.generators)),
            ca.SX(0))

    def _optimize(self, generator_statuses: str) -> OptimizeResult:
        """Runs IPOPT for a given set of enabled generators.
        :param generator_statuses: Binary generator on/off bitstring.
        :return: IPOPT result with normalized ``fun``, ``penalty``, and ``total`` attributes.
        """
        bounds = self.problem.get_bounds(generator_statuses)
        initial_point = self.get_initial_point(bounds)
        solution = self.solver(x0=initial_point, lbx=[bound[0] for bound in bounds], ubx=[bound[1] for bound in bounds],
            lbg=[constraint.lb for constraint in self.constraints], ubg=[constraint.ub for constraint in self.constraints])
        stats = self.solver.stats()
        result = OptimizeResult(x=np.array(solution["x"]).reshape(-1), success=stats["success"], message=stats["return_status"])
        result.fun = self.get_generation_cost(generator_statuses, result.x)
        result.penalty = self.get_penalty(result.x)
        result.total = result.fun + result.penalty
        return result
