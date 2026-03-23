"""Continuous-variable optimizers for fixed generator commitment states."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

import casadi as ca
import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import NonlinearConstraint

from .EvaluationResult import EvaluationResult
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
    :var max_time_s: Wall-clock limit in seconds.
    :var feasibility_tolerance: Maximum penalty still treated as feasible when comparing candidates.
    :var best_result_callback: Callback invoked whenever the best cached result across all assignments improves.
    :var cache: Map from generator-status bitstring to cached optimization result or the current incumbent during an active run.
    :var best_result: Best cached result across all assignments optimized by this object.
    :var tracking_start_time: Wall-clock timestamp when the current tracked run started.
    """
    problem: PowerFlowProblem
    penalty_mult: float
    max_time_s: float
    feasibility_tolerance: float = 1e-10
    best_result_callback: Callable[[EvaluationResult], None] | None = None
    cache: dict[str, EvaluationResult] = field(default_factory=dict)
    best_result: EvaluationResult | None = field(init=False, default=None)
    tracking_start_time: float = field(init=False, repr=False, default=0)

    def optimize(self, generator_statuses: str) -> EvaluationResult:
        """Returns cached result or runs the solver for the given generator statuses.
        :param generator_statuses: Binary generator on/off bitstring.
        :return: Cached or newly computed optimizer result for the status pattern.
        """
        if generator_statuses not in self.cache:
            self._optimize(generator_statuses)
        return self.cache[generator_statuses]

    @abstractmethod
    def _optimize(self, generator_statuses: str):
        """Finds optimal continuous variables for a given set of enabled generators. Updates cache as it solves.
        :param generator_statuses: Binary generator on/off bitstring.
        """

    def get_initial_point(self, bounds: Sequence[NDArray[float]]) -> list[float]:
        """Returns initial point for the optimization.
        :param bounds: Per-variable lower and upper bounds.
        :return: Midpoints for active/reactive-power bounds followed by all-ones voltages and all-zero angles.
        """
        active_bounds, reactive_bounds, voltage_bounds, angle_bounds = self.problem.split_params(bounds)
        return [np.average(bound) for bound in active_bounds + reactive_bounds] + [1] * len(voltage_bounds) + [0] * len(angle_bounds)

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

    def consider_candidate(self, generator_statuses: str, params: Sequence[float]) -> EvaluationResult:
        """Makes a candidate solution from parameters. Updates cached result if it is better.
        :param generator_statuses: Binary generator on/off bitstring for the run.
        :param params: Full continuous optimization vector.
        :return: Result metrics for ``params``.
        """
        eval_time = time.perf_counter() - self.tracking_start_time
        params = np.array(params).reshape(-1)
        objective = self.get_cost(generator_statuses, params)
        penalty = self.get_penalty(params)
        result = EvaluationResult(eval_time, generator_statuses, params.tolist(), objective, penalty, objective + penalty)
        if result.is_better_than(self.cache.get(generator_statuses), self.feasibility_tolerance):
            self.cache[generator_statuses] = result
            if result.is_better_than(self.best_result, self.feasibility_tolerance):
                self.best_result = result
                if self.best_result_callback is not None:
                    self.best_result_callback(result)
        return result

    def get_cost(self, generator_statuses: str, params: Sequence[float]) -> float:
        """Returns total objective for the given statuses and continuous parameters.
        :param generator_statuses: Binary generator on/off bitstring.
        :param params: Full continuous optimization vector.
        :return: Generation cost plus voltage-deviation penalty.
        """
        active_powers, _, voltages, _ = self.problem.split_params(params)
        return self.problem.get_total_cost(generator_statuses, active_powers, voltages)

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

    def _optimize(self, generator_statuses: str):
        """Runs SLSQP for a given set of enabled generators.
        :param generator_statuses: Binary generator on/off bitstring.
        """
        def cost_function(params: Sequence[float]) -> float:
            result = self.consider_candidate(generator_statuses, params)
            if time.perf_counter() - self.tracking_start_time > self.max_time_s:
                raise TimeoutError
            return result.fun

        bounds = self.problem.get_bounds(generator_statuses)
        initial_point = self.get_initial_point(bounds)
        constraints = [NonlinearConstraint(self.evaluate_equality_constraints, 0, 0), NonlinearConstraint(self.evaluate_inequality_constraints, -np.inf, 0)]
        self.tracking_start_time = time.perf_counter()
        try:
            result = optimize.minimize(cost_function, initial_point, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 2 ** 31 - 1})
        except TimeoutError:
            return
        self.consider_candidate(generator_statuses, result.x)
        self.cache[generator_statuses].final = True
        self.cache[generator_statuses].success = result.success
        self.cache[generator_statuses].message = result.message


class UpdateCallback(ca.Callback):
    """Tracks IPOPT iterates and requests termination after the configured time limit.
    :var _optimizer: Owning optimizer whose incumbent tracking is updated.
    :var _num_params: Number of decision variables in the NLP.
    :var _num_constraints: Number of nonlinear constraints in the NLP.
    :var generator_statuses: Generator-status bitstring for the active solve.
    """
    _optimizer: ContinuousPowerOptimizer
    _num_params: int
    _num_constraints: int
    generator_statuses: str | None

    def __init__(self, optimizer: ContinuousPowerOptimizer, num_params: int, num_constraints: int):
        """Constructs callback for one optimizer instance.
        :param optimizer: Optimizer whose incumbent tracking should be updated.
        :param num_params: Number of decision variables in the NLP.
        :param num_constraints: Number of nonlinear constraints in the NLP.
        :return: Nothing.
        """
        ca.Callback.__init__(self)
        self._optimizer = optimizer
        self._num_params = num_params
        self._num_constraints = num_constraints
        self.generator_statuses = None
        self.construct(f"{type(optimizer).__name__.lower()}_callback")

    def get_n_in(self) -> int:
        """Returns number of callback inputs.
        :return: Number of NLP solver outputs passed into the callback.
        """
        return ca.nlpsol_n_out()

    def get_n_out(self) -> int:
        """Returns number of callback outputs.
        :return: One scalar stop flag.
        """
        return 1

    def get_sparsity_in(self, i: int) -> ca.Sparsity:
        """Returns sparsity for one callback input.
        :param i: Input position.
        :return: Dense sparsity pattern matching the corresponding NLP output.
        """
        sizes = {"x": self._num_params, "f": 1, "g": self._num_constraints, "lam_x": self._num_params, "lam_g": self._num_constraints, "lam_p": 0}
        return ca.Sparsity.dense(sizes[ca.nlpsol_out(i)], 1)

    def get_sparsity_out(self, i: int) -> ca.Sparsity:
        """Returns sparsity for one callback output.
        :param i: Output position.
        :return: Scalar sparsity pattern for the stop flag.
        """
        return ca.Sparsity.scalar()

    def eval(self, args: Sequence[ca.DM]) -> list[int]:
        """Updates incumbent tracking from one IPOPT iterate and optionally requests termination.
        :param args: NLP solver outputs for the current iterate.
        :return: One-element list whose value is nonzero when IPOPT should stop.
        """
        assert self.generator_statuses is not None, "Tracking must be initialized before callback evaluation."
        self._optimizer.consider_candidate(self.generator_statuses, args[0])
        return [time.perf_counter() - self._optimizer.tracking_start_time >= self._optimizer.max_time_s]


@dataclass
class CasadiOptimizer(ContinuousPowerOptimizer):
    """Optimizes continuous variables for fixed generator commitments with CasADi and IPOPT.
    :var solver: Reusable IPOPT-backed nonlinear solver for the problem structure.
    :var constraints: Symbolic constraints together with their valid ranges.
    :var update_callback: CasADi callback used to track incumbents and enforce ``max_time_s``.
    :var silent: Whether to suppress IPOPT and CasADi solver output.
    :var max_iter: Optional IPOPT iteration limit. ``None`` leaves IPOPT at its own default.
    """
    silent: bool = False
    max_iter: int | None = None
    solver: ca.Function = field(init=False, repr=False)
    constraints: list[Constraint] = field(init=False, repr=False)
    update_callback: UpdateCallback = field(init=False, repr=False)

    def __post_init__(self):
        """Builds reusable symbolic problem representation after initialization."""
        vars = ca.SX.sym("params", 2 * len(self.problem.generators) + 2 * len(self.problem.graph))
        active_powers, reactive_powers, voltages, angles = self.problem.split_params(vars)
        self.constraints = self._build_constraints(active_powers, reactive_powers, voltages, angles)
        self.update_callback = UpdateCallback(self, vars.size1(), len(self.constraints))
        options = {"error_on_fail": False, "iteration_callback": self.update_callback, "iteration_callback_step": 1}
        if self.silent:
            options |= {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        if self.max_iter is not None:
            options["ipopt.max_iter"] = self.max_iter
        problem = {"x": vars, "f": self._build_objective(active_powers, voltages), "g": ca.vertcat(*(constraint.expression for constraint in self.constraints))}
        self.solver = ca.nlpsol("casadi", "ipopt", problem, options)

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
            constraints.extend([Constraint(real_gen_power - node_data["load"].real - sum(real_flows, ca.SX(0)), 0, 0),
                Constraint(imag_gen_power - node_data["load"].imag - sum(imag_flows, ca.SX(0)), 0, 0)])
        return constraints

    def _build_objective(self, active_powers: ca.SX, voltages: ca.SX) -> ca.SX:
        """Builds symbolic objective without commitment constants.
        :param active_powers: Symbolic active generation values.
        :param voltages: Symbolic voltage magnitudes.
        :return: Symbolic objective expression.
        """
        generation_cost = \
            sum((gen.cost_terms[0] * active_powers[i] ** 2 + gen.cost_terms[1] * active_powers[i] for i, gen in enumerate(self.problem.generators)), ca.SX(0))
        voltage_deviation_cost = self.problem.voltage_deviation_mult * sum(((voltages[i] - 1) ** 2 for i in range(len(self.problem.graph))), ca.SX(0))
        return generation_cost + voltage_deviation_cost

    def _optimize(self, generator_statuses: str):
        """Runs IPOPT for a given set of enabled generators.
        :param generator_statuses: Binary generator on/off bitstring.
        """
        bounds = self.problem.get_bounds(generator_statuses)
        initial_point = self.get_initial_point(bounds)
        self.tracking_start_time = time.perf_counter()
        self.update_callback.generator_statuses = generator_statuses
        solution = self.solver(x0=initial_point, lbx=[bound[0] for bound in bounds], ubx=[bound[1] for bound in bounds],
                               lbg=[constraint.lb for constraint in self.constraints], ubg=[constraint.ub for constraint in self.constraints])
        stats = self.solver.stats()
        self.consider_candidate(generator_statuses, solution["x"])
        if stats["return_status"] == "User_Requested_Stop":
            return
        self.cache[generator_statuses].final = True
        self.cache[generator_statuses].success = stats["success"]
        self.cache[generator_statuses].message = stats["return_status"]
