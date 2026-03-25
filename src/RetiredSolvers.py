"""Retired continuous-variable solver implementations kept only for debugging."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import casadi as ca
import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import NonlinearConstraint

from .ContinuousPowerOptimizer import Constraint, ContinuousPowerOptimizer, UpdateCallback


@dataclass
class SLSQPOptimizer(ContinuousPowerOptimizer):
    """Optimizes continuous variables for fixed generator commitments with SLSQP."""

    def _optimize(self, generator_statuses: str):
        """Runs SLSQP for a given set of enabled generators.
        :param generator_statuses: Binary generator on/off bitstring.
        """
        def cost_function(params: Sequence[float]) -> float:
            result = self.consider_candidate(generator_statuses, params)
            if time.perf_counter() - start_time > self.max_time_s:
                raise TimeoutError
            return result.fun

        bounds = self.problem.get_bounds(generator_statuses)
        initial_point = self.get_initial_point(bounds)
        constraints = [NonlinearConstraint(self.evaluate_equality_constraints, 0, 0), NonlinearConstraint(self.evaluate_inequality_constraints, -np.inf, 0)]
        start_time = time.perf_counter()
        try:
            result = optimize.minimize(cost_function, initial_point, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 2 ** 31 - 1})
        except TimeoutError:
            self.cache[generator_statuses].extra["opt_time"] = self.max_time_s
            return
        self.consider_candidate(generator_statuses, result.x)
        self.cache[generator_statuses].extra |= {"opt_time": time.perf_counter() - start_time, "final": True, "success": result.success, "message": result.message}


class UpdateCallbackRectangular(UpdateCallback):
    """Tracks IPOPT iterates for rectangular-voltage optimization.
    :var _optimizer: Owning rectangular optimizer whose incumbent tracking is updated.
    """
    _optimizer: CasadiOptimizerRectangular

    def eval(self, args: Sequence[ca.DM]) -> list[int]:
        """Updates incumbent tracking from one IPOPT iterate and optionally requests termination.
        :param args: NLP solver outputs for the current iterate.
        :return: One-element list whose value is nonzero when IPOPT should stop.
        """
        return super().eval([ca.DM(self._optimizer.to_polar_params(args[0])), *args[1:]])


@dataclass
class CasadiOptimizerRectangular(ContinuousPowerOptimizer):
    """Optimizes continuous variables with CasADi and IPOPT using rectangular voltage coordinates.
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
    update_callback: UpdateCallbackRectangular = field(init=False, repr=False)

    def __post_init__(self):
        """Builds reusable symbolic problem representation after initialization."""
        vars = ca.SX.sym("params", 2 * len(self.problem.generators) + 2 * len(self.problem.graph))
        active_powers, reactive_powers, voltage_reals, voltage_imags = self.problem.split_params(vars)
        self.constraints = self._build_constraints(active_powers, reactive_powers, voltage_reals, voltage_imags)
        self.update_callback = UpdateCallbackRectangular(self, vars.size1(), len(self.constraints))
        options = {"error_on_fail": False, "iteration_callback": self.update_callback, "iteration_callback_step": 1}
        if self.silent:
            options |= {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        if self.max_iter is not None:
            options["ipopt.max_iter"] = self.max_iter
        problem = {"x": vars, "f": self._build_objective(active_powers, voltage_reals, voltage_imags), "g": ca.vertcat(*(constraint.expression for constraint in self.constraints))}
        self.solver = ca.nlpsol("casadi_rectangular", "ipopt", problem, options)

    def _build_constraints(self, active_powers: ca.SX, reactive_powers: ca.SX, voltage_reals: ca.SX, voltage_imags: ca.SX) -> list[Constraint]:
        """Builds CasADi constraints using rectangular voltage coordinates.
        :param active_powers: Symbolic active generation values.
        :param reactive_powers: Symbolic reactive generation values.
        :param voltage_reals: Symbolic voltage real parts.
        :param voltage_imags: Symbolic voltage imaginary parts.
        :return: Symbolic constraints paired with matching lower and upper bounds.
        """
        constraints = [Constraint(voltage_imags[0], 0, 0), Constraint(voltage_reals[0], 0, np.inf)]
        for node_label, node_data in self.problem.graph.nodes(data=True):
            i = node_data["node_ind"]
            voltage_magnitude_2 = voltage_reals[i] ** 2 + voltage_imags[i] ** 2
            constraints.append(Constraint(voltage_magnitude_2, node_data["voltage_range"][0] ** 2, node_data["voltage_range"][1] ** 2))
            real_flows = []
            imag_flows = []
            for _, neighbor_label, line_data in self.problem.graph.edges(node_label, data=True):
                neighbor_data = self.problem.graph.nodes[neighbor_label]
                j = neighbor_data["node_ind"]
                voltage_overlap = voltage_reals[i] * voltage_reals[j] + voltage_imags[i] * voltage_imags[j]
                cross_term = voltage_reals[i] * voltage_imags[j] - voltage_imags[i] * voltage_reals[j]
                alpha = line_data["admittance"].real
                beta = line_data["admittance"].imag
                real_flows.append(alpha * (voltage_magnitude_2 - voltage_overlap) + beta * cross_term)
                imag_flows.append(-beta * (voltage_magnitude_2 - voltage_overlap) + alpha * cross_term)
                if i < j:
                    current_magnitude_2 = abs(line_data["admittance"]) ** 2 * ((voltage_reals[i] - voltage_reals[j]) ** 2 + (voltage_imags[i] - voltage_imags[j]) ** 2)
                    constraints.append(Constraint(current_magnitude_2 - line_data["capacity"] ** 2, -np.inf, 0))

            real_gen_power = sum((active_powers[gen_ind] for gen_ind in node_data["gen_inds"]), ca.SX(0))
            imag_gen_power = sum((reactive_powers[gen_ind] for gen_ind in node_data["gen_inds"]), ca.SX(0))
            constraints.extend([Constraint(real_gen_power - node_data["load"].real - sum(real_flows, ca.SX(0)), 0, 0),
                                Constraint(imag_gen_power - node_data["load"].imag - sum(imag_flows, ca.SX(0)), 0, 0)])
        return constraints

    def _build_objective(self, active_powers: ca.SX, voltage_reals: ca.SX, voltage_imags: ca.SX) -> ca.SX:
        """Builds symbolic objective without commitment constants.
        :param active_powers: Symbolic active generation values.
        :param voltage_reals: Symbolic voltage real parts.
        :param voltage_imags: Symbolic voltage imaginary parts.
        :return: Symbolic objective expression.
        """
        generation_cost = sum((gen.cost_terms[0] * active_powers[i] ** 2 + gen.cost_terms[1] * active_powers[i] for i, gen in enumerate(self.problem.generators)), ca.SX(0))
        voltage_deviation_cost = self.problem.voltage_deviation_mult * sum(((voltage_reals[i] ** 2 + voltage_imags[i] ** 2 - 1) ** 2 for i in range(len(self.problem.graph))), ca.SX(0))
        return generation_cost + voltage_deviation_cost

    def _optimize(self, generator_statuses: str):
        """Runs IPOPT for a given set of enabled generators.
        :param generator_statuses: Binary generator on/off bitstring.
        """
        bounds = self._get_rectangular_bounds(generator_statuses)
        initial_point = self.get_initial_point(bounds)
        self.update_callback.generator_statuses = generator_statuses
        self.update_callback.start_time = time.perf_counter()
        solution = self.solver(x0=initial_point, lbx=[bound[0] for bound in bounds], ubx=[bound[1] for bound in bounds],
                               lbg=[constraint.lb for constraint in self.constraints], ubg=[constraint.ub for constraint in self.constraints])
        stats = self.solver.stats()
        self.consider_candidate(generator_statuses, self.to_polar_params(solution["x"]))
        self.cache[generator_statuses].extra["opt_time"] = time.perf_counter() - self.update_callback.start_time
        if stats["return_status"] == "User_Requested_Stop":
            return
        self.cache[generator_statuses].extra |= {"final": True, "success": stats["success"], "message": stats["return_status"]}

    def _get_rectangular_bounds(self, generator_statuses: str) -> list[NDArray[float]]:
        """Returns bounds for rectangular-voltage optimization variables.
        :param generator_statuses: Binary generator on/off bitstring.
        :return: Bounds for active/reactive generation together with rectangular voltage components.
        """
        active_bounds, reactive_bounds, voltage_bounds, _ = self.problem.split_params(self.problem.get_bounds(generator_statuses))
        voltage_real_bounds = [np.array((-bound[1], bound[1])) for bound in voltage_bounds]
        voltage_imag_bounds = [np.array((-bound[1], bound[1])) for bound in voltage_bounds]
        return list(active_bounds) + list(reactive_bounds) + voltage_real_bounds + voltage_imag_bounds

    def to_polar_params(self, params: Sequence[float]) -> NDArray[float]:
        """Converts rectangular optimizer variables to the canonical returned parameter vector.
        :param params: Full rectangular optimization vector.
        :return: Canonical parameter vector ordered as active powers, reactive powers, voltages, and angles.
        """
        params = np.array(params).reshape(-1)
        active_powers, reactive_powers, voltage_reals, voltage_imags = self.problem.split_params(params)
        voltages = np.sqrt(voltage_reals ** 2 + voltage_imags ** 2)
        angles = np.arctan2(voltage_imags, voltage_reals)
        return np.concatenate((active_powers, reactive_powers, voltages, angles))
