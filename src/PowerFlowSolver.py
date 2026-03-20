"""Classical and hybrid solvers for ``PowerFlowProblem`` instances."""

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy import random
from pyscipopt import Eventhdlr, Model, SCIP_EVENTTYPE, sin, cos, quicksum
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

from . import utils
from .ContinuousPowerOptimizer import ContinuousPowerOptimizer
from .EvaluationResult import EvaluationResult
from .HistoryEntry import HistoryEntry
from .PowerFlowProblem import PowerFlowProblem, PowerFlowSolution
from .Sampler import ExactSampler
from .VariationalQuantumProgram import VariationalQuantumProgram


class HistoryEventHandler(Eventhdlr):
    """Records incumbent solutions each time SCIP finds a better one."""

    def __init__(self, variables: dict[str, list], progress_path: Path, start_time: float) -> None:
        """Initializes event handler state.
        :param variables: Structured variable container returned by model builder.
        :param progress_path: Path for persisting incumbent data and full history.
        :param start_time: Wall-clock reference from ``time.perf_counter()`` at solve start.
        """
        super().__init__()
        self.variables = variables
        self.progress_path = progress_path
        self.start_time = start_time
        self.history: list[HistoryEntry] = []

    def eventinit(self) -> None:
        """Registers event subscription for incumbent updates."""
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self) -> None:
        """Removes event subscription for incumbent updates."""
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event: object) -> None:
        """Appends current time and evaluation result for each incumbent event.
        :param event: Event payload provided by SCIP.
        """
        current_time = time.perf_counter() - self.start_time
        solution = ClassicalSolver.extract_solution(self.model, self.variables)
        params = np.concatenate((solution.active_powers, solution.reactive_powers, solution.voltages, solution.angles)).tolist()
        evaluation_result = EvaluationResult(solution.generator_statuses, params, solution.cost, 0, solution.cost)
        self.history.append(HistoryEntry(current_time, None, evaluation_result))
        pd.to_pickle(self.history, self.progress_path)


class PowerFlowSolver(ABC):
    """Base class for power grid problem solvers.
    :var name: Canonical solver name used in file naming.
    """
    name: str

    @abstractmethod
    def solve(self, problem: PowerFlowProblem, progress_path: Path | None = None) -> PowerFlowSolution:
        """Solves a given power grid optimization problem and returns its solution.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Optional path for persisting incumbent progress snapshots.
        :return: Solution object produced by the solver.
        """
        pass


@dataclass
class ClassicalSolver(PowerFlowSolver):
    """Uses SCIP library to solve power grid problems classically.
    :var name: Canonical solver name used in file naming.
    :var silent: Whether SCIP output is suppressed while solving.
    """
    name: str = "scip"
    silent: bool = False

    def build_model_power_flow(self, problem: PowerFlowProblem) -> tuple[Model, dict[str, list]]:
        """Builds model based on problem description.
        :param problem: Optimization problem to encode in SCIP.
        :return: SCIP model and grouped variable containers.
        """
        model = Model("PowerFlowAC")
        if self.silent:
            model.hideOutput()
        cost_terms = []
        variables = defaultdict(lambda: [[] for _ in range(len(problem.graph))])
        for node_label, node_data in problem.graph.nodes(data=True):
            for i, gen in enumerate(node_data["generators"]):
                u = model.addVar(vtype="B", name=f"u_{node_label}_{i}")
                p = model.addVar(name=f"p_{node_label}_{i}")
                q = model.addVar(name=f"q_{node_label}_{i}")
                model.addCons(p >= gen.power_range[0] * u, name=f"p_min_{node_label}_{i}")
                model.addCons(p <= gen.power_range[1] * u, name=f"p_max_{node_label}_{i}")
                model.addCons(q >= gen.reactive_power_range[0] * u, name=f"q_min_{node_label}_{i}")
                model.addCons(q <= gen.reactive_power_range[1] * u, name=f"q_max_{node_label}_{i}")
                cost_terms.append(gen.cost_terms[0] * p ** 2 + gen.cost_terms[1] * p + gen.cost_terms[2] * u)
                variables["u"][node_data["node_ind"]].append(u)
                variables["p"][node_data["node_ind"]].append(p)
                variables["q"][node_data["node_ind"]].append(q)
            variables["v"][node_data["node_ind"]] = model.addVar(lb=node_data["voltage_range"][0], ub=node_data["voltage_range"][1], name=f"v_{node_label}")
            variables["d"][node_data["node_ind"]] = model.addVar(lb=node_data["angle_range"][0], ub=node_data["angle_range"][1], name=f"d_{node_label}")
            cost_terms.append(problem.voltage_deviation_mult * (variables["v"][node_data["node_ind"]] - 1) ** 2)

        model.addCons(variables["d"][0] == 0, name="fixed angle")
        for node_label, node_data in problem.graph.nodes(data=True):
            real_flows = []
            imag_flows = []
            for _, neighbor_label, line_data in problem.graph.edges(node_label, data=True):
                neighbor_data = problem.graph.nodes[neighbor_label]
                delta = variables["d"][node_data["node_ind"]] - variables["d"][neighbor_data["node_ind"]]
                v_i = variables["v"][node_data["node_ind"]]
                v_j = variables["v"][neighbor_data["node_ind"]]
                alpha = line_data["admittance"].real
                beta = line_data["admittance"].imag
                real_flows.append(alpha * v_i ** 2 - v_i * v_j * (alpha * cos(delta) + beta * sin(delta)))
                imag_flows.append(-beta * v_i ** 2 + v_i * v_j * (beta * cos(delta) - alpha * sin(delta)))
                if node_data["node_ind"] < neighbor_data["node_ind"]:
                    abs_current_2 = abs(line_data["admittance"]) ** 2 * (v_i ** 2 + v_j ** 2 - 2 * v_i * v_j * cos(delta))
                    model.addCons(abs_current_2 - line_data["capacity"] ** 2 <= 0, name=f"capacity_{node_label}_{neighbor_label}")

            model.addCons(quicksum(variables["p"][node_data["node_ind"]]) - node_data["load"].real - quicksum(real_flows) == 0,
                          name=f"net_power_real_{node_label}")
            model.addCons(quicksum(variables["q"][node_data["node_ind"]]) - node_data["load"].imag - quicksum(imag_flows) == 0,
                          name=f"net_power_imag_{node_label}")

        set_nonlinear_objective(model, quicksum(cost_terms), sense="minimize")
        return model, variables

    @staticmethod
    def extract_solution(model: Model, variables: dict[str, list]) -> PowerFlowSolution:
        """Extracts optimized variables from model and fills out solution instance.
        :param model: Solved SCIP model with at least one solution.
        :param variables: Grouped variable containers returned by model builder.
        :return: Extracted power-flow solution.
        """
        best_solution = model.getBestSol()
        all_u = sum(variables["u"], [])
        generator_statuses = "".join([str(int(model.getSolVal(best_solution, var))) for var in all_u])
        all_p = sum(variables["p"], [])
        active_powers = np.array([model.getSolVal(best_solution, var) for var in all_p])
        all_q = sum(variables["q"], [])
        reactive_powers = np.array([model.getSolVal(best_solution, var) for var in all_q])
        voltages = np.array([model.getSolVal(best_solution, var) for var in variables["v"]])
        angles = np.array([model.getSolVal(best_solution, var) for var in variables["d"]])
        cost = model.getSolObjVal(best_solution)
        return PowerFlowSolution(generator_statuses, active_powers, reactive_powers, voltages, angles, cost)

    def solve(self, problem: PowerFlowProblem, progress_path: Path | None = None) -> PowerFlowSolution:
        """Solves given problem and returns its solution.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Optional path for persisting incumbent progress snapshots.
        :return: Final solution with bound-history metadata.
        """
        t1 = time.perf_counter()
        model, variables = self.build_model_power_flow(problem)
        if progress_path is not None:
            history_handler = HistoryEventHandler(variables, progress_path, t1)
            model.includeEventhdlr(history_handler, "incumbent_history", "Records incumbent solution history.")
        model.optimize()

        status = str(model.getStatus())
        if status == "infeasible":
            raise AssertionError("Infeasible instance")
                
        solution = ClassicalSolver.extract_solution(model, variables)
        if progress_path is not None:
            solution.history = history_handler.history
        solution.extra["solve_status"] = status
        return solution


@dataclass
class HybridSolver(PowerFlowSolver):
    """Optimizes binary variables on a quantum computer. Continuous variables are optimized classically by the problem.
    :var name: Canonical solver name used in file naming.
    :var vqp: Variational quantum program used for binary-variable search.
    :var inner_optimizer_factory: Factory that creates continuous optimizers for a given problem.
    :var seed: Optional random seed for initial quantum-parameter sampling.
    :var feasibility_tolerance: Feasibility tolerance; history stores only entries with penalty below this threshold.
    :var exact_final_expectation: Whether to compute the exact final bitstring distribution and expectation after optimization.
    """
    name: str = field(init=False)
    vqp: VariationalQuantumProgram
    inner_optimizer_factory: Callable[[PowerFlowProblem], ContinuousPowerOptimizer]
    seed: int = None
    feasibility_tolerance: float = 1e-10
    exact_final_expectation: bool = False

    def __post_init__(self) -> None:
        """Initializes derived solver metadata."""
        inner_optimizer_type = self.inner_optimizer_factory.func if isinstance(self.inner_optimizer_factory, partial) else self.inner_optimizer_factory
        self.name = {"SLSQPOptimizer": "slsqp", "CasadiOptimizer": "casadi"}[inner_optimizer_type.__name__]

    def solve(self, problem: PowerFlowProblem, progress_path: Path | None = None) -> PowerFlowSolution:
        """Optimizes quantum parameters and returns the last recorded feasible solution.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Optional path for persisting incumbent progress snapshots.
        :return: Last recorded feasible solution obtained from sampled binary and optimized continuous parameters.
        """
        def get_assignment_cost_tracked(generator_statuses: str) -> float:
            nonlocal best_result
            optimized_result = inner_optimizer.optimize(generator_statuses)
            if optimized_result.is_better_than(best_result, self.feasibility_tolerance):
                best_result = optimized_result
                history.append(HistoryEntry(self.vqp.get_current_classical_time(), self.vqp.num_jobs, optimized_result))
                if progress_path is not None:
                    pd.to_pickle(history, progress_path)
            return optimized_result.total

        inner_optimizer = self.inner_optimizer_factory(problem)
        rng = random.default_rng(self.seed)
        initial_angles = rng.uniform(-np.pi, np.pi, len(self.vqp.circuit.parameters))
        history = []
        best_result = None
        result = self.vqp.optimize_parameters(get_assignment_cost_tracked, initial_angles)
        assert result.success, f"Angle optimization failed: {result.message}"

        assert len(history) > 0, "Hybrid solver did not record any feasible history entry."
        final_result = history[-1].evaluation_result
        active_powers, reactive_powers, voltages, angles = problem.split_params(np.array(final_result.params))
        solution = PowerFlowSolution(final_result.generator_statuses, active_powers, reactive_powers, voltages, angles, final_result.fun, history)

        if self.exact_final_expectation:
            exact_sampler = ExactSampler()
            solution.extra["final_probs"] = exact_sampler.get_sample_probabilities(self.vqp.circuit, result.x)
            solution.extra["cost_expectation"] = \
                utils.get_cost_expectation(lambda bitstring: inner_optimizer.optimize(bitstring).total, solution.extra["final_probs"])
        return solution
