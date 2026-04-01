"""Classical and hybrid solvers for ``PowerFlowProblem`` instances."""

import shutil
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from math import inf
from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace, Categorical
from numpy import random
from pyscipopt import Eventhdlr, Model, SCIP_EVENTTYPE, SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING, sin, cos, quicksum
from pyscipopt.recipes.nonlinear import set_nonlinear_objective
from smac import AlgorithmConfigurationFacade, Scenario
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialValue

from . import utils
from .ContinuousPowerOptimizer import ContinuousPowerOptimizer
from .EvaluationResult import EvaluationResult
from .HistoryEntry import HistoryEntry
from .PowerFlowProblem import PowerFlowProblem
from .Sampler import ExactSampler
from .VariationalQuantumProgram import VariationalQuantumProgram


class HistoryEventHandler(Eventhdlr):
    """Records incumbent solutions each time SCIP finds a better one."""

    def __init__(self, variables: dict[str, list], progress_path: Path, start_time: float):
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

    def eventinit(self):
        """Registers event subscription for incumbent updates."""
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        """Removes event subscription for incumbent updates."""
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event: object):
        """Appends current time and evaluation result for each incumbent event.
        :param event: Event payload provided by SCIP.
        """
        current_time = time.perf_counter() - self.start_time
        self.history.append(HistoryEntry(current_time, None, SCIPSolver.extract_evaluation_result(self.model, self.variables)))
        pd.to_pickle(self.history, self.progress_path)


class PowerFlowSolver(ABC):
    """Base class for power grid problem solvers.
    :var name: Canonical solver name used in file naming.
    """
    feasibility_tolerance: float
    name: str

    @abstractmethod
    def solve(self, problem: PowerFlowProblem, progress_path: Path) -> tuple[list[HistoryEntry], dict[str, Any]]:
        """Solves a given power grid optimization problem and returns its history together with solver-specific extras.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Path for persisting incumbent progress snapshots.
        :return: Solver history whose last entry is the final incumbent together with solver-specific extra information.
        """
        pass


def get_inner_optimizer_extra(inner_optimizer: ContinuousPowerOptimizer) -> dict[str, float | int]:
    """Returns common summary metrics for solvers backed by the inner continuous optimizer.
    :param inner_optimizer: Inner continuous optimizer whose cache holds one entry per optimized bitstring.
    :return: Average inner optimization time together with the number of optimized bitstrings.
    """
    return {"avg_inner": sum(result.extra["opt_time"] for result in inner_optimizer.cache.values()) / len(inner_optimizer.cache),
            "optimized_bitstrings": len(inner_optimizer.cache)}


@dataclass
class SCIPSolver(PowerFlowSolver):
    """Uses SCIP library to solve power grid problems classically.
    :var name: Canonical solver name used in file naming.
    :var silent: Whether SCIP output is suppressed while solving.
    :var seed: Randomization seed passed to SCIP.
    :var feasibility_tolerance: SCIP primal feasibility tolerance used internally during solving.
    """
    feasibility_tolerance: float = 1e-10
    silent: bool = False
    seed: int | None = None
    name: str = "scip"

    def build_model_power_flow(self, problem: PowerFlowProblem) -> tuple[Model, dict[str, list]]:
        """Builds model based on problem description.
        :param problem: Optimization problem to encode in SCIP.
        :return: SCIP model and grouped variable containers.
        """
        model = Model("PowerFlowAC")
        model.setRealParam("numerics/feastol", self.feasibility_tolerance)
        model.setStringParam("nlp/solver", "ipopt")
        model.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)
        model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
        model.setSeparating(SCIP_PARAMSETTING.FAST)
        model.setPresolve(SCIP_PARAMSETTING.FAST)
        if self.silent:
            model.hideOutput()
        if self.seed:
            model.setIntParam("randomization/randomseedshift", self.seed)
            model.setIntParam("randomization/permutationseed", self.seed)
            model.setIntParam("randomization/lpseed", self.seed)
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
    def extract_evaluation_result(model: Model, variables: dict[str, list]) -> EvaluationResult:
        """Extracts optimized variables from model and builds evaluation-result data.
        :param model: Solved SCIP model with at least one solution.
        :param variables: Grouped variable containers returned by model builder.
        :return: Extracted evaluation result for the current incumbent.
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
        return EvaluationResult(generator_statuses, np.concatenate((active_powers, reactive_powers, voltages, angles)).tolist(), cost, 0, cost)

    def solve(self, problem: PowerFlowProblem, progress_path: Path) -> tuple[list[HistoryEntry], dict[str, Any]]:
        """Solves given problem and returns its incumbent history.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Path for persisting incumbent progress snapshots.
        :return: Solver history whose last entry is the final incumbent together with an extra-info dict.
        """
        t1 = time.perf_counter()
        model, variables = self.build_model_power_flow(problem)
        history_handler = HistoryEventHandler(variables, progress_path, t1)
        model.includeEventhdlr(history_handler, "incumbent_history", "Records incumbent solution history.")
        model.optimize()
        assert str(model.getStatus()) != "infeasible", "Infeasible instance"
        pd.to_pickle(history_handler.history, progress_path)
        return history_handler.history, {}


@dataclass
class SmacSolver(PowerFlowSolver):
    """Optimizes binary generator assignments with SMAC3 and continuous variables with the inner optimizer.
    :var name: Canonical solver name used in file naming.
    :var inner_optimizer_factory: Factory that creates continuous optimizers for a given problem.
    :var seed: Optional random seed for SMAC3.
    :var feasibility_tolerance: Feasibility tolerance passed through to the inner optimizer.
    :var silent: Whether SMAC3 output is suppressed.
    """
    inner_optimizer_factory: Callable[[PowerFlowProblem], ContinuousPowerOptimizer]
    feasibility_tolerance: float = 1e-10
    silent: bool = False
    seed: int | None = None
    name: str = "smac"

    def solve(self, problem: PowerFlowProblem, progress_path: Path) -> tuple[list[HistoryEntry], dict[str, Any]]:
        """Optimizes binary assignments with SMAC3 and returns the incumbent history.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Path for persisting incumbent progress snapshots.
        :return: Solver history whose last entry is the final incumbent together with solver-specific extra information.
        """
        def update_history(new_result: EvaluationResult):
            history.append(HistoryEntry(time.perf_counter() - start_time, None, new_result))
            pd.to_pickle(history, progress_path)

        history = []
        inner_optimizer = self.inner_optimizer_factory(problem)
        inner_optimizer.feasibility_tolerance = self.feasibility_tolerance
        inner_optimizer.best_result_callback = update_history
        num_bitstrings = 2 ** len(problem.generators)
        start_time = time.perf_counter()
        output_directory = progress_path.parent / f".smac_{progress_path.stem}"
        shutil.rmtree(output_directory, ignore_errors=True)
        optimizer = self._build_optimizer(problem, output_directory)
        while len(inner_optimizer.cache) < num_bitstrings:
            trial = optimizer.ask()
            generator_statuses = self._config_to_generator_statuses(trial.config, len(problem.generators))
            assert generator_statuses not in inner_optimizer.cache, "SMAC solver asked for an already known bitstring."
            result = inner_optimizer.optimize(generator_statuses)
            optimizer.tell(trial, TrialValue(cost=result.total))
        assert len(history) > 0, "SMAC solver did not record any history entry."
        return history, get_inner_optimizer_extra(inner_optimizer)

    def _build_optimizer(self, problem: PowerFlowProblem, output_directory: Path) -> AlgorithmConfigurationFacade:
        """Builds the SMAC3 optimizer for binary generator assignments.
        :param problem: Power-flow optimization problem whose generators define the binary search space.
        :param output_directory: Temporary SMAC3 output directory.
        :return: Configured SMAC3 optimizer instance.
        """
        scenario = Scenario(self._get_configspace(problem), deterministic=True, n_trials=2 ** 31 - 1, output_directory=output_directory, seed=self.seed)
        kwargs = {}
        if self.silent:
            kwargs["logging_level"] = False
        kwargs["config_selector"] = ConfigSelector(scenario, retries=inf)
        return AlgorithmConfigurationFacade(scenario, self._dummy_target_function, **kwargs)

    def _get_configspace(self, problem: PowerFlowProblem) -> ConfigurationSpace:
        """Builds the binary configuration space for generator commitment variables.
        :param problem: Power-flow optimization problem whose generators define the binary variables.
        :return: Configuration space with one categorical ``{0, 1}`` variable per generator.
        """
        configspace = ConfigurationSpace(seed=self.seed)
        configspace.add([Categorical(f"u_{i}", [0, 1]) for i in range(len(problem.generators))])
        return configspace

    @staticmethod
    def _dummy_target_function(config: Configuration, seed: int = 0) -> float:
        """Raises if SMAC3 tries to use the target-function path instead of ask/tell.
        :param config: Unused configuration.
        :param seed: Unused seed value.
        :return: Never returns.
        """
        raise RuntimeError("SMAC target function should not be called in ask/tell mode.")

    @staticmethod
    def _config_to_generator_statuses(config: Configuration, num_generators: int) -> str:
        """Converts a SMAC3 configuration to the generator-status bitstring used by the inner optimizer.
        :param config: Binary configuration sampled by SMAC3.
        :param num_generators: Number of generator-status bits.
        :return: Generator-status bitstring.
        """
        return "".join(str(int(config[f"u_{i}"])) for i in range(num_generators))


@dataclass
class UniformSolver(PowerFlowSolver):
    """Samples generator assignments uniformly and optimizes them classically until all assignments are seen.
    :var name: Canonical solver name used in file naming.
    :var inner_optimizer_factory: Factory that creates continuous optimizers for a given problem.
    :var seed: Optional random seed for uniform bitstring sampling.
    :var feasibility_tolerance: Feasibility tolerance passed through to the inner optimizer.
    """
    inner_optimizer_factory: Callable[[PowerFlowProblem], ContinuousPowerOptimizer]
    feasibility_tolerance: float = 1e-10
    seed: int | None = None
    name: str = "uniform"

    def solve(self, problem: PowerFlowProblem, progress_path: Path) -> tuple[list[HistoryEntry], dict[str, Any]]:
        """Samples bitstrings uniformly and returns the incumbent history.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Path for persisting incumbent progress snapshots.
        :return: Solver history whose last entry is the final incumbent together with solver-specific extra information.
        """
        def update_history(new_result: EvaluationResult):
            history.append(HistoryEntry(time.perf_counter() - start_time, None, new_result))
            pd.to_pickle(history, progress_path)

        history = []
        inner_optimizer = self.inner_optimizer_factory(problem)
        inner_optimizer.feasibility_tolerance = self.feasibility_tolerance
        inner_optimizer.best_result_callback = update_history
        num_bitstrings = 2 ** len(problem.generators)
        rng = random.default_rng(self.seed)
        start_time = time.perf_counter()
        while len(inner_optimizer.cache) < num_bitstrings:
            generator_statuses = format(rng.integers(num_bitstrings), f"0{len(problem.generators)}b")
            if generator_statuses in inner_optimizer.cache:
                continue
            inner_optimizer.optimize(generator_statuses)
        assert len(history) > 0, "Uniform solver did not record any history entry."
        return history, get_inner_optimizer_extra(inner_optimizer)


@dataclass
class HybridSolver(PowerFlowSolver):
    """Optimizes binary variables on a quantum computer. Continuous variables are optimized classically by the problem.
    :var name: Canonical solver name used in file naming.
    :var vqp: Variational quantum program used for binary-variable search.
    :var inner_optimizer_factory: Factory that creates continuous optimizers for a given problem.
    :var seed: Optional random seed for initial quantum-parameter sampling.
    :var feasibility_tolerance: Feasibility tolerance; history stores only entries with penalty below this threshold.
    """
    vqp: VariationalQuantumProgram
    inner_optimizer_factory: Callable[[PowerFlowProblem], ContinuousPowerOptimizer]
    feasibility_tolerance: float = 1e-10
    seed: int | None = None
    name: str = "hybrid"

    def solve(self, problem: PowerFlowProblem, progress_path: Path, exact_final_expectation: bool = False) -> tuple[list[HistoryEntry], dict[str, Any]]:
        """Optimizes quantum parameters and returns the feasible incumbent history.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Path for persisting incumbent progress snapshots.
        :param exact_final_expectation: Whether to compute the exact final bitstring distribution and expectation after optimization.
        :return: Tuple of feasible incumbent history and optional extra hybrid-run information.
        """
        def update_history(new_result: EvaluationResult):
            history.append(HistoryEntry(self.vqp.get_current_classical_time(), self.vqp.num_jobs, new_result))
            pd.to_pickle(history, progress_path)

        history = []
        inner_optimizer = self.inner_optimizer_factory(problem)
        inner_optimizer.feasibility_tolerance = self.feasibility_tolerance
        inner_optimizer.best_result_callback = update_history
        rng = random.default_rng(self.seed)
        initial_angles = rng.uniform(-np.pi, np.pi, len(self.vqp.circuit.parameters))
        result = self.vqp.optimize_parameters(lambda generator_statuses: inner_optimizer.optimize(generator_statuses).total, initial_angles)
        assert result.success, f"Angle optimization failed: {result.message}"
        assert len(history) > 0, "Hybrid solver did not record any feasible history entry."
        extra = get_inner_optimizer_extra(inner_optimizer) | {"total_jobs": self.vqp.num_jobs}

        if exact_final_expectation:
            exact_sampler = ExactSampler()
            final_probs = exact_sampler.get_sample_probabilities(self.vqp.circuit, result.x)
            inner_optimizer.best_result_callback = None
            cost_expectation = utils.get_cost_expectation(lambda bitstring: inner_optimizer.optimize(bitstring).total, final_probs)
            extra |= {"final_probs": final_probs, "cost_expectation": cost_expectation}
        return history, extra
