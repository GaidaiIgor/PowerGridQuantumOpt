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
from numpy import ndarray, random
from pyscipopt import Eventhdlr, Model, SCIP_EVENTTYPE, SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING, sin, cos, quicksum
from pyscipopt.recipes.nonlinear import set_nonlinear_objective
from smac import AlgorithmConfigurationFacade, Scenario
from smac.main.config_selector import ConfigSelector
from smac.runhistory import TrialValue

from . import utils
from .ContinuousPowerOptimizer import ContinuousPowerOptimizer, get_optimizer_stats
from .EvaluationResult import EvaluationResult
from .HistoryEntry import HistoryEntry
from .PowerFlowProblem import PowerFlowProblem
from .Sampler import ExactSampler
from .VariationalQuantumProgram import VariationalQuantumProgram


class PowerFlowSolver(ABC):
    """Base class for power grid problem solvers.
    :var name: Canonical solver name used in file naming.
    """
    violation_tolerance: float
    name: str

    @abstractmethod
    def solve(self, problem: PowerFlowProblem, progress_path: Path) -> tuple[list[HistoryEntry], dict[str, Any]]:
        """Solves a given power grid optimization problem and returns its history together with solver-specific extras.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Path for persisting incumbent progress snapshots.
        :return: Solver history whose last entry is the final incumbent together with solver-specific extra information.
        """
        pass


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
        self.history.append(HistoryEntry(current_time, None, {}, SCIPSolver.extract_evaluation_result(self.model, self.variables)))
        pd.to_pickle(self.history, self.progress_path)


@dataclass
class SCIPSolver(PowerFlowSolver):
    """Uses SCIP library to solve power grid problems classically.
    :var name: Canonical solver name used in file naming.
    :var silent: Whether SCIP output is suppressed while solving.
    :var seed: Randomization seed passed to SCIP.
    :var violation_tolerance: SCIP primal feasibility tolerance used internally during solving.
    """
    violation_tolerance: float = 1e-10
    silent: bool = False
    seed: int | None = None
    name: str = "scip"

    def build_model_power_flow(self, problem: PowerFlowProblem) -> tuple[Model, dict[str, list]]:
        """Builds model based on problem description.
        :param problem: Optimization problem to encode in SCIP.
        :return: SCIP model and grouped variable containers.
        """
        model = Model("PowerFlowAC")
        model.setRealParam("numerics/feastol", self.violation_tolerance)
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
    :var violation_tolerance: Violation tolerance passed through to the inner optimizer.
    :var silent: Whether SMAC3 output is suppressed.
    """
    inner_optimizer_factory: Callable[[PowerFlowProblem], ContinuousPowerOptimizer]
    violation_tolerance: float = 1e-10
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
            """Appends a new incumbent history entry.
            :param new_result: Improved incumbent returned by the inner optimizer.
            """
            current_time = time.perf_counter()
            stats = get_optimizer_stats(inner_optimizer)
            history.append(HistoryEntry(current_time - start_time, None, stats.copy(), new_result))
            pd.to_pickle(history, progress_path)

        history = []
        inner_optimizer = self.inner_optimizer_factory(problem)
        inner_optimizer.violation_tolerance = self.violation_tolerance
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
        return history, get_optimizer_stats(inner_optimizer)

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
    """Samples generator assignments uniformly and optimizes them classically until all assignments are seen or the solver time limit is exceeded.
    :var name: Canonical solver name used in file naming.
    :var inner_optimizer_factory: Factory that creates continuous optimizers for a given problem.
    :var max_time: Maximum wall-clock solve time in seconds, or ``None`` to disable the cap.
    :var violation_tolerance: Violation tolerance passed through to the inner optimizer.
    :var seed: Optional random seed for uniform bitstring sampling.
    """
    inner_optimizer_factory: Callable[[PowerFlowProblem], ContinuousPowerOptimizer]
    max_time: float | None = None
    violation_tolerance: float = 1e-10
    seed: int | None = None
    name: str = "uniform"

    def solve(self, problem: PowerFlowProblem, progress_path: Path) -> tuple[list[HistoryEntry], dict[str, Any]]:
        """Samples bitstrings uniformly and returns the incumbent history.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Path for persisting incumbent progress snapshots.
        :return: Solver history whose last entry is the final incumbent reached before exhausting bitstrings or hitting ``max_time``, together with solver-specific extra information.
        """
        def update_history(new_result: EvaluationResult):
            """Appends a new incumbent history entry.
            :param new_result: Improved incumbent returned by the inner optimizer.
            """
            current_time = time.perf_counter()
            stats = get_optimizer_stats(inner_optimizer)
            history.append(HistoryEntry(current_time - start_time, None, stats.copy(), new_result))
            pd.to_pickle(history, progress_path)

        history = []
        inner_optimizer = self.inner_optimizer_factory(problem)
        inner_optimizer.violation_tolerance = self.violation_tolerance
        inner_optimizer.best_result_callback = update_history
        num_bitstrings = 2 ** len(problem.generators)
        rng = random.default_rng(self.seed)
        start_time = time.perf_counter()
        while len(inner_optimizer.cache) < num_bitstrings:
            if self.max_time is not None and time.perf_counter() - start_time > self.max_time:
                break
            generator_statuses = format(rng.integers(num_bitstrings), f"0{len(problem.generators)}b")
            if generator_statuses in inner_optimizer.cache:
                continue
            inner_optimizer.optimize(generator_statuses)
        assert len(history) > 0, "Uniform solver did not record any history entry."
        return history, get_optimizer_stats(inner_optimizer)


@dataclass
class HybridSolver(PowerFlowSolver):
    """Optimizes binary variables on a quantum computer. Continuous variables are optimized classically by the problem.
    :var vqp: Variational quantum program used for binary-variable search.
    :var inner_optimizer_factory: Factory that creates continuous optimizers for a given problem.
    :var violation_tolerance: Violation tolerance; history stores only entries with violation below this threshold.
    :var analyze_expectations: Whether to compute post-optimization expectation analysis.
    :var seed: Optional random seed for initial quantum-parameter sampling.
    :var name: Canonical solver name used in file naming.
    :var max_classical_time: Maximum classical angle-optimization time in seconds for the hybrid run, or ``None`` to disable the cap.
    :var max_process_time: Maximum process time in seconds for the hybrid run, or ``None`` to disable the cap.
    """
    vqp: VariationalQuantumProgram
    inner_optimizer_factory: Callable[[PowerFlowProblem], ContinuousPowerOptimizer]
    violation_tolerance: float = 1e-10
    analyze_expectations: bool = False
    seed: int | None = None
    name: str = "hybrid"
    max_classical_time: float | None = None
    max_process_time: float | None = None

    def solve(self, problem: PowerFlowProblem, progress_path: Path, initial_angles: ndarray | None = None) -> tuple[list[HistoryEntry], dict[str, Any]]:
        """Optimizes quantum parameters and returns the feasible incumbent history.
        :param problem: Power-flow optimization problem to solve.
        :param progress_path: Path for persisting incumbent progress snapshots.
        :param initial_angles: Initial quantum parameter vector, or ``None`` to sample it randomly.
        :return: Tuple of feasible incumbent history and optional extra hybrid-run information, including VQA angle-optimization classical time.
        """
        def update_history(new_result: EvaluationResult):
            """Appends a new incumbent history entry.
            :param new_result: Improved incumbent returned by the inner optimizer.
            """
            stats = get_optimizer_stats(inner_optimizer)
            classical_time = time.perf_counter() - start_time - self.vqp.quantum_time
            history.append(HistoryEntry(classical_time, self.vqp.num_jobs, stats.copy(), new_result))
            pd.to_pickle(history, progress_path)

        def get_inner_result(generator_statuses: str, max_classical_time: float | None = self.max_classical_time) -> EvaluationResult:
            """Returns inner-optimizer result for one generator-status bitstring and aborts angle optimization when its classical-time cap is exceeded.
            :param generator_statuses: Binary generator on/off bitstring.
            :param max_classical_time: Classical angle-optimization time cap in seconds, or ``None`` to disable time checks.
            :return: Inner-optimizer result for the given status pattern.
            """
            if max_classical_time is not None and time.perf_counter() - start_time - self.vqp.quantum_time > max_classical_time:
                raise TimeoutError("Classical angle optimization time exceeded limit.")
            if max_classical_time is not None and time.perf_counter() - start_time > self.max_process_time - 60:
                raise AssertionError(f"Ran out of process time. Quantum time = {self.vqp.quantum_time}")
            return inner_optimizer.optimize(generator_statuses)

        def get_cost(generator_statuses: str, max_classical_time: float | None = self.max_classical_time) -> float:
            """Returns total cost for one generator-status bitstring.
            :param generator_statuses: Binary generator on/off bitstring.
            :param max_classical_time: Classical angle-optimization time cap in seconds, or ``None`` to disable time checks.
            :return: Total objective value for the given status pattern.
            """
            return get_inner_result(generator_statuses, max_classical_time).total

        def get_cost_inverse(generator_statuses: str, max_classical_time: float | None = self.max_classical_time) -> float:
            """Returns the negative inverse of the total cost for one generator-status bitstring.
            :param generator_statuses: Binary generator on/off bitstring.
            :param max_classical_time: Classical angle-optimization time cap in seconds, or ``None`` to disable time checks.
            :return: Negative inverse transformed objective value for the given status pattern.
            """
            return -1 / get_inner_result(generator_statuses, max_classical_time).total

        inner_optimizer = self.inner_optimizer_factory(problem)
        inner_optimizer.violation_tolerance = self.violation_tolerance
        inner_optimizer.best_result_callback = update_history
        num_bitstrings = 2 ** len(problem.generators)

        if initial_angles is None:
            # initial_angles = random.default_rng(self.seed).uniform(-np.pi, np.pi, len(self.vqp.circuit.parameters))
            initial_angles = np.zeros(len(self.vqp.circuit.parameters))
        active_cost = get_cost_inverse

        history = []
        extra = {}
        try:
            start_time = time.perf_counter()
            result = self.vqp.optimize_parameters(active_cost, initial_angles)
            extra |= {"classical_opt_time": time.perf_counter() - start_time - self.vqp.quantum_time, "total_opt_jobs": self.vqp.num_jobs}
            assert result.success, f"Angle optimization failed: {result.message}"

            while len(inner_optimizer.cache) < num_bitstrings:
                self.vqp.get_function_expectation(active_cost, result.x)
        except TimeoutError:
            pass
        assert len(history) > 0, "Hybrid solver did not record any feasible history entry."
        
        best_result = min(inner_optimizer.cache.values(), key=lambda cached_result: cached_result.total)
        assert np.isclose(best_result.total, history[-1].result.total), \
            f"Lowest overall: fun={best_result.fun}; violation={best_result.violation}. Lowest feasible: fun={history[-1].result.fun}."

        extra |= get_optimizer_stats(inner_optimizer)
        if self.analyze_expectations:
            assert extra.get("classical_opt_time") is not None, "Expectation analysis requires completed angle optimization."
            inner_optimizer.best_result_callback = None
            cost_function_untimed = partial(get_cost_inverse, max_classical_time=None)
            uniform_probs = {format(i, f"0{len(problem.generators)}b"): 1 / num_bitstrings for i in range(num_bitstrings)}
            uniform_expectation = utils.get_function_expectation(cost_function_untimed, uniform_probs)
            best_total = min(cached_result.total for cached_result in inner_optimizer.cache.values())
            uniform_expectation *= -best_total

            exact_sampler = ExactSampler()
            opt_probs = exact_sampler.get_sample_probabilities(self.vqp.circuit, result.x)
            opt_expectation = utils.get_function_expectation(cost_function_untimed, opt_probs) * -best_total

            extra |= {"final_probs": opt_probs, "ar_uniform": uniform_expectation, "ar_opt": opt_expectation}
        return history, extra

    def get_feasible_probs(self, feasible_bitstrings: set[str], probs: dict[str, float]) -> dict[str, float]:
        """Removes infeasible bitstrings from a probability distribution and renormalizes it.
        :param feasible_bitstrings: Bitstrings that should be kept in the distribution.
        :param probs: Probability distribution over bitstrings.
        :return: Feasible-only renormalized probability distribution.
        """
        feasible_probs = {generator_statuses: probability for generator_statuses, probability in probs.items() if generator_statuses in feasible_bitstrings}
        total_probability = sum(feasible_probs.values())
        return {generator_statuses: probability / total_probability for generator_statuses, probability in feasible_probs.items()}
