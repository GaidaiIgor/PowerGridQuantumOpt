from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy import random
from pyscipopt import Model, sin, cos, quicksum
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

import utils
from ContinuousPowerOptimizer import ContinuousPowerOptimizer
from PowerFlowProblem import PowerFlowProblem, PowerFlowSolution
from Sampler import ExactSampler
from VariationalQuantumProgram import VariationalQuantumProgram


class PowerFlowSolver(ABC):
    """ Base class for power grid problem solvers. """

    @abstractmethod
    def solve(self, problem: PowerFlowProblem) -> PowerFlowSolution:
        """ Solves a given power grid optimization problem and returns its solution. """
        pass


class ClassicalSolver(PowerFlowSolver):
    """ Uses SCIP library to solve power grid problems classically. """

    @staticmethod
    def build_model_power_flow(problem: PowerFlowProblem) -> tuple[Model, dict[str, list]]:
        """ Builds model based on problem description. """
        model = Model("PowerFlowAC")
        cost_terms = []
        variables = defaultdict(lambda: [[] for _ in range(len(problem.graph))])
        for node_label, node_data in problem.graph.nodes(data=True):
            for i, gen in enumerate(node_data["generators"]):
                u = model.addVar(vtype="B", name=f"u_{node_label}_{i}")
                p = model.addVar(lb=0, ub=gen.power_range[1], name=f"p_{node_label}_{i}")
                q = model.addVar(lb=0, ub=gen.reactive_power_range[1], name=f"q_{node_label}_{i}")
                model.addCons(p >= gen.power_range[0] * u, name=f"p_min_{node_label}_{i}")
                model.addCons(p <= gen.power_range[1] * u, name=f"p_max_{node_label}_{i}")
                model.addCons(q >= gen.reactive_power_range[0] * u, name=f"q_min_{node_label}_{i}")
                model.addCons(q <= gen.reactive_power_range[1] * u, name=f"q_max_{node_label}_{i}")
                cost_terms.append(gen.cost_terms[0] * p * p + gen.cost_terms[1] * p + gen.cost_terms[2] * u)
                variables["u"][node_data["node_ind"]].append(u)
                variables["p"][node_data["node_ind"]].append(p)
                variables["q"][node_data["node_ind"]].append(q)
            variables["v"][node_data["node_ind"]] = model.addVar(lb=node_data["voltage_range"][0], ub=node_data["voltage_range"][1], name=f"v_{node_label}")
            variables["d"][node_data["node_ind"]] = model.addVar(lb=node_data["angle_range"][0], ub=node_data["angle_range"][1], name=f"d_{node_label}")

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
                real_flow = alpha * v_i * v_i - v_i * v_j * (alpha * cos(delta) + beta * sin(delta))
                imag_flow = -beta * v_i * v_i + v_i * v_j * (beta * cos(delta) - alpha * sin(delta))
                real_flows.append(real_flow)
                imag_flows.append(imag_flow)
                if node_data["node_ind"] < neighbor_data["node_ind"]:
                    abs_current_2 = abs(line_data["admittance"]) ** 2 * (v_i * v_i + v_j * v_j - 2 * v_i * v_j * cos(delta))
                    model.addCons(abs_current_2 <= line_data["capacity"] ** 2, name=f"capacity_{node_label}_{neighbor_label}")

            model.addCons(quicksum(variables["p"][node_data["node_ind"]]) - node_data["load"].real - quicksum(real_flows) == 0,
                          name=f"net_power_real_{node_label}")
            model.addCons(quicksum(variables["q"][node_data["node_ind"]]) - node_data["load"].imag - quicksum(imag_flows) == 0,
                          name=f"net_power_imag_{node_label}")

        set_nonlinear_objective(model, quicksum(cost_terms), sense="minimize")
        return model, variables

    @staticmethod
    def extract_solution(model: Model, variables: dict[str, list]) -> PowerFlowSolution:
        """ Extracts optimized variables from model and fills out solution instance. """
        assert model.getNSols() > 0, "Failed to find a feasible solution"
        all_u = sum(variables["u"], [])
        generator_statuses = "".join([str(int(model.getVal(var))) for var in all_u])
        all_p = sum(variables["p"], [])
        active_powers = np.array([model.getVal(var) for var in all_p])
        all_q = sum(variables["q"], [])
        reactive_powers = np.array([model.getVal(var) for var in all_q])
        voltages = np.array([model.getVal(var) for var in variables["v"]])
        angles = np.array([model.getVal(var) for var in variables["d"]])
        cost = model.getObjVal()
        return PowerFlowSolution(generator_statuses, active_powers, reactive_powers, voltages, angles, cost)

    def solve(self, problem: PowerFlowProblem) -> PowerFlowSolution:
        """ Solves given problem and returns its solution. """
        model, variables = ClassicalSolver.build_model_power_flow(problem)
        model.optimize()
        solution = ClassicalSolver.extract_solution(model, variables)
        return solution


@dataclass
class HybridSolver(PowerFlowSolver):
    """ Optimizes binary variables on a quantum computer. Continuous variables are optimized classically by the problem. """
    vqp: VariationalQuantumProgram
    inner_optimizer_factory: Callable[[PowerFlowProblem], ContinuousPowerOptimizer]
    seed: int = None

    def solve(self, problem: PowerFlowProblem) -> PowerFlowSolution:
        inner_optimizer = self.inner_optimizer_factory(problem)
        rng = random.default_rng(self.seed)
        initial_angles = rng.uniform(-np.pi, np.pi, len(self.vqp.circuit.parameters))
        result = self.vqp.optimize_parameters(inner_optimizer.get_optimized_cost, initial_angles)
        assert result.success, f"Angle optimization failed: {result.message}"

        best_sample = min(inner_optimizer.cache.items(), key=lambda pair: pair[1].total)
        active_powers, reactive_powers, voltages, angles = inner_optimizer.split_params(best_sample[1].x)
        solution = PowerFlowSolution(best_sample[0], active_powers, reactive_powers, voltages, angles, best_sample[1].fun)
        solution.extra["opt_result"] = best_sample[1]

        exact_sampler = ExactSampler()
        solution.extra["final_probs"] = exact_sampler.get_sample_probabilities(self.vqp.circuit, result.x)
        solution.extra["cost_expectation"] = utils.get_cost_expectation(inner_optimizer.get_optimized_cost, solution.extra["final_probs"])
        solution.extra["num_jobs"] = result.nfev
        return solution
