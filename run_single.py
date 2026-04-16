"""Runs one power-flow instance with the configured solver."""

import pickle
from pathlib import Path

import common.debug as debug
import numpy as np
from networkx import Graph

from common.utils import get_solver
from src.Generator import Generator
from src.PowerFlowProblem import PowerFlowProblem
from src.PowerFlowSolver import HybridSolver
from src.utils import my_format


def run_single() -> None:
    """Runs the configured solver on one stored problem instance."""
    # problem = get_power_flow_ac_problem()
    index = 49
    solver_id = "hybrid"
    num_layers = 1
    voltage_deviation_mult = 10
    analyze_expectations = False
    data_path = Path("data/5/capacity_100")
    with (data_path / f"{index}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file), voltage_deviation_mult)

    # debug.set_all_edge_capacities(problem, 100)
    # debug.set_all_node_voltage_ranges(problem, (1, 100))
    # debug.set_all_generator_p_min(problem, 0)

    # solver = ClassicalSolver()
    solver = get_solver(len(problem.generators), solver_id, num_layers, analyze_expectations)

    inner_solver = solver.inner_optimizer_factory(problem)
    inner_solver.optimize("11110")

    progress_folder = Path(".progress")
    progress_folder.mkdir(exist_ok=True)
    progress_path = progress_folder / f"{index}.pkl"
    history, extra = solver.solve(problem, progress_path)

    print("\nSolution:")
    debug.print_evaluation_result(problem, history[-1].result)
    print(f"Job index: {history[-1].job_ind}")
    if "total_jobs" in extra:
        print(f"Total jobs: {extra["total_jobs"]}")
    if "optimized_bitstrings" in extra:
        print(f"Optimized bitstrings: {extra["optimized_bitstrings"]}")
    if isinstance(solver, HybridSolver) and solver.analyze_expectations:
        print(f"Optimized probabilities: {my_format(extra["final_probs"])}")
        print(f"AR uniform total: {extra["ar_uniform_total"]}")
        print(f"AR uniform fun: {extra["ar_uniform_fun"]}")
        print(f"AR opt total: {extra["ar_opt_total"]}")
        print(f"AR opt fun: {extra["ar_opt_fun"]}")


def get_power_flow_ac_problem() -> PowerFlowProblem:
    """Builds a small manual AC power-flow problem for debugging.
    :return: Hand-constructed power-flow problem.
    """
    voltage_range = (0, 10)
    angle_range = (-np.pi, np.pi)
    graph = Graph()

    graph.add_node(0, generators=[Generator((0, 100), (-100, 100), (0, 1, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_node(1, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_edge(0, 1, capacity=100, admittance=1 + 1j)

    # graph.add_node(0, generators=[Generator((0, 30), (0, 0), (0, 10, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(1, generators=[Generator((0, 10), (0, 0), (0, 20, 1))], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(2, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_edge(0, 1, capacity=10, admittance=1)
    # graph.add_edge(0, 2, capacity=5, admittance=1)
    # graph.add_edge(1, 2, capacity=10, admittance=1)

    return PowerFlowProblem(graph)


if __name__ == "__main__":
    run_single()
