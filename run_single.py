"""Runs one power-flow instance with the configured solver."""

import pickle
import time
from itertools import product
from pathlib import Path

import common.debug as debug
import numpy as np
from networkx import Graph

from common.utils import get_solver
from src.Generator import Generator
from src.PowerFlowProblem import PowerFlowProblem
from src.utils import my_format


def run_single():
    """Runs the configured solver on one stored problem instance."""
    solver_id = "hybrid"
    violation_tolerance = 1e-10
    silent = True
    seed = 0
    violation_mult = 10 ** 7
    max_inner_time_s = 30
    max_classical_time = None
    num_layers = 1
    sampler_id = "exact"
    # sampler_id = "finite"
    shots = 1000
    analyze_expectations = True
    max_process_time = None
    data_path = Path("data/5")
    instance = 7
    voltage_deviation_mult = 10

    with (data_path / f"{instance}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file), voltage_deviation_mult)
    num_generators = len(problem.generators)
    np.random.seed(seed)

    # debug.set_all_edge_capacities(problem, 100)
    # debug.set_all_node_voltage_ranges(problem, (1, 100))
    # debug.set_all_generator_p_min(problem, 0)

    solver = get_solver(solver_id, violation_tolerance, silent, seed, violation_mult, max_inner_time_s, max_classical_time, num_generators, num_layers,
                        sampler_id, shots, analyze_expectations, max_process_time)

    # inner_solver = solver.inner_optimizer_factory(problem)
    # inner_solver.optimize("11110")

    progress_folder = Path(".progress")
    progress_folder.mkdir(exist_ok=True)
    progress_path = progress_folder / f"{instance}.pkl"

    angle_results = []
    best_result = None
    counter = 0
    for initial_angles in product((-0.1, 0.1), repeat=len(solver.vqp.circuit.parameters)):
        history, extra = solver.solve(problem, progress_path, np.array(initial_angles))
        angle_results.append((initial_angles, extra["ar_opt"]))
        if best_result is None or extra["ar_opt"] > best_result[1]:
            best_result = (initial_angles, extra["ar_opt"], history, extra)
        print(f"Initial angles {counter}: {my_format(initial_angles)}; final AR opt: {extra["ar_opt"]}")
        counter += 1

    print("\nAR opt ranking by initial angles:")
    for initial_angles, ar_opt in sorted(angle_results, key=lambda item: item[1], reverse=True):
        print(f"Initial angles: {my_format(initial_angles)}; final AR opt: {ar_opt}")

    initial_angles, _, history, extra = best_result

    print("\nSolution:")
    print(f"Initial angles: {my_format(initial_angles)}")
    debug.print_evaluation_result(problem, history[-1].result)
    print(f"Optimized bitstrings: {extra["optimized_bitstrings"]}")
    print(f"Solution found job ind: {history[-1].job_ind}")
    print(f"Total opt jobs: {extra["total_opt_jobs"]}")
    if solver.analyze_expectations:
        print(f"Optimized probabilities: {my_format(extra["final_probs"])}")
        print(f"AR uniform: {extra["ar_uniform"]}")
        print(f"AR opt: {extra["ar_opt"]}")


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
    start_time = time.perf_counter()
    run_single()
    print(f"Elapsed time: {time.perf_counter() - start_time}")
