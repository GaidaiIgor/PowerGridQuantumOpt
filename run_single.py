"""Runs one power-flow instance with the configured solver."""

import pickle
from functools import partial
from pathlib import Path

import numpy as np
from networkx import Graph
from qiskit.primitives import StatevectorSampler

import debug
from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.Generator import Generator
from src.PowerFlowProblem import PowerFlowProblem
from src.PowerFlowSolver import HybridSolver, PowerFlowSolver, SCIPSolver
from src.Sampler import MySamplerV2
from src.VariationalQuantumProgram import VariationalQuantumProgram
from src.utils import my_format


def run_single() -> None:
    """Runs the configured solver on one stored problem instance."""
    # problem = get_power_flow_ac_problem()
    index = 49
    voltage_deviation_mult = 10
    exact_final_expectation = False
    data_path = Path("data/5/capacity_100")
    with (data_path / f"{index}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file), voltage_deviation_mult)

    # debug.set_all_edge_capacities(problem, 100)
    # debug.set_all_node_voltage_ranges(problem, (1, 100))
    # debug.set_all_generator_p_min(problem, 0)

    # solver = ClassicalSolver()
    solver = get_solver(len(problem.generators))

    inner_solver = solver.inner_optimizer_factory(problem)
    inner_solver.optimize("11110")

    progress_folder = data_path / f".progress_{solver.name}"
    progress_folder.mkdir(exist_ok=True)
    progress_path = progress_folder / f"{index}.pkl"
    if isinstance(solver, HybridSolver):
        history, extra = solver.solve(problem, progress_path, exact_final_expectation)
    else:
        history, extra = solver.solve(problem, progress_path)

    print("\nSolution:")
    debug.print_evaluation_result(problem, history[-1].result)
    print(f"Job index: {history[-1].job_ind}")
    if "total_jobs" in extra:
        print(f"Total jobs: {extra['total_jobs']}")
    if exact_final_expectation:
        print(f"Optimized probabilities: {my_format(extra['final_probs'])}")
        print(f"Optimized expectation: {extra['cost_expectation']}")


def get_solver(num_generators: int) -> PowerFlowSolver:
    """Builds the configured solver for a problem size.
    :param num_generators: Number of generators or qubits in the target instance.
    :return: Solver configured for the current experiment.
    """
    max_inner_time_s = 30
    penalty_mult = 10
    feasibility_tolerance = 1e-10
    silent = True
    seed = 0

    vqp = get_variational_quantum_program(num_generators)
    inner_optimizer_factory = partial(CasadiOptimizer, penalty_mult=penalty_mult, max_time_s=max_inner_time_s, silent=True)

    solver = SCIPSolver(feasibility_tolerance, silent, seed, "scip_adaptive")
    # solver = SmacSolver(inner_optimizer_factory, feasibility_tolerance, silent, seed)
    # solver = UniformSolver(inner_optimizer_factory, feasibility_tolerance, seed)
    # solver = HybridSolver(vqp, inner_optimizer_factory, feasibility_tolerance, seed)
    return solver


def get_variational_quantum_program(num_qubits: int) -> VariationalQuantumProgram:
    """Builds the variational quantum program used by hybrid solvers.
    :param num_qubits: Number of qubits used by the program.
    :return: Configured variational quantum program.
    """
    entangler = AllToAllEntangler(num_qubits)
    mixer = ZXMixer(num_qubits)
    num_layers = 1

    # sampler = ExactSampler()
    sampler = MySamplerV2(StatevectorSampler(default_shots=1000))
    # sampler = IonQSampler("simulator", 1000, None)
    # sampler = IonQSampler("qpu.forte-enterprise-1", 1000, None)

    return VariationalQuantumProgram(num_layers, [entangler, mixer], sampler)


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
