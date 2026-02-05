import time

import numpy as np
from networkx import Graph

from CircuitLayer import AllToAllEntangler, ZXMixer
from ContinuousPowerOptimizer import ContinuousPowerOptimizer
from Generator import Generator
from PowerFlowProblem import PowerFlowProblem
from PowerFlowSolver import HybridSolver
from Sampler import ExactSampler
from VariationalQuantumProgram import VariationalQuantumProgram
from utils import my_format


def get_power_flow_ac_problem() -> PowerFlowProblem:
    voltage_range = (0, 10)
    angle_range = (-np.pi, np.pi)
    graph = Graph()

    graph.add_node(0, generators=[Generator((0, 100), (0, 100), (0, 1, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_node(1, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_edge(0, 1, capacity=100, admittance=1+1j)

    # graph.add_node(0, generators=[Generator((0, 30), (0, 0), (0, 10, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(1, generators=[Generator((0, 10), (0, 0), (0, 20, 1))], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(2, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_edge(0, 1, capacity=10, admittance=1)
    # graph.add_edge(0, 2, capacity=5, admittance=1)
    # graph.add_edge(1, 2, capacity=10, admittance=1)

    return PowerFlowProblem(graph)


def get_variational_quantum_program(num_qubits: int) -> VariationalQuantumProgram:
    entangler = AllToAllEntangler(num_qubits)
    mixer = ZXMixer(num_qubits)
    num_layers = 1

    sampler = ExactSampler()
    # sampler = MySamplerV2(StatevectorSampler(default_shots=1000))
    # sampler = IonQSampler("simulator", 1000, None)
    # sampler = IonQSampler("qpu.forte-enterprise-1", 1000, None)

    return VariationalQuantumProgram(num_layers, [entangler, mixer], sampler)


def get_hybrid_solver(num_generators: int) -> HybridSolver:
    vqp = get_variational_quantum_program(num_generators)
    penalty_mult = 10
    inner_optimizer_factory = lambda problem: ContinuousPowerOptimizer(problem, penalty_mult)
    seed = 0
    return HybridSolver(vqp, inner_optimizer_factory, seed)


def main():
    problem = get_power_flow_ac_problem()

    # solver = ClassicalSolver()
    solver = get_hybrid_solver(len(problem.generators))

    solution = solver.solve(problem)
    print("\nSolution:")
    print(solution)

    if isinstance(solver, HybridSolver):
        print(f"Optimized probabilities: {my_format(solution.extra["final_probs"])}")
        print(f"Optimized expectation: {solution.extra["cost_expectation"]}")
        print(f"Number of jobs: {solution.extra["num_jobs"]}")

        print("=== Best sample ===")
        print(f"Inner optimization successful: {solution.extra["opt_result"].success}")
        print(f"Penalty: {solution.extra["opt_result"].penalty}")


if __name__ == "__main__":
    t1 = time.perf_counter()
    main()
    t2 = time.perf_counter()
    print(f"Elapsed time {t2 - t1} seconds")
