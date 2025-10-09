import time
from functools import partial

import numpy as np
import qiskit
from networkx import Graph
from numpy import random
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_ionq import IonQProvider

import src.utils as utils
from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.PowerGrid import GeneratorCommitmentProblem, Generator, SimplePowerFlowProblem, PowerFlowACProblem
from src.Sampler import ExactSampler, MySamplerV2, IonQSampler
from src.VariationalCircuit import VariationalCircuit


def get_generator_commitment_problem() -> GeneratorCommitmentProblem:
    generators = np.array([Generator((15, 20), (0, 0), (0, 1, 10)),
                           Generator((0, 10), (0, 0), (1, 0, 1))])
    load = 10

    # generators = np.array([Generator((100, 600), (0.002, 10, 500)),
    #                        Generator((100, 400), (0.0025, 8, 300)),
    #                        Generator((50, 200), (0.005, 6, 100))])
    # load = 170
    #
    # generators = np.array([Generator((150, 455), (0.00048, 16.19, 1000)),
    #                        Generator((150, 455), (0.00031, 17.26, 970)),
    #                        Generator((20, 130), (0.002, 16.6, 700)),
    #                        Generator((20, 130), (0.00211, 16.5, 680)),
    #                        Generator((25, 162), (0.00398, 19.7, 450)),
    #                        Generator((20, 80), (0.00712, 22.26, 370)),
    #                        Generator((25, 85), (0.00079, 27.74, 480)),
    #                        Generator((10, 55), (0.00413, 25.92, 660)),
    #                        Generator((10, 55), (0.00222, 27.27, 665)),
    #                        Generator((10, 55), (0.00173, 27.79, 670))
    #                        ])
    # load = 700

    problem = GeneratorCommitmentProblem(generators, load)
    return problem


def get_power_flow_ac_problem() -> PowerFlowACProblem:
    voltage_range = (0, 100)
    angle_range = (0, 0)
    graph = Graph()

    graph.add_node(0, generators=[Generator((0, 150), (0, 0), (0, 10, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_node(1, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_edge(0, 1, capacity=1000, admittance=1)

    # graph.add_node(0, generators=[Generator((0, 30), (0, 0), (0, 10, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(1, generators=[Generator((0, 10), (0, 0), (0, 20, 1))], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(2, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_edge(0, 1, capacity=10, admittance=1)
    # graph.add_edge(0, 2, capacity=5, admittance=1)
    # graph.add_edge(1, 2, capacity=10, admittance=1)

    return PowerFlowACProblem(graph)


def main():
    problem = get_generator_commitment_problem()
    # problem = get_power_flow_ac_problem()
    penalty_mult = 10
    num_gen = len(problem.generators)

    entangler = AllToAllEntangler(num_gen)
    mixer = ZXMixer(num_gen)
    num_layers = 1

    # sampler = ExactSampler()
    sampler = MySamplerV2(StatevectorSampler(default_shots=1000))
    # sampler = IonQSampler("simulator", 1000, None)
    # sampler = IonQSampler("qpu.forte-enterprise-1", 1000, None)

    seed = 0
    rng = random.default_rng(seed)

    vqa = VariationalCircuit(num_layers, [entangler, mixer], sampler)
    initial_angles = rng.uniform(-np.pi, np.pi, len(vqa.circuit.parameters))
    cost_function = partial(problem.evaluate, penalty_mult=penalty_mult)

    # probs = {'01': 0.446, '10': 0.451, '11': 0.085, '00': 0.018}
    # expectation = utils.get_cost_expectation(cost_function, probs)
    # print(f"Expectation: {expectation}")

    result = vqa.optimize_parameters(cost_function, initial_angles)

    exact_sampler = ExactSampler()
    final_probs = exact_sampler.get_sample_probabilities(vqa.circuit, result.x)
    final_expectation = utils.get_cost_expectation(cost_function, final_probs)
    print(f"Angle optimization successful: {result.success}")
    print(f"Optimized angles: {result.x}")
    print(f"Optimized probabilities: {final_probs}")
    print(f"Optimized expectation: {final_expectation}")
    print(f"Number of jobs: {result.nfev}")

    best_sample = min(problem.optimize_power.cache.items(), key=lambda pair: pair[1].total)
    print("=== Best sample ===")
    print(f"Power optimization successful: {best_sample[1].success}")
    print(f"Generators selected: {best_sample[0]}")
    print(f"Optimized power: {best_sample[1].x}")
    print(f"Optimized cost: {best_sample[1].fun}")
    print(f"Penalty: {best_sample[1].penalty}")


if __name__ == "__main__":
    t1 = time.perf_counter()
    main()
    t2 = time.perf_counter()
    print(f"Elapsed time {t2 - t1} seconds")
