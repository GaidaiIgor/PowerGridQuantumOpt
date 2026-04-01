"""Provides shared solver utilities for experiment entry points."""

from functools import partial

from qiskit.primitives import StatevectorSampler

from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.PowerFlowSolver import PowerFlowSolver, SCIPSolver
from src.Sampler import MySamplerV2
from src.VariationalQuantumProgram import VariationalQuantumProgram


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
