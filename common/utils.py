"""Provides shared solver utilities for experiment entry points."""

from functools import partial

from qiskit.primitives import StatevectorSampler

from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.PowerFlowSolver import PowerFlowSolver, SCIPSolver, SmacSolver, UniformSolver, HybridSolver
from src.Sampler import MySamplerV2
from src.VariationalQuantumProgram import VariationalQuantumProgram

SOLVER_IDS = ("scip", "smac", "uniform", "hybrid")


def get_solver(num_generators: int, solver_id: str, num_layers: int = 1, analyze_expectations: bool = False, max_classical_time: float = 0) \
    -> PowerFlowSolver:
    """Builds the configured solver for a problem size.
    :param num_generators: Number of generators or qubits in the target instance.
    :param solver_id: Solver identifier. Must be one of ``"scip"``, ``"smac"``, ``"uniform"``, or ``"hybrid"``.
    :param num_layers: Number of repeated ansatz blocks used by the hybrid solver.
    :param analyze_expectations: Whether hybrid solvers should compute post-optimization expectation analysis.
    :param max_classical_time: Maximum total classical time in seconds for hybrid runs.
    :return: Solver configured for the current experiment.
    """
    max_inner_time_s = 30
    penalty_mult = 10
    feasibility_tolerance = 1e-10
    silent = True
    seed = 0

    if solver_id == "scip":
        return SCIPSolver(feasibility_tolerance, silent, seed)

    inner_optimizer_factory = partial(CasadiOptimizer, penalty_mult=penalty_mult, max_time_s=max_inner_time_s, silent=True)
    if solver_id == "smac":
        return SmacSolver(inner_optimizer_factory, feasibility_tolerance, silent, seed)
    if solver_id == "uniform":
        return UniformSolver(inner_optimizer_factory, feasibility_tolerance, seed)
    if solver_id == "hybrid":
        vqp = get_variational_quantum_program(num_generators, num_layers)
        return HybridSolver(vqp, inner_optimizer_factory, analyze_expectations, max_classical_time, feasibility_tolerance, seed)
    raise ValueError(f"Unsupported solver {solver_id}. Expected one of " + ", ".join(SOLVER_IDS) + ".")


def get_variational_quantum_program(num_qubits: int, num_layers: int) -> VariationalQuantumProgram:
    """Builds the variational quantum program used by hybrid solvers.
    :param num_qubits: Number of qubits used by the program.
    :param num_layers: Number of repeated ansatz blocks.
    :return: Configured variational quantum program.
    """
    entangler = AllToAllEntangler(num_qubits)
    mixer = ZXMixer(num_qubits)

    # sampler = ExactSampler()
    sampler = MySamplerV2(StatevectorSampler(default_shots=1000))
    # sampler = IonQSampler("simulator", 1000, None)
    # sampler = IonQSampler("qpu.forte-enterprise-1", 1000, None)

    return VariationalQuantumProgram(num_layers, [entangler, mixer], sampler)
