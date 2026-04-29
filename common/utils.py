"""Provides shared solver utilities for experiment entry points."""

from functools import partial

from qiskit.primitives import StatevectorSampler

from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.PowerFlowSolver import PowerFlowSolver, SCIPSolver, SmacSolver, UniformSolver, HybridSolver
from src.Sampler import ExactSampler, IonQSampler, MySamplerV2, Sampler
from src.VariationalQuantumProgram import VariationalQuantumProgram

SOLVER_IDS = ("scip", "smac", "uniform", "hybrid")
SAMPLER_IDS = ("exact", "finite", "ionq-simulator", "ionq-hardware")


def get_solver(num_generators: int, solver_id: str, num_layers: int = 1, analyze_expectations: bool = False, max_classical_time: float | None = None,
               sampler_id: str = "finite", shots: int = 1000, violation_mult: float = 10 ** 7, seed: int = 0) -> PowerFlowSolver:
    """Builds the configured solver for a problem size.
    :param num_generators: Number of generators or qubits in the target instance.
    :param solver_id: Solver identifier. Must be one of ``"scip"``, ``"smac"``, ``"uniform"``, or ``"hybrid"``.
    :param num_layers: Number of repeated ansatz blocks used by the hybrid solver.
    :param analyze_expectations: Whether hybrid solvers should compute post-optimization expectation analysis.
    :param max_classical_time: Maximum classical angle-optimization time in seconds for hybrid runs, or ``None`` to disable the cap.
    :param sampler_id: Sampler identifier for hybrid runs.
    :param shots: Number of shots for sampling-based backends.
    :param violation_mult: Multiplication factor for objective constraint violation. Total objective = objective + mult * violation.
    :param seed: Randomness seed for the solver.
    :return: Solver configured for the current experiment.
    """
    max_inner_time_s = 30
    violation_tolerance = 1e-10
    silent = True

    if solver_id == "scip":
        return SCIPSolver(violation_tolerance, silent, seed)

    inner_optimizer_factory = partial(CasadiOptimizer, violation_mult=violation_mult, max_time_s=max_inner_time_s, silent=True)
    if solver_id == "smac":
        return SmacSolver(inner_optimizer_factory, violation_tolerance, silent, seed)
    if solver_id == "uniform":
        return UniformSolver(inner_optimizer_factory, max_classical_time, violation_tolerance, seed)
    if solver_id == "hybrid":
        vqp = get_variational_quantum_program(num_generators, num_layers, sampler_id, shots, seed)
        return HybridSolver(vqp, inner_optimizer_factory, analyze_expectations, max_classical_time, violation_tolerance, seed)
    raise ValueError(f"Unsupported solver {solver_id}. Expected one of " + ", ".join(SOLVER_IDS) + ".")


def get_variational_quantum_program(num_qubits: int, num_layers: int, sampler_id: str = "finite", shots: int = 1000, seed: int | None = None) \
    -> VariationalQuantumProgram:
    """Builds the variational quantum program used by hybrid solvers.
    :param num_qubits: Number of qubits used by the program.
    :param num_layers: Number of repeated ansatz blocks.
    :param sampler_id: Sampler identifier for circuit evaluation.
    :param shots: Number of shots for sampling-based backends.
    :param seed: Randomness seed for sampling-based backends.
    :return: Configured variational quantum program.
    """
    entangler = AllToAllEntangler(num_qubits)
    mixer = ZXMixer(num_qubits)
    sampler = get_sampler(sampler_id, shots, seed)
    return VariationalQuantumProgram(num_layers, [entangler, mixer], sampler)


def get_sampler(sampler_id: str, shots: int, seed: int | None = None) -> Sampler:
    """Builds the configured sampler backend.
    :param sampler_id: Sampler identifier.
    :param shots: Number of shots for sampling-based backends.
    :param seed: Randomness seed for sampling-based backends.
    :return: Configured sampler backend.
    """
    match sampler_id:
        case "exact":
            return ExactSampler()
        case "finite":
            return MySamplerV2(StatevectorSampler(default_shots=shots, seed=seed))
        case "ionq-simulator":
            return IonQSampler("simulator", shots, None)
        case "ionq-hardware":
            return IonQSampler("qpu.forte-enterprise-1", shots, None)
    raise ValueError(f"Unsupported sampler {sampler_id}. Expected one of " + ", ".join(SAMPLER_IDS) + ".")
