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


def get_solver(solver_id: str, violation_tolerance: float = 1e-10, silent: bool = True, seed: int = 0, violation_mult: float = 10 ** 7,
               max_inner_time_s: float = 30, max_classical_time: float | None = None, num_generators: int = 5, num_layers: int = 1, sampler_id: str = "finite",
               shots: int = 1000, analyze_expectations: bool = False, max_process_time: float | None = None) -> PowerFlowSolver:
    """Builds the configured solver for a problem size.
    :param solver_id: Solver identifier. Must be one of ``"scip"``, ``"smac"``, ``"uniform"``, or ``"hybrid"``.
    :param violation_tolerance: Maximum tolerated constraint violation.
    :param silent: Whether solver output should be suppressed.
    :param seed: Randomness seed for the solver.
    :param violation_mult: Multiplication factor for objective constraint violation. Total objective = objective + mult * violation.
    :param max_inner_time_s: Maximum inner optimization time in seconds.
    :param max_classical_time: Maximum classical angle-optimization time in seconds for hybrid runs, or ``None`` to disable the cap.
    :param num_generators: Number of generators or qubits in the target instance.
    :param num_layers: Number of repeated ansatz blocks used by the hybrid solver.
    :param sampler_id: Sampler identifier for hybrid runs.
    :param shots: Number of shots for sampling-based backends.
    :param analyze_expectations: Whether hybrid solvers should compute post-optimization expectation analysis.
    :param max_process_time: Maximum process time in seconds for hybrid runs, or ``None`` to disable the cap.
    :return: Solver configured for the current experiment.
    """
    if solver_id == "scip":
        return SCIPSolver(violation_tolerance, silent, seed)

    inner_optimizer_factory = partial(CasadiOptimizer, violation_mult=violation_mult, max_time_s=max_inner_time_s, silent=silent)
    if solver_id == "smac":
        return SmacSolver(inner_optimizer_factory, violation_tolerance, silent, seed)
    if solver_id == "uniform":
        return UniformSolver(inner_optimizer_factory, max_classical_time, violation_tolerance, seed)
    if solver_id == "hybrid":
        vqp = get_variational_quantum_program(num_generators, num_layers, sampler_id, shots, seed)
        return HybridSolver(vqp, inner_optimizer_factory, violation_tolerance, analyze_expectations, seed, "hybrid", max_classical_time, max_process_time)
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
