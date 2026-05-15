"""Variational quantum program for optimizing generator commitment distributions."""

import time
from functools import partial
from typing import Callable, Mapping, Sequence

import noisyopt
import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import SPSA, ADAM, QNSPSA
from qiskit_algorithms.utils import algorithm_globals
from scipy import optimize
from scipy.optimize import OptimizeResult

from . import utils
from .CircuitLayer import CircuitLayer
from .Sampler import Sampler, ExactSampler

OPTIMIZATION_METHOD_IDS = ("auto", "l-bfgs-b", "spsa", "qnspsa", "adam", "noisyopt", "ax")


class VariationalQuantumProgram:
    """Represents a parameterized quantum program with classical parameter optimization.
    :var num_layers: Number of repeated layer blocks in the ansatz.
    :var layer_types: Ordered layer templates composed in each block.
    :var sampler: Sampling backend used to estimate output probabilities.
    :var optimization_method: Classical angle-optimization method.
    :var circuit: Fully constructed parameterized quantum circuit.
    :var expectation_time: Accumulated time spent evaluating expectation sample probabilities.
    :var expectation_jobs: Number of expectation-evaluation quantum jobs.
    :var fidelity_time: Accumulated time spent evaluating QNSPSA fidelity probabilities.
    :var fidelity_jobs: Number of QNSPSA fidelity-evaluation quantum jobs.
    :var seed: Optional seed for optimizer-local randomness.
    """
    num_layers: int
    layer_types: list[CircuitLayer]
    sampler: Sampler
    optimization_method: str
    circuit: QuantumCircuit
    expectation_time: float
    expectation_jobs: int
    fidelity_time: float
    fidelity_jobs: int
    seed: int | None

    def __init__(self, num_layers: int, layer_types: list[CircuitLayer], sampler: Sampler, optimization_method: str = "auto", seed: int | None = None):
        """Appends configured layer blocks and initializes program state.
        :param num_layers: Number of repeated ansatz blocks.
        :param layer_types: Layer templates composed once per block.
        :param sampler: Sampling backend used for probability estimation.
        :param optimization_method: Classical angle-optimization method.
        :param seed: Optional seed for optimizer-local randomness.
        """
        if optimization_method not in OPTIMIZATION_METHOD_IDS:
            raise ValueError(f"Unsupported optimization method {optimization_method}. Expected one of " + ", ".join(OPTIMIZATION_METHOD_IDS) + ".")

        self.num_layers = num_layers
        self.layer_types = layer_types
        self.sampler = sampler
        self.optimization_method = optimization_method
        self.circuit = self.build_circuit()
        self.seed = seed

    def build_circuit(self) -> QuantumCircuit:
        """Builds the layered ansatz circuit from configured layer templates.
        :return: Parameterized ansatz circuit.
        """
        qc = QuantumCircuit(self.layer_types[0].num_qubits)
        qc.h(range(qc.num_qubits))
        for i in range(self.num_layers):
            for layer_type in self.layer_types:
                qc.compose(layer_type.get_circuit(str(i)), inplace=True)
        return qc

    def optimize_parameters(self, target_function: Callable[[str], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes variational parameters of the circuit to minimize expectation of target function and returns optimized parameter values.
        :param target_function: Function mapping sampled bitstrings to target scalar to minimize.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata.
        """
        self.reset_stats()
        get_target_stats = partial(self.get_function_stats, target_function)
        get_target_expectation = lambda angles: get_target_stats(angles)[0]
        match self.get_optimization_method():
            case "l-bfgs-b":
                result = self.optimize_parameters_l_bfgs_b(get_target_expectation, initial_angles)
            case "spsa":
                result = self.optimize_parameters_spsa(get_target_expectation, initial_angles)
            case "qnspsa":
                result = self.optimize_parameters_qnspsa(get_target_expectation, initial_angles)
            case "adam":
                result = self.optimize_parameters_adam(get_target_expectation, initial_angles)
            case "noisyopt":
                result = self.optimize_parameters_noisyopt(get_target_expectation, initial_angles)
            case "ax":
                result = self.optimize_parameters_ax(get_target_stats, len(self.circuit.parameters))
        result.expectation_time = self.expectation_time
        return result

    def reset_stats(self):
        """Resets quantum-evaluation counters before a parameter optimization run."""
        self.expectation_time = 0
        self.expectation_jobs = 0
        self.fidelity_time = 0
        self.fidelity_jobs = 0

    def get_function_stats(self, function: Callable[[str], float], param_vals: Sequence[float]) -> tuple[float, float]:
        """Evaluates expectation and standard error for given circuit parameter values.
        :param function: Function mapping sampled bitstrings to costs.
        :param param_vals: Parameter values assigned to the ansatz circuit.
        :return: Expected cost and standard error of the sampled mean estimate.
        """
        t1 = time.perf_counter()
        probabilities = self.sampler.get_sample_probabilities(self.circuit, param_vals)
        self.expectation_time += time.perf_counter() - t1
        self.expectation_jobs += 1
        return utils.get_function_stats(function, probabilities, self.sampler.get_num_shots())

    def get_optimization_method(self) -> str:
        """Returns the concrete optimization method selected for this run.
        :return: Optimization method name.
        """
        if self.optimization_method == "auto":
            return "l-bfgs-b" if isinstance(self.sampler, ExactSampler) else "spsa"
        if self.optimization_method in OPTIMIZATION_METHOD_IDS:
            return self.optimization_method
        raise ValueError(f"Unsupported optimization method {self.optimization_method}. Expected one of " + ", ".join(OPTIMIZATION_METHOD_IDS) + ".")

    @staticmethod
    def optimize_parameters_l_bfgs_b(objective: Callable[[Sequence[float]], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes parameters with SciPy L-BFGS-B.
        :param objective: Objective function mapping angle vectors to expected cost.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata.
        """
        return optimize.minimize(objective, initial_angles, method="L-BFGS-B", options={"maxiter": np.iinfo(np.int32).max})

    @staticmethod
    def optimize_parameters_spsa(objective: Callable[[Sequence[float]], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes parameters with Qiskit SPSA.
        :param objective: Objective function mapping angle vectors to expected cost.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata.
        """
        result = SPSA(maxiter=1000).minimize(objective, initial_angles)
        result.success = True
        return result

    def optimize_parameters_qnspsa(self, objective: Callable[[Sequence[float]], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes parameters with Qiskit quantum natural SPSA.
        :param objective: Objective function mapping angle vectors to expected cost.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata."""
        def fidelity(angles: ndarray, shifted_angles: ndarray) -> float:
            """Estimates fidelity between ansatz states prepared with two angle vectors.
            :param angles: First angle vector used by the inverse ansatz.
            :param shifted_angles: Second angle vector used by the forward ansatz.
            :return: Probability of returning to the all-zero state after compute-uncompute."""
            t1 = time.perf_counter()
            overlap_circuit = self.circuit.assign_parameters(shifted_angles)
            overlap_circuit.compose(self.circuit.assign_parameters(angles).inverse(), inplace=True)
            probabilities = self.sampler.get_sample_probabilities(overlap_circuit, [])
            self.fidelity_time += time.perf_counter() - t1
            self.fidelity_jobs += 1
            return probabilities.get("0" * self.circuit.num_qubits, 0)

        if self.seed is not None:
            algorithm_globals.random_seed = self.seed
        result = QNSPSA(fidelity).minimize(objective, initial_angles)
        result.success = True
        return result

    def optimize_parameters_adam(self, objective: Callable[[Sequence[float]], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes parameters with Qiskit ADAM and simultaneous-perturbation gradients.
        :param objective: Objective function mapping angle vectors to expected cost.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata.
        """
        def estimate_gradient(angles: ndarray) -> ndarray:
            """Estimates a stochastic gradient from one simultaneous perturbation.
            :param angles: Center point for the gradient estimate.
            :return: SPSA-style gradient estimate at ``angles``.
            """
            delta = rng.choice((-1, 1), size=len(angles))
            return (objective(angles + perturbation * delta) - objective(angles - perturbation * delta)) / (2 * perturbation) * delta

        perturbation = 0.1
        rng = np.random.default_rng(self.seed)
        result = ADAM(maxiter=1000, lr=0.03, tol=0).minimize(objective, initial_angles, jac=estimate_gradient)
        result.success = True
        return result

    @staticmethod
    def optimize_parameters_noisyopt(objective: Callable[[Sequence[float]], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes parameters with ``noisyopt`` compass search.
        :param objective: Objective function mapping angle vectors to expected cost.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata.
        """
        return noisyopt.minimizeCompass(objective, initial_angles, errorcontrol=False, deltatol=1e-4)

    @staticmethod
    def optimize_parameters_ax(objective: Callable[[Sequence[float]], tuple[float, float]], num_params: int) -> OptimizeResult:
        """Optimizes parameters with Ax using BoTorch-backed Bayesian optimization.
        :param objective: Objective function mapping angle vectors to expected cost and SEM.
        :param num_params: Number of circuit parameters to optimize.
        :return: Optimization result including optimized angles and metadata.
        """
        from logging import WARNING

        from ax.api.client import Client
        from ax.api.configs import RangeParameterConfig
        from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
        from ax.utils.common.logger import set_stderr_log_level

        initialization_budget = max(50, 5 * num_params)
        min_trials = initialization_budget + max(50, 3 * num_params)
        max_trials = 1000
        window_size = max(20, 2 * num_params)
        improvement_bar = 1e-3

        set_stderr_log_level(WARNING)
        client = Client()
        parameters = [RangeParameterConfig(name=f"angle_{i}", bounds=(-np.pi, np.pi), parameter_type="float") for i in range(num_params)]
        client.configure_experiment(parameters=parameters, name="variational_quantum_program_angles")
        client.configure_optimization(objective="-expectation")
        client.configure_generation_strategy(method="quality", initialization_budget=initialization_budget,
                                             min_observed_initialization_trials=initialization_budget, initialize_with_center=True)
        global_stopping_strategy = ImprovementGlobalStoppingStrategy(min_trials=min_trials, window_size=window_size, improvement_bar=improvement_bar)
        for trial_count in range(max_trials):
            trial_index, parameters = next(iter(client.get_next_trials(max_trials=1).items()))
            expectation, sem = objective(VariationalQuantumProgram.ax_parameters_to_angles(parameters))
            client.complete_trial(trial_index, raw_data={"expectation": (expectation, sem)})
            success, stop_message = global_stopping_strategy.should_stop_optimization(experiment=client._experiment)
            if success:
                break
        parameters, values, trial_index, _ = client.get_best_parameterization()
        message = f"Ax success={success}; trial_count={trial_count}; Best trial_index={trial_index}; stop_message={stop_message}"
        return OptimizeResult(x=VariationalQuantumProgram.ax_parameters_to_angles(parameters), fun=values["expectation"][0], success=success, message=message,
                              nit=trial_count, nfev=trial_count)

    @staticmethod
    def ax_parameters_to_angles(parameters: Mapping[str, int | float | str | bool]) -> ndarray:
        """Converts Ax parameter mapping to the ordered circuit-angle vector.
        :param parameters: Ax parameter mapping keyed by angle name.
        :return: Ordered angle vector.
        """
        return np.array([parameters[f"angle_{i}"] for i in range(len(parameters))])
