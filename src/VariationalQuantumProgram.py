"""Variational quantum program for optimizing generator commitment distributions."""

import time
from functools import partial
from typing import Callable, Mapping, Sequence

import noisyopt
import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import SPSA, ADAM
from scipy import optimize
from scipy.optimize import OptimizeResult

from . import utils
from .CircuitLayer import CircuitLayer
from .Sampler import Sampler, ExactSampler

OPTIMIZATION_METHOD_IDS = ("auto", "l-bfgs-b", "spsa", "adam", "noisyopt", "ax")


class VariationalQuantumProgram:
    """Represents a parameterized quantum program with classical parameter optimization.
    :var num_layers: Number of repeated layer blocks in the ansatz.
    :var layer_types: Ordered layer templates composed in each block.
    :var sampler: Sampling backend used to estimate output probabilities.
    :var optimization_method: Classical angle-optimization method.
    :var circuit: Fully constructed parameterized quantum circuit.
    :var quantum_time: Accumulated time spent evaluating sampled quantum probabilities.
    :var num_jobs: Number of quantum computer jobs.
    """
    num_layers: int
    layer_types: list[CircuitLayer]
    sampler: Sampler
    optimization_method: str
    circuit: QuantumCircuit
    quantum_time: float
    num_jobs: int

    def __init__(self, num_layers: int, layer_types: list[CircuitLayer], sampler: Sampler, optimization_method: str = "auto"):
        """Appends configured layer blocks and initializes program state.
        :param num_layers: Number of repeated ansatz blocks.
        :param layer_types: Layer templates composed once per block.
        :param sampler: Sampling backend used for probability estimation.
        :param optimization_method: Classical angle-optimization method.
        """
        if optimization_method not in OPTIMIZATION_METHOD_IDS:
            raise ValueError(f"Unsupported optimization method {optimization_method}. Expected one of " + ", ".join(OPTIMIZATION_METHOD_IDS) + ".")

        self.num_layers = num_layers
        self.layer_types = layer_types
        self.sampler = sampler
        self.optimization_method = optimization_method
        self.circuit = self.build_circuit()
        self.quantum_time = 0
        self.num_jobs = 0

    def build_circuit(self) -> QuantumCircuit:
        """Builds the layered ansatz circuit from configured layer templates.
        :return: Parameterized ansatz circuit.
        """
        qc = QuantumCircuit(self.layer_types[0].num_qubits)
        qc.h(range(qc.num_qubits))
        # qc.barrier()
        for i in range(self.num_layers):
            for layer_type in self.layer_types:
                qc.compose(layer_type.get_circuit(str(i)), inplace=True)
                # qc.barrier()
        return qc

    def get_function_stats(self, function: Callable[[str], float], param_vals: Sequence[float]) -> tuple[float, float]:
        """Evaluates expectation and standard error for given circuit parameter values.
        :param function: Function mapping sampled bitstrings to costs.
        :param param_vals: Parameter values assigned to the ansatz circuit.
        :return: Expected cost and standard error of the sampled mean estimate.
        """
        t1 = time.perf_counter()
        probabilities = self.sampler.get_sample_probabilities(self.circuit, param_vals)
        self.quantum_time += time.perf_counter() - t1
        self.num_jobs += 1
        return utils.get_function_stats(function, probabilities, self.sampler.get_num_shots())

    def optimize_parameters(self, target_function: Callable[[str], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes variational parameters of the circuit to minimize expectation of target function and returns optimized parameter values.
        :param target_function: Function mapping sampled bitstrings to target scalar to minimize.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata.
        """
        self.quantum_time = 0
        self.num_jobs = 0
        get_target_stats = partial(self.get_function_stats, target_function)
        get_target_expectation = lambda angles: get_target_stats(angles)[0]
        match self.get_optimization_method():
            case "l-bfgs-b":
                result = self.optimize_parameters_l_bfgs_b(get_target_expectation, initial_angles)
            case "spsa":
                result = self.optimize_parameters_spsa(get_target_expectation, initial_angles)
            case "adam":
                result = self.optimize_parameters_adam(get_target_expectation, initial_angles)
            case "noisyopt":
                result = self.optimize_parameters_noisyopt(get_target_expectation, initial_angles)
            case "ax":
                result = self.optimize_parameters_ax(get_target_stats)
        result.quantum_time = self.quantum_time
        return result

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

    @staticmethod
    def optimize_parameters_adam(objective: Callable[[Sequence[float]], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes parameters with Qiskit ADAM.
        :param objective: Objective function mapping angle vectors to expected cost.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata.
        """
        result = ADAM().minimize(objective, initial_angles)
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
    def optimize_parameters_ax(objective: Callable[[Sequence[float]], tuple[float, float]]) -> OptimizeResult:
        """Optimizes parameters with Ax using BoTorch-backed Bayesian optimization.
        :param objective: Objective function mapping angle vectors to expected cost and SEM.
        :return: Optimization result including optimized angles and metadata.
        """
        from logging import WARNING

        from ax.api.client import Client
        from ax.api.configs import RangeParameterConfig
        from ax.utils.common.logger import set_stderr_log_level

        ax_trial_count = 1000
        set_stderr_log_level(WARNING)
        client = Client()
        parameters = [RangeParameterConfig(name=f"angle_{i}", bounds=(-np.pi, np.pi), parameter_type="float") for i in range(len(initial_angles))]
        client.configure_experiment(parameters=parameters, name="variational_quantum_program_angles")
        client.configure_optimization(objective="-expectation")
        for _ in range(ax_trial_count):
            trial_index, parameters = next(iter(client.get_next_trials(max_trials=1).items()))
            client.complete_trial(trial_index, raw_data={"expectation": objective(VariationalQuantumProgram.ax_parameters_to_angles(parameters))})
        parameters, values, trial_index, _ = client.get_best_parameterization()
        return OptimizeResult(x=VariationalQuantumProgram.ax_parameters_to_angles(parameters), fun=values["expectation"][0], success=True,
                              message=f"Ax completed {ax_trial_count} trials; best trial index {trial_index}.", nit=ax_trial_count, nfev=ax_trial_count)

    @staticmethod
    def ax_parameters_to_angles(parameters: Mapping[str, int | float | str | bool]) -> ndarray:
        """Converts Ax parameter mapping to the ordered circuit-angle vector.
        :param parameters: Ax parameter mapping keyed by angle name.
        :return: Ordered angle vector.
        """
        return np.array([parameters[f"angle_{i}"] for i in range(len(parameters))])
