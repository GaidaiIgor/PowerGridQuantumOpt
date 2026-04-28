"""Variational quantum program for optimizing generator commitment distributions."""

import time
from functools import partial
from typing import Callable, Sequence

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


class VariationalQuantumProgram:
    """Represents a parameterized quantum program with classical parameter optimization.
    :var num_layers: Number of repeated layer blocks in the ansatz.
    :var layer_types: Ordered layer templates composed in each block.
    :var sampler: Sampling backend used to estimate output probabilities.
    :var circuit: Fully constructed parameterized quantum circuit.
    :var quantum_time: Accumulated time spent evaluating sampled quantum probabilities.
    :var num_jobs: Number of quantum computer jobs.
    """

    def __init__(self, num_layers: int, layer_types: list[CircuitLayer], sampler: Sampler):
        """Appends configured layer blocks and initializes program state.
        :param num_layers: Number of repeated ansatz blocks.
        :param layer_types: Layer templates composed once per block.
        :param sampler: Sampling backend used for probability estimation.
        """
        self.num_layers = num_layers
        self.layer_types = layer_types
        self.sampler = sampler
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

    def get_probabilities(self, param_vals: Sequence[float]) -> dict[str, float]:
        """Evaluates output probabilities for given circuit parameter values.
        :param param_vals: Parameter values assigned to the ansatz circuit.
        :return: Bitstring-probability mapping returned by the configured sampler.
        """
        t1 = time.perf_counter()
        probabilities = self.sampler.get_sample_probabilities(self.circuit, param_vals)
        self.quantum_time += time.perf_counter() - t1
        self.num_jobs += 1
        return probabilities

    def get_cost_expectation(self, cost_function: Callable[[str], float], param_vals: Sequence[float]) -> float:
        """Evaluates expectation of the cost function for given circuit parameter values.
        :param cost_function: Function mapping sampled bitstrings to costs.
        :param param_vals: Parameter values assigned to the ansatz circuit.
        :return: Expected cost for the sampled output distribution.
        """
        probabilities = self.get_probabilities(param_vals)
        return utils.get_cost_expectation(cost_function, probabilities)

    def optimize_parameters(self, cost_function: Callable[[str], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes variational parameters of the circuit to minimize expectation of cost function and returns optimized parameter values.
        :param cost_function: Function mapping sampled bitstrings to costs.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata.
        """
        self.quantum_time = 0
        self.num_jobs = 0
        objective = partial(self.get_cost_expectation, cost_function)
        # objective = lambda params: -1 / self.get_cost_expectation(cost_function, params)

        if isinstance(self.sampler, ExactSampler):
            result = optimize.minimize(objective, initial_angles, method="SLSQP", options={"maxiter": np.iinfo(np.int32).max})
        else:
            # opt = SPSA(maxiter=100000)
            # opt = ADAM()
            # result = opt.minimize(objective, initial_angles)
            # result.success = True

            result = noisyopt.minimizeCompass(objective, initial_angles, errorcontrol=False, deltatol=1e-4)
        result.quantum_time = self.quantum_time
        return result
