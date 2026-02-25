"""Variational quantum program for optimizing generator commitment distributions."""

import time
from functools import partial
from typing import Callable, Sequence

import noisyopt
import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
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
    :var classical_time: Accumulated CPU time spent evaluating classical expectation calculations.
    :var num_jobs: Number of quantum computer jobs.
    :var exp_eval_start_time: Start timestamp of currently running expectation evaluation, if any.
    """

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

    def __init__(self, num_layers: int, layer_types: list[CircuitLayer], sampler: Sampler) -> None:
        """Appends configured layer blocks and initializes program state.
        :param num_layers: Number of repeated ansatz blocks.
        :param layer_types: Layer templates composed once per block.
        :param sampler: Sampling backend used for probability estimation.
        """
        self.num_layers = num_layers
        self.layer_types = layer_types
        self.sampler = sampler
        self.circuit = self.build_circuit()
        self.classical_time = 0.0
        self.num_jobs = 0
        self.exp_eval_start_time: float | None = None

    def get_current_classical_time(self) -> float:
        """Returns accumulated classical-time including ongoing expectation evaluation.
        :return: Classical-time elapsed in seconds.
        """
        return self.classical_time + (time.perf_counter() - self.exp_eval_start_time if self.exp_eval_start_time is not None else 0.0)

    def get_cost_expectation(self, cost_function: Callable[[str], float], param_vals: Sequence[float]) -> float:
        """Evaluates expectation of the cost function for given circuit parameter values.
        :param cost_function: Function mapping sampled bitstrings to costs.
        :param param_vals: Parameter values assigned to the ansatz circuit.
        :return: Expected cost for the sampled output distribution.
        """
        probabilities = self.sampler.get_sample_probabilities(self.circuit, param_vals)
        self.num_jobs += 1
        self.exp_eval_start_time = time.perf_counter()
        expectation = utils.get_cost_expectation(cost_function, probabilities)
        self.classical_time += time.perf_counter() - self.exp_eval_start_time
        self.exp_eval_start_time = None
        return expectation

    def optimize_parameters(self, cost_function: Callable[[str], float], initial_angles: ndarray) -> OptimizeResult:
        """Optimizes variational parameters of the circuit to minimize expectation of cost function and returns optimized parameter values.
        :param cost_function: Function mapping sampled bitstrings to costs.
        :param initial_angles: Initial parameter vector for classical optimization.
        :return: Optimization result including optimized angles and metadata.
        """
        self.classical_time = 0.0
        self.num_jobs = 0
        self.exp_eval_start_time = None
        objective = partial(self.get_cost_expectation, cost_function)

        if isinstance(self.sampler, ExactSampler):
            result = optimize.minimize(objective, initial_angles, method="SLSQP", options={"maxiter": np.iinfo(np.int32).max})
        else:
            result = noisyopt.minimizeCompass(objective, initial_angles, errorcontrol=False)
        result.classical_eval_time = self.classical_time
        return result
