from typing import Callable

import qiskit
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider


def get_cost_expectation(cost_function: Callable[[str], float], probabilities: dict[str, float]):
    """ Evaluates expectation of the cost function for a given probability distribution. """
    expectation = sum(cost_function(bitstring) * probability for bitstring, probability in probabilities.items())
    return expectation


def get_job_cost(circuit: QuantumCircuit, nfev: int) -> tuple[int, int, int, float]:
    native_backend = IonQProvider().get_backend("qpu.forte-1", gateset="native")
    circuit_transpiled = qiskit.transpile(circuit, native_backend)
    num_one_qubit_gates = sum(1 for instruction in circuit_transpiled.data if instruction.operation.num_qubits == 1)
    num_two_qubit_gates = sum(1 for instruction in circuit_transpiled.data if instruction.operation.num_qubits == 2)
    return num_one_qubit_gates, num_two_qubit_gates, nfev, (num_one_qubit_gates * 0.0001645 + num_two_qubit_gates * 0.0011213) * nfev

