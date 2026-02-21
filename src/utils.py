"""Shared utility helpers for expectations, resource estimates, and formatting."""

from typing import Callable, Mapping, Any

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider


def get_cost_expectation(cost_function: Callable[[str], float], probabilities: dict[str, float]) -> float:
    """Evaluates expectation of the cost function for a given probability distribution.
    :param cost_function: Function that maps a bitstring sample to its cost.
    :param probabilities: Bitstring-probability mapping.
    :return: Expected cost under the provided distribution.
    """
    expectation = sum(cost_function(bitstring) * probability for bitstring, probability in probabilities.items())
    return expectation


def get_job_cost(circuit: QuantumCircuit, nfev: int) -> tuple[int, int, int, float]:
    """Estimates IonQ hardware execution cost from transpiled one/two-qubit gate counts.
    :param circuit: Quantum circuit whose native-gateset costs are estimated.
    :param nfev: Number of circuit evaluations to scale the per-evaluation cost.
    :return: One-qubit gate count, two-qubit gate count, evaluation count, and estimated total USD cost.
    """
    native_backend = IonQProvider().get_backend("qpu.forte-1", gateset="native")
    circuit_transpiled = qiskit.transpile(circuit, native_backend)
    num_one_qubit_gates = sum(1 for instruction in circuit_transpiled.data if instruction.operation.num_qubits == 1)
    num_two_qubit_gates = sum(1 for instruction in circuit_transpiled.data if instruction.operation.num_qubits == 2)
    return num_one_qubit_gates, num_two_qubit_gates, nfev, (num_one_qubit_gates * 0.0001645 + num_two_qubit_gates * 0.0011213) * nfev


def my_format(obj: Any, sig: int = 3, *, _top: bool = True) -> str:
    """Formats nested objects compactly with configurable significant digits for floats.
    :param obj: Object to format (scalars, arrays, mappings, and sequences are supported).
    :param sig: Number of significant digits used for float formatting.
    :param _top: Whether formatting occurs at top-level context for string handling.
    :return: Human-readable compact string representation.
    """
    if isinstance(obj, np.generic):
        obj = obj.item()
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()

    if isinstance(obj, float):
        return format(obj, f".{sig}g")
    if obj is None or isinstance(obj, (bool, int)):
        return repr(obj)
    if isinstance(obj, str):
        return obj if _top else repr(obj)

    if isinstance(obj, Mapping):
        return "{" + ", ".join(f"{my_format(k, sig, _top=False)}: {my_format(v, sig, _top=False)}" for k, v in obj.items()) + "}"

    if isinstance(obj, list):
        return "[" + ", ".join(my_format(v, sig, _top=False) for v in obj) + "]"

    if isinstance(obj, tuple):
        inner = ", ".join(my_format(v, sig, _top=False) for v in obj)
        if len(obj) == 1:
            inner += ","
        return "(" + inner + ")"

    return str(obj) if _top else repr(obj)
