"""Building blocks for parameterized quantum circuit layers used in VQA."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


@dataclass
class CircuitLayer(ABC):
    """A layer of a VQA.
    :var num_qubits: Number of qubits in the circuit.
    """
    num_qubits: int

    @staticmethod
    def connect_pair(qc: QuantumCircuit, i: int, j: int, angle: Parameter) -> None:
        """Appends an RZZ coupling gate between two qubits.
        :param qc: Circuit that receives the new gate.
        :param i: Index of the first qubit.
        :param j: Index of the second qubit.
        :param angle: Rotation parameter for the RZZ gate.
        """
        qc.rzz(angle, i, j)

    @abstractmethod
    def get_circuit(self, name_suffix: str) -> QuantumCircuit:
        """Returns a Qiskit circuit for this layer and suffixes parameter names.
        :param name_suffix: Suffix appended to created parameter names.
        :return: Circuit implementing the layer.
        """
        pass


class AllToAllEntangler(CircuitLayer):
    """Entangler that applies parameterized ZZ couplings to all qubit pairs."""

    def get_circuit(self, name_suffix: str) -> QuantumCircuit:
        """Builds an all-to-all ZZ entangling layer.
        :param name_suffix: Suffix appended to created parameter names.
        :return: Circuit with one parameterized ZZ gate per qubit pair.
        """
        qc = QuantumCircuit(self.num_qubits)
        params = ParameterVector(f"G_{name_suffix}", self.num_qubits * (self.num_qubits - 1) // 2)
        for ind, (i, j) in enumerate(combinations(range(self.num_qubits), 2)):
            self.connect_pair(qc, i, j, params[ind])
        return qc


class ButterflyEntangler(CircuitLayer):
    """Recursive butterfly-style ZZ entangler over contiguous qubit ranges."""

    def connect_qubits(self, qc: QuantumCircuit, qubit_range: tuple[int, int], name_suffix: str) -> None:
        """Applies butterfly couplings recursively over a half-open qubit range.
        :param qc: Circuit that receives coupling gates.
        :param qubit_range: Half-open index range ``(start, stop)`` to entangle.
        :param name_suffix: Suffix appended to created parameter names.
        """
        range_len = qubit_range[1] - qubit_range[0]
        if range_len < 2:
            return
        step = 2 ** (math.ceil(math.log2(range_len)) - 1)
        params = ParameterVector(f"G_{name_suffix}_{qubit_range[0]}{qubit_range[1]}", range_len - step)
        for ind, start in enumerate(range(qubit_range[0], qubit_range[0] + len(params))):
            self.connect_pair(qc, start, start + step, params[ind])
        self.connect_qubits(qc, (qubit_range[0], sum(qubit_range) // 2), name_suffix)
        self.connect_qubits(qc, (sum(qubit_range) // 2, qubit_range[1]), name_suffix)

    def get_circuit(self, name_suffix: str) -> QuantumCircuit:
        """Builds a butterfly entangling layer spanning all qubits.
        :param name_suffix: Suffix appended to created parameter names.
        :return: Circuit with butterfly-pattern ZZ couplings.
        """
        qc = QuantumCircuit(self.num_qubits)
        self.connect_qubits(qc, (0, self.num_qubits), name_suffix)
        return qc


class ZXMixer(CircuitLayer):
    """Mixer layer composed of RZ and RX rotations on every qubit."""

    def get_circuit(self, name_suffix: str) -> QuantumCircuit:
        """Builds the parameterized ZX mixer circuit.
        :param name_suffix: Suffix appended to created parameter names.
        :return: Circuit with one RZ and one RX rotation per qubit.
        """
        qc = QuantumCircuit(self.num_qubits)
        params = ParameterVector(f"B_{name_suffix}", 2 * self.num_qubits)
        for i in range(self.num_qubits):
            qc.rz(params[i], i)
        for i in range(self.num_qubits):
            qc.rx(params[self.num_qubits + i], i)
        return qc
