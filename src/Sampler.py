"""Sampling backends used by the variational quantum optimization loop."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import Statevector
from qiskit_ionq import IonQProvider


class Sampler(ABC):
    """Base class for samplers."""

    @abstractmethod
    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        """Assigns given values to given parameterized circuit, executes it and returns dictionary where keys are bitstrings are values are their sampling probabilities.
        :param circuit: Parameterized circuit to evaluate.
        :param param_vals: Parameter values assigned to the circuit.
        :return: Bitstring-probability mapping.
        """
        pass


class ExactSampler(Sampler):
    """Calculates exact probabilities of each generator_statuses."""

    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        """Returns exact probabilities from the bound circuit statevector.
        :param circuit: Parameterized circuit to evaluate.
        :param param_vals: Parameter values assigned to the circuit.
        :return: Exact bitstring-probability mapping from statevector amplitudes.
        """
        bound = circuit.assign_parameters(param_vals)
        return Statevector(bound).probabilities_dict()


@dataclass
class MySamplerV2(Sampler):
    """Uses sampler compatible with BaseSamplerV2 interface.
    :var sampler: Backend sampler object implementing the ``BaseSamplerV2`` interface.
    """
    sampler: BaseSamplerV2

    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        """Returns empirical probabilities using a ``BaseSamplerV2`` implementation.
        :param circuit: Parameterized circuit to evaluate.
        :param param_vals: Parameter values assigned to the circuit.
        :return: Bitstring-probability mapping estimated from shot counts.
        """
        measured_circuit = circuit.measure_all(inplace=False)
        counts = self.sampler.run([(measured_circuit, param_vals)]).result()[0].data.meas.get_counts()
        probabilities = {key: value / self.sampler.default_shots for key, value in counts.items()}
        return probabilities


class IonQSampler(Sampler):
    """Uses IonQ's hardware or cloud simulators to get probability distribution."""

    def __init__(self, backend_name: str, shots: int = 1000, noise_model: str = None):
        """Initializes an IonQ backend and optional noise model for sampling.
        :param backend_name: IonQ backend name, e.g. simulator or QPU identifier.
        :param shots: Number of shots used per sampling job.
        :param noise_model: Optional IonQ noise model name for simulator runs.
        """
        self.backend_name = backend_name
        self.backend = IonQProvider().get_backend(backend_name)
        self.shots = shots
        self.samples = 0
        if noise_model is not None:
            self.backend.set_options(noise_model=noise_model)

    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        """Executes the circuit on IonQ backend and return normalized bitstring frequencies.
        :param circuit: Parameterized circuit to evaluate.
        :param param_vals: Parameter values assigned to the circuit.
        :return: Bitstring-probability mapping estimated from backend counts.
        """
        print(f"Sample #{self.samples}")
        print(f"Parameters: {param_vals}")
        bound = circuit.assign_parameters(param_vals)
        bound.measure_all()
        result = self.backend.run(bound, shots=self.shots).result()
        counts = result.get_counts()
        counts = {key.rjust(circuit.num_qubits, "0"): value / self.shots for key, value in counts.items()}
        print(f"Probabilities: {counts}")
        self.samples += 1
        return counts
