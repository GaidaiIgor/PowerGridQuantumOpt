from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import Statevector
from qiskit_ionq import IonQProvider


class Sampler(ABC):
    """ Base class for samplers. """

    @abstractmethod
    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        """ Assigns given values to given parameterized circuit, executes it and returns dictionary where keys are bitstrings are values are their sampling probabilities. """
        pass


class ExactSampler(Sampler):
    """ Calculates exact probabilities of each generator_statuses. """

    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        bound = circuit.assign_parameters(param_vals)
        return Statevector(bound).probabilities_dict()


@dataclass
class MySamplerV2(Sampler):
    """ Uses sampler compatible with BaseSamplerV2 interface. """
    sampler: BaseSamplerV2

    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
        measured_circuit = circuit.measure_all(inplace=False)
        counts = self.sampler.run([(measured_circuit, param_vals)]).result()[0].data.meas.get_counts()
        probabilities = {key: value / self.sampler.default_shots for key, value in counts.items()}
        return probabilities


class IonQSampler(Sampler):
    """ Uses IonQ's hardware or cloud simulators to get probability distribution. """

    def __init__(self, backend_name: str, shots: int = 1000, noise_model: str = None):
        self.backend_name = backend_name
        self.backend = IonQProvider().get_backend(backend_name)
        self.shots = shots
        self.samples = 0
        if noise_model is not None:
            self.backend.set_options(noise_model=noise_model)

    def get_sample_probabilities(self, circuit: QuantumCircuit, param_vals: Sequence[float]) -> dict[str, float]:
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
