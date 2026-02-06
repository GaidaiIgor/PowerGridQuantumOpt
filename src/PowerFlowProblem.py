from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Sequence

import numpy as np
from networkx import Graph
from numpy.typing import NDArray

from .utils import my_format


class PowerFlowProblem:
    """ AC-OPF-UC problem. """

    def __init__(self, graph: Graph):
        """ Graph nodes should have the following properties:
        1) generators: list[Generator]. List of generator instances located at a given node.
        2) load: complex. Total load at a given node.
        3) voltage_range: tuple[float, float]. Range of allowed voltage magnitudes at this node.
        4) angle_range: tuple[float, float]. Range of allowed phase angles at this node.
        Graph edges should have the following properties:
        1) capacity: float > 0. Maximum absolute value of power that can be routed through a given edge.
        2) admittance: complex. Line admittance.
        The following additional properties control mapping between node parameters and the overall optimization vector.
        1) gen_inds: list[int]. List of generator indices in self.generators corresponding to generators at this node.
        Collects generators from all nodes into a single generators list. """
        self.graph = graph
        self.generators = []
        for i, (_, data) in enumerate(sorted(self.graph.nodes(data=True))):
            data["node_ind"] = i
            data["gen_inds"] = list(range(len(self.generators), len(self.generators) + len(data["generators"])))
            self.generators += data["generators"]

    def get_bounds(self, generator_status: str) -> list[NDArray[float]]:
        """ Returns list of NDArray of 2 elements: [min, max], i.e. bounds on each continuous parameter for a given set of generator statuses. """
        bounds_active = [np.array(gen.power_range) * int(generator_status[i]) for i, gen in enumerate(self.generators)]
        bounds_reactive = [np.array(gen.reactive_power_range) * int(generator_status[i]) for i, gen in enumerate(self.generators)]
        bounds_voltage = [0] * len(self.graph)
        bounds_angle = [0] * len(self.graph)
        for node, data in self.graph.nodes(data=True):
            bounds_voltage[data["node_ind"]] = np.array(data["voltage_range"])
            bounds_angle[data["node_ind"]] = np.array(data["angle_range"])
        bounds = bounds_active + bounds_reactive + bounds_voltage + bounds_angle
        return bounds

    def evaluate_constraints(self, active_powers: NDArray[float], reactive_powers: NDArray[float], voltages: NDArray[float], angles: NDArray[float]) \
            -> list[float]:
        """
        Evaluates all constraints other than bounds, i.e. power balance at each node (generated params + incoming - outgoing - load == 0)
        and line capacities (|I_ij| <= max capacity).
        Returns a list with constraint values. First equality constraints (len = 2 * len(voltages) + 1), then inequality constraints (>= 0 is feasible for all).
        """
        complex_powers = active_powers + 1j * reactive_powers
        voltage_phasors = voltages * np.exp(1j * angles)

        equality_constraints = [angles[0]]
        inequality_constraints = []
        for node_label, node_data in self.graph.nodes(data=True):
            generated_power = np.sum(complex_powers[node_data["gen_inds"]])
            outgoing_line_powers = []
            for _, neighbor_label, line_data in self.graph.edges(node_label, data=True):
                neighbor_data = self.graph.nodes[neighbor_label]
                volt_diff = voltage_phasors[node_data["node_ind"]] - voltage_phasors[neighbor_data["node_ind"]]
                current_phasor = line_data["admittance"] * volt_diff
                line_power = voltage_phasors[node_data["node_ind"]] * np.conj(current_phasor)
                outgoing_line_powers.append(line_power)
                if node_data["node_ind"] < neighbor_data["node_ind"]:
                    inequality_constraints.append(line_data["capacity"] - np.abs(current_phasor))

            power_balance = generated_power - node_data["load"] - np.sum(outgoing_line_powers)
            equality_constraints.append(np.real(power_balance))
            equality_constraints.append(np.imag(power_balance))
        return equality_constraints + inequality_constraints

    def get_generation_cost(self, generator_statuses: str, active_powers: Sequence[float]) -> float:
        """ Returns the total cost of generation for a given set of enabled generators at given power outputs. """
        return sum(int(status) * gen.generation_cost(power) for status, gen, power in zip(generator_statuses, self.generators, active_powers))


@dataclass
class PowerFlowSolution:
    """ Represents solution to a power grid network. """
    generator_statuses: str
    active_powers: NDArray[float]
    reactive_powers: NDArray[float]
    voltages: NDArray[float]
    angles: NDArray[float]
    cost: float
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        """ Prints solution. """
        buf = StringIO()
        with redirect_stdout(buf):
            print(f"Generator statuses: {self.generator_statuses}")
            print(f"Active powers     : {my_format(self.active_powers)}")
            print(f"Reactive powers   : {my_format(self.reactive_powers)}")
            print(f"Voltages          : {my_format(self.voltages)}")
            print(f"Phase angles      : {my_format(self.angles)}")
            print(f"Optimized cost    : {my_format(self.cost)}")
        return buf.getvalue()
