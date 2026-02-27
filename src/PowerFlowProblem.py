"""Data structures and constraint/cost evaluation for AC power-flow optimization."""

import math
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Sequence

import numpy as np
from networkx import Graph
from numpy.typing import NDArray

from .utils import my_format


class PowerFlowProblem:
    """Represents an AC-OPF-UC instance on a graph.
    :var graph: NetworkX graph with node and edge electrical metadata.
    :var generators: Flat list of generator objects collected from all nodes.
    """

    def __init__(self, graph: Graph):
        """Constructs power flow problem from graph.
        :param graph: Graph whose nodes contain generator/load/voltage/angle data and edges contain capacity/admittance data.
        Graph nodes should have the following properties:
        1) generators: list[Generator]. List of generator instances located at a given node.
        2) load: complex. Total load at a given node.
        3) voltage_range: tuple[float, float]. Range of allowed voltage magnitudes at this node.
        4) angle_range: tuple[float, float]. Range of allowed phase angles at this node.
        Graph edges should have the following properties:
        1) capacity: float > 0. Maximum absolute value of current that can be routed through a given edge.
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
        """Returns list of NDArray of 2 elements: [min, max], i.e. bounds on each continuous parameter for a given set of generator statuses.
        :param generator_status: Binary on/off string for all generators.
        :return: Bounds for active/reactive generation, voltages, and angles.
        """
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
        :param active_powers: Active generation values for all generators.
        :param reactive_powers: Reactive generation values for all generators.
        :param voltages: Voltage magnitudes for all nodes.
        :param angles: Voltage phase angles for all nodes.
        :return: Constraint values. First equality constraints (len = 2 * len(voltages) + 1), then inequality constraints (>= 0 is feasible for all).
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
        """Returns the total cost of generation for a given set of enabled generators at given power outputs.
        :param generator_statuses: Binary on/off string for all generators.
        :param active_powers: Active power outputs for all generators.
        :return: Total generation cost.
        """
        return sum(int(status) * gen.generation_cost(power) for status, gen, power in zip(generator_statuses, self.generators, active_powers))

    @staticmethod
    def check_infeasible(graph: Graph) -> str | None:
        """Returns infeasibility reason detected by fast necessary-condition pre-checks on graph metadata.
        :param graph: Graph whose node and generator bounds are inspected.
        :return: Explanation string when pre-check detects infeasibility, otherwise ``None``.
        """
        total_load_p = sum(node_data["load"].real for _, node_data in graph.nodes(data=True))
        total_load_q = sum(node_data["load"].imag for _, node_data in graph.nodes(data=True))
        generators = [gen for _, node_data in graph.nodes(data=True) for gen in node_data["generators"]]
        min_gen_p_min = min(gen.power_range[0] for gen in generators)
        total_gen_p_max = sum(gen.power_range[1] for gen in generators)
        min_gen_q_min = min(gen.reactive_power_range[0] for gen in generators)
        total_gen_q_max = sum(gen.reactive_power_range[1] for gen in generators)
        reasons = []

        if total_gen_p_max < total_load_p:
            reasons.append(f"Active generation upper bound ({total_gen_p_max:.6g}) is below total active load ({total_load_p:.6g}).")
        if min_gen_p_min > total_load_p:
            reasons.append(f"Smallest generator minimum ({min_gen_p_min:.6g}) is above total active load ({total_load_p:.6g}).")
        if total_gen_q_max < total_load_q:
            reasons.append(f"Reactive generation upper bound ({total_gen_q_max:.6g}) is below total reactive load ({total_load_q:.6g}).")
        if min_gen_q_min > total_load_q:
            reasons.append(f"Reactive generation lower bound ({min_gen_q_min:.6g}) is above total reactive load ({total_load_q:.6g}).")
        for node_label, node_data in graph.nodes(data=True):
            p_load = node_data["load"].real
            q_load = node_data["load"].imag
            local_p_max = sum(gen.power_range[1] for gen in node_data["generators"])
            local_q_min = sum(gen.reactive_power_range[0] for gen in node_data["generators"])
            local_q_max = sum(gen.reactive_power_range[1] for gen in node_data["generators"])
            required_import_p = max(0, p_load - local_p_max)
            if q_load > local_q_max:
                required_import_q = q_load - local_q_max
            elif q_load < local_q_min:
                required_import_q = local_q_min - q_load
            else:
                required_import_q = 0
            required_import_apparent = math.sqrt(required_import_p ** 2 + required_import_q ** 2)
            max_node_voltage = node_data["voltage_range"][1]
            min_import_current = required_import_apparent / max_node_voltage
            adjacent_capacity = sum(line_data["capacity"] for _, _, line_data in graph.edges(node_label, data=True))
            if adjacent_capacity < min_import_current:
                reasons.append(f"Adjacent edge capacity sum at node {node_label} is below required import current ")
        return " ".join(reasons) if len(reasons) > 0 else None


@dataclass
class PowerFlowSolution:
    """Represents solution to a power grid network.
    :var generator_statuses: Binary on/off string for all generators.
    :var active_powers: Active power outputs for all generators.
    :var reactive_powers: Reactive power outputs for all generators.
    :var voltages: Voltage magnitudes for all nodes.
    :var angles: Voltage phase angles for all nodes.
    :var cost: Objective value of the solution.
    :var history: Incumbent history entries with ``time`` and ``objective`` plus solver-specific metadata.
    :var extra: Additional solver-specific metadata.
    """
    generator_statuses: str
    active_powers: NDArray[float]
    reactive_powers: NDArray[float]
    voltages: NDArray[float]
    angles: NDArray[float]
    cost: float
    history: list[dict[str, float | int]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Returns a formatted multi-line representation of the solution.
        :return: Human-readable summary string.
        """
        buf = StringIO()
        with redirect_stdout(buf):
            print(f"Generator statuses: {self.generator_statuses}")
            print(f"Active powers     : {my_format(self.active_powers)}")
            print(f"Reactive powers   : {my_format(self.reactive_powers)}")
            print(f"Voltages          : {my_format(self.voltages)}")
            print(f"Phase angles      : {my_format(self.angles)}")
            print(f"Optimized cost    : {my_format(self.cost)}")
        return buf.getvalue()
