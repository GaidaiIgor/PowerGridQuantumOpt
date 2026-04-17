"""Debug utilities."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from cattrs.preconf.json import make_converter

from src.EvaluationResult import EvaluationResult
from src.HistoryEntry import HistoryEntry
from src.PowerFlowProblem import PowerFlowProblem


def save_instance_human_readable(instance_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Reads one serialized instance and saves it as readable JSON.
    :param instance_path: Path to input instance ``.pkl`` file.
    :param output_path: Optional output file path. Defaults to ``<instance>.json``.
    :return: Path to written JSON file.
    """
    def encode_complex(value: complex) -> dict[str, float]:
        """Encodes complex value into JSON-serializable mapping.
        :param value: Complex value to encode.
        :return: Mapping with real and imaginary parts.
        """
        return {"real": value.real, "imag": value.imag}

    source_path = Path(instance_path)
    destination_path = source_path.with_suffix(".json") if output_path is None else Path(output_path)
    with source_path.open("rb") as file:
        graph = pickle.load(file)
    problem = PowerFlowProblem(graph)

    nodes = [
        {
            "label": node_label,
            "load": encode_complex(node_data["load"]),
            "voltage_range": list(node_data["voltage_range"]),
            "angle_range": list(node_data["angle_range"]),
            "generators": [
                {
                    "power_range": list(generator.power_range),
                    "reactive_power_range": list(generator.reactive_power_range),
                    "cost_terms": list(generator.cost_terms),
                }
                for generator in node_data["generators"]
            ],
        }
        for node_label, node_data in sorted(problem.graph.nodes(data=True))
    ]
    edges = [
        {
            "u": u,
            "v": v,
            "capacity": edge_data["capacity"],
            "admittance": encode_complex(edge_data["admittance"]),
        }
        for u, v, edge_data in sorted(problem.graph.edges(data=True))
    ]
    payload = {
        "num_nodes": problem.graph.number_of_nodes(),
        "num_edges": problem.graph.number_of_edges(),
        "num_generators": len(problem.generators),
        "nodes": nodes,
        "edges": edges,
    }
    destination_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination_path


def set_all_edge_capacities(problem: PowerFlowProblem, capacity: float):
    """Sets all edge capacities in a power-flow problem to the same value.
    :param problem: Power-flow problem whose graph edges are modified in place.
    :param capacity: Capacity value assigned to each edge.
    """
    for _, _, edge_data in problem.graph.edges(data=True):
        edge_data["capacity"] = capacity


def set_all_node_voltage_ranges(problem: PowerFlowProblem, voltage_range: tuple[float, float]):
    """Sets voltage ranges for all nodes in a power-flow problem to the same range.
    :param problem: Power-flow problem whose node voltage ranges are modified in place.
    :param voltage_range: Voltage range assigned to each node.
    """
    for _, node_data in problem.graph.nodes(data=True):
        node_data["voltage_range"] = voltage_range


def set_all_generator_p_min(problem: PowerFlowProblem, p_min: float):
    """Sets active-power lower bound for all generators in a power-flow problem.
    :param problem: Power-flow problem whose generator active-power ranges are modified in place.
    :param p_min: Lower bound assigned to active-power range of each generator.
    """
    for _, node_data in problem.graph.nodes(data=True):
        for generator in node_data["generators"]:
            generator.power_range = (p_min, generator.power_range[1])


def print_solution_from_csv(csv_path: str | Path, instance_index: int):
    """Reads one problem instance and its persisted CSV solution, then prints it.
    :param csv_path: Path to a specific csv file.
    :param instance_index: Instance index identifying both ``<index>.pkl`` and the matching CSV row.
    """
    solutions_path = Path(csv_path)
    dataset_path = solutions_path.parent
    with (dataset_path / f"{instance_index}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file))

    solutions_df = pd.read_csv(solutions_path, dtype={"instance": "Int64", "generators": "string"})
    solution_row = solutions_df.loc[solutions_df["instance"].astype(int) == instance_index].iloc[0]
    converter = make_converter()
    history = converter.loads(solution_row["history"], list[HistoryEntry])
    for entry_ind, entry in enumerate(history):
        print(f"===========================================================================")
        print(f"History entry {entry_ind}: time={entry.time:.6g}, job_ind={entry.job_ind}")
        print_evaluation_result(problem, entry.result)


def print_evaluation_result(problem: PowerFlowProblem, result: EvaluationResult):
    """Prints node and line values for one power-flow problem and its solution. Positive active power is produced. Negative active power is spent.
    :param problem: Power-flow instance that defines graph topology, loads, and bounds.
    :param result: Evaluation result printed against ``problem`` bounds and line capacities.
    """
    active_powers, reactive_powers, voltages, angles = problem.split_params(np.array(result.params))
    bounds = problem.get_bounds(result.generator_statuses)
    bounds_active, bounds_reactive, bounds_voltage, bounds_angle = problem.split_params(bounds)
    voltage_phasors = voltages * np.exp(1j * angles)

    total_load = -sum(node_data["load"] for _, node_data in problem.graph.nodes(data=True))
    total_generation_p_min = sum(active_bounds[0] for active_bounds in bounds_active)
    total_generation_p_max = sum(active_bounds[1] for active_bounds in bounds_active)
    total_generation_q_min = sum(reactive_bounds[0] for reactive_bounds in bounds_reactive)
    total_generation_q_max = sum(reactive_bounds[1] for reactive_bounds in bounds_reactive)
    print(f"Objective: {result.fun:.3g}")
    print(f"Violation: {result.violation:.3g}")
    print(f"Generator assignments: {result.generator_statuses}")
    print(f"Total load: P: {total_load.real}, Q: {total_load.imag}")
    print(f"Total generation range: P: {total_generation_p_min:.3g} <= total <= {total_generation_p_max:.3g}, "
          f"Q: {total_generation_q_min:.3g} <= total <= {total_generation_q_max:.3g}")
    for node_label, node_data in problem.graph.nodes(data=True):
        node_ind = node_data["node_ind"]
        voltage_bounds = bounds_voltage[node_ind]
        angle_bounds = bounds_angle[node_ind]
        load_power = -node_data["load"]
        print(f"Node {node_label}:")
        print(f"  Load: P: {load_power.real:.3g}, Q: {load_power.imag:.3g}")
        print(f"  Voltage: {voltage_bounds[0]:.3g} <= {voltages[node_ind]:.3g} <= {voltage_bounds[1]:.3g}")
        print(f"  Angle: {angle_bounds[0]:.3g} <= {angles[node_ind]:.3g} <= {angle_bounds[1]:.3g}")
        for gen_index in node_data["gen_inds"]:
            active_bounds = bounds_active[gen_index]
            reactive_bounds = bounds_reactive[gen_index]
            print(f"  Generator {gen_index}: P: {active_bounds[0]:.3g} <= {active_powers[gen_index]:.3g} <= {active_bounds[1]:.3g}, "
                  f"Q: {reactive_bounds[0]:.3g} <= {reactive_powers[gen_index]:.3g} <= {reactive_bounds[1]:.3g}")
        total_generation = np.sum(active_powers[node_data["gen_inds"]]) + 1j * np.sum(reactive_powers[node_data["gen_inds"]])
        print(f"  Total generation: P: {total_generation.real:.3g}, Q: {total_generation.imag:.3g}")
        total_line_power = 0
        for _, neighbor_label, line_data in problem.graph.edges(node_label, data=True):
            neighbor_data = problem.graph.nodes[neighbor_label]
            voltage_diff = voltage_phasors[node_ind] - voltage_phasors[neighbor_data["node_ind"]]
            current_phasor = line_data["admittance"] * voltage_diff
            line_power = voltage_phasors[node_ind] * np.conj(current_phasor)
            signed_line_power = -line_power
            total_line_power += signed_line_power
            print(f"  Line {node_label}--{neighbor_label}: Capacity {np.abs(current_phasor):.3g} <= {line_data['capacity']:.3g}; "
                  f"Power P: {signed_line_power.real:.3g}, Q: {signed_line_power.imag:.3g}")
        print(f"  Total line power: P: {total_line_power.real:.3g}, Q: {total_line_power.imag:.3g}")
        total_node_power_balance = total_generation + load_power + total_line_power
        print(f"  Total node power balance: P: {total_node_power_balance.real:.3g}, Q: {total_node_power_balance.imag:.3g}")
        print("")
