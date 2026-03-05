"""Debug utilities."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.PowerFlowProblem import PowerFlowProblem, PowerFlowSolution


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


def set_all_edge_capacities(problem: PowerFlowProblem, capacity: float) -> None:
    """Sets all edge capacities in a power-flow problem to the same value.
    :param problem: Power-flow problem whose graph edges are modified in place.
    :param capacity: Capacity value assigned to each edge.
    """
    for _, _, edge_data in problem.graph.edges(data=True):
        edge_data["capacity"] = capacity


def set_all_node_voltage_ranges(problem: PowerFlowProblem, voltage_range: tuple[float, float]) -> None:
    """Sets voltage ranges for all nodes in a power-flow problem to the same range.
    :param problem: Power-flow problem whose node voltage ranges are modified in place.
    :param voltage_range: Voltage range assigned to each node.
    """
    for _, node_data in problem.graph.nodes(data=True):
        node_data["voltage_range"] = voltage_range


def set_all_generator_p_min(problem: PowerFlowProblem, p_min: float) -> None:
    """Sets active-power lower bound for all generators in a power-flow problem.
    :param problem: Power-flow problem whose generator active-power ranges are modified in place.
    :param p_min: Lower bound assigned to active-power range of each generator.
    """
    for _, node_data in problem.graph.nodes(data=True):
        for generator in node_data["generators"]:
            generator.power_range = (p_min, generator.power_range[1])


def print_solution_from_csv(data_path: str | Path, instance_index: int) -> None:
    """Reads one problem instance and its persisted CSV solution, then prints it.
    :param data_path: Path to a specific ``.solutions_*.csv`` file.
    :param instance_index: Instance index identifying both ``<index>.pkl`` and the matching CSV row.
    """
    solutions_path = Path(data_path)
    dataset_path = solutions_path.parent
    with (dataset_path / f"{instance_index}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file))

    solutions_df = pd.read_csv(solutions_path, dtype={"instance": "Int64", "generator_assignments": "string"})
    solution_row = solutions_df.loc[solutions_df["instance"].astype(int) == instance_index].iloc[0]
    params = np.fromstring(solution_row["continuous_parameters"].strip("[]"), sep=",")
    active_powers, reactive_powers, voltages, angles = problem.split_params(params)
    solution = PowerFlowSolution(solution_row["generator_assignments"], active_powers, reactive_powers, voltages, angles, solution_row["cost"])
    solution.print(problem)
