"""Generates datasets for power-flow optimization experiments."""

import argparse
from pathlib import Path

from src import PowerFlowProblemGenerator


def generate_dataset(num_generators: int, num_instances: int, output_path: Path, average_node_degree: float | None = None):
    """Generates a dataset of random power-flow instances.
    :param num_generators: Number of generators to place in each generated problem family.
    :param num_instances: Number of instances to generate.
    :param output_path: Destination directory for generated instance files.
    :param average_node_degree: Optional target average graph degree; generator default is used when omitted.
    """
    problem_generator = PowerFlowProblemGenerator()
    generate_kwargs = {"voltage_range": (0, 100), "output_folder": output_path, "strictness_factor": 1.2}
    if average_node_degree is not None:
        generate_kwargs["average_node_degree"] = average_node_degree
    problem_generator.generate_instances(num_generators, num_instances, **generate_kwargs)


def parse_cli_args() -> tuple[int, int, Path, float | None]:
    """Parses command-line arguments for dataset generation.
    :return: Generator count, instance count, destination directory, and optional target average graph degree.
    """
    parser = argparse.ArgumentParser(description="Generates random power-flow instances.")
    parser.add_argument("-ng", "--num-generators", type=int, required=True, help="Number of generators to use when generating instances.")
    parser.add_argument("-ni", "--num-instances", type=int, required=True, help="Number of instances to generate.")
    parser.add_argument("-op", "--output-path", type=Path, required=True, help="Directory where generated instances will be written.")
    parser.add_argument("-and", "--average-node-degree", type=float, default=None, help="Target average graph degree. Omit to use the generator default.")
    args = parser.parse_args()
    return args.num_generators, args.num_instances, args.output_path, args.average_node_degree


if __name__ == "__main__":
    generate_dataset(*parse_cli_args())
