"""Generates datasets for power-flow optimization experiments."""

import argparse
from pathlib import Path

from src import PowerFlowProblemGenerator


def generate_dataset(num_generators: int, num_instances: int, output_path: Path) -> None:
    """Generates a dataset of random power-flow instances.
    :param num_generators: Number of generators to place in each generated problem family.
    :param num_instances: Number of instances to generate.
    :param output_path: Destination directory for generated instance files.
    """
    problem_generator = PowerFlowProblemGenerator()
    problem_generator.generate_instances(num_generators, num_instances, voltage_range=(0, 100), output_folder=output_path, strictness_factor=1.2)


def parse_cli_args() -> tuple[int, int, Path]:
    """Parses command-line arguments for dataset generation.
    :return: Generator count, instance count, and destination directory.
    """
    parser = argparse.ArgumentParser(description="Generates random power-flow instances.")
    parser.add_argument("num_generators", type=int, help="Number of generators to use when generating instances.")
    parser.add_argument("num_instances", type=int, help="Number of instances to generate.")
    parser.add_argument("output_path", type=Path, help="Directory where generated instances will be written.")
    args = parser.parse_args()
    return args.num_generators, args.num_instances, args.output_path


if __name__ == "__main__":
    generate_dataset(*parse_cli_args())
