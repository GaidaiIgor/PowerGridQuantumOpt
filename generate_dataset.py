"""Generates datasets for power-flow optimization experiments."""

from src import PowerFlowProblemGenerator


def generate_dataset() -> None:
    """Generates a dataset of random power-flow instances."""
    num_generators = 5
    problem_generator = PowerFlowProblemGenerator()
    problem_generator.generate_instances(num_generators, 100, voltage_range=(0, 100), output_folder=f"data/{num_generators}", strictness_factor=1.2)


if __name__ == "__main__":
    generate_dataset()
