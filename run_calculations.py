import os
import pickle
import shutil
import time
from concurrent.futures import TimeoutError as FutureTimeoutError, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from networkx import Graph
from pebble import ProcessPool
from tqdm import tqdm

import debug
from src import PowerFlowProblemGenerator
from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.ContinuousPowerOptimizer import ContinuousPowerOptimizer
from src.Generator import Generator
from src.PowerFlowProblem import PowerFlowProblem
from src.PowerFlowSolver import ClassicalSolver, HybridSolver, PowerFlowSolver
from src.Sampler import ExactSampler
from src.VariationalQuantumProgram import VariationalQuantumProgram
from src.utils import my_format


def get_power_flow_ac_problem() -> PowerFlowProblem:
    voltage_range = (0, 10)
    angle_range = (-np.pi, np.pi)
    graph = Graph()

    graph.add_node(0, generators=[Generator((0, 100), (-100, 100), (0, 1, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_node(1, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    graph.add_edge(0, 1, capacity=100, admittance=1 + 1j)

    # graph.add_node(0, generators=[Generator((0, 30), (0, 0), (0, 10, 1))], load=0, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(1, generators=[Generator((0, 10), (0, 0), (0, 20, 1))], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_node(2, generators=[], load=10, voltage_range=voltage_range, angle_range=angle_range)
    # graph.add_edge(0, 1, capacity=10, admittance=1)
    # graph.add_edge(0, 2, capacity=5, admittance=1)
    # graph.add_edge(1, 2, capacity=10, admittance=1)

    return PowerFlowProblem(graph)


def generate_dataset():
    problem_generator = PowerFlowProblemGenerator()
    problem_generator.generate_instances(5, 100, output_folder="data/5", strictness_factor=1.2)


def get_variational_quantum_program(num_qubits: int) -> VariationalQuantumProgram:
    entangler = AllToAllEntangler(num_qubits)
    mixer = ZXMixer(num_qubits)
    num_layers = 1

    sampler = ExactSampler()
    # sampler = MySamplerV2(StatevectorSampler(default_shots=1000))
    # sampler = IonQSampler("simulator", 1000, None)
    # sampler = IonQSampler("qpu.forte-enterprise-1", 1000, None)

    return VariationalQuantumProgram(num_layers, [entangler, mixer], sampler)


def get_hybrid_solver(num_generators: int) -> HybridSolver:
    vqp = get_variational_quantum_program(num_generators)
    penalty_mult = 10
    inner_optimizer_factory = lambda problem: ContinuousPowerOptimizer(problem, penalty_mult)
    seed = 0
    return HybridSolver(vqp, inner_optimizer_factory, seed)


def run_single():
    # problem = get_power_flow_ac_problem()
    index = 5
    data_path = Path(f"data/5")
    with (data_path / f"{index}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file))

    debug.set_all_edge_capacities(problem, 100)
    # debug.set_all_node_voltage_ranges(problem, (0.5, 1.5))
    # debug.set_all_generator_p_min(problem, 0)

    solver = ClassicalSolver()
    # solver = get_hybrid_solver(len(problem.generators))
    progress_folder = data_path / ".progress"
    progress_folder.mkdir(exist_ok=True)
    progress_path = progress_folder / f"{index}.pkl"
    solution = solver.solve(problem, progress_path=progress_path)
    print("\nSolution:")
    print(solution)

    if isinstance(solver, HybridSolver):
        print(f"Optimized probabilities: {my_format(solution.extra["final_probs"])}")
        print(f"Optimized expectation: {solution.extra["cost_expectation"]}")
        print(f"Number of jobs: {solution.extra["num_jobs"]}")

        print("=== Best sample ===")
        print(f"Inner optimization successful: {solution.extra["opt_result"].success}")
        print(f"Penalty: {solution.extra["opt_result"].penalty}")


def run_instance(folder: Path, index: int, solver: PowerFlowSolver) -> tuple[int, str, list[float], float, list[dict[str, float | int]]]:
    """Solves one instance and returns a serialized result row.
    :param folder: Path to the dataset folder.
    :param index: Instance index to solve.
    :param solver: Solver used for the instance.
    :return: Tuple ``(index, generator_assignments, continuous_parameters, cost, history)``.
    """
    with (folder / f"{index}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file))
    progress_path = folder / ".progress" / f"{index}.pkl"
    solution = solver.solve(problem, progress_path=progress_path)
    continuous_params = np.concatenate((solution.active_powers, solution.reactive_powers, solution.voltages, solution.angles)).tolist()
    return index, solution.generator_statuses, continuous_params, solution.cost, solution.history


def load_progress_snapshot(progress_path: Path) -> tuple[str | None, list[float] | None, float, list[dict[str, float | int]] | None]:
    """Loads persisted worker progress for one instance.
    :param progress_path: Path to pickle snapshot written by a worker process.
    :return: Tuple ``(generator_assignments, continuous_parameters, cost, history)`` recovered from the snapshot.
    """
    if not progress_path.exists():
        return None, None, np.nan, None
    with progress_path.open("rb") as file:
        payload = pickle.load(file)
    return payload["incumbent"]["generator_assignments"], payload["incumbent"]["continuous_parameters"], payload["history"][-1]["objective"], payload["history"]


def run_parallel() -> None:
    """Runs selected instances in parallel and persists each completed result to CSV."""
    num_generators = 5
    data_folder = Path(f"data/{num_generators}")
    instance_indices = list(range(12))
    absent_only = True
    timeout_s = 300

    # solver = ClassicalSolver(silent=True)
    solver = get_hybrid_solver(num_generators)

    solver_name = type(solver).__name__.removesuffix("Solver").lower()
    solutions_path = data_folder / f".solutions_{solver_name}.csv"
    columns = ["generator_assignments", "continuous_parameters", "cost", "history", "error"]
    if solutions_path.exists():
        existing_df = pd.read_csv(solutions_path, dtype={"generator_assignments": "string"})
        existing_df = existing_df.reindex(columns=columns)
    else:
        existing_df = pd.DataFrame(columns=columns)

    if absent_only:
        filled_mask = existing_df["generator_assignments"].notna()
        filled_index_set = set(existing_df.index[filled_mask].tolist())
        instance_indices = [index for index in instance_indices if index not in filled_index_set]

    if len(instance_indices) == 0:
        print("No instance indices selected for run_parallel.")
        return

    progress_folder = data_folder / ".progress"
    shutil.rmtree(progress_folder, ignore_errors=True)
    progress_folder.mkdir()

    workers = min(max(1, (os.cpu_count() or 1) // 2), len(instance_indices))
    print(f"Using {workers} worker(s).")
    rows = existing_df.to_dict(orient="index")
    timeout_count = 0
    error_count = 0
    with ProcessPool(max_workers=workers) as pool:
        future_to_metadata = {pool.schedule(run_instance, args=(data_folder, index, solver), timeout=timeout_s): index for index in instance_indices}
        for future in tqdm(as_completed(future_to_metadata), total=len(future_to_metadata), smoothing=0.0):
            index = future_to_metadata[future]
            progress_path = progress_folder / f"{index}.pkl"
            try:
                _, generator_assignments, continuous_params, cost, history = future.result()
                error = None
            except Exception as ex:
                generator_assignments, continuous_params, cost, history = load_progress_snapshot(progress_path)
                if isinstance(ex, FutureTimeoutError):
                    timeout_count += 1
                    error = f"Timeout after {timeout_s}s"
                else:
                    error_count += 1
                    error = f"{type(ex).__name__}: {ex}"
            rows[index] = {
                "generator_assignments": generator_assignments,
                "continuous_parameters": continuous_params,
                "cost": cost,
                "history": history,
                "error": error,
            }
            max_index = max(rows.keys())
            pd.DataFrame.from_dict(rows, orient="index").sort_index().reindex(range(max_index + 1)).to_csv(solutions_path, index=False)
    print(f"Run complete: {timeout_count} timeout(s), {error_count} other failure(s).")


if __name__ == "__main__":
    t1 = time.perf_counter()

    # debug.save_instance_human_readable("data/5/5.pkl")

    # generate_dataset()
    # run_single()
    run_parallel()

    t2 = time.perf_counter()
    print(f"Elapsed time {t2 - t1} seconds")
