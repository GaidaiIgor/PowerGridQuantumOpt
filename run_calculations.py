import os
import pickle
import shutil
import sys
import time
from contextlib import contextmanager
from functools import partial
from concurrent.futures import TimeoutError as FutureTimeoutError, as_completed
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from networkx import Graph
from pebble import ProcessPool
from cattrs.preconf.json import make_converter
from qiskit.primitives import StatevectorSampler
from tqdm import tqdm

import debug
from src import PowerFlowProblemGenerator, LognormalSpec
from src.CircuitLayer import AllToAllEntangler, ZXMixer
from src.ContinuousPowerOptimizer import ContinuousPowerOptimizer, CasadiOptimizer
from src.HistoryEntry import HistoryEntry
from src.Generator import Generator
from src.PowerFlowProblem import PowerFlowProblem
from src.PowerFlowSolver import ClassicalSolver, HybridSolver, PowerFlowSolver
from src.Sampler import ExactSampler, MySamplerV2
from src.VariationalQuantumProgram import VariationalQuantumProgram
from src.utils import my_format


def generate_dataset():
    num_generators = 5
    problem_generator = PowerFlowProblemGenerator()
    problem_generator.generate_instances(num_generators, 100, voltage_range=(0, 100), output_folder=f"data/{num_generators}/capacity_100",
                                         strictness_factor=1.2, capacity_spec=LognormalSpec(100, 2))


def run_single():
    # problem = get_power_flow_ac_problem()
    index = 0
    voltage_deviation_mult = 10
    data_path = Path(f"data/5/capacity_100")
    with (data_path / f"{index}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file), voltage_deviation_mult)

    # debug.set_all_edge_capacities(problem, 100)
    # debug.set_all_node_voltage_ranges(problem, (1, 100))
    # debug.set_all_generator_p_min(problem, 0)

    # solver = ClassicalSolver()
    solver = get_hybrid_solver(len(problem.generators))

    progress_folder = data_path / f".progress_{solver.name}"
    progress_folder.mkdir(exist_ok=True)
    progress_path = progress_folder / f"{index}.pkl"
    solution = solver.solve(problem, progress_path=progress_path)
    print("\nSolution:")
    debug.print_power_flow_solution(problem, solution)

    if isinstance(solver, HybridSolver) and solver.exact_final_expectation:
        print(f"Optimized probabilities: {my_format(solution.extra["final_probs"])}")
        print(f"Optimized expectation: {solution.extra["cost_expectation"]}")
        print(f"Number of jobs: {solution.history[-1].num_jobs}")


def get_hybrid_solver(num_generators: int) -> HybridSolver:
    max_inner_time_s = 30
    penalty_mult = 10
    seed = 0
    vqp = get_variational_quantum_program(num_generators)
    inner_optimizer_factory = partial(CasadiOptimizer, penalty_mult=penalty_mult, max_time_s=max_inner_time_s, silent=True)
    return HybridSolver(vqp, inner_optimizer_factory, seed)


def get_variational_quantum_program(num_qubits: int) -> VariationalQuantumProgram:
    entangler = AllToAllEntangler(num_qubits)
    mixer = ZXMixer(num_qubits)
    num_layers = 1

    # sampler = ExactSampler()
    sampler = MySamplerV2(StatevectorSampler(default_shots=1000))
    # sampler = IonQSampler("simulator", 1000, None)
    # sampler = IonQSampler("qpu.forte-enterprise-1", 1000, None)

    return VariationalQuantumProgram(num_layers, [entangler, mixer], sampler)


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


def run_parallel() -> None:
    """Runs selected instances in parallel and persists each completed result to CSV."""
    num_generators = 5
    data_folder = Path(f"data/{num_generators}/capacity_100")
    instance_indices = list(range(100))
    voltage_deviation_mult = 10
    absent_only = True
    timeout_s = 1800

    # solver = ClassicalSolver(silent=True)
    solver = get_hybrid_solver(num_generators)

    solutions_path = data_folder / f".solutions_{solver.name}.csv"
    columns = ["instance", "generator_assignments", "continuous_parameters", "cost", "penalty", "num_jobs", "history", "error"]
    if solutions_path.exists():
        existing_df = pd.read_csv(solutions_path, dtype={"instance": "Int64", "generator_assignments": "string"})
        existing_df = existing_df.reindex(columns=columns)
    else:
        existing_df = pd.DataFrame(columns=columns)

    if absent_only:
        filled_mask = existing_df["generator_assignments"].notna()
        filled_index_set = set(existing_df.loc[filled_mask, "instance"].astype(int).tolist())
        instance_indices = [index for index in instance_indices if index not in filled_index_set]

    if len(instance_indices) == 0:
        print("No instance indices selected for run_parallel.")
        return

    progress_folder = data_folder / f".progress_{solver.name}"
    shutil.rmtree(progress_folder, ignore_errors=True)
    progress_folder.mkdir()

    workers = min(max(1, (os.cpu_count() or 1) // 2), len(instance_indices))
    print(f"Using {workers} worker(s).")
    rows = existing_df.set_index("instance").to_dict(orient="index")
    converter = make_converter()
    timeout_count = 0
    error_count = 0
    with ProcessPool(max_workers=workers) as pool:
        future_to_metadata = \
            {pool.schedule(run_instance, args=(data_folder, index, solver, voltage_deviation_mult), timeout=timeout_s): index for index in instance_indices}
        for future in tqdm(as_completed(future_to_metadata), total=len(future_to_metadata), smoothing=0.0):
            index = future_to_metadata[future]
            progress_path = progress_folder / f"{index}.pkl"
            log_path = progress_folder / f"{index}.txt"
            try:
                _, generator_assignments, continuous_params, cost, penalty, num_jobs, history = future.result()
                error = None
            except Exception as ex:
                history = pd.read_pickle(progress_path) if progress_path.exists() else None
                if history is None:
                    generator_assignments, continuous_params, cost, penalty, num_jobs = None, None, None, None, None
                else:
                    last_result = history[-1].evaluation_result
                    generator_assignments = last_result.generator_statuses
                    continuous_params = last_result.params
                    cost = last_result.fun
                    penalty = last_result.penalty
                    num_jobs = history[-1].num_jobs
                if isinstance(ex, FutureTimeoutError):
                    timeout_count += 1
                    error = f"Timeout after {timeout_s}s"
                else:
                    error_count += 1
                    error = f"{type(ex).__name__}: {ex}"
            if log_path.stat().st_size == 0:
                log_path.unlink()
            rows[index] = {"generator_assignments": generator_assignments, "continuous_parameters": continuous_params, "cost": cost, "penalty": penalty,
                           "num_jobs": num_jobs, "history": converter.dumps(history) if history is not None else None, "error": error}
            output_df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("instance").reset_index().sort_values("instance")
            output_df["num_jobs"] = output_df["num_jobs"].astype("Int64")
            output_df.to_csv(solutions_path, index=False)
    print(f"Run complete: {timeout_count} timeout(s), {error_count} other failure(s).")

def run_instance(data_folder: Path, index: int, solver: PowerFlowSolver, voltage_deviation_mult: float) \
    -> tuple[int, str, list[float], float, float, int | None, list[HistoryEntry]]:
    """Solves one instance and returns a serialized result row.
    :param data_folder: Path to the dataset folder.
    :param index: Instance index to solve.
    :param solver: Solver used for the instance.
    :param voltage_deviation_mult: Multiplier applied to squared voltage deviation from ``1`` in the objective.
    :return: Tuple ``(index, generator_assignments, continuous_parameters, cost, penalty, num_jobs, history)``.
    """
    progress_folder = data_folder / f".progress_{solver.name}"
    log_path = progress_folder / f"{index}.txt"
    with redirect_worker_output(log_path):
        with (data_folder / f"{index}.pkl").open("rb") as file:
            problem = PowerFlowProblem(pickle.load(file), voltage_deviation_mult)
        progress_path = progress_folder / f"{index}.pkl"
        solution = solver.solve(problem, progress_path=progress_path)
        continuous_params = np.concatenate((solution.active_powers, solution.reactive_powers, solution.voltages, solution.angles)).tolist()
        penalty = solution.history[-1].evaluation_result.penalty if isinstance(solver, HybridSolver) else 0
        num_jobs = solution.history[-1].num_jobs if isinstance(solver, HybridSolver) and len(solution.history) > 0 else None
        return index, solution.generator_statuses, continuous_params, solution.cost, penalty, num_jobs, solution.history


@contextmanager
def redirect_worker_output(log_path: Path) -> Iterator[None]:
    """Redirects process stdout and stderr to a worker log file.
    :param log_path: Path to the worker log file.
    :return: Yields while worker output is redirected to the log file.
    """
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        with log_path.open("w") as log_file:
            log_fd = log_file.fileno()
            os.dup2(log_fd, 1)
            os.dup2(log_fd, 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


if __name__ == "__main__":
    t1 = time.perf_counter()

    # debug.save_instance_human_readable("data/5/5.pkl")

    # generate_dataset()
    run_single()
    # run_parallel()

    # debug.print_solution_from_csv("data/5/capacity_100/.solutions_casadi.csv", 2)

    t2 = time.perf_counter()
    print(f"Elapsed time {t2 - t1} seconds")
