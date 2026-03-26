import os
import pickle
import shutil
import sys
import time
from contextlib import contextmanager
from functools import partial
from concurrent.futures import TimeoutError as FutureTimeoutError, as_completed
from pathlib import Path
from typing import Any, Iterator

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
from src.ContinuousPowerOptimizer import CasadiOptimizer
from src.HistoryEntry import HistoryEntry
from src.Generator import Generator
from src.PowerFlowProblem import PowerFlowProblem
from src.PowerFlowSolver import SCIPSolver, HybridSolver, PowerFlowSolver, SmacSolver, UniformSolver
from src.Sampler import MySamplerV2
from src.VariationalQuantumProgram import VariationalQuantumProgram
from src.utils import my_format


def generate_dataset():
    num_generators = 5
    problem_generator = PowerFlowProblemGenerator()
    problem_generator.generate_instances(num_generators, 100, voltage_range=(0, 100), output_folder=f"data/{num_generators}", strictness_factor=1.2)


def run_single():
    # problem = get_power_flow_ac_problem()
    index = 49
    voltage_deviation_mult = 10
    exact_final_expectation = False
    data_path = Path(f"data/5/capacity_100")
    with (data_path / f"{index}.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file), voltage_deviation_mult)

    # debug.set_all_edge_capacities(problem, 100)
    # debug.set_all_node_voltage_ranges(problem, (1, 100))
    # debug.set_all_generator_p_min(problem, 0)

    # solver = ClassicalSolver()
    solver = get_solver(len(problem.generators))

    inner_solver = solver.inner_optimizer_factory(problem)
    inner_solver.optimize("11110")

    progress_folder = data_path / f".progress_{solver.name}"
    progress_folder.mkdir(exist_ok=True)
    progress_path = progress_folder / f"{index}.pkl"
    if isinstance(solver, HybridSolver):
        history, extra = solver.solve(problem, progress_path, exact_final_expectation)
    else:
        history, extra = solver.solve(problem, progress_path)

    print("\nSolution:")
    debug.print_evaluation_result(problem, history[-1].result)
    print(f"Job index: {history[-1].job_ind}")
    if "total_jobs" in extra:
        print(f"Total jobs: {extra["total_jobs"]}")
    if exact_final_expectation:
        print(f"Optimized probabilities: {my_format(extra["final_probs"])}")
        print(f"Optimized expectation: {extra["cost_expectation"]}")


def get_solver(num_generators: int) -> PowerFlowSolver:
    max_inner_time_s = 30
    penalty_mult = 10
    feasibility_tolerance = 1e-10
    silent = True
    seed = 0

    vqp = get_variational_quantum_program(num_generators)
    inner_optimizer_factory = partial(CasadiOptimizer, penalty_mult=penalty_mult, max_time_s=max_inner_time_s, silent=True)

    solver = SCIPSolver(feasibility_tolerance, silent, seed)
    # solver = SmacSolver(inner_optimizer_factory, feasibility_tolerance, silent, seed)
    # solver = UniformSolver(inner_optimizer_factory, feasibility_tolerance, seed)
    # solver = HybridSolver(vqp, inner_optimizer_factory, feasibility_tolerance, seed)
    return solver


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


def run_parallel():
    """Runs selected instances in parallel and persists each completed result to CSV."""
    num_generators = 5
    data_folder = Path(f"data/{num_generators}")
    instance_indices = list(range(100))
    voltage_deviation_mult = 10
    absent_only = True
    timeout_s = 1800
    solver = get_solver(num_generators)

    solutions_path = data_folder / f".solutions_{solver.name}.csv"
    columns = ["instance", "generator_assignments", "continuous_parameters", "cost", "penalty", "job_ind", "total_jobs", "avg_inner", "history", "error"]
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
                history, extra = future.result()
                error = None
            except Exception as ex:
                history = pd.read_pickle(progress_path) if progress_path.exists() else None
                extra = {}
                if isinstance(ex, FutureTimeoutError):
                    timeout_count += 1
                    error = f"Timeout after {timeout_s}s"
                else:
                    error_count += 1
                    error = f"{type(ex).__name__}: {ex}"

            if log_path.stat().st_size == 0:
                log_path.unlink()
            row = {"history": None, "error": error}
            if history is not None:
                last_result = history[-1].result
                row |= {"generator_assignments": last_result.generator_statuses,
                        "continuous_parameters": last_result.params,
                        "cost": last_result.fun,
                        "penalty": last_result.penalty,
                        "job_ind": history[-1].job_ind,
                        "total_jobs": extra.get("total_jobs"),
                        "avg_inner": extra.get("avg_inner"),
                        "history": converter.dumps(history)}
            rows[index] = row
            output_df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("instance").reset_index().reindex(columns=columns).sort_values("instance")
            output_df["job_ind"] = output_df["job_ind"].astype("Int64")
            output_df["total_jobs"] = output_df["total_jobs"].astype("Int64")
            output_df.to_csv(solutions_path, index=False)

    print(f"Run complete: {timeout_count} timeout(s), {error_count} other failure(s).")
    avg_inner_values = pd.to_numeric(output_df["avg_inner"], errors="coerce")
    total_jobs_values = pd.to_numeric(output_df["total_jobs"], errors="coerce")
    infeasible_count = (pd.to_numeric(output_df["penalty"], errors="coerce") > solver.feasibility_tolerance).sum()
    print(f"Inner optimization time: avg={avg_inner_values.mean()}, min={avg_inner_values.min()}, max={avg_inner_values.max()}")
    print(f"Total jobs: avg={total_jobs_values.mean()}, min={total_jobs_values.min()}, max={total_jobs_values.max()}")
    print(f"Unfeasible instances: {infeasible_count}")


def run_instance(data_folder: Path, index: int, solver: PowerFlowSolver, voltage_deviation_mult: float) -> tuple[list[HistoryEntry], dict[str, Any]]:
    """Solves one instance and returns its solver history and extras.
    :param data_folder: Path to the dataset folder.
    :param index: Instance index to solve.
    :param solver: Solver used for the instance.
    :param voltage_deviation_mult: Multiplier applied to squared voltage deviation from ``1`` in the objective.
    :return: Solver history whose last entry is the final incumbent together with solver-specific extras.
    """
    progress_folder = data_folder / f".progress_{solver.name}"
    log_path = progress_folder / f"{index}.txt"
    with redirect_worker_output(log_path):
        with (data_folder / f"{index}.pkl").open("rb") as file:
            problem = PowerFlowProblem(pickle.load(file), voltage_deviation_mult)
        progress_path = progress_folder / f"{index}.pkl"
        return solver.solve(problem, progress_path=progress_path)


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
    # debug.print_solution_from_csv("data/5/capacity_100/.solutions_casadi.csv", 49)

    # generate_dataset()
    # run_single()
    run_parallel()

    t2 = time.perf_counter()
    print(f"Elapsed time {t2 - t1} seconds")
