import time
from concurrent.futures import TimeoutError as FutureTimeoutError, as_completed
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from networkx import Graph
from pebble import ProcessPool
from tqdm import tqdm

from src import PowerFlowProblemGenerator, LognormalSpec
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
    # problem_generator.generate_instances(5, num_instances=100, output_folder="data/5")
    problem_generator.generate_instances(5, num_instances=100, output_folder="data/5", generator_ref_p_spec=LognormalSpec(100, 2),
                                         generator_reactive_range=(0.8, 0.9), capacity_spec=LognormalSpec(100, 2), voltage_range=(0, 100))


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
    with Path("data/5/1.pkl").open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file))

    solver = ClassicalSolver()
    # solver = get_hybrid_solver(len(problem.generators))

    solution = solver.solve(problem)
    print("\nSolution:")
    print(solution)

    if isinstance(solver, HybridSolver):
        print(f"Optimized probabilities: {my_format(solution.extra["final_probs"])}")
        print(f"Optimized expectation: {solution.extra["cost_expectation"]}")
        print(f"Number of jobs: {solution.extra["num_jobs"]}")
        print(f"Total classical time: {solution.classical_time}")

        print("=== Best sample ===")
        print(f"Inner optimization successful: {solution.extra["opt_result"].success}")
        print(f"Penalty: {solution.extra["opt_result"].penalty}")


def run_instance(folder: str, index: int, solver: PowerFlowSolver) -> tuple[int, str | None, list[float] | None, float, float, str | None]:
    """Solve one indexed instance and return generator assignments, continuous parameters, objective, timing, and optional error."""
    try:
        with (Path(folder) / f"{index}.pkl").open("rb") as file:
            problem = PowerFlowProblem(pickle.load(file))
        solution = solver.solve(problem)
        continuous_params = np.concatenate((solution.active_powers, solution.reactive_powers, solution.voltages, solution.angles)).tolist()
        return index, solution.generator_statuses, continuous_params, solution.cost, solution.classical_time, None
    except Exception as ex:
        return index, None, None, np.nan, np.nan, f"{type(ex).__name__}: {ex}"


def run_parallel() -> None:
    """Runs selected instances in parallel and persists each completed result to CSV immediately."""
    folder = Path("data/5")
    output_path = folder / ".solutions.csv"
    instance_indices = list(range(100))
    absent_only = True
    timeout_s = 300

    solver = ClassicalSolver(silent=True)
    # solver = get_hybrid_solver(5)

    columns = ["generator_assignments", "continuous_parameters", "cost", "classical_time", "error"]
    if output_path.exists():
        existing_df = pd.read_csv(output_path, index_col="index")
        existing_df = existing_df.reindex(columns=columns)
    else:
        existing_df = pd.DataFrame(columns=columns)

    if absent_only:
        existing_index_set = set(existing_df.index.tolist())
        if existing_df.empty:
            failed_index_set = set()
        else:
            error_mask = existing_df["error"].notna()
            failed_index_set = set(existing_df.index[error_mask].tolist())
        instance_indices = [index for index in instance_indices if index not in existing_index_set or index in failed_index_set]

    if len(instance_indices) == 0:
        print("No instance indices selected for run_parallel.")
        return

    workers = min(max(1, (os.cpu_count() or 1) // 2), len(instance_indices))
    rows = existing_df.to_dict(orient="index")
    with ProcessPool(max_workers=workers) as pool:
        future_to_index = {pool.schedule(run_instance, args=(str(folder), index, solver), timeout=timeout_s): index for index in instance_indices}
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), smoothing=0.0):
            index = future_to_index[future]
            try:
                _, generator_assignments, continuous_params, cost, classical_time, error = future.result()
            except FutureTimeoutError:
                generator_assignments, continuous_params, cost, classical_time, error = None, None, np.nan, np.nan, f"Timeout after {timeout_s}s"
            except Exception as ex:
                generator_assignments, continuous_params, cost, classical_time, error = None, None, np.nan, np.nan, f"{type(ex).__name__}: {ex}"
            rows[index] = {
                "generator_assignments": generator_assignments,
                "continuous_parameters": continuous_params,
                "cost": cost,
                "classical_time": classical_time,
                "error": error,
            }
            pd.DataFrame.from_dict(rows, orient="index").sort_index().to_csv(output_path, index_label="index")


if __name__ == "__main__":
    t1 = time.perf_counter()

    # generate_dataset()
    run_single()
    # run_parallel()

    t2 = time.perf_counter()
    print(f"Elapsed time {t2 - t1} seconds")
