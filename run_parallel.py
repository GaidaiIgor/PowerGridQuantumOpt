"""Runs multiple stored power-flow instances in parallel."""

import argparse
import os
import pickle
import shutil
import sys
from collections.abc import Iterator
from concurrent.futures import TimeoutError as FutureTimeoutError, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from cattrs.preconf.json import make_converter
from pebble import ProcessPool
from tqdm import tqdm

from common.utils import SOLVER_IDS, get_solver
from src.HistoryEntry import HistoryEntry
from src.PowerFlowProblem import PowerFlowProblem
from src.PowerFlowSolver import PowerFlowSolver


def run_parallel() -> None:
    """Runs selected instances in parallel and persists each completed result to CSV."""
    args = parse_cli_args()
    instance_indices = list(range(120))
    voltage_deviation_mult = 10
    absent_only = True
    timeout_s = args.timeout * 3600
    max_classical_time_s = args.max_classical_time * 3600
    num_generators = read_num_generators(args.data_folder)
    solver = get_solver(num_generators, args.solver, args.num_layers, args.analyze_expectations, max_classical_time_s)

    solutions_path = Path(".solutions.csv")
    columns = ["instance", "generators", "cont_params", "cost", "violation", "job_ind", "total_jobs", "optimized_bitstrings", "total_inner", "max_inner",
               "ar_uniform_total", "ar_uniform_fun", "ar_opt_total", "ar_opt_fun", "error", "history"]
    if solutions_path.exists():
        existing_df = pd.read_csv(solutions_path, dtype={"instance": "Int64", "generators": "string"}).reindex(columns=columns)
    else:
        existing_df = pd.DataFrame(columns=columns)

    if absent_only:
        filled_mask = existing_df["generators"].notna()
        filled_index_set = set(existing_df.loc[filled_mask, "instance"].astype(int).tolist())
        instance_indices = [index for index in instance_indices if index not in filled_index_set]

    if not instance_indices:
        print("No instance indices selected for run_parallel.")
        return

    progress_folder = Path(".progress")
    shutil.rmtree(progress_folder, ignore_errors=True)
    progress_folder.mkdir()

    workers = min(os.cpu_count(), len(instance_indices))
    print(f"Using {workers} worker(s).")
    rows = existing_df.set_index("instance").to_dict(orient="index")
    converter = make_converter()
    timeout_count = 0
    error_count = 0
    with ProcessPool(max_workers=workers) as pool:
        future_to_metadata = \
            {pool.schedule(run_instance, args=(args.data_folder, index, solver, voltage_deviation_mult), timeout=timeout_s): index
             for index in instance_indices}
        for future in tqdm(as_completed(future_to_metadata), total=len(future_to_metadata), smoothing=0):
            index = future_to_metadata[future]
            progress_path = progress_folder / f"{index}.pkl"
            log_path = progress_folder / f"{index}.txt"
            try:
                history, extra = future.result()
                error = None
            except Exception as ex:
                if progress_path.exists():
                    history = pd.read_pickle(progress_path)
                    extra = history[-1].optimizer_stats
                else:
                    history = None
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
                row |= {"generators": last_result.generator_statuses,
                        "cont_params": last_result.params,
                        "cost": last_result.fun,
                        "violation": last_result.violation,
                        "job_ind": history[-1].job_ind,
                        "total_jobs": extra.get("total_jobs"),
                        "optimized_bitstrings": extra.get("optimized_bitstrings"),
                        "total_inner": extra.get("total_inner"),
                        "max_inner": extra.get("max_inner"),
                        "ar_uniform_total": extra.get("ar_uniform_total"),
                        "ar_uniform_fun": extra.get("ar_uniform_fun"),
                        "ar_opt_total": extra.get("ar_opt_total"),
                        "ar_opt_fun": extra.get("ar_opt_fun"),
                        "history": converter.dumps(history)}
            rows[index] = row
            output_df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("instance").reset_index().reindex(columns=columns).sort_values("instance")
            output_df["job_ind"] = output_df["job_ind"].astype("Int64")
            output_df["total_jobs"] = output_df["total_jobs"].astype("Int64")
            output_df["optimized_bitstrings"] = output_df["optimized_bitstrings"].astype("Int64")
            output_df.to_csv(solutions_path, index=False)

    print(f"Run complete: {timeout_count} timeout(s), {error_count} other failure(s).")
    print("\nAll instances:")
    print_stats(output_df, solver.violation_tolerance)
    print("\nFastest 100 instances:")
    fastest_100_df = output_df.loc[pd.to_numeric(output_df["total_inner"], errors="coerce").nsmallest(100).index]
    print_stats(fastest_100_df, solver.violation_tolerance)


def parse_cli_args() -> argparse.Namespace:
    """Parses command-line arguments for ``run_parallel``.
    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Runs multiple stored power-flow instances in parallel.")
    parser.add_argument("-df", "--data-folder", required=True, type=Path, help="Folder containing stored power-flow instance pickle files.")
    parser.add_argument("-s", "--solver", required=True, choices=SOLVER_IDS, help="Solver to run.")
    parser.add_argument("-nl", "--num-layers", default=1, type=int, help="Number of repeated ansatz blocks for the hybrid solver.")
    parser.add_argument("-ae", "--analyze-expectations", action="store_true", help="Enables post-optimization expectation analysis for hybrid runs.")
    parser.add_argument("-mct", "--max-classical-time", default=0, type=float, help="Maximum total classical time in hours for hybrid runs.")
    parser.add_argument("-t", "--timeout", required=True, type=float, help="Per-instance timeout in hours.")
    return parser.parse_args()


def read_num_generators(data_folder: Path) -> int:
    """Reads generator count from the first stored instance in a dataset folder.
    :param data_folder: Path to the dataset folder.
    :return: Number of generators stored in the dataset.
    """
    first_instance_path = next(data_folder.glob("*.pkl"), None)
    assert first_instance_path is not None, f"No instance files found in {data_folder}."
    with first_instance_path.open("rb") as file:
        problem = PowerFlowProblem(pickle.load(file), 0)
    return len(problem.generators)


def run_instance(data_folder: Path, index: int, solver: PowerFlowSolver, voltage_deviation_mult: float) -> tuple[list[HistoryEntry], dict[str, Any]]:
    """Solves one instance and returns its solver history and extras.
    :param data_folder: Path to the dataset folder.
    :param index: Instance index to solve.
    :param solver: Solver used for the instance.
    :param voltage_deviation_mult: Multiplier applied to squared voltage deviation from ``1`` in the objective.
    :return: Solver history whose last entry is the final incumbent together with solver-specific extras.
    """
    progress_folder = Path(".progress")
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


def print_stats(df: pd.DataFrame, violation_tolerance: float):
    """Prints summary statistics for a result dataframe.
    :param df: Result dataframe whose summary statistics are printed.
    :param violation_tolerance: Violation threshold above which an instance is considered infeasible.
    """
    total_jobs_values = pd.to_numeric(df["total_jobs"], errors="coerce")
    optimized_bitstring_values = pd.to_numeric(df["optimized_bitstrings"], errors="coerce")
    total_inner_values = pd.to_numeric(df["total_inner"], errors="coerce") / 3600
    max_inner_values = pd.to_numeric(df["max_inner"], errors="coerce")
    infeasible_count = (pd.to_numeric(df["violation"], errors="coerce") > violation_tolerance).sum()
    print(f"Total jobs: avg={total_jobs_values.mean()}, max={total_jobs_values.max()}")
    print(f"Optimized bitstrings: avg={optimized_bitstring_values.mean()}, max={optimized_bitstring_values.max()}")
    print(f"Total inner optimization time (h): avg={total_inner_values.mean()}, max={total_inner_values.max()}")
    print(f"Max inner optimization time (s): avg={max_inner_values.mean()}, max={max_inner_values.max()}")
    print(f"Infeasible instances: {infeasible_count}")


if __name__ == "__main__":
    run_parallel()
