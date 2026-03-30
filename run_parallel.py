"""Runs multiple stored power-flow instances in parallel."""

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

from run_single import get_solver
from src.HistoryEntry import HistoryEntry
from src.PowerFlowProblem import PowerFlowProblem
from src.PowerFlowSolver import PowerFlowSolver


def run_parallel() -> None:
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
        existing_df = pd.read_csv(solutions_path, dtype={"instance": "Int64", "generator_assignments": "string"}).reindex(columns=columns)
    else:
        existing_df = pd.DataFrame(columns=columns)

    if absent_only:
        filled_mask = existing_df["generator_assignments"].notna()
        filled_index_set = set(existing_df.loc[filled_mask, "instance"].astype(int).tolist())
        instance_indices = [index for index in instance_indices if index not in filled_index_set]

    if not instance_indices:
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
            {pool.schedule(run_instance, args=(data_folder, index, solver, voltage_deviation_mult), timeout=timeout_s): index
             for index in instance_indices}
        for future in tqdm(as_completed(future_to_metadata), total=len(future_to_metadata), smoothing=0):
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
    run_parallel()
