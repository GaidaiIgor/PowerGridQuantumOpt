# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Modified for PowerGridQuantumOpt local optimizer experiments.

from __future__ import annotations

from collections.abc import Callable
from csv import DictReader, DictWriter
from dataclasses import dataclass
from math import erf, pi, sqrt
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy import ndarray
from qiskit_algorithms.optimizers.optimizer import POINT, Optimizer, OptimizerResult, OptimizerSupportLevel


StatsObjective = Callable[[POINT], tuple[float, float]]
GradientFunction = Callable[[POINT], POINT]
StatsGradientFunction = Callable[[POINT], tuple[POINT, float, float]]


@dataclass(slots=True)
class EvalStats:
    mean: float
    sem: float


@dataclass(slots=True)
class GradientStats:
    gradient: ndarray
    gradient_sem_scale: float
    z_score: float
    plus: EvalStats
    minus: EvalStats
    num_objective_evals: int


class SEMADAM(Optimizer):
    """SEM-aware Adam/AMSGRAD optimizer with built-in SPSA gradient estimates.

    The objective function must return `(mean, sem)`, where `mean` is the sampled
    expectation estimate and `sem` is the standard error of that sampled mean.

    If `jac` is not supplied, the optimizer uses a two-sided SPSA gradient:
        [f(x + c delta) - f(x - c delta)] / (2 c) * delta

    The SEM values are used to estimate the signal-to-noise ratio of the finite
    difference:
        z = (mean_plus - mean_minus) / sqrt(sem_plus^2 + sem_minus^2)

    The Adam moment update is then confidence-weighted. Low-SNR SPSA gradients
    influence the moving averages less, can optionally be skipped, and can
    optionally be resampled along the same SPSA direction.

    This is still Adam-SPSA, not magic. It merely stops treating a meaningless
    finite difference and a high-confidence finite difference as equally sacred.
    """

    _OPTIONS: ClassVar[tuple[str, ...]] = (
        "maxiter",
        "tol",
        "lr",
        "beta_1",
        "beta_2",
        "noise_factor",
        "eps",
        "amsgrad",
        "snapshot_dir",
        "perturbation",
        "wrap_angles",
        "snr_weighting",
        "snr_z0",
        "min_snr_to_update",
        "max_resamplings",
        "target_snr",
        "best_by",
        "best_confidence_z",
        "seed",
    )

    _maxiter: int
    _snapshot_dir: str | None
    _tol: float
    _lr: float
    _beta_1: float
    _beta_2: float
    _noise_factor: float
    _eps: float
    _amsgrad: bool
    _perturbation: float
    _wrap_angles: bool
    _snr_weighting: bool
    _snr_z0: float
    _min_snr_to_update: float
    _max_resamplings: int
    _target_snr: float
    _best_by: str
    _best_confidence_z: float
    _seed: int | None
    _rng: np.random.Generator
    _t: int
    _m: ndarray
    _v: ndarray
    _v_eff: ndarray

    def __init__(
        self,
        maxiter: int = 10000,
        tol: float = 1e-6,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        noise_factor: float = 1e-8,
        eps: float = 1e-10,
        amsgrad: bool = False,
        snapshot_dir: str | None = None,
        perturbation: float = 0.1,
        wrap_angles: bool = True,
        snr_weighting: bool = True,
        snr_z0: float = 1.0,
        min_snr_to_update: float = 0.0,
        max_resamplings: int = 1,
        target_snr: float = 1.0,
        best_by: str = "ucb",
        best_confidence_z: float = 1.0,
        seed: int | None = None,
    ):
        """Initializes the SEM-aware Adam optimizer.

        Args:
            maxiter: Maximum number of Adam iterations.
            tol: Step-size tolerance used for termination.
            lr: Learning rate.
            beta_1: Exponential decay rate for first moment.
            beta_2: Exponential decay rate for second moment.
            noise_factor: Stabilizing denominator offset.
            eps: Numerical finite-difference epsilon used only if a fallback
                numerical gradient is needed.
            amsgrad: Whether to use AMSGRAD.
            snapshot_dir: Optional directory used to write moment snapshots.
            perturbation: SPSA perturbation size `c`.
            wrap_angles: If true, wrap parameters to [-pi, pi) after each step.
            snr_weighting: If true, confidence-weight Adam moment updates.
            snr_z0: SNR scale in weight z^2 / (z^2 + z0^2).
            min_snr_to_update: If abs(z) is below this value, skip the Adam
                update for this iteration. Keep 0 unless you have evidence.
            max_resamplings: Number of repeated SPSA pair evaluations along the
                same direction before computing the gradient. 1 means no extra
                resampling beyond the initial plus/minus pair.
            target_snr: If max_resamplings > 1, stop resampling once abs(z)
                reaches this target.
            best_by: How to choose the returned incumbent. One of:
                "mean" = lowest sampled mean,
                "ucb" = lowest mean + z * sem, conservative,
                "lcb" = lowest mean - z * sem, optimistic.
            best_confidence_z: z used by "ucb" or "lcb" incumbent scoring.
            seed: Random seed for SPSA perturbations.
        """
        super().__init__()
        self._options.update(
            {
                "maxiter": maxiter,
                "tol": tol,
                "lr": lr,
                "beta_1": beta_1,
                "beta_2": beta_2,
                "noise_factor": noise_factor,
                "eps": eps,
                "amsgrad": amsgrad,
                "snapshot_dir": snapshot_dir,
                "perturbation": perturbation,
                "wrap_angles": wrap_angles,
                "snr_weighting": snr_weighting,
                "snr_z0": snr_z0,
                "min_snr_to_update": min_snr_to_update,
                "max_resamplings": max_resamplings,
                "target_snr": target_snr,
                "best_by": best_by,
                "best_confidence_z": best_confidence_z,
                "seed": seed,
            }
        )

        self._maxiter = maxiter
        self._snapshot_dir = snapshot_dir
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad
        self._perturbation = perturbation
        self._wrap_angles = wrap_angles
        self._snr_weighting = snr_weighting
        self._snr_z0 = snr_z0
        self._min_snr_to_update = min_snr_to_update
        self._max_resamplings = max_resamplings
        self._target_snr = target_snr
        self._best_by = best_by
        self._best_confidence_z = best_confidence_z
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._validate_options()

        self._t = 0
        self._m = np.zeros(1)
        self._v = np.zeros(1)
        if self._amsgrad:
            self._v_eff = np.zeros(1)

        if self._snapshot_dir:
            with Path(self._snapshot_dir, "adam_params.csv").open("w", newline="") as csv_file:
                fieldnames = ["v", "v_eff", "m", "t"] if self._amsgrad else ["v", "m", "t"]
                writer = DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

    @property
    def settings(self) -> dict[str, Any]:
        """Returns constructor settings that recreate this optimizer."""
        return {
            "maxiter": self._maxiter,
            "tol": self._tol,
            "lr": self._lr,
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "noise_factor": self._noise_factor,
            "eps": self._eps,
            "amsgrad": self._amsgrad,
            "snapshot_dir": self._snapshot_dir,
            "perturbation": self._perturbation,
            "wrap_angles": self._wrap_angles,
            "snr_weighting": self._snr_weighting,
            "snr_z0": self._snr_z0,
            "min_snr_to_update": self._min_snr_to_update,
            "max_resamplings": self._max_resamplings,
            "target_snr": self._target_snr,
            "best_by": self._best_by,
            "best_confidence_z": self._best_confidence_z,
            "seed": self._seed,
        }

    def get_support_level(self) -> dict[str, OptimizerSupportLevel]:
        """Returns Qiskit support levels for optimizer inputs."""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.supported,
        }

    def minimize(
        self,
        fun: StatsObjective,
        x0: POINT,
        jac: GradientFunction | StatsGradientFunction | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> OptimizerResult:
        """Minimizes an objective returning `(mean, sem)`.

        Args:
            fun: Objective function returning `(mean, sem)`.
            x0: Initial parameter vector.
            jac: Optional gradient function. If supplied, it may return either:
                gradient
                or
                (gradient, gradient_sem_scale, z_score)

                If omitted, this optimizer uses its built-in SEM-aware SPSA
                gradient estimator.
            bounds: Ignored, accepted for Qiskit API compatibility.

        Returns:
            OptimizerResult. Additional attributes are attached:
                sem, nfev, njev, best_x, best_fun, best_sem, last_fun, last_sem,
                skipped_updates, mean_abs_z, mean_weight
        """
        del bounds

        params = self._as_angle_array(x0)
        initial_stats = self._evaluate(fun, params)
        best_x = params.copy()
        best_stats = initial_stats
        last_stats = initial_stats

        derivative, gradient_sem_scale, z_score, gradient_evals = self._compute_gradient(fun, params, jac)
        self._t = 0
        self._m = np.zeros(np.shape(derivative))
        self._v = np.zeros(np.shape(derivative))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(derivative))

        nfev = 1 + gradient_evals
        njev = 1
        skipped_updates = 0
        z_values: list[float] = []
        weights: list[float] = []

        params_new = params.copy()
        while self._t < self._maxiter:
            if self._t > 0:
                derivative, gradient_sem_scale, z_score, gradient_evals = self._compute_gradient(fun, params, jac)
                nfev += gradient_evals
                njev += 1

            self._t += 1
            z_abs = abs(float(z_score)) if np.isfinite(z_score) else 0.0
            z_values.append(z_abs)

            weight = self._gradient_weight(z_abs)
            weights.append(weight)

            if z_abs < self._min_snr_to_update:
                skipped_updates += 1
                params_new = params.copy()
            else:
                weighted_derivative = weight * derivative
                self._m = self._beta_1 * self._m + (1 - self._beta_1) * weighted_derivative
                self._v = self._beta_2 * self._v + (1 - self._beta_2) * weight * derivative * derivative

                lr_eff = self._lr * np.sqrt(1 - self._beta_2**self._t) / (1 - self._beta_1**self._t)
                if self._amsgrad:
                    self._v_eff = np.maximum(self._v_eff, self._v)
                    params_new = params - lr_eff * self._m.flatten() / (np.sqrt(self._v_eff.flatten()) + self._noise_factor)
                else:
                    params_new = params - lr_eff * self._m.flatten() / (np.sqrt(self._v.flatten()) + self._noise_factor)

                params_new = self._wrap(params_new)

            new_stats = self._evaluate(fun, params_new)
            nfev += 1
            last_stats = new_stats

            if self._is_better(new_stats, best_stats):
                best_x = params_new.copy()
                best_stats = new_stats

            if self._snapshot_dir:
                self.save_params(self._snapshot_dir)

            if np.linalg.norm(params - params_new) < self._tol:
                params = params_new
                break

            params = params_new

        result = OptimizerResult()
        result.x = best_x
        result.fun = best_stats.mean
        result.sem = best_stats.sem
        result.nfev = nfev
        result.njev = njev
        result.nit = self._t
        result.best_x = best_x
        result.best_fun = best_stats.mean
        result.best_sem = best_stats.sem
        result.last_x = params
        result.last_fun = last_stats.mean
        result.last_sem = last_stats.sem
        result.skipped_updates = skipped_updates
        result.mean_abs_z = float(np.mean(z_values)) if z_values else 0.0
        result.mean_weight = float(np.mean(weights)) if weights else 0.0
        return result

    def estimate_gradient_spsa(self, fun: StatsObjective, point: POINT) -> GradientStats:
        """Estimates a SEM-aware SPSA gradient at `point`.

        This is public so you can call it from experiments if you want to inspect
        z-scores, SEMs, or finite-difference behavior directly.
        """
        point_arr = self._as_angle_array(point)
        delta = self._rng.choice((-1.0, 1.0), size=len(point_arr))
        x_plus = self._wrap(point_arr + self._perturbation * delta)
        x_minus = self._wrap(point_arr - self._perturbation * delta)

        plus_stats: list[EvalStats] = []
        minus_stats: list[EvalStats] = []

        for _ in range(self._max_resamplings):
            plus_stats.append(self._evaluate(fun, x_plus))
            minus_stats.append(self._evaluate(fun, x_minus))
            plus = self._combine_stats(plus_stats)
            minus = self._combine_stats(minus_stats)
            z_score = self._finite_difference_z(plus, minus)
            if abs(z_score) >= self._target_snr:
                break

        plus = self._combine_stats(plus_stats)
        minus = self._combine_stats(minus_stats)
        diff = plus.mean - minus.mean
        diff_sem = sqrt(plus.sem * plus.sem + minus.sem * minus.sem)
        gradient_sem_scale = diff_sem / (2.0 * self._perturbation)
        gradient = diff / (2.0 * self._perturbation) * delta
        z_score = diff / diff_sem if diff_sem > 0 else np.inf * np.sign(diff)

        return GradientStats(
            gradient=gradient,
            gradient_sem_scale=float(gradient_sem_scale),
            z_score=float(z_score),
            plus=plus,
            minus=minus,
            num_objective_evals=2 * len(plus_stats),
        )

    def save_params(self, snapshot_dir: str):
        """Saves current iteration parameters to adam_params.csv."""
        with Path(snapshot_dir, "adam_params.csv").open("a", newline="") as csv_file:
            if self._amsgrad:
                writer = DictWriter(csv_file, fieldnames=["v", "v_eff", "m", "t"])
                writer.writerow({"v": self._v, "v_eff": self._v_eff, "m": self._m, "t": self._t})
            else:
                writer = DictWriter(csv_file, fieldnames=["v", "m", "t"])
                writer.writerow({"v": self._v, "m": self._m, "t": self._t})

    def load_params(self, load_dir: str):
        """Loads iteration parameters from adam_params.csv."""
        with Path(load_dir, "adam_params.csv").open(newline="") as csv_file:
            fieldnames = ["v", "v_eff", "m", "t"] if self._amsgrad else ["v", "m", "t"]
            reader = DictReader(csv_file, fieldnames=fieldnames)
            for line in reader:
                v = line["v"]
                if self._amsgrad:
                    v_eff = line["v_eff"]
                m = line["m"]
                t = line["t"]

        self._v = np.fromstring(v[1:-1], dtype=float, sep=" ")
        if self._amsgrad:
            self._v_eff = np.fromstring(v_eff[1:-1], dtype=float, sep=" ")
        self._m = np.fromstring(m[1:-1], dtype=float, sep=" ")
        self._t = int(np.fromstring(t[1:-1], dtype=int, sep=" "))

    @staticmethod
    def probability_less(mean1: float, sem1: float, mean2: float, sem2: float) -> float:
        """Approximate P(mu1 < mu2) assuming independent normal mean estimates."""
        denom = sqrt(sem1 * sem1 + sem2 * sem2)
        if denom <= 0:
            if mean1 < mean2:
                return 1.0
            if mean1 > mean2:
                return 0.0
            return 0.5
        z = (mean2 - mean1) / denom
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    @staticmethod
    def wrap_angles(x: POINT) -> ndarray:
        """Wrap angles to [-pi, pi)."""
        arr = np.asarray(x, dtype=float)
        return ((arr + pi) % (2.0 * pi)) - pi

    def _compute_gradient(
        self,
        fun: StatsObjective,
        params: ndarray,
        jac: GradientFunction | StatsGradientFunction | None,
    ) -> tuple[ndarray, float, float, int]:
        if jac is None:
            gradient_stats = self.estimate_gradient_spsa(fun, params)
            return gradient_stats.gradient, gradient_stats.gradient_sem_scale, gradient_stats.z_score, gradient_stats.num_objective_evals

        jac_value = jac(params)
        if isinstance(jac_value, tuple) and len(jac_value) == 3:
            gradient, gradient_sem_scale, z_score = jac_value
            return np.asarray(gradient, dtype=float), float(gradient_sem_scale), float(z_score), 0

        return np.asarray(jac_value, dtype=float), 0.0, np.inf, 0

    def _evaluate(self, fun: StatsObjective, params: ndarray) -> EvalStats:
        mean, sem = fun(self._wrap(params))
        mean = float(mean)
        sem = float(sem)
        if not np.isfinite(mean):
            raise ValueError(f"objective returned non-finite mean: {mean}")
        if not np.isfinite(sem) or sem < 0:
            raise ValueError(f"objective returned invalid SEM: {sem}")
        return EvalStats(mean=mean, sem=sem)

    @staticmethod
    def _combine_stats(stats: list[EvalStats]) -> EvalStats:
        if not stats:
            raise ValueError("cannot combine empty stats")
        if len(stats) == 1:
            return stats[0]

        means = np.asarray([s.mean for s in stats], dtype=float)
        sems = np.asarray([s.sem for s in stats], dtype=float)
        combined_mean = float(np.mean(means))
        combined_sem = float(np.sqrt(np.sum(sems * sems)) / len(stats))
        return EvalStats(mean=combined_mean, sem=combined_sem)

    @staticmethod
    def _finite_difference_z(plus: EvalStats, minus: EvalStats) -> float:
        denom = sqrt(plus.sem * plus.sem + minus.sem * minus.sem)
        diff = plus.mean - minus.mean
        if denom <= 0:
            return float(np.inf * np.sign(diff))
        return float(diff / denom)

    def _gradient_weight(self, z_abs: float) -> float:
        if not self._snr_weighting:
            return 1.0
        if not np.isfinite(z_abs):
            return 1.0
        return float((z_abs * z_abs) / (z_abs * z_abs + self._snr_z0 * self._snr_z0))

    def _is_better(self, candidate: EvalStats, incumbent: EvalStats) -> bool:
        return self._score(candidate) < self._score(incumbent)

    def _score(self, stats: EvalStats) -> float:
        if self._best_by == "mean":
            return stats.mean
        if self._best_by == "ucb":
            return stats.mean + self._best_confidence_z * stats.sem
        if self._best_by == "lcb":
            return stats.mean - self._best_confidence_z * stats.sem
        raise ValueError(f"unknown best_by={self._best_by!r}")

    def _as_angle_array(self, point: POINT) -> ndarray:
        return self._wrap(np.asarray(point, dtype=float).flatten())

    def _wrap(self, point: POINT) -> ndarray:
        arr = np.asarray(point, dtype=float).flatten()
        if self._wrap_angles:
            return self.wrap_angles(arr)
        return arr

    def _validate_options(self) -> None:
        if self._maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self._tol < 0:
            raise ValueError("tol must be nonnegative")
        if self._lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= self._beta_1 < 1:
            raise ValueError("beta_1 must be in [0, 1)")
        if not 0 <= self._beta_2 < 1:
            raise ValueError("beta_2 must be in [0, 1)")
        if self._noise_factor <= 0:
            raise ValueError("noise_factor must be positive")
        if self._perturbation <= 0:
            raise ValueError("perturbation must be positive")
        if self._snr_z0 <= 0:
            raise ValueError("snr_z0 must be positive")
        if self._min_snr_to_update < 0:
            raise ValueError("min_snr_to_update must be nonnegative")
        if self._max_resamplings <= 0:
            raise ValueError("max_resamplings must be positive")
        if self._target_snr < 0:
            raise ValueError("target_snr must be nonnegative")
        if self._best_by not in {"mean", "ucb", "lcb"}:
            raise ValueError("best_by must be one of: 'mean', 'ucb', 'lcb'")
