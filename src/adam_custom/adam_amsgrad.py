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

"""Provides a local copy of the Qiskit Adam and AMSGRAD optimizers."""
from __future__ import annotations

from collections.abc import Callable
from csv import DictReader, DictWriter
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy import ndarray
from qiskit_algorithms.optimizers.optimizer import POINT, Optimizer, OptimizerResult, OptimizerSupportLevel


class ADAM(Optimizer):
    """Implements Adam and AMSGRAD optimizers using the Qiskit Optimizer interface.
    :var _OPTIONS: Optimizer option names accepted by the constructor.
    :var _maxiter: Maximum number of optimization iterations.
    :var _snapshot_dir: Optional directory used to write parameter snapshots.
    :var _tol: Step-size tolerance used for termination.
    :var _lr: Learning rate.
    :var _beta_1: Exponential decay rate for the first moment estimate.
    :var _beta_2: Exponential decay rate for the second moment estimate.
    :var _noise_factor: Stabilizing denominator offset.
    :var _eps: Finite-difference epsilon used when no analytic gradient is supplied.
    :var _amsgrad: Whether to use the AMSGRAD variant.
    :var _t: Current time step.
    :var _m: First moment vector.
    :var _v: Second moment vector.
    :var _v_eff: AMSGRAD effective second moment vector."""
    _OPTIONS: ClassVar[tuple[str, ...]] = ("maxiter", "tol", "lr", "beta_1", "beta_2", "noise_factor", "eps", "amsgrad", "snapshot_dir")
    _maxiter: int
    _snapshot_dir: str | None
    _tol: float
    _lr: float
    _beta_1: float
    _beta_2: float
    _noise_factor: float
    _eps: float
    _amsgrad: bool
    _t: int
    _m: ndarray
    _v: ndarray
    _v_eff: ndarray

    def __init__(self, maxiter: int = 10000, tol: float = 1e-6, lr: float = 1e-3, beta_1: float = 0.9, beta_2: float = 0.99,
                 noise_factor: float = 1e-8, eps: float = 1e-10, amsgrad: bool = False, snapshot_dir: str | None = None):
        """Initializes optimizer options and runtime moment estimates.
        :param maxiter: Maximum number of optimization iterations.
        :param tol: Step-size tolerance used for termination.
        :param lr: Learning rate.
        :param beta_1: Exponential decay rate for the first moment estimate.
        :param beta_2: Exponential decay rate for the second moment estimate.
        :param noise_factor: Stabilizing denominator offset.
        :param eps: Finite-difference epsilon used when no analytic gradient is supplied.
        :param amsgrad: Whether to use the AMSGRAD variant.
        :param snapshot_dir: Optional directory used to write parameter snapshots."""
        super().__init__()
        self._options.update({"maxiter": maxiter, "tol": tol, "lr": lr, "beta_1": beta_1, "beta_2": beta_2, "noise_factor": noise_factor, "eps": eps,
                              "amsgrad": amsgrad, "snapshot_dir": snapshot_dir})
        self._maxiter = maxiter
        self._snapshot_dir = snapshot_dir
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad

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
        """Returns constructor settings that recreate this optimizer.
        :return: Optimizer settings dictionary."""
        return {"maxiter": self._maxiter, "tol": self._tol, "lr": self._lr, "beta_1": self._beta_1, "beta_2": self._beta_2,
                "noise_factor": self._noise_factor, "eps": self._eps, "amsgrad": self._amsgrad, "snapshot_dir": self._snapshot_dir}

    def get_support_level(self) -> dict[str, OptimizerSupportLevel]:
        """Returns the Qiskit support levels for optimizer inputs.
        :return: Support levels for gradients, bounds, and initial points."""
        return {"gradient": OptimizerSupportLevel.supported, "bounds": OptimizerSupportLevel.ignored, "initial_point": OptimizerSupportLevel.supported}

    def minimize(self, fun: Callable[[POINT], float], x0: POINT, jac: Callable[[POINT], POINT] | None = None,
                 bounds: list[tuple[float, float]] | None = None) -> OptimizerResult:
        """Minimizes a scalar objective function.
        :param fun: Scalar objective function to minimize.
        :param x0: Initial parameter vector.
        :param jac: Optional gradient function for the scalar objective.
        :param bounds: Optional variable bounds accepted for Qiskit API compatibility.
        :return: Optimization result containing the final point, final value, and evaluation count."""
        if jac is None:
            jac = Optimizer.wrap_function(Optimizer.gradient_num_diff, (fun, self._eps, self._max_evals_grouped))

        derivative = jac(x0)
        self._t = 0
        self._m = np.zeros(np.shape(derivative))
        self._v = np.zeros(np.shape(derivative))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(derivative))

        params = params_new = x0
        while self._t < self._maxiter:
            if self._t > 0:
                derivative = jac(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
            lr_eff = self._lr * np.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
            if self._amsgrad:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = params - lr_eff * self._m.flatten() / (np.sqrt(self._v_eff.flatten()) + self._noise_factor)
            else:
                params_new = params - lr_eff * self._m.flatten() / (np.sqrt(self._v.flatten()) + self._noise_factor)

            if self._snapshot_dir:
                self.save_params(self._snapshot_dir)

            if np.linalg.norm(params - params_new) < self._tol:
                break

            params = params_new

        result = OptimizerResult()
        result.x = params_new
        result.fun = fun(params_new)
        result.nfev = self._t
        return result

    def save_params(self, snapshot_dir: str):
        """Saves current iteration parameters to adam_params.csv.
        :param snapshot_dir: Directory that receives the parameter snapshot."""
        with Path(snapshot_dir, "adam_params.csv").open("a", newline="") as csv_file:
            if self._amsgrad:
                writer = DictWriter(csv_file, fieldnames=["v", "v_eff", "m", "t"])
                writer.writerow({"v": self._v, "v_eff": self._v_eff, "m": self._m, "t": self._t})
            else:
                writer = DictWriter(csv_file, fieldnames=["v", "m", "t"])
                writer.writerow({"v": self._v, "m": self._m, "t": self._t})

    def load_params(self, load_dir: str):
        """Loads iteration parameters from adam_params.csv.
        :param load_dir: Directory containing the parameter snapshot."""
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
