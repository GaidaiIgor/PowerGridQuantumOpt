"""Provides noisy SPSA-style optimization for variational angle parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, pi, sqrt
from typing import Callable, Literal, Sequence

import numpy as np
from numpy import float64, int64
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult


type Objective = Callable[[NDArray[float64]], tuple[float, float]]


@dataclass(slots=True)
class EvalStats:
    """Stores one objective evaluation result.
    mean is the sampled expectation estimate; sem is its standard error."""
    mean: float
    sem: float


@dataclass(slots=True)
class OptimizerState:
    """Stores per-iteration optimizer progress.
    iteration counts completed updates; x is the current wrapped vector; current_value and current_sem describe the local estimate.
    best_x, best_value, and best_sem describe the incumbent; jobs is the objective-call count; accepted and rejected count proposals.
    step_norm, grad_norm, learning_rate, and perturbation record update scales; message carries termination status."""
    iteration: int
    x: NDArray[float64]
    current_value: float
    current_sem: float
    best_x: NDArray[float64]
    best_value: float
    best_sem: float
    jobs: int
    accepted: int
    rejected: int
    step_norm: float
    grad_norm: float
    learning_rate: float
    perturbation: float
    message: str = ""


type Callback = Callable[[OptimizerState], bool | None]


@dataclass(slots=True)
class _GradientEval:
    """Stores paired SPSA probe results.
    grad is the estimated gradient; x_minus and x_plus are probe points; y_minus and y_plus are their objective statistics."""
    grad: NDArray[float64]
    x_minus: NDArray[float64]
    x_plus: NDArray[float64]
    y_minus: EvalStats
    y_plus: EvalStats


@dataclass(slots=True)
class NoisyAngleOptimizer:
    """Minimizes high-dimensional noisy angle objectives with Adam-preconditioned SPSA.
    maxiter and miniter bound iteration count; a0, c0, stability_offset, alpha, and gamma set schedules.
    beta1, beta2, adam_eps, amsgrad, max_step_norm, and max_component_step configure Adam and clipping.
    perturbation_distribution and perturbation_dims choose SPSA probes; blocking, blocking_z, and max_consecutive_rejections control acceptance.
    shrink_on_rejection scales rejected learning rates.
    evaluate_x0, evaluate_final, reevaluate_best_every, best_reevaluations, keep_best_by, and confidence_z manage incumbent evaluation.
    window, relative_improvement_tol, step_tol, and grad_tol set convergence checks; restarts and restart_radius configure blocked-run restarts.
    restart_shrink scales restart radius after each restart.
    seed initializes optimizer randomness; dtype chooses internal array dtype; history stores OptimizerState snapshots."""
    maxiter: int = 1000
    miniter: int = 50

    a0: float | None = None
    c0: float | None = None
    stability_offset: float = 10
    alpha: float = 0.602
    gamma: float = 0.101

    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    amsgrad: bool = True
    max_step_norm: float | None = None
    max_component_step: float | None = 0.25

    perturbation_distribution: Literal["rademacher", "normal"] = "rademacher"
    perturbation_dims: int | None = None

    blocking: bool = False
    blocking_z: float = 1
    max_consecutive_rejections: int = 25
    shrink_on_rejection: float = 0.7

    evaluate_x0: bool = True
    evaluate_final: bool = True
    reevaluate_best_every: int = 25
    best_reevaluations: int = 1
    keep_best_by: Literal["mean", "lcb", "ucb"] = "mean"
    confidence_z: float = 1

    window: int = 40
    relative_improvement_tol: float = 1e-4
    step_tol: float = 1e-8
    grad_tol: float = 0

    restarts: int = 0
    restart_radius: float = 0.25
    restart_shrink: float = 0.5

    seed: int | None = None
    dtype: type = float64
    history: list[OptimizerState] = field(default_factory=list, init=False)

    def minimize(self, objective: Objective, x0: Sequence[float] | NDArray[float64], callback: Callback | None = None) -> OptimizeResult:
        """Minimizes objective from x0 and returns an OptimizeResult with incumbent, evaluation counts, and history.
        objective maps wrapped angles to mean and SEM; x0 supplies initial angles; callback can stop the run by returning True."""
        self._validate_options()

        rng = default_rng(self.seed)
        x = self.wrap_angles(np.asarray(x0, dtype=self.dtype))
        if x.ndim != 1:
            raise ValueError("x0 must be a one-dimensional sequence of angles")
        if len(x) == 0:
            raise ValueError("x0 must contain at least one parameter")

        d = len(x)
        perturbation_dims = self._normalized_perturbation_dims(d)
        jobs = 0
        accepted = 0
        rejected = 0
        total_restarts_used = 0
        consecutive_rejections = 0
        self.history.clear()
        best_trace = []

        if self.evaluate_x0:
            y0 = self._evaluate(objective, x)
            jobs += 1
            current_value = y0.mean
            current_sem = y0.sem
            best_x = x.copy()
            best_value = y0.mean
            best_sem = y0.sem
        else:
            current_value = inf
            current_sem = inf
            best_x = x.copy()
            best_value = inf
            best_sem = inf

        a0 = self._initial_learning_rate(d)
        c0 = self._initial_perturbation()
        m = np.zeros(d, dtype=self.dtype)
        v = np.zeros(d, dtype=self.dtype)
        vhat_max = np.zeros(d, dtype=self.dtype)
        message = "maximum iterations reached"
        success = False

        for k in range(1, self.maxiter + 1):
            ak = a0 / ((k + self.stability_offset) ** self.alpha)
            ck = c0 / (k ** self.gamma)
            gradient_eval = self._spsa_gradient(objective, x, ck, rng, perturbation_dims)
            grad = gradient_eval.grad
            jobs += 2

            best_x, best_value, best_sem = self._consider_candidate(gradient_eval.x_minus, gradient_eval.y_minus, best_x, best_value, best_sem)
            best_x, best_value, best_sem = self._consider_candidate(gradient_eval.x_plus, gradient_eval.y_plus, best_x, best_value, best_sem)

            midpoint_value = 0.5 * (gradient_eval.y_minus.mean + gradient_eval.y_plus.mean)
            midpoint_sem = 0.5 * sqrt(gradient_eval.y_minus.sem ** 2 + gradient_eval.y_plus.sem ** 2)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            m_hat = m / (1 - self.beta1 ** k)
            v_hat = v / (1 - self.beta2 ** k)
            if self.amsgrad:
                vhat_max = np.maximum(vhat_max, v_hat)
                denom = np.sqrt(vhat_max) + self.adam_eps
            else:
                denom = np.sqrt(v_hat) + self.adam_eps

            step = self._clip_step(-ak * m_hat / denom)
            proposal = self.wrap_angles(x + step)
            step_norm = np.linalg.norm(step)
            grad_norm = np.linalg.norm(grad)
            accept = True
            proposal_stats = None

            if self.blocking:
                proposal_stats = self._evaluate(objective, proposal)
                jobs += 1
                threshold = self.blocking_z * sqrt(current_sem ** 2 + proposal_stats.sem ** 2)
                accept = proposal_stats.mean <= current_value + threshold

            if accept:
                x = proposal
                accepted += 1
                consecutive_rejections = 0

                if proposal_stats is not None:
                    current_value = proposal_stats.mean
                    current_sem = proposal_stats.sem
                    best_x, best_value, best_sem = self._consider_candidate(x, proposal_stats, best_x, best_value, best_sem)
                else:
                    current_value = midpoint_value
                    current_sem = midpoint_sem
            else:
                rejected += 1
                consecutive_rejections += 1
                if self.shrink_on_rejection < 1:
                    a0 *= self.shrink_on_rejection

            if self.reevaluate_best_every > 0 and k % self.reevaluate_best_every == 0 and np.isfinite(best_value):
                best_stats = self._reevaluate(objective, best_x, self.best_reevaluations)
                jobs += self.best_reevaluations
                best_value = best_stats.mean
                best_sem = best_stats.sem

            state = OptimizerState(iteration=k, x=x.copy(), current_value=current_value, current_sem=current_sem, best_x=best_x.copy(),
                best_value=best_value, best_sem=best_sem, jobs=jobs, accepted=accepted, rejected=rejected, step_norm=step_norm, grad_norm=grad_norm,
                learning_rate=ak, perturbation=ck, message=message)
            self.history.append(state)
            best_trace.append(best_value)

            if callback is not None and callback(state):
                message = "stopped by callback"
                success = True
                break

            if k >= self.miniter:
                converged, why = self._check_convergence(best_trace, step_norm, grad_norm)
                if converged:
                    message = why
                    success = True
                    break

            should_restart = self.blocking and self.restarts > 0 and consecutive_rejections >= self.max_consecutive_rejections \
                and total_restarts_used < self.restarts
            if should_restart:
                radius = self.restart_radius * (self.restart_shrink ** total_restarts_used)
                x = self.wrap_angles(best_x + rng.normal(0, radius, size=d))
                m.fill(0)
                v.fill(0)
                vhat_max.fill(0)
                consecutive_rejections = 0
                total_restarts_used += 1

        final_stats = None
        if self.evaluate_final:
            final_stats = self._evaluate(objective, x)
            jobs += 1
            current_value = final_stats.mean
            current_sem = final_stats.sem
            best_x, best_value, best_sem = self._consider_candidate(x, final_stats, best_x, best_value, best_sem)

        return OptimizeResult(x=best_x, fun=best_value, sem=best_sem, success=success, message=message, nit=len(self.history), nfev=jobs,
            accepted=accepted, rejected=rejected, restarts=total_restarts_used, last_x=x.copy(), last_value=current_value, last_sem=current_sem,
            final_evaluation=final_stats, history=self.history.copy())

    def _validate_options(self):
        """Validates optimizer options before a run."""
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self.miniter < 0:
            raise ValueError("miniter must be nonnegative")
        if not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.stability_offset < 0:
            raise ValueError("stability_offset must be nonnegative")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.gamma < 0:
            raise ValueError("gamma must be nonnegative")
        if self.best_reevaluations <= 0:
            raise ValueError("best_reevaluations must be positive")
        if self.window <= 1:
            raise ValueError("window must be greater than 1")
        if not 0 < self.shrink_on_rejection <= 1:
            raise ValueError("shrink_on_rejection must be in (0, 1]")

    @staticmethod
    def wrap_angles(x: Sequence[float] | NDArray[float64]) -> NDArray[float64]:
        """Wraps angle vector x to [-pi, pi) and returns a float64 array."""
        arr = np.asarray(x, dtype=float64)
        return ((arr + pi) % (2 * pi)) - pi

    def _normalized_perturbation_dims(self, d: int) -> int:
        """Returns the active perturbation count for a d-dimensional parameter vector."""
        if self.perturbation_dims is None:
            return d
        if self.perturbation_dims <= 0:
            raise ValueError("perturbation_dims must be positive")
        return min(self.perturbation_dims, d)

    def _evaluate(self, objective: Objective, x: NDArray[float64]) -> EvalStats:
        """Evaluates objective at wrapped x and returns normalized statistics."""
        mean, sem = objective(self.wrap_angles(x))
        mean = float(mean)
        sem = float(sem)
        if not np.isfinite(mean):
            raise ValueError(f"objective returned non-finite mean: {mean}")
        if not np.isfinite(sem) or sem < 0:
            raise ValueError(f"objective returned invalid SEM: {sem}")
        return EvalStats(mean=mean, sem=sem)

    def _initial_learning_rate(self, d: int) -> float:
        """Returns the configured or dimension-scaled starting learning rate for dimension d."""
        if self.a0 is not None:
            if self.a0 <= 0:
                raise ValueError("a0 must be positive")
            return self.a0
        return min(0.25, 1 / sqrt(max(d, 1)))

    def _initial_perturbation(self) -> float:
        """Returns the configured or default SPSA perturbation radius."""
        if self.c0 is not None:
            if self.c0 <= 0:
                raise ValueError("c0 must be positive")
            return self.c0
        return 0.1

    def _spsa_gradient(self, objective: Objective, x: NDArray[float64], c: float, rng: Generator, perturbation_dims: int) -> _GradientEval:
        """Estimates an SPSA gradient at x with perturbation radius c.
        objective supplies noisy values; rng draws perturbations; perturbation_dims limits the perturbed coordinate subset."""
        d = len(x)
        selected = self._select_perturbed_indices(d, perturbation_dims, rng)
        delta = np.zeros(d, dtype=float64)

        if self.perturbation_distribution == "rademacher":
            delta[selected] = rng.choice((-1, 1), size=len(selected))
        elif self.perturbation_distribution == "normal":
            raw = rng.normal(size=len(selected))
            raw = np.where(np.abs(raw) < 0.1, np.sign(raw + 1e-12) * 0.1, raw)
            delta[selected] = raw
        else:
            raise ValueError(f"unknown perturbation_distribution={self.perturbation_distribution}")

        x_plus = self.wrap_angles(x + c * delta)
        x_minus = self.wrap_angles(x - c * delta)
        y_plus = self._evaluate(objective, x_plus)
        y_minus = self._evaluate(objective, x_minus)
        grad = np.zeros(d, dtype=float64)
        grad[selected] = (y_plus.mean - y_minus.mean) / (2 * c * delta[selected])
        if perturbation_dims < d:
            grad *= d / perturbation_dims
        return _GradientEval(grad=grad.astype(self.dtype), x_minus=x_minus, x_plus=x_plus, y_minus=y_minus, y_plus=y_plus)

    def _select_perturbed_indices(self, d: int, perturbation_dims: int, rng: Generator) -> NDArray[int64]:
        """Returns sorted coordinate indices to perturb within a d-dimensional vector."""
        if perturbation_dims >= d:
            return np.arange(d, dtype=int64)
        return np.sort(rng.choice(d, size=perturbation_dims, replace=False))

    def _consider_candidate(self, candidate_x: NDArray[float64], candidate_stats: EvalStats, best_x: NDArray[float64], best_value: float,
            best_sem: float) -> tuple[NDArray[float64], float, float]:
        """Compares a candidate against the incumbent and returns the selected x, value, and SEM."""
        if self._is_better(candidate_stats.mean, candidate_stats.sem, best_value, best_sem):
            return candidate_x.copy(), candidate_stats.mean, candidate_stats.sem
        return best_x, best_value, best_sem

    def _is_better(self, value: float, sem: float, best_value: float, best_sem: float) -> bool:
        """Returns whether value and sem improve on the incumbent under the selected comparison rule."""
        if not np.isfinite(best_value):
            return True
        if self.keep_best_by == "mean":
            return value < best_value
        if self.keep_best_by == "lcb":
            return value - self.confidence_z * sem < best_value - self.confidence_z * best_sem
        if self.keep_best_by == "ucb":
            return value + self.confidence_z * sem < best_value + self.confidence_z * best_sem
        raise ValueError(f"unknown keep_best_by={self.keep_best_by}")

    def _clip_step(self, step: NDArray[float64]) -> NDArray[float64]:
        """Clips step by component and vector norm limits and returns an array with the configured dtype."""
        out = np.asarray(step, dtype=float64).copy()
        if self.max_component_step is not None:
            out = np.clip(out, -self.max_component_step, self.max_component_step)
        if self.max_step_norm is not None:
            norm = np.linalg.norm(out)
            if norm > self.max_step_norm > 0:
                out *= self.max_step_norm / norm
        return out.astype(self.dtype)

    def _reevaluate(self, objective: Objective, x: NDArray[float64], n: int) -> EvalStats:
        """Evaluates objective n times at x and returns the averaged mean with combined SEM."""
        if n <= 0:
            raise ValueError("number of reevaluations must be positive")

        means = np.empty(n, dtype=float64)
        sems = np.empty(n, dtype=float64)
        for i in range(n):
            stats = self._evaluate(objective, x)
            means[i] = stats.mean
            sems[i] = stats.sem

        return EvalStats(mean=np.mean(means), sem=np.sqrt(np.sum(sems * sems)) / n)

    def _check_convergence(self, best_trace: list[float], step_norm: float, grad_norm: float) -> tuple[bool, str]:
        """Checks convergence from incumbent trace, step norm, and gradient norm, then returns status and reason."""
        if self.step_tol > 0 and step_norm < self.step_tol:
            return True, "step norm below tolerance"
        if self.grad_tol > 0 and grad_norm < self.grad_tol:
            return True, "gradient norm below tolerance"
        if len(best_trace) >= self.window:
            old = best_trace[-self.window]
            new = best_trace[-1]
            rel_improvement = (old - new) / max(1, abs(old))
            if 0 <= rel_improvement < self.relative_improvement_tol:
                return True, "relative improvement below tolerance"
        return False, ""


if __name__ == "__main__":
    def toy_noisy_objective(theta: NDArray[float64]) -> tuple[float, float]:
        """Returns noisy sample mean and SEM for a toy cosine objective.
        theta supplies wrapped angles; the return value is the sampled expectation estimate and its SEM."""
        true_mean = np.mean(1 - np.cos(theta))
        samples = true_mean + rng_for_objective.normal(0, 0.2, size=shots)
        return float(np.mean(samples)), float(np.std(samples, ddof=1) / sqrt(shots))

    shots = 1000
    rng_for_objective = default_rng(456)
    dim = 100
    rng = default_rng(123)
    x0 = rng.uniform(-pi, pi, size=dim)
    optimizer = NoisyAngleOptimizer(maxiter=500, miniter=100, seed=123, perturbation_dims=None, blocking=False, reevaluate_best_every=25,
        best_reevaluations=2)
    result = optimizer.minimize(toy_noisy_objective, x0)
    print("success:", result.success)
    print("message:", result.message)
    print("fun:", result.fun)
    print("sem:", result.sem)
    print("jobs:", result.nfev)
