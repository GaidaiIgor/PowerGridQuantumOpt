from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps, partial
from typing import Callable, Hashable, Any, Concatenate

import networkx as nx
import numpy as np
from networkx import Graph
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import OptimizeResult, LinearConstraint, NonlinearConstraint


def cached[K: Hashable, **P, R](func: Callable[Concatenate[K, P], R]) -> Callable[Concatenate[K, P], R]:
    """ Function decorator that adds saving of function's evaluation result in function's internal dict under the value of the function's first argument and reads it from the
    dict if the function is called again with the same first argument. """
    @wraps(func)
    def wrapper(key: K, *args: P.args, **kwargs: P.kwargs) -> R:
        if key not in wrapper.cache:
            wrapper.cache[key] = func(key, *args, **kwargs)
        return wrapper.cache[key]

    wrapper.cache = {}
    return wrapper


def get_penalty(params: list[float], constraints: list[LinearConstraint | NonlinearConstraint], mult: float = 1e1) -> float:
    """ Evaluates penalty term for a given optimization parameter vector and list of constraints. """
    penalty = 0
    for constraint in constraints:
        if isinstance(constraint, LinearConstraint):
            residuals = constraint.residual(params)
            val = np.concatenate((*residuals, ))
        elif isinstance(constraint, NonlinearConstraint):
            val = constraint.fun(params)
        penalty += mult * np.sum(np.minimum(val, 0) ** 2)
    return penalty


@dataclass
class Generator:
    """
    Describes a generator.
    :var power_range: (min, max) values of active power (p) for this generator.
    :var reactive_power_range: (min, max) values of reactive power for this generator.
    :var cost_terms: (a, b, c) terms of quadratic generation cost function (ap^2 + bp + c).
    """
    power_range: tuple[float, float]
    reactive_power_range: tuple[float, float]
    cost_terms: tuple[float, float, float]

    def generation_cost(self, power: float) -> float:
        return self.cost_terms[0] * power ** 2 + self.cost_terms[1] * power + self.cost_terms[2]


@dataclass
class PowerGridProblem(ABC):
    """ Generic protocol for a power grid problem. """
    generators: NDArray[Generator]

    def __init_subclass__(cls, **kwargs):
        cls.optimize_power = cached(cls.optimize_power)

    @abstractmethod
    def optimize_power(self, generator_statuses: str, penalty_mult: float = 10) -> OptimizeResult:
        """ Optimizes continuous problem variables for given generator statuses. Returns full OptimizeResult. Cached. """
        pass

    def evaluate(self, generator_statuses: str, penalty_mult: float = 10) -> float:
        """ Returns the value of the cost function after optimization. """
        return self.optimize_power(generator_statuses, penalty_mult).total


@dataclass
class GeneratorCommitmentProblem(PowerGridProblem):
    """ Describes a unit commitment problem in a power grid.
    I.e. given a set of generators, which ones should be enabled and at what power in order to meet target load using the smallest operation cost. """
    load: float

    def optimize_power(self, generator_statuses: str, penalty_mult: float = 10) -> OptimizeResult:
        """ Finds optimal generation cost for a given generator assignment. """
        def generation_cost_total(powers: list[float]) -> float:
            return sum(gen.generation_cost(power) for gen, power in zip(enabled_generators, powers))

        enabled_generators = self.generators[[int(val) == 1 for val in generator_statuses]]
        initial_point = [gen.power_range[1] for gen in enabled_generators]
        bounds = [gen.power_range for gen in enabled_generators] if enabled_generators.size > 0 else None
        constraints = [{"type": "ineq", "fun": lambda powers: sum(powers) - self.load}]
        result = optimize.minimize(generation_cost_total, initial_point, method="SLSQP", bounds=bounds, constraints=constraints)
        result.penalty = get_penalty(result.x, constraints, penalty_mult)
        result.total = result.fun + result.penalty
        return result


class SimplePowerFlowProblem(PowerGridProblem):
    """ Generalized version of GeneratorCommitmentProblem, where locations of generators and loads are taken into account.
    Specifically, the problem is described by a graph, where nodes represent neighborhoods that can include generators and loads.
    Generated power is consumed by local loads. Any excess power can be transferred to adjacent nodes to supplement their generators.
    Edges represent power lines between neighborhoods and have finite capacities, so routing the generated power to the loads now becomes a problem too. """

    def __init__(self, graph: Graph):
        """ Graph nodes should have the following properties:
        1) generators: list[Generator]. List of generator instances located at a given node.
        2) load: float >= 0. Total load at a given node.
        Graph edges should have the following properties:
        1) capacity: float > 0. Maximum power that can be routed through a given edge.
        The following additional properties will be automatically added to the graph:
        1) var_inds: list[int], added to nodes. List of indices in the optimization vector to which generator's power outputs at this node map.
        2) var_ind: int, added to edges. Index in the optimization vector to which power flow through this edge maps.
        3) start: node key type, added to edges. Defines positive flow direction by choosing an arbitrary end of a given edge as start.
        Collects generators from all nodes into a single generators list. """
        self.graph = graph
        generators = np.array([gen for _, gens in self.graph.nodes(data="generators") for gen in gens])
        super().__init__(generators)

        var_ind = 0
        for _, data in self.graph.nodes(data=True):
            data["var_inds"] = list(range(var_ind, var_ind + len(data["generators"])))
            var_ind += len(data["generators"])
        nx.set_edge_attributes(self.graph, {(u, v): {"var_ind": i + var_ind, "start": u} for i, (u, v) in enumerate(self.graph.edges)})

    def evaluate_power_balance(self, powers: list[float], node_label: Hashable) -> float:
        """ Evaluates power balance at a given node, i.e. sum of all generated params + incoming - outgoing - load. """
        power_balance = 0
        this_node = self.graph.nodes[node_label]
        for gen_ind in this_node["var_inds"]:
            power_balance += powers[gen_ind]
        for _, v, data in self.graph.edges(node_label, data=True):
            power_balance += powers[data["var_ind"]] * (-1) ** (node_label == data["start"])
        power_balance -= this_node["load"]
        return power_balance

    def get_constraints(self) -> list[dict[str, Any]]:
        """ Provides a list of inequality constraints. Each constraint requires power balance at a given node to be non-negative.
        Positive power balance (i.e. excess power) is allowed, since it is assumed that each node has a power sink. """
        constraints = []
        for label in self.graph.nodes:
            constraints.append({"type": "ineq", "fun": partial(self.evaluate_power_balance, node_label=label)})
        return constraints

    def optimize_power(self, bitstring: str) -> OptimizeResult:
        """ Finds optimal power vector for a given set of enabled generators, defined by the generator_statuses. """
        def generation_cost_total(powers: list[float]) -> float:
            return sum(gen.generation_cost(power) for gen, power in zip(self.generators, powers))

        bounds = [np.array(gen.power_range) * int(bitstring[i]) for i, gen in enumerate(self.generators)]
        bounds += [(-cap, cap) for _, _, cap in self.graph.edges(data="capacity")]
        initial_point = [bound[1] for bound in bounds[:len(self.generators)]]
        initial_point += [0] * len(self.graph.edges)
        constraints = self.get_constraints()
        result = optimize.minimize(generation_cost_total, initial_point, method="SLSQP", bounds=bounds, constraints=constraints)
        result.penalty = get_penalty(result.x, constraints)
        result.total = result.fun + result.penalty
        return result


class PowerFlowACProblem(PowerGridProblem):
    """ Physical version of SimplePowerFlowProblem, where power is complex, lines are not lossless and line flows have to satisfy physical constraints. """

    def __init__(self, graph: Graph):
        """ Graph nodes should have the following properties:
        1) generators: list[Generator]. List of generator instances located at a given node.
        2) load: complex. Total load at a given node.
        3) voltage_range: tuple[float, float]. Range of allowed voltage magnitudes at this node.
        4) angle_range: tuple[float, float]. Range of allowed phase angles at this node.
        Graph edges should have the following properties:
        1) capacity: float > 0. Maximum absolute value of power that can be routed through a given edge.
        2) admittance: complex. Line admittance.
        The following additional properties control mapping between node parameters and the overall optimization vector.
        1) gen_inds: list[int]. List of generator indices in self.generators corresponding to generators at this node.
        Collects generators from all nodes into a single generators list. """
        self.graph = graph
        generators = []
        for i, (_, data) in enumerate(sorted(self.graph.nodes(data=True))):
            data["node_ind"] = i
            data["gen_inds"] = list(range(len(generators), len(generators) + len(data["generators"])))
            generators += data["generators"]
        super().__init__(generators)

    def get_bounds(self, generator_status: str) -> list[NDArray[float]]:
        """ Returns list of NDArray of 2 elements: [min, max], i.e. bounds on each optimization parameter. """
        bounds_active = [np.array(gen.power_range) * int(generator_status[i]) for i, gen in enumerate(self.generators)]
        bounds_reactive = [np.array(gen.reactive_power_range) * int(generator_status[i]) for i, gen in enumerate(self.generators)]
        bounds_voltage = [0] * len(self.graph)
        bounds_angle = [0] * len(self.graph)
        for node, data in self.graph.nodes(data=True):
            bounds_voltage[data["node_ind"]] = np.array(data["voltage_range"])
            bounds_angle[data["node_ind"]] = np.array(data["angle_range"])
        bounds = bounds_active + bounds_reactive + bounds_voltage + bounds_angle
        return bounds

    @staticmethod
    def get_initial_point(bounds: list[NDArray[float]]) -> list[float]:
        """ Returns initial point for the optimization. """
        initial_point = [np.average(bound) for bound in bounds]
        return initial_point

    @staticmethod
    def convert_bounds_to_constraints(bounds: list[NDArray[float]]) -> LinearConstraint:
        """ Converts bounds to a linear constraint object. """
        A = np.eye(len(bounds))
        bounds_matrix = np.array(bounds)
        constraint = LinearConstraint(A, bounds_matrix[:, 0], bounds_matrix[:, 1])
        return constraint

    def evaluate_line_powers(self, params: list[float]) -> list[complex]:
        voltage_magnitudes = np.array(params[2 * len(self.generators):2 * len(self.generators) + len(self.graph)])
        phase_angles = np.array(params[2 * len(self.generators) + len(self.graph):])
        voltages = voltage_magnitudes * np.exp(1j * phase_angles)
        line_powers = []
        for node_label, node_data in self.graph.nodes(data=True):
            for _, neighbor, line_data in self.graph.edges(node_label, data=True):
                current = line_data["admittance"] * (voltages[node_data["node_ind"]] - voltages[self.graph.nodes[neighbor]["node_ind"]])
                line_power = voltages[node_data["node_ind"]] * np.conj(current)
                line_powers.append(line_power)
        return line_powers

    def evaluate_constraints(self, params: list[float]) -> list[float]:
        """
        Evaluates all constraints, i.e. power balance at each node (generated params + incoming - outgoing - load >= 0) and line capacities (|S_ij| <= max capacity).
        :param params: Vector of optimization parameters.
        Includes real power of each generator, then reactive power of each generator, then voltage of each node, then phase angle of each node.
        Total length is 2G + 2N, where G is the total number of generators in the graph and N is the number of nodes.
        :return: List of constraint values (>= 0 is feasible).
        """
        active_powers = np.array(params[:len(self.generators)])
        reactive_powers = np.array(params[len(self.generators):2 * len(self.generators)])
        voltage_magnitudes = np.array(params[2 * len(self.generators):2 * len(self.generators) + len(self.graph)])
        phase_angles = np.array(params[2 * len(self.generators) + len(self.graph):])
        powers = active_powers + 1j * reactive_powers
        voltages = voltage_magnitudes * np.exp(1j * phase_angles)

        constraints = []
        for node_label, node_data in self.graph.nodes(data=True):
            generated_power = np.sum(powers[node_data["gen_inds"]])
            line_powers = []
            for _, neighbor, line_data in self.graph.edges(node_label, data=True):
                current = line_data["admittance"] * (voltages[node_data["node_ind"]] - voltages[self.graph.nodes[neighbor]["node_ind"]])
                line_power = voltages[node_data["node_ind"]] * np.conj(current)
                constraints.append(line_data["capacity"] - np.abs(line_power))
                line_powers.append(line_power)
            power_balance = generated_power - node_data["load"] - np.sum(line_powers)
            constraints.append(np.real(power_balance))
            constraints.append(np.imag(power_balance))
        return constraints

    def get_generation_cost(self, generator_statuses: str, params: list[float]) -> float:
        """ Returns the total cost of generation for a given set of enabled generators at given optimization parameters. """
        active_powers = params[:len(self.generators)]
        return sum(int(status) * gen.generation_cost(power) for status, gen, power in zip(generator_statuses, self.generators, active_powers))

    def optimize_power(self, generator_statuses: str, penalty_mult: float = 10) -> OptimizeResult:
        """ Finds optimal power vector for a given set of enabled generators, defined by the generator_statuses. """
        bounds = self.get_bounds(generator_statuses)
        initial_point = self.get_initial_point(bounds)
        constraints = [self.convert_bounds_to_constraints(bounds), NonlinearConstraint(self.evaluate_constraints, 0, np.inf)]
        # constraints = [{"type": "ineq", "fun": self.evaluate_constraints}]
        cost_function = partial(self.get_generation_cost, generator_statuses)
        options = {"maxiter": 2 ** 31 - 1}
        result = optimize.minimize(cost_function, initial_point, method="SLSQP", constraints=constraints, options=options)
        result.penalty = get_penalty(result.x, constraints, penalty_mult)
        result.total = result.fun + result.penalty
        return result
