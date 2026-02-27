"""Utilities for generating random annotated graphs for ``PowerFlowProblem`` instances."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from networkx import Graph
from scipy.optimize import brentq
from scipy.stats import norm

from .Generator import Generator
from .PowerFlowProblem import PowerFlowProblem
from .validation import validate_bounds


@dataclass
class LognormalSpec:
    """Lognormal sampling configuration.
    :var mean: Target mean of the lognormal distribution.
    :var spread_factor: Multiplicative half-width around mean.
    :var spread_mass: Probability mass inside ``[mean / spread_factor, mean * spread_factor]``.
    :var _mu: Normal-space mean parameter.
    :var _sigma: Normal-space standard deviation parameter.
    """
    mean: float
    spread_factor: float
    spread_mass: float = 0.9
    _mu: float = field(init=False, repr=False)
    _sigma: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validates and precomputes normal-space parameters used for sampling."""
        assert self.mean > 0, f"lognormal.mean must be positive, got {self.mean}."
        assert self.spread_factor >= 1, f"lognormal spread_factor must be >= 1, got {self.spread_factor}."
        assert 0 < self.spread_mass <= 1, f"lognormal.spread_mass must satisfy 0 < spread_mass <= 1, got {self.spread_mass}."
        assert not (self.spread_factor == 1 and self.spread_mass != 1), f"lognormal.spread_mass must be 1 when spread_factor is 1, got {self.spread_mass}."

        if self.spread_factor == 1:
            sigma = 0
        else:
            def interval_mass(sigma: float) -> float:
                if sigma == 0:
                    return 1
                z_hi = 0.5 * sigma + log_spread / sigma
                z_low = 0.5 * sigma - log_spread / sigma
                return norm.cdf(z_hi) - norm.cdf(z_low)

            log_spread = math.log(self.spread_factor)
            high = 1
            objective = lambda sigma: interval_mass(sigma) - self.spread_mass
            while objective(high) > 0:
                high *= 2
            sigma = brentq(objective, 0, high)

        self._mu = math.log(self.mean) - 0.5 * sigma ** 2
        self._sigma = sigma

    @property
    def mu(self) -> float:
        """Returns precomputed normal-space ``mu`` used for lognormal sampling.
        :return: Normal-space mean parameter.
        """
        return self._mu

    @property
    def sigma(self) -> float:
        """Returns precomputed normal-space ``sigma`` used for lognormal sampling.
        :return: Normal-space standard deviation parameter.
        """
        return self._sigma

    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Samples one lognormal value from this configuration.
        :param rng: Optional random generator; default generator is used when omitted.
        :return: One sampled positive value.
        """
        rng = rng or np.random.default_rng()
        return rng.lognormal(self.mu, self.sigma)


class PowerFlowProblemGenerator:
    """Factory for random AC power-flow problem instances."""

    def __init__(self, *, random_seed: int | None = None):
        """Initializes a reproducible random number stream.
        :param random_seed: Optional seed for deterministic generation.
        """
        self._rng = np.random.default_rng(random_seed)

    def generate_instances(
        self,
        num_generators: int,
        num_instances: int,
        *,
        # graph parameters
        average_node_degree: float = 4,
        degree_bias: float = 0.3,
        voltage_range: tuple[float, float] = (0.95, 1.05),
        angle_range: tuple[float, float] = (-math.pi, math.pi),
        # load parameters
        load_s_spec: LognormalSpec | None = None,
        load_react_frac_range: tuple[float, float] = (0, 0.1),
        # generator parameters
        generator_density: float = 1,
        generator_s_range_ref_spec: LognormalSpec | None = None,
        generator_s_range_len_spec: LognormalSpec | None = None,
        generator_react_frac_range: tuple[float, float] = (0, 0.1),
        symmetric_q_range: bool = True,
        cost_a_spec: LognormalSpec | None = None,
        cost_b_spec: LognormalSpec | None = None,
        cost_c_spec: LognormalSpec | None = None,
        # line parameters
        impedance_spec: LognormalSpec | None = None,
        line_react_frac_range: tuple[float, float] = (0.9, 1),
        scale_lines: bool = True,
        capacity_spec: LognormalSpec | None = None,
        # miscellaneous parameters
        output_folder: str | Path = "data",
        check_basic_feasibility: bool = True,
    ) -> list[Path]:
        """Generates and persists random power-flow graph instances.
        :param num_generators: Total number of generators to place across all nodes.
        :param num_instances: Number of independent graph instances to generate.
        :param average_node_degree: Target average degree for geometric graph radius selection.
        :param degree_bias: Bias strength toward high-degree nodes during generator placement.
        :param voltage_range: Allowed voltage magnitude range for every node.
        :param angle_range: Allowed voltage phase-angle range for every node.
        :param load_s_spec: Distribution spec for apparent load magnitude ``|S|`` on each node.
        :param load_react_frac_range: Uniform distribution range for load reactive fraction, i.e. Q / |S|
        :param generator_density: Target ratio of generators per node used to infer node count.
        :param generator_s_range_ref_spec: Lognormal distribution spec for reference points (ref) of generator's apparent power magnitude range.
        :param generator_s_range_len_spec: Lognormal distribution spec for multiplicative length factor (M) for generator's apparent-power magnitude range.
            Range is calculated as [ref / (1 + M); ref * (1 + M)].
        :param generator_react_frac_range: Uniform distribution range for generator reactive fraction, i.e. Q_ref / |S_ref|.
        :param symmetric_q_range: Whether to enforce symmetric generator reactive bounds around zero.
        :param cost_a_spec: Distribution spec for quadratic cost coefficient ``a``.
        :param cost_b_spec: Distribution spec for linear cost coefficient ``b``.
        :param cost_c_spec: Distribution spec for constant cost coefficient ``c``.
        :param impedance_spec: Distribution spec for sampled line-impedance magnitudes ``|Z|``.
        :param line_react_frac_range: Uniform distribution range for line reactance fraction ``X / |Z|``.
        :param scale_lines: Whether to multiply sampled line resistance/reactance by ``line_length / median_length``.
        :param capacity_spec: Distribution spec for line current capacities.
        :param output_folder: Destination directory for serialized graph files.
        :param check_basic_feasibility: Whether to skip generated instances that fail the fast aggregate feasibility pre-check.
        :return: Paths to serialized generated graph files.
        """
        assert num_generators > 0, f"num_generators must be positive, got {num_generators}."
        assert num_instances > 0, f"num_instances must be positive, got {num_instances}."
        assert generator_density > 0, f"generator_density must be positive, got {generator_density}."
        assert average_node_degree >= 0, f"average_node_degree must be non-negative, got {average_node_degree}."
        validate_bounds("load_react_frac_range", load_react_frac_range, min_value=-1, max_value=1, include_min=False, include_max=False)
        validate_bounds("generator_react_frac_range", generator_react_frac_range, min_value=-1, max_value=1, include_min=False, include_max=False)
        validate_bounds("line_react_frac_range", line_react_frac_range, min_value=-1, max_value=1, include_min=True, include_max=True)
        validate_bounds("degree_bias", degree_bias, min_value=0, max_value=1, include_min=True, include_max=True)

        load_s_spec = load_s_spec or LognormalSpec(1, 10)
        generator_s_range_ref_spec = generator_s_range_ref_spec or LognormalSpec(load_s_spec.mean * 2, load_s_spec.spread_factor)
        generator_s_range_len_spec = generator_s_range_len_spec or LognormalSpec(0.5, 1.2)
        num_nodes = max(1, int(math.ceil(num_generators / generator_density)))
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        generator_p_range_ref_mean = self._get_active_power_mean(generator_s_range_ref_spec.mean, generator_react_frac_range)
        cost_specs = self._resolve_cost_specs(generator_p_range_ref_mean, cost_a_spec, cost_b_spec, cost_c_spec)
        impedance_distribution = impedance_spec or LognormalSpec(0.01, 10)
        capacity_distribution = capacity_spec or LognormalSpec(4 * load_s_spec.mean / average_node_degree, 2)

        generated_paths = []
        skipped_infeasible = 0
        while len(generated_paths) < num_instances:
            graph = self._generate_graph(num_nodes, average_node_degree)
            self._annotate_nodes(graph, num_generators, degree_bias, load_s_spec, load_react_frac_range, generator_s_range_ref_spec, generator_s_range_len_spec,
                                 generator_react_frac_range, cost_specs, voltage_range, angle_range, symmetric_q_range)
            self._annotate_edges(graph, capacity_distribution, impedance_distribution, line_react_frac_range, scale_lines)
            if check_basic_feasibility and PowerFlowProblem.check_infeasible(graph) is not None:
                skipped_infeasible += 1
                continue
            index = len(generated_paths)
            file_path = output_path / f"{index}.pkl"
            with file_path.open("wb") as file:
                pickle.dump(graph, file)
            generated_paths.append(file_path)
        print(f"Generation complete. {skipped_infeasible} infeasible instances skipped.")
        return generated_paths

    @staticmethod
    def _get_active_power_mean(apparent_mean: float, generator_react_frac_range: tuple[float, float]) -> float:
        """Calculates expected active power from apparent-power mean and reactive-fraction range.
        :param apparent_mean: Mean apparent power ``|S|``.
        :param generator_react_frac_range: Uniform reactive-fraction range used in generator sampling.
        :return: Expected active power ``P`` corresponding to sampled ``|S|`` and reactive fraction.
        """
        lower, upper = generator_react_frac_range
        if lower == upper:
            factor = math.sqrt(1 - lower ** 2)
        else:
            primitive = lambda value: 0.5 * (value * math.sqrt(1 - value ** 2) + math.asin(value))
            factor = (primitive(upper) - primitive(lower)) / (upper - lower)
        return apparent_mean * factor

    def _resolve_cost_specs(
        self, ref_power_mean: float, cost_a_spec: LognormalSpec | None, cost_b_spec: LognormalSpec | None, cost_c_spec: LognormalSpec | None
    ) -> tuple[LognormalSpec, LognormalSpec, LognormalSpec]:
        """Resolves generator cost distribution specs, filling unspecified values with defaults.
        :param ref_power_mean: Mean reference power scale used to normalize cost defaults.
        :param cost_a_spec: Optional override for quadratic coefficient distribution.
        :param cost_b_spec: Optional override for linear coefficient distribution.
        :param cost_c_spec: Optional override for constant coefficient distribution.
        :return: Fully resolved ``(a, b, c)`` distribution specs.
        """
        base_cost = 1
        default_a = LognormalSpec(base_cost / ref_power_mean ** 2, 2)
        default_b = LognormalSpec(base_cost / ref_power_mean, 1.5)
        default_c = LognormalSpec(base_cost, 2)
        return cost_a_spec or default_a, cost_b_spec or default_b, cost_c_spec or default_c

    def _generate_graph(self, num_nodes: int, average_node_degree: float) -> Graph:
        """Creates a geometric graph and ensure edge lengths and connectivity are set.
        :param num_nodes: Number of nodes to place in the geometric graph.
        :param average_node_degree: Target average node degree for radius computation.
        :return: Connected graph with edge-length attributes.
        """
        radius = self._solve_radius(num_nodes, average_node_degree)
        graph = nx.random_geometric_graph(num_nodes, radius, seed=int(self._rng.integers(0, 2 ** 31 - 1)))
        positions = nx.get_node_attributes(graph, "pos")
        for u, v in graph.edges:
            graph.edges[u, v]["length"] = math.dist(positions[u], positions[v])
        self._connect_components(graph, positions)
        return graph

    @staticmethod
    def _solve_radius(num_nodes: int, average_node_degree: float) -> float:
        """Solves boundary-aware radius from target expected degree with a library root finder.
        :param num_nodes: Number of nodes in the random geometric graph.
        :param average_node_degree: Target expected node degree.
        :return: Radius ``r`` such that geometric graph generated with that ``r`` has specified expected degree.
        """
        if num_nodes == 1:
            assert average_node_degree == 0, f"average_node_degree must be 0 when num_nodes == 1, got {average_node_degree}."
            return 0
        max_node_degree = num_nodes - 1
        assert average_node_degree <= max_node_degree, "average_node_degree must not exceed number of nodes - 1."

        target_probability = average_node_degree / max_node_degree
        equation = lambda radius: PowerFlowProblemGenerator._distance_cdf_unit_square(radius) - target_probability
        return brentq(equation, 0, math.sqrt(2))

    @staticmethod
    def _distance_cdf_unit_square(radius: float) -> float:
        """Returns CDF ``D(r)`` of distance between two random points in the unit square.
        Formula source: https://mathworld.wolfram.com/SquareLinePicking.html
        :param radius: Distance threshold ``r``.
        :return: Probability `P(||X - Y|| <= r)` for `X, Y ~ U([0, 1]^2)`.
        """
        if radius < 0:
            return 0
        if radius <= 1:
            return 0.5 * radius ** 4 - 8 / 3 * radius ** 3 + math.pi * radius ** 2
        if radius < math.sqrt(2):
            root = math.sqrt(radius ** 2 - 1)
            return -0.5 * radius ** 4 - 4 * radius ** 2 * math.atan(root) + 4 / 3 * (2 * radius ** 2 + 1) * root + (math.pi - 2) * radius ** 2 + 1 / 3
        return 1

    def _connect_components(self, graph: Graph, positions: dict[int, tuple[float, float]]) -> None:
        """Connects disconnected components with nearest-node bridging edges.
        :param graph: Graph to modify in place.
        :param positions: Node-position mapping used for nearest-distance checks.
        """
        components = [set(component) for component in nx.connected_components(graph)]
        base_component = components[0]
        for component in components[1:]:
            u, v, length = self._closest_nodes(base_component, component, positions)
            graph.add_edge(u, v, length=length)
            base_component |= component

    @staticmethod
    def _closest_nodes(first_component: set[int], second_component: set[int], positions: dict[int, tuple[float, float]]) -> tuple[int, int, float]:
        """Finds the closest pair of nodes between two components.
        :param first_component: Node set of the first connected component.
        :param second_component: Node set of the second connected component.
        :param positions: Node-position mapping.
        :return: Closest node pair and their Euclidean distance.
        """
        best_pair: tuple[int, int] | None = None
        best_distance = math.inf
        for first_node in first_component:
            for second_node in second_component:
                distance = math.dist(positions[first_node], positions[second_node])
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (first_node, second_node)
        if best_pair is None:
            raise ValueError("Failed to find nodes to connect graph components.")
        return best_pair[0], best_pair[1], best_distance

    def _annotate_nodes(
        self,
        graph: Graph,
        num_generators: int,
        beta: float,
        load_s_spec: LognormalSpec,
        load_react_frac_range: tuple[float, float],
        generator_s_range_ref_spec: LognormalSpec,
        generator_s_range_len_spec: LognormalSpec,
        generator_react_frac_range: tuple[float, float],
        cost_specs: tuple[LognormalSpec, LognormalSpec, LognormalSpec],
        voltage_range: tuple[float, float],
        angle_range: tuple[float, float],
        symmetric_q_range: bool,
    ) -> None:
        """Annotates graph nodes with load, limits and sampled generators.
        :param graph: Graph whose nodes are updated in place.
        :param num_generators: Total number of generators to distribute over nodes.
        :param beta: Degree-bias coefficient used in placement probabilities.
        :param load_s_spec: Distribution spec for node apparent-load magnitude.
        :param load_react_frac_range: Fraction range for reactive load derivation.
        :param generator_s_range_ref_spec: Distribution spec for generator reference apparent power.
        :param generator_s_range_len_spec: Distribution spec for apparent-power range length multiplier.
        :param generator_react_frac_range: Fraction range for reactive capability derivation.
        :param cost_specs: Triplet of distribution specs for quadratic cost coefficients.
        :param voltage_range: Voltage magnitude bounds assigned to each node.
        :param angle_range: Voltage phase-angle bounds assigned to each node.
        :param symmetric_q_range: Whether to enforce symmetric generator reactive bounds around zero.
        """
        nodes = sorted(graph.nodes)
        degrees = np.array([graph.degree[node] for node in nodes], dtype=float)
        probabilities = self._generator_placement_probabilities(degrees, beta)
        generators_per_node = self._rng.multinomial(num_generators, probabilities)

        for node, count in zip(nodes, generators_per_node, strict=True):
            load_s = load_s_spec.sample(self._rng)
            load_reactive_frac = self._rng.uniform(*load_react_frac_range)
            load_q = load_s * load_reactive_frac
            load_p = load_s * math.sqrt(1 - load_reactive_frac ** 2)
            generators = [
                self._sample_generator(generator_s_range_ref_spec, generator_s_range_len_spec, generator_react_frac_range, cost_specs, symmetric_q_range)
                for _ in range(int(count))
            ]
            graph.nodes[node]["generators"] = generators
            graph.nodes[node]["load"] = complex(load_p, load_q)
            graph.nodes[node]["voltage_range"] = tuple(voltage_range)
            graph.nodes[node]["angle_range"] = tuple(angle_range)

    def _generator_placement_probabilities(self, degrees: np.ndarray, beta: float) -> np.ndarray:
        """Computes degree-biased node probabilities for generator placement.
        :param degrees: Node degree array aligned with node ordering.
        :param beta: Bias coefficient in ``[0, 1]`` controlling preference for high-degree nodes.
        :return: Normalized placement probability vector.
        """
        if len(degrees) == 0:
            raise ValueError("Graph has no nodes.")
        max_degree = np.max(degrees)
        if beta == 1:
            mask = (degrees == max_degree).astype(float)
            return mask / np.sum(mask)
        k = beta / (1 - beta)
        weights = np.exp(k * (degrees - max_degree))
        return weights / np.sum(weights)

    def _sample_generator(
        self,
        generator_s_range_ref_spec: LognormalSpec,
        generator_s_range_len_spec: LognormalSpec,
        generator_react_frac_range: tuple[float, float],
        cost_specs: tuple[LognormalSpec, LognormalSpec, LognormalSpec],
        symmetric_q_range: bool,
    ) -> Generator:
        """Samples one generator with active/reactive ranges and quadratic cost terms.
        :param generator_s_range_ref_spec: Distribution spec for reference apparent power.
        :param generator_s_range_len_spec: Distribution spec for multiplicative range length.
        :param generator_react_frac_range: Fraction range for reactive capability derivation.
        :param cost_specs: Triplet of distribution specs for quadratic cost coefficients.
        :param symmetric_q_range: Whether to enforce symmetric generator reactive bounds around zero.
        :return: Sampled generator object.
        """
        s_range_ref = generator_s_range_ref_spec.sample(self._rng)
        reactive_factor = self._rng.uniform(*generator_react_frac_range)
        q_range_ref = s_range_ref * reactive_factor
        p_range_ref = s_range_ref * math.sqrt(1 - reactive_factor ** 2)
        length_mult = 1 + generator_s_range_len_spec.sample(self._rng)
        p_min = p_range_ref / length_mult
        p_max = p_range_ref * length_mult
        q_min = q_range_ref / length_mult
        q_max = q_range_ref * length_mult
        if symmetric_q_range:
            q_abs_max = max(abs(q_min), abs(q_max))
            q_min = -q_abs_max
            q_max = q_abs_max
        a = cost_specs[0].sample(self._rng)
        b = cost_specs[1].sample(self._rng)
        c = cost_specs[2].sample(self._rng)
        return Generator(power_range=(p_min, p_max), reactive_power_range=(q_min, q_max), cost_terms=(a, b, c))

    def _annotate_edges(
        self,
        graph: Graph,
        capacity_spec: LognormalSpec,
        impedance_spec: LognormalSpec,
        line_react_frac_range: tuple[float, float],
        scale_lines: bool,
    ) -> None:
        """Annotates graph edges with admittance and current capacity attributes.
        :param graph: Graph whose edges are updated in place.
        :param capacity_spec: Distribution spec for edge current capacity.
        :param impedance_spec: Distribution spec for sampled line-impedance magnitudes ``|Z|``.
        :param line_react_frac_range: Uniform distribution range for line reactance fraction ``X / |Z|``.
        :param scale_lines: Whether to multiply sampled line resistance by ``line_length / median_length``.
        """
        lengths = [edge_data["length"] for _, _, edge_data in graph.edges(data=True)]
        median_length = np.median(lengths)

        for _, _, edge_data in graph.edges(data=True):
            impedance = impedance_spec.sample(self._rng)
            if scale_lines:
                impedance *= edge_data["length"] / median_length
            react_frac = self._rng.uniform(*line_react_frac_range)
            resistance = impedance * math.sqrt(1 - react_frac ** 2)
            reactance = impedance * react_frac
            edge_data["admittance"] = 1 / complex(resistance, reactance)
            edge_data["capacity"] = capacity_spec.sample(self._rng)
