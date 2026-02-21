"""Utilities for generating random annotated graphs for ``PowerFlowProblem`` instances."""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from networkx import Graph
from scipy.stats import norm

from .Generator import Generator
from .validation import validate_bounds


@dataclass
class LognormalSpec:
    """Lognormal sampling configuration.
    :var mean: Target mean of the lognormal distribution.
    :var spread_factor: Multiplicative half-width around mean used for spread calibration.
    :var spread_ref: Probability mass inside ``[mean / spread_factor, mean * spread_factor]``.
    :var _mu: Precomputed normal-space mean parameter.
    :var _sigma: Precomputed normal-space standard deviation parameter.
    """

    mean: float
    spread_factor: float
    spread_ref: float = 0.9
    _mu: float = field(init=False, repr=False)
    _sigma: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validates and precomputes normal-space parameters used for sampling."""
        if self.mean <= 0:
            raise ValueError(f"lognormal.mean must be positive, got {self.mean}.")
        if self.spread_factor < 1:
            raise ValueError(f"lognormal spread_factor must be >= 1, got {self.spread_factor}.")
        if not (0 < self.spread_ref <= 1):
            raise ValueError(f"lognormal.spread_ref must satisfy 0 < spread_ref <= 1, got {self.spread_ref}.")
        if self.spread_factor == 1.0 and self.spread_ref != 1.0:
            raise ValueError(f"lognormal.spread_ref must be 1 when spread_factor is 1, got {self.spread_ref}.")

        if self.spread_factor == 1.0:
            sigma = 0.0
        else:
            log_spread = math.log(self.spread_factor)

            def interval_mass(sigma: float) -> float:
                z_hi = 0.5 * sigma + log_spread / sigma
                z_low = 0.5 * sigma - log_spread / sigma
                return float(norm.cdf(z_hi) - norm.cdf(z_low))

            low, high = 0.0, 1.0
            while interval_mass(high) > self.spread_ref:
                high *= 2.0
            for _ in range(100):
                sigma = 0.5 * (low + high)
                if interval_mass(sigma) > self.spread_ref:
                    low = sigma
                else:
                    high = sigma
            sigma = 0.5 * (low + high)

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
        return float(rng.lognormal(self.mu, self.sigma))


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
        *,
        num_instances: int = 1,
        output_folder: str | Path = "data",
        generator_density: float = 2.0,
        average_node_degree: float = 3.0,
        degree_bias: float = 0.3,
        load_p_spec: LognormalSpec | None = None,
        load_reactive_range: tuple[float, float] = (0.0, 0.1),
        generator_ref_p_spec: LognormalSpec | None = None,
        generator_len_p_spec: LognormalSpec | None = None,
        generator_reactive_range: tuple[float, float] = (0.0, 0.1),
        cost_a_spec: LognormalSpec | None = None,
        cost_b_spec: LognormalSpec | None = None,
        cost_c_spec: LognormalSpec | None = None,
        voltage_range: tuple[float, float] = (0.95, 1.05),
        angle_range: tuple[float, float] = (-math.pi, math.pi),
        base_resistance: float = 0.01,
        min_edge_length: float = 0.01,
        line_reactive_range: tuple[float, float] = (0.95, 0.999),
        negative_reactance_prob: float = 0.0,
        capacity_spec: LognormalSpec | None = None,
    ) -> list[Path]:
        """Generates and persists random power-flow graph instances.
        :param num_generators: Total number of generators to place across all nodes.
        :param num_instances: Number of independent graph instances to generate.
        :param output_folder: Destination directory for serialized graph files.
        :param generator_density: Target ratio of generators per node used to infer node count.
        :param average_node_degree: Target average degree for geometric graph radius selection.
        :param degree_bias: Bias strength toward high-degree nodes during generator placement.
        :param load_p_spec: Distribution spec for active load on each node.
        :param load_reactive_range: Fraction range used to derive reactive load from active load.
        :param generator_ref_p_spec: Distribution spec for generator reference active power.
        :param generator_len_p_spec: Distribution spec for multiplicative active-power range length.
        :param generator_reactive_range: Fraction range used to derive reactive limits from active max.
        :param cost_a_spec: Distribution spec for quadratic cost coefficient ``a``.
        :param cost_b_spec: Distribution spec for linear cost coefficient ``b``.
        :param cost_c_spec: Distribution spec for constant cost coefficient ``c``.
        :param voltage_range: Allowed voltage magnitude range for every node.
        :param angle_range: Allowed voltage phase-angle range for every node.
        :param base_resistance: Reference line resistance used before distance scaling.
        :param min_edge_length: Minimum relative edge-length floor factor.
        :param line_reactive_range: Fraction range used to derive line reactance from resistance.
        :param negative_reactance_prob: Probability of sampling negative reactance for an edge.
        :param capacity_spec: Distribution spec for line current capacities.
        :return: Paths to serialized generated graph files.
        """
        assert num_generators > 0, f"num_generators must be positive, got {num_generators}."
        assert num_instances > 0, f"num_instances must be positive, got {num_instances}."
        assert generator_density > 0, f"generator_density must be positive, got {generator_density}."
        assert average_node_degree >= 0, f"average_node_degree must be non-negative, got {average_node_degree}."
        validate_bounds("load_reactive_range", load_reactive_range, min_value=0.0, max_value=1.0)
        validate_bounds("generator_reactive_range", generator_reactive_range, min_value=0.0, max_value=1.0)
        validate_bounds("line_reactive_range", line_reactive_range, min_value=0.0, max_value=1.0, include_max=True)
        validate_bounds("negative_reactance_prob", negative_reactance_prob, min_value=0.0, max_value=1.0, include_max=True)
        validate_bounds("degree_bias", degree_bias, min_value=0.0, max_value=1.0, include_max=True)

        load_p_spec = load_p_spec or LognormalSpec(10.0, 10.0)
        generator_ref_p_spec = generator_ref_p_spec or load_p_spec
        generator_len_p_spec = generator_len_p_spec or LognormalSpec(0.5, 1.2)
        num_nodes = max(1, int(math.ceil(num_generators / generator_density)))
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        cost_specs = self._resolve_cost_specs(generator_ref_p_spec.mean, cost_a_spec, cost_b_spec, cost_c_spec)
        capacity_distribution = capacity_spec or LognormalSpec(self._get_default_capacity_mean(average_node_degree, load_p_spec.mean, load_reactive_range), 2.0)

        generated_paths: list[Path] = []
        for index in range(num_instances):
            graph = self._generate_graph(num_nodes, average_node_degree)
            self._annotate_nodes(graph, num_generators, degree_bias, load_p_spec, load_reactive_range, generator_ref_p_spec, generator_len_p_spec,
                                 generator_reactive_range, cost_specs, voltage_range, angle_range)
            self._annotate_edges(graph, capacity_distribution, base_resistance, min_edge_length, line_reactive_range, negative_reactance_prob)
            file_path = output_path / f"{index}.pkl"
            with file_path.open("wb") as file:
                pickle.dump(graph, file)
            generated_paths.append(file_path)
        return generated_paths

    def _generate_graph(self, num_nodes: int, average_node_degree: float) -> Graph:
        """Creates a geometric graph and ensure edge lengths and connectivity are set.
        :param num_nodes: Number of nodes to place in the geometric graph.
        :param average_node_degree: Target average node degree for radius computation.
        :return: Connected graph with edge-length attributes.
        """
        radius = math.sqrt(average_node_degree / ((num_nodes - 1) * math.pi))
        graph = nx.random_geometric_graph(num_nodes, radius, seed=int(self._rng.integers(0, 2 ** 31 - 1)))
        positions = nx.get_node_attributes(graph, "pos")
        for u, v in graph.edges:
            graph.edges[u, v]["length"] = self._distance(positions[u], positions[v])
        self._connect_components(graph, positions)
        return graph

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

    def _annotate_nodes(
        self,
        graph: Graph,
        num_generators: int,
        beta: float,
        load_p_spec: LognormalSpec,
        load_reactive_range: tuple[float, float],
        generator_ref_p_spec: LognormalSpec,
        generator_len_p_spec: LognormalSpec,
        generator_reactive_range: tuple[float, float],
        cost_specs: tuple[LognormalSpec, LognormalSpec, LognormalSpec],
        voltage_range: tuple[float, float],
        angle_range: tuple[float, float],
    ) -> None:
        """Annotates graph nodes with load, limits and sampled generators.
        :param graph: Graph whose nodes are updated in place.
        :param num_generators: Total number of generators to distribute over nodes.
        :param beta: Degree-bias coefficient used in placement probabilities.
        :param load_p_spec: Distribution spec for node active load.
        :param load_reactive_range: Fraction range for reactive load derivation.
        :param generator_ref_p_spec: Distribution spec for generator reference active power.
        :param generator_len_p_spec: Distribution spec for active-power range length multiplier.
        :param generator_reactive_range: Fraction range for reactive capability derivation.
        :param cost_specs: Triplet of distribution specs for quadratic cost coefficients.
        :param voltage_range: Voltage magnitude bounds assigned to each node.
        :param angle_range: Voltage phase-angle bounds assigned to each node.
        """
        nodes = sorted(graph.nodes)
        degrees = np.array([graph.degree[node] for node in nodes], dtype=float)
        probabilities = self._generator_placement_probabilities(degrees, beta)
        generators_per_node = self._rng.multinomial(num_generators, probabilities)

        for node, count in zip(nodes, generators_per_node, strict=True):
            load_p = load_p_spec.sample(self._rng)
            load_reactive_frac = float(self._rng.uniform(*load_reactive_range))
            load_q = load_p * load_reactive_frac / math.sqrt(1.0 - load_reactive_frac ** 2)
            generators = [self._sample_generator(generator_ref_p_spec, generator_len_p_spec, generator_reactive_range, cost_specs) for _ in range(int(count))]
            graph.nodes[node]["generators"] = generators
            graph.nodes[node]["load"] = complex(load_p, load_q)
            graph.nodes[node]["voltage_range"] = tuple(voltage_range)
            graph.nodes[node]["angle_range"] = tuple(angle_range)

    def _annotate_edges(
        self,
        graph: Graph,
        capacity_spec: LognormalSpec,
        base_resistance: float,
        min_edge_length_factor: float,
        reactive_factor_range: tuple[float, float],
        negative_reactance_probability: float,
    ) -> None:
        """Annotates graph edges with admittance and current capacity attributes.
        :param graph: Graph whose edges are updated in place.
        :param capacity_spec: Distribution spec for edge current capacity.
        :param base_resistance: Reference resistance used before length scaling.
        :param min_edge_length_factor: Floor factor applied to median edge length.
        :param reactive_factor_range: Fraction range for reactance derivation from resistance.
        :param negative_reactance_probability: Probability of flipping reactance sign.
        """
        lengths = [edge_data["length"] for _, _, edge_data in graph.edges(data=True)]
        median_length = float(np.median(lengths))
        length_floor = median_length * min_edge_length_factor

        for _, _, edge_data in graph.edges(data=True):
            length = max(float(edge_data["length"]), length_floor)
            resistance = base_resistance * length / median_length
            reactive_factor = float(self._rng.uniform(*reactive_factor_range))
            reactance = resistance * reactive_factor / math.sqrt(1.0 - reactive_factor ** 2)
            if self._rng.random() < negative_reactance_probability:
                reactance = -reactance
            edge_data["admittance"] = 1.0 / complex(resistance, reactance)
            edge_data["capacity"] = capacity_spec.sample(self._rng)

    def _sample_generator(
        self,
        ref_p_spec: LognormalSpec,
        len_p_spec: LognormalSpec,
        reactive_range: tuple[float, float],
        cost_specs: tuple[LognormalSpec, LognormalSpec, LognormalSpec],
    ) -> Generator:
        """Samples one generator with active/reactive ranges and quadratic cost terms.
        :param ref_p_spec: Distribution spec for reference active power.
        :param len_p_spec: Distribution spec for multiplicative range length.
        :param reactive_range: Fraction range for reactive capability derivation.
        :param cost_specs: Triplet of distribution specs for quadratic cost coefficients.
        :return: Sampled generator object.
        """
        reference_p = ref_p_spec.sample(self._rng)
        length_mult = 1.0 + len_p_spec.sample(self._rng)
        p_min = reference_p / length_mult
        p_max = reference_p * length_mult
        reactive_factor = float(self._rng.uniform(*reactive_range))
        q_limit = p_max * reactive_factor / math.sqrt(1.0 - reactive_factor ** 2)
        a = cost_specs[0].sample(self._rng)
        b = cost_specs[1].sample(self._rng)
        c = cost_specs[2].sample(self._rng)
        return Generator(power_range=(p_min, p_max), reactive_power_range=(-q_limit, q_limit), cost_terms=(a, b, c))

    def _generator_placement_probabilities(self, degrees: np.ndarray, beta: float) -> np.ndarray:
        """Computes degree-biased node probabilities for generator placement.
        :param degrees: Node degree array aligned with node ordering.
        :param beta: Bias coefficient in ``[0, 1]`` controlling preference for high-degree nodes.
        :return: Normalized placement probability vector.
        """
        if len(degrees) == 0:
            raise ValueError("Graph has no nodes.")
        max_degree = float(np.max(degrees))
        if beta == 1.0:
            mask = (degrees == max_degree).astype(float)
            return mask / np.sum(mask)
        k = beta / (1.0 - beta)
        weights = np.exp(k * (degrees - max_degree))
        return weights / np.sum(weights)

    def _resolve_cost_specs(
        self, reference_p_mean: float, cost_a_spec: LognormalSpec | None, cost_b_spec: LognormalSpec | None, cost_c_spec: LognormalSpec | None
    ) -> tuple[LognormalSpec, LognormalSpec, LognormalSpec]:
        """Resolves generator cost distribution specs, filling unspecified values with defaults.
        :param reference_p_mean: Mean reference active power used to scale defaults.
        :param cost_a_spec: Optional override for quadratic coefficient distribution.
        :param cost_b_spec: Optional override for linear coefficient distribution.
        :param cost_c_spec: Optional override for constant coefficient distribution.
        :return: Fully resolved ``(a, b, c)`` distribution specs.
        """
        base_cost = 1.0
        default_a = LognormalSpec(base_cost / reference_p_mean ** 2, 2.0)
        default_b = LognormalSpec(base_cost / reference_p_mean, 1.5)
        default_c = LognormalSpec(base_cost, 2.0)
        return cost_a_spec or default_a, cost_b_spec or default_b, cost_c_spec or default_c

    def _get_default_capacity_mean(self, average_node_degree: float, load_p_mean: float, load_reactive_range: tuple[float, float]) -> float:
        """Calculates default line-capacity mean from degree and expected apparent load.
        :param average_node_degree: Target average node degree.
        :param load_p_mean: Mean active load used in expectation calculation.
        :param load_reactive_range: Fraction range used to derive apparent-load expectation.
        :return: Default mean capacity value.
        """
        a, b = load_reactive_range
        if a == b:
            apparent_load_mean = load_p_mean / math.sqrt(1.0 - a ** 2)
        else:
            apparent_load_mean = load_p_mean * (math.asin(b) - math.asin(a)) / (b - a)
        return 2.0 * apparent_load_mean / average_node_degree

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
                distance = PowerFlowProblemGenerator._distance(positions[first_node], positions[second_node])
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (first_node, second_node)
        if best_pair is None:
            raise ValueError("Failed to find nodes to connect graph components.")
        return best_pair[0], best_pair[1], best_distance

    @staticmethod
    def _distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """Computes Euclidean distance between two points.
        :param point1: First 2D point.
        :param point2: Second 2D point.
        :return: Euclidean distance between points.
        """
        return float(math.dist(point1, point2))
