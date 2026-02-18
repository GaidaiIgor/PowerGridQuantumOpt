"""Utilities for generating random annotated graphs for ``PowerFlowProblem`` instances."""

from __future__ import annotations

import math
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

    Parameters
    ----------
    mean:
        Target arithmetic mean of the sampled lognormal random variable. Must be positive.
    spread_factor:
        spread_ref fraction of samples will land in a range [mean / spread_factor, mean * spread_factor]
    spread_ref:
        Reference factor for spread_factor.
    """

    mean: float
    spread_factor: float
    spread_ref: float = 0.9
    _mu: float = field(init=False, repr=False)
    _sigma: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and precompute normal-space parameters used for sampling."""
        assert self.mean > 0, f"lognormal.mean must be positive, got {self.mean}."
        assert self.spread_factor >= 1, f"lognormal spread_factor must be >= 1, got {self.spread_factor}."
        assert 0 <= self.spread_ref < 1, f"lognormal.spread_ref must satisfy 0 <= spread_ref < 1, got {self.spread_ref}."

        if self.spread_factor == 1.0:
            mu = math.log(self.mean)
            sigma = 0.0
        else:
            quantile_probability = 0.5 * (1.0 + self.spread_ref)
            z_value = float(norm.ppf(quantile_probability))
            log_spread = math.log(self.spread_factor)

            discriminant = z_value**2 - 2.0 * log_spread
            if discriminant <= 0:
                sigma = log_spread / z_value
            else:
                sigma = z_value - math.sqrt(discriminant)

            mu = math.log(self.mean) - 0.5 * sigma**2

        self._mu = mu
        self._sigma = sigma

    @property
    def mu(self) -> float:
        """Return precomputed normal-space ``mu`` used for lognormal sampling."""
        return self._mu

    @property
    def sigma(self) -> float:
        """Return precomputed normal-space ``sigma`` used for lognormal sampling."""
        return self._sigma


class RandomPowerFlowProblemGenerator:
    """Factory for random AC power-flow problem instances."""

    def __init__(self, *, random_seed: int | None = None):
        """Initialize a reproducible random number stream."""
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
        capacity_spec: LognormalSpec | None = None
    ) -> list[Path]:
        """Generate and persist random power-flow graph instances.

        Parameters
        ----------
        num_generators:
            Total number of generators to place in each generated graph.
        num_instances:
            Number of independent instances to generate.
        output_folder:
            Destination directory where graph files are written.
        generator_density:
            Target generators-per-node ratio.
        average_node_degree:
            Target mean node degree.
        degree_bias:
            Bias in generator placement probability by node degree.
            ``0`` means no bias (all generators distributed uniformly); ``1`` means maximum bias (only max-degree have generators).
        load_p_spec:
            Lognormal spec for active-power demand at each node.
        load_reactive_range:
            Uniform sampling interval for load reactive-power factor (Q / |S|).
        generator_ref_p_spec:
            Lognormal spec for generator reference active power (geometric mean of power range).
        generator_len_p_spec:
            1 + (sample from generator_len_p_spec) will define multiplicative power range half-length around mean.
        generator_reactive_range:
            Uniform sampling interval for generator reactive-power factor (Q / |S|).
        cost_a_spec:
            Lognormal spec for quadratic cost term ``a``.
        cost_b_spec:
            Lognormal spec for linear cost term ``b``.
        cost_c_spec:
            Lognormal spec for constant cost term ``c``.
        voltage_range:
            Allowed node-voltage magnitude range.
        angle_range:
            Allowed node-voltage angle range (radians).
        base_resistance:
            Base line resistance (scaled with line length).
        min_edge_length:
            Fraction of median edge length used as a floor for effective length when scaling base_resistance.
        line_reactive_range:
            Uniform interval for edge reactive factor (|X| / |Z|).
        negative_reactance_prob:
            Probability that sampled line reactance sign is flipped.
        capacity_spec:
            Lognormal spec for edge capacity. If omitted, a default is derived from expected apparent load and graph degree.

        Returns
        -------
        list[Path]
            Output file paths in generation order.
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

        cost_specs = self._resolve_cost_specs(
            generator_ref_p_spec.mean,
            cost_a_spec,
            cost_b_spec,
            cost_c_spec,
        )
        capacity_distribution = capacity_spec or LognormalSpec(
            self._get_default_capacity_mean(average_node_degree, load_p_spec.mean, load_reactive_range),
            2.0,
        )

        generated_paths: list[Path] = []
        for index in range(num_instances):
            graph = self._generate_graph(num_nodes, average_node_degree)
            self._annotate_nodes(
                graph,
                num_generators,
                degree_bias,
                load_p_spec,
                load_reactive_range,
                generator_ref_p_spec,
                generator_len_p_spec,
                generator_reactive_range,
                cost_specs,
                voltage_range,
                angle_range,
            )
            self._annotate_edges(graph, capacity_distribution, base_resistance, min_edge_length, line_reactive_range, negative_reactance_prob)

            file_path = output_path / f"{index}.gml"
            nx.write_gml(graph, file_path, stringizer=repr)
            generated_paths.append(file_path)

        return generated_paths

    def _generate_graph(self, num_nodes: int, average_node_degree: float) -> Graph:
        """Create a geometric graph and ensure edge lengths and connectivity are set."""
        if num_nodes == 1:
            graph = nx.Graph()
            graph.add_node(0, pos=(0.5, 0.5))
            return graph

        radius = math.sqrt(average_node_degree / ((num_nodes - 1) * math.pi))
        graph = nx.random_geometric_graph(num_nodes, radius, seed=int(self._rng.integers(0, 2**31 - 1)))

        positions = nx.get_node_attributes(graph, "pos")
        for u, v in graph.edges:
            graph.edges[u, v]["length"] = self._distance(positions[u], positions[v])

        self._connect_components(graph, positions)
        return graph

    def _connect_components(self, graph: Graph, positions: dict[int, tuple[float, float]]) -> None:
        """Connect disconnected components with nearest-node bridging edges."""
        components = [set(component) for component in nx.connected_components(graph)]
        if not components:
            return

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
        """Annotate graph nodes with load, limits and sampled generators."""
        nodes = sorted(graph.nodes)
        degrees = np.array([graph.degree[node] for node in nodes], dtype=float)
        probabilities = self._generator_placement_probabilities(degrees, beta)
        generators_per_node = self._rng.multinomial(num_generators, probabilities)

        for node, count in zip(nodes, generators_per_node, strict=True):
            load_p = self._sample_lognormal(load_p_spec)
            load_reactive_frac = float(self._rng.uniform(*load_reactive_range))
            load_q = load_p * load_reactive_frac / math.sqrt(1.0 - load_reactive_frac ** 2)

            generators = [
                self._sample_generator(
                    generator_ref_p_spec,
                    generator_len_p_spec,
                    generator_reactive_range,
                    cost_specs,
                )
                for _ in range(int(count))
            ]

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
        """Annotate graph edges with admittance and current capacity attributes."""
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
            edge_data["capacity"] = self._sample_lognormal(capacity_spec)

    def _sample_generator(
        self,
        ref_p_spec: LognormalSpec,
        len_p_spec: LognormalSpec,
        reactive_range: tuple[float, float],
        cost_specs: tuple[LognormalSpec, LognormalSpec, LognormalSpec],
    ) -> Generator:
        """Sample one generator with active/reactive ranges and quadratic cost terms."""
        reference_p = self._sample_lognormal(ref_p_spec)
        length_mult = 1.0 + self._sample_lognormal(len_p_spec)

        p_min = reference_p / length_mult
        p_max = reference_p * length_mult

        reactive_factor = float(self._rng.uniform(*reactive_range))
        q_limit = p_max * reactive_factor / math.sqrt(1.0 - reactive_factor ** 2)

        a = self._sample_lognormal(cost_specs[0])
        b = self._sample_lognormal(cost_specs[1])
        c = self._sample_lognormal(cost_specs[2])

        return Generator(
            power_range=(p_min, p_max),
            reactive_power_range=(-q_limit, q_limit),
            cost_terms=(a, b, c),
        )

    def _sample_lognormal(self, spec: LognormalSpec) -> float:
        """Draw one lognormal sample from precompiled normal-space parameters."""
        return float(self._rng.lognormal(spec.mu, spec.sigma))

    def _generator_placement_probabilities(self, degrees: np.ndarray, beta: float) -> np.ndarray:
        """Compute degree-biased node probabilities for generator placement."""
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
        self,
        reference_p_mean: float,
        cost_a_spec: LognormalSpec | None,
        cost_b_spec: LognormalSpec | None,
        cost_c_spec: LognormalSpec | None,
    ) -> tuple[LognormalSpec, LognormalSpec, LognormalSpec]:
        """Resolve generator cost distribution specs, filling unspecified values with defaults."""
        base_cost = 1.0
        default_a = LognormalSpec(base_cost / reference_p_mean ** 2, 2.0)
        default_b = LognormalSpec(base_cost / reference_p_mean, 1.5)
        default_c = LognormalSpec(base_cost, 2.0)

        return (
            cost_a_spec or default_a,
            cost_b_spec or default_b,
            cost_c_spec or default_c,
        )

    def _get_default_capacity_mean(self, average_node_degree: float, load_p_mean: float, load_reactive_range: tuple[float, float]) -> float:
        """Calculate default line-capacity mean from degree and expected apparent load."""
        a, b = load_reactive_range
        if a == b:
            apparent_load_mean = load_p_mean / math.sqrt(1.0 - a ** 2)
        else:
            apparent_load_mean = load_p_mean * (math.asin(b) - math.asin(a)) / (b - a)
        return 2.0 * apparent_load_mean / average_node_degree

    @staticmethod
    def _closest_nodes(first_component: set[int], second_component: set[int], positions: dict[int, tuple[float, float]]) -> tuple[int, int, float]:
        """Find the closest pair of nodes between two components."""
        best_pair: tuple[int, int] | None = None
        best_distance = math.inf
        for first_node in first_component:
            for second_node in second_component:
                distance = RandomPowerFlowProblemGenerator._distance(positions[first_node], positions[second_node])
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (first_node, second_node)
        if best_pair is None:
            raise ValueError("Failed to find nodes to connect graph components.")
        return best_pair[0], best_pair[1], best_distance

    @staticmethod
    def _distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
        """Compute Euclidean distance between two points."""
        return float(math.dist(point1, point2))
