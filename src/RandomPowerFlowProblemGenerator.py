"""Utilities for generating random annotated graphs for ``PowerFlowProblem`` instances."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from networkx import Graph
from scipy.stats import norm

from .Generator import Generator
from .PowerFlowProblem import PowerFlowProblem


@dataclass(frozen=True)
class LognormalSpec:
    """Lognormal distribution parametrized by mean and spread factor."""

    mean: float
    spread_factor: float


class RandomPowerFlowProblemGenerator:
    """Factory for random AC power-flow problem instances."""

    def __init__(self, *, spread_ref: float = 0.9, random_seed: int | None = None):
        """Initialize generator-level defaults and a reproducible random number stream."""
        self.spread_ref = spread_ref
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
        load_p_spec: LognormalSpec = LognormalSpec(10.0, 10.0),
        load_reactive_range: tuple[float, float] = (0.0, 0.1),
        generator_ref_p_spec: LognormalSpec = LognormalSpec(10.0, 10.0),
        generator_len_p_spec: LognormalSpec = LognormalSpec(0.5, 1.2),
        generator_reactive_range: tuple[float, float] = (0.0, 0.1),
        base_cost: float = 1.0,
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
        """Generate and persist random instances as graph files.

        Returns the list of output file paths in generation order.
        """

        self._validate_positive("num_generators", num_generators)
        self._validate_positive("num_instances", num_instances)
        self._validate_positive("generator_density", generator_density)
        self._validate_non_negative("average_node_degree", average_node_degree)
        self._validate_range("load_reactive_range", load_reactive_range)
        self._validate_range("generator_reactive_range)", generator_reactive_range)
        self._validate_range("line_reactive_range", line_reactive_range)
        self._validate_probability("negative_reactance_prob", negative_reactance_prob)
        self._validate_probability("degree_bias", degree_bias, include_one=True)
        self._validate_probability("spread_ref", self.spread_ref)

        num_nodes = max(1, int(math.ceil(num_generators / generator_density)))
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        cost_specs = self._resolve_cost_specs(
            generator_ref_p_spec.mean,
            base_cost,
            cost_a_spec,
            cost_b_spec,
            cost_c_spec,
        )
        capacity_distribution = capacity_spec or LognormalSpec(
            self._get_default_capacity_mean(average_node_degree, load_p_spec.mean, load_reactive_range),
            2.0,
        )

        generated_paths: list[Path] = []
        normalized_extension = "json".lstrip(".")

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
            self._annotate_edges(
                graph,
                capacity_distribution,
                base_resistance,
                min_edge_length,
                line_reactive_range,
                negative_reactance_prob,
            )

            file_path = output_path / f"{index}.{normalized_extension}"
            self._write_graph(graph, file_path)
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
        load_rpf_range: tuple[float, float],
        generator_reference_p_spec: LognormalSpec,
        generator_plfv_spec: LognormalSpec,
        generator_rpf_range: tuple[float, float],
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
            load_rpf = self._sample_uniform(load_rpf_range)
            load_q = load_p * load_rpf / math.sqrt(1.0 - load_rpf**2)

            generators = [
                self._sample_generator(generator_reference_p_spec, generator_plfv_spec, generator_rpf_range, cost_specs)
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
        median_length = float(np.median(lengths)) if lengths else 1.0
        length_floor = median_length * min_edge_length_factor

        for _, _, edge_data in graph.edges(data=True):
            length = max(float(edge_data["length"]), length_floor)
            resistance = base_resistance * length / median_length

            reactive_factor = self._sample_uniform(reactive_factor_range)
            reactance = resistance * reactive_factor / math.sqrt(1.0 - reactive_factor**2)
            if self._rng.random() < negative_reactance_probability:
                reactance = -reactance

            edge_data["admittance"] = 1.0 / complex(resistance, reactance)
            edge_data["capacity"] = self._sample_lognormal(capacity_spec)

    def _sample_generator(
        self,
        reference_p_spec: LognormalSpec,
        plfv_spec: LognormalSpec,
        rpf_range: tuple[float, float],
        cost_specs: tuple[LognormalSpec, LognormalSpec, LognormalSpec],
    ) -> Generator:
        """Sample one generator with active/reactive ranges and quadratic cost terms."""
        reference_p = self._sample_lognormal(reference_p_spec)
        plf = 1.0 + self._sample_lognormal(plfv_spec)

        p_min = reference_p / plf
        p_max = reference_p * plf

        rpf = self._sample_uniform(rpf_range)
        q_limit = p_max * rpf / math.sqrt(1.0 - rpf**2)

        a = self._sample_lognormal(cost_specs[0])
        b = self._sample_lognormal(cost_specs[1])
        c = self._sample_lognormal(cost_specs[2])

        return Generator(
            power_range=(p_min, p_max),
            reactive_power_range=(-q_limit, q_limit),
            cost_terms=(a, b, c),
        )

    def _sample_lognormal(self, spec: LognormalSpec) -> float:
        """Draw one lognormal sample from the custom ``LognormalSpec`` parameterization."""
        mu, sigma = self._lognormal_mu_sigma(spec.mean, spec.spread_factor)
        return float(self._rng.lognormal(mu, sigma))

    def _lognormal_mu_sigma(self, mean: float, spread_factor: float) -> tuple[float, float]:
        """Convert mean/spread-factor parameters into normal-space ``(mu, sigma)``."""
        self._validate_positive("lognormal.mean", mean)
        self._validate_spread_factor(spread_factor)

        if spread_factor == 1.0:
            return math.log(mean), 0.0

        quantile_probability = 0.5 * (1.0 + self.spread_ref)
        z_value = float(norm.ppf(quantile_probability))
        log_spread = math.log(spread_factor)

        discriminant = z_value**2 - 2.0 * log_spread
        if discriminant <= 0:
            sigma = log_spread / z_value
        else:
            sigma = z_value - math.sqrt(discriminant)

        mu = math.log(mean) - 0.5 * sigma**2
        return mu, sigma

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
        base_cost: float,
        cost_a_spec: LognormalSpec | None,
        cost_b_spec: LognormalSpec | None,
        cost_c_spec: LognormalSpec | None,
    ) -> tuple[LognormalSpec, LognormalSpec, LognormalSpec]:
        """Resolve generator cost distribution specs, filling unspecified values with defaults."""
        if cost_a_spec is not None and cost_b_spec is not None and cost_c_spec is not None:
            return cost_a_spec, cost_b_spec, cost_c_spec

        default_a = LognormalSpec(base_cost / reference_p_mean ** 2, 2.0)
        default_b = LognormalSpec(base_cost / reference_p_mean, 1.5)
        default_c = LognormalSpec(base_cost, 2.0)

        return (
            cost_a_spec or default_a,
            cost_b_spec or default_b,
            cost_c_spec or default_c,
        )

    def _get_default_capacity_mean(
        self,
        average_node_degree: float,
        load_p_mean: float,
        load_reactive_range: tuple[float, float],
    ) -> float:
        """Calculate default line-capacity mean from degree and expected apparent load."""
        a, b = load_reactive_range
        if a == b:
            apparent_mean = load_p_mean / math.sqrt(1.0 - a**2)
        else:
            apparent_mean = load_p_mean * (math.asin(b) - math.asin(a)) / (b - a)

        if average_node_degree == 0:
            return 2.0 * apparent_mean

        return 2.0 * apparent_mean / average_node_degree

    def _write_graph(self, graph: Graph, file_path: Path) -> None:
        """Serialize an annotated graph to JSON node-link format."""
        data = nx.node_link_data(graph)

        for node_data in data["nodes"]:
            node_data["load"] = self._serialize_complex(node_data["load"])
            node_data["generators"] = [
                {
                    "power_range": list(generator.power_range),
                    "reactive_power_range": list(generator.reactive_power_range),
                    "cost_terms": list(generator.cost_terms),
                }
                for generator in node_data["generators"]
            ]

        for edge_data in data["links"]:
            edge_data["admittance"] = self._serialize_complex(edge_data["admittance"])

        with file_path.open("w", encoding="utf-8") as output_stream:
            json.dump(data, output_stream, indent=2)

    @staticmethod
    def build_problem_from_graph(graph: Graph) -> PowerFlowProblem:
        """Build a ``PowerFlowProblem`` from an already annotated graph."""
        return PowerFlowProblem(graph)

    @staticmethod
    def _serialize_complex(value: complex) -> dict[str, float]:
        """Convert a complex value to a JSON-friendly ``{"real", "imag"}`` mapping."""
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}

    def _sample_uniform(self, value_range: tuple[float, float]) -> float:
        """Draw one uniform sample from a closed interval with degenerate-range support."""
        low, high = value_range
        if low == high:
            return float(low)
        return float(self._rng.uniform(low, high))

    @staticmethod
    def _closest_nodes(
        first_component: set[int],
        second_component: set[int],
        positions: dict[int, tuple[float, float]],
    ) -> tuple[int, int, float]:
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

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        """Validate that ``value`` is strictly positive."""
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        """Validate that ``value`` is non-negative."""
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")

    @staticmethod
    def _validate_probability(name: str, value: float, *, include_one: bool = False) -> None:
        """Validate that ``value`` lies in [0, 1) or [0, 1] depending on ``include_one``."""
        upper = 1.0 if include_one else 1.0 - np.finfo(float).eps
        if value < 0 or value > upper:
            comparator = "<= 1" if include_one else "< 1"
            raise ValueError(f"{name} must satisfy 0 <= {name} {comparator}, got {value}.")

    @staticmethod
    def _validate_range(name: str, value_range: tuple[float, float]) -> None:
        """Validate an ordered interval fully contained in [0, 1)."""
        low, high = value_range
        if low > high:
            raise ValueError(f"{name} must be an ordered range (low <= high), got {value_range}.")
        if low < 0 or high >= 1:
            raise ValueError(f"{name} must be fully within [0, 1), got {value_range}.")

    @staticmethod
    def _validate_spread_factor(spread_factor: float) -> None:
        """Validate a lognormal spread factor (must be at least 1)."""
        if spread_factor < 1:
            raise ValueError(f"lognormal spread_factor must be >= 1, got {spread_factor}.")
