from dataclasses import dataclass


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
