from dataclasses import dataclass

@dataclass(slots=True)
class RuleProfile:
    rule: int

    entropy: float
    symmetry: float
    complexity: float

    restoration: float
    recovery_probability: float
    recovery_time: float

    scar_size: float
    scar_duration: float

    edge_density: float
    activity: float

    nearest_rules: list[int]

    benchmark_score: float
