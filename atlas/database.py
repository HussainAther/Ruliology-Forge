@dataclass(frozen=True)
class RuleProfile:
    rule: int
    width: int
    steps: int
    seed: int | None
    restoration_coefficient: float
    final_restoration: float
    recovery_probability: float
    density: float
    row_entropy_mean: float
    row_entropy_std: float
    temporal_activity: float
    horizontal_edge_density: float
    bilateral_symmetry: float
    mean_components: float
    max_components: int
    occupied_fraction: float
    compression_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
