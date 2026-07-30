from ruliology_forge.atlas import (
    AtlasConfig,
    aggregate_rule_profiles,
    atlas_markdown,
    build_resilience_atlas,
)
from ruliology_forge.benchmarks import BenchmarkScenario, BenchmarkSuite, timing_suite


def test_timing_suite_varies_only_time():
    suite = timing_suite(steps=41, interval=10, perturb_radius=2)
    assert [scenario.perturb_time for scenario in suite.scenarios] == [10, 20, 30, 40]
    signatures = {
        (
            scenario.perturbation,
            scenario.perturb_radius,
            scenario.initial_condition,
            scenario.initial_density,
            scenario.noise_probability,
            scenario.boundary,
        )
        for scenario in suite.scenarios
    }
    assert len(signatures) == 1


def test_tiny_atlas_contains_features_profiles_and_correlations():
    suite = BenchmarkSuite(
        "tiny-atlas",
        (
            BenchmarkScenario("early", perturb_time=4, perturb_radius=1),
            BenchmarkScenario("late", perturb_time=8, perturb_radius=1),
        ),
    )
    result = build_resilience_atlas(
        AtlasConfig(rules=(0, 90), repeats=2, width=21, steps=12, seed=7),
        suite=suite,
    )
    assert len(result.observations) == 8
    assert len(result.rule_profiles) == 2
    assert result.observations[0]["control_density"] >= 0
    assert "control_trivial_convergence" in result.observations[0]
    assert result.correlations
    assert result.manifest["suite_fingerprint"] == suite.fingerprint()
    assert "Ruliology Forge Resilience Atlas" in atlas_markdown(result)


def test_aggregate_profiles_flags_trivial_convergence():
    rows = [
        {
            "rule": 0,
            "scenario": "a",
            "restoration_coefficient": 1.0,
            "shift_tolerant_restoration": 1.0,
            "scar_duration": 0,
            "final_scar_size": 0,
            "peak_divergence": 0.0,
            "recovery_time": 0,
            "recovered": True,
            "control_trivial_convergence": True,
            "control_density": 0.0,
            "control_row_entropy_mean": 0.0,
            "control_row_entropy_std": 0.0,
            "control_temporal_activity": 0.0,
            "control_horizontal_edge_density": 0.0,
            "control_bilateral_symmetry": 1.0,
            "control_mean_components": 0.0,
            "control_max_components": 0,
            "control_occupied_fraction": 0.0,
            "control_compression_ratio": 0.1,
        }
    ]
    profile = aggregate_rule_profiles(rows)[0]
    assert profile["trivial_convergence_probability"] == 1.0
