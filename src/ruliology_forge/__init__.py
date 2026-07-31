"""Ruliology Forge: computational resilience tools for rule-based systems."""

from .analysis import (
    AtlasAnalysisConfig,
    AtlasAnalysisResult,
    analyze_atlas_directory,
    classify_rule_profiles,
    identify_statistical_anomalies,
    rank_rule_categories,
)
from .atlas import (
    AtlasConfig,
    AtlasResult,
    aggregate_rule_profiles,
    build_resilience_atlas,
)
from .benchmarks import (
    BenchmarkScenario,
    BenchmarkSuite,
    benchmark_rules,
    rank_benchmark_rows,
    standard_suite,
    timing_suite,
)
from .eca import ECARule, initial_state, simulate_eca
from .experiments import (
    ExperimentConfig,
    PerturbationResult,
    run_perturbation_experiment,
    scan_rules,
)
from .exporters import (
    BundleFiles,
    CoordinateSystem,
    HOUDINI_SCHEMA_VERSION,
    HoudiniBundleManifest,
    export_houdini_bundle,
)
from .metrics import (
    ResilienceSummary,
    divergence_auc,
    final_restoration,
    hamming_divergence,
    recovery_time,
    restoration_coefficient,
    scar_metrics,
    shift_tolerant_divergence,
    summarize_resilience,
    xor_difference,
)
from .perturb import perturb_state
from .plotting import plot_divergence, plot_trajectory

__all__ = [
    "AtlasAnalysisConfig",
    "AtlasAnalysisResult",
    "AtlasConfig",
    "AtlasResult",
    "BenchmarkScenario",
    "BenchmarkSuite",
    "BundleFiles",
    "CoordinateSystem",
    "ECARule",
    "ExperimentConfig",
    "HOUDINI_SCHEMA_VERSION",
    "HoudiniBundleManifest",
    "PerturbationResult",
    "ResilienceSummary",
    "aggregate_rule_profiles",
    "analyze_atlas_directory",
    "benchmark_rules",
    "build_resilience_atlas",
    "classify_rule_profiles",
    "divergence_auc",
    "export_houdini_bundle",
    "final_restoration",
    "hamming_divergence",
    "identify_statistical_anomalies",
    "initial_state",
    "perturb_state",
    "plot_divergence",
    "plot_trajectory",
    "rank_benchmark_rows",
    "rank_rule_categories",
    "recovery_time",
    "restoration_coefficient",
    "run_perturbation_experiment",
    "scan_rules",
    "scar_metrics",
    "shift_tolerant_divergence",
    "simulate_eca",
    "standard_suite",
    "summarize_resilience",
    "timing_suite",
    "xor_difference",
]