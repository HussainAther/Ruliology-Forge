# Computational Restoration in Elementary Cellular Automata: code and data

Paper-specific benchmark code and data for the Ruliology-Forge repository. This folder is self-contained
and independent of the `ruliology_forge` package: the manuscript's results come from the standalone scripts here.

Code and outputs for the manuscript "Computational Restoration in Elementary Cellular
Automata: A Quantitative Benchmark of Recovery Across Rule Space."

**Citation / DOI:** [ADD after the tagged release is archived]
**License:** MIT for code (see the repository `LICENSE`). Data and derived tables: [ADD if you want a separate data license, e.g. CC BY 4.0]

## Quick check (about 15 seconds)

    pip install -r requirements.txt
    python scripts/verify_reproduction.py

This re-runs the primary 98,304-trial benchmark and the 88-class sensitivity analysis and
confirms that every shipped value is reproduced.

## Layout

| Path | Contents |
|---|---|
| `scripts/run_restoration_benchmark_v2.py` | Primary benchmark (NumPy, pandas, SciPy). Defaults reproduce the primary protocol: width 200, 40 post-perturbation steps, perturbation width 20, perturbation times 25/50/75, 128 trials per rule per time, periodic boundary, random-replacement operator, density 0.5, seed 20260816. |
| `scripts/run_restoration_benchmark_v2_filtered.py` | Identical to the script above plus a `--rules` option to run a subset of rules (used for the Section 5.9 subset reruns). Defaults and per-rule computation are unchanged. |
| `scripts/eca_equivalence_class_analysis.py` | 88-class sensitivity analysis (Sections 4.5 and 5.6). |
| `scripts/bootstrap_ci_comparison_script.py` | Wald vs. bootstrap confidence intervals for headline rules. |
| `scripts/generate_restoration_animation.py` | Supplementary Movie S1 (needs matplotlib; not re-run during verification). |
| `scripts/verify_reproduction.py` | Reproduction check described above. |
| `data/primary_benchmark/` | The four primary CSV outputs: `rule_metrics_aggregated.csv` (256 rows), `rule_metrics_by_perturbation_time.csv` (768 rows), `symmetry_validation.csv`, `timing_rank_stability.csv`. |
| `data/analysis_outputs/` | Derived tables: Wolfram-class comparison (`wolfram_class_comparison.xlsx`, `rule_metrics_with_wolfram_class.csv`), `equivalence_class_sensitivity.csv`, bootstrap comparison, symmetry summaries, empirical grouping, correlations. |
| `data/confirmatory_5_9/` | Outputs of the three Section 5.9 confirmatory reruns. |
| `MANIFEST.sha256` | SHA-256 checksum of every file. Verify with `sha256sum -c MANIFEST.sha256`. |

## Commands

Run from this folder (`paper/eca_restoration/` in the Ruliology-Forge repository).

    # Primary benchmark (about 15 s)
    python scripts/run_restoration_benchmark_v2.py --out out_primary

    # Section 5.9 confirmatory reruns (same seed, only the stated parameter changed)
    python scripts/run_restoration_benchmark_v2_filtered.py --out out_neg --trials 2048                    # about 2.5 min
    python scripts/run_restoration_benchmark_v2_filtered.py --out out_sym --trials 4096 --rules 14,84
    python scripts/run_restoration_benchmark_v2_filtered.py --out out_ext --post-steps 200 --rules 2,4,57,161

    # 88-class sensitivity analysis
    python scripts/eca_equivalence_class_analysis.py data/analysis_outputs/rule_metrics_with_wolfram_class.csv \
        --metric-col relative_restoration_mean --out equivalence_class_sensitivity.csv

    # Bootstrap comparison for headline rules 105, 161, 57, 36
    python scripts/bootstrap_ci_comparison_script.py

Row order can differ from the shipped file when `--rules` lists rules in a different order
(the shipped symmetry-outlier output lists rule 84 before 14); sorted by rule and
perturbation time the values are identical.

## Notes

* All random streams derive deterministically from the seed (`seed + 1000003 * rule` for
  initial conditions, plus `7919 * perturb_time` for the perturbation stream). The confirmatory
  reruns reuse the primary seed, so larger samples can contain the primary-benchmark trials as
  subsets (manuscript Section 5.9).
* Verified with Python 3.12.3, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1. The primary benchmark,
  all three confirmatory reruns and the bootstrap table were re-run and matched the shipped
  files (bootstrap values to the 6 decimals stored in the CSV).
* The rule-to-Wolfram-class assignment is stored in `rule_metrics_with_wolfram_class.csv`
  (column `wolfram_class`); the source is recorded in `wolfram_class_comparison.xlsx`
  ("Key Findings"). Equivalence classes (reflection and 0/1 conjugation) are computed in
  `eca_equivalence_class_analysis.py` and give 88 classes (8 of size 1, 36 of size 2, 44 of size 4).
