# Ruliology Forge

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)]()

**Ruliology Forge** is an open-source Python toolkit for studying computational resilience in rule-based systems.

The first release focuses on Elementary Cellular Automata (ECA): simulating rules, applying localized perturbations, comparing control and perturbed trajectories, and measuring recovery with a Restoration Coefficient.

This repository is intentionally scoped as a clean, reproducible starter toolkit. Broader biological, morphogenetic, and shell-pattern work can be added later as optional research modules once datasets and claims are curated.

## Why this exists

Many cellular automata studies focus on growth: what pattern does a rule produce from an initial condition?

Ruliology Forge asks a complementary question:

> What happens when a rule-generated pattern is disrupted?

Some rules rapidly return to their baseline behavior. Others diverge permanently. Some preserve global structure while carrying localized scars. This toolkit provides the machinery to explore those regimes.

## Current features

- Elementary Cellular Automata simulation for rules 0 through 255
- Single-cell and random initial conditions
- Periodic or fixed boundary conditions
- Perturbation operators:
  - bit flip
  - void / zeroing
  - random mix
- Control vs perturbed trajectory comparison
- XOR difference maps
- Normalized Hamming divergence
- Restoration Coefficient `R`
- Rule-space scans
- Basic plotting helpers
- Tests and example scripts

## Installation

```bash
git clone https://github.com/HussainAther/ruliology-forge.git
cd ruliology-forge
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

## Quick start

```python
from ruliology_forge import run_perturbation_experiment, plot_trajectory

result = run_perturbation_experiment(
    rule=110,
    width=201,
    steps=200,
    perturb_time=80,
    perturb_radius=5,
    perturbation="bit_flip",
)

print(result.restoration_coefficient)
plot_trajectory(result.control, title="Rule 110 control")
plot_trajectory(result.difference, title="Rule 110 XOR difference")
```

## Command-line usage

Scan all 256 ECA rules:

```bash
ruliology scan --output results/eca_scan.csv
```

Run a single perturbation experiment and save figures:

```bash
ruliology experiment --rule 110 --output-dir results/rule110
```

## Repository structure

```text
ruliology-forge/
  README.md
  LICENSE
  CITATION.cff
  pyproject.toml
  src/ruliology_forge/
    __init__.py
    eca.py
    perturb.py
    metrics.py
    experiments.py
    plotting.py
    cli.py
  examples/
    quickstart_rule110.py
    scan_all_rules.py
  tests/
    test_eca.py
    test_metrics.py
    test_experiments.py
  docs/
    project_plan.md
    first_issues.md
  results/
    .gitkeep
```

## Core metric

The normalized divergence at time `t` is:

```text
D(t) = HammingDistance(control[t], perturbed[t]) / lattice_width
```

The Restoration Coefficient is:

```text
R = 1 - mean(D(t))
```

where the mean is taken over the post-perturbation recovery window.

Interpretation:

- `R = 1.0`: exact restoration / no divergence
- `R near 0.0`: persistent divergence
- intermediate `R`: partial recovery, scarring, or structured deviation

## Roadmap

### v0.1 — ECA resilience toolkit

- Stable ECA engine
- Perturbation experiments
- Restoration metrics
- Rule scanning
- Figures and CSV export
- Tests and documentation

### v0.2 — richer automata

- Totalistic automata
- 2D Life-like automata
- Additional perturbation geometries
- Shift-tolerant restoration metrics

### v0.3 — research workflows

- Batch experiment configs
- Parallel scans
- Reproducible figure pipelines
- Dataset export
- Notebook tutorials

### Future research directions

- Morphogenesis-inspired repair models
- Artificial life experiments
- Shell-pattern simulations
- Repair-aware computational architectures
- Rule-space maps of robustness and fragility

## Good first issues

See [`docs/first_issues.md`](docs/first_issues.md).

## Citation

If you use this project in research, please cite the repository using [`CITATION.cff`](CITATION.cff).

## License

MIT License. See [`LICENSE`](LICENSE).
