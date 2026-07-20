# Changelog

## 0.3.0

- Added grouped scan summaries with confidence intervals and recovery rates.
- Added deterministic Cartesian parameter grids.
- Added parameter sweeps with optional process-based parallel execution.
- Added an evolutionary search over the 256 elementary cellular automata rules.
- Added `summarize`, `sweep`, and `evolve` CLI commands.
- Added tests for aggregation, parameter grids, and repeated sweeps.

## 0.2.0

- Added reproducible perturbation timing and independent random streams.
- Added scar and shift-tolerant resilience metrics.
- Added richer metadata and trajectory exports.

## 0.4.0 - Benchmarking and robust rankings

- Added named, fingerprinted benchmark suites.
- Added a balanced standard ECA resilience benchmark.
- Added cross-scenario robust rule ranking with a transparent score.
- Added benchmark manifests, raw observations, rankings, and Markdown reports.
- Added the `ruliology benchmark` CLI command.
- Added benchmark API tests and deterministic suite fingerprinting.
