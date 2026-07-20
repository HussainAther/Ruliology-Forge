from ruliology_forge.analysis import parameter_grid, summarize_scan
from ruliology_forge.experiments import ExperimentConfig, run_parameter_sweep


def test_parameter_grid_cartesian_product():
    grid = parameter_grid(rule=[30, 90], perturb_radius=[1, 2])
    assert len(grid) == 4
    assert {row["rule"] for row in grid} == {30, 90}


def test_summarize_scan():
    rows = [
        {"rule": 30, "restoration_coefficient": 0.8, "recovered": True, "recovery_time": 2,
         "peak_divergence": 0.2, "final_scar_size": 1},
        {"rule": 30, "restoration_coefficient": 0.6, "recovered": False, "recovery_time": None,
         "peak_divergence": 0.4, "final_scar_size": 3},
    ]
    summary = summarize_scan(rows)[0]
    assert summary["n"] == 2
    assert summary["mean_restoration"] == 0.7
    assert summary["recovery_probability"] == 0.5


def test_small_parameter_sweep():
    config = ExperimentConfig(rule=90, width=31, steps=20, perturb_time=5, perturb_radius=1)
    rows = run_parameter_sweep([config], repeats=2, jobs=1)
    assert len(rows) == 2
    assert all(row["rule"] == 90 for row in rows)
