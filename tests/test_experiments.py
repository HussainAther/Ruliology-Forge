from ruliology_forge import run_perturbation_experiment, scan_rules


def test_run_experiment_shapes():
    result = run_perturbation_experiment(rule=110, width=51, steps=40, perturb_time=10)
    assert result.control.shape == (40, 51)
    assert result.perturbed.shape == (40, 51)
    assert result.difference.shape == (40, 51)
    assert result.divergence.shape == (40,)
    assert 0 <= result.restoration_coefficient <= 1


def test_scan_rules_small_range():
    rows = scan_rules([0, 90, 110], width=51, steps=40, perturb_time=10)
    assert len(rows) == 3
    assert rows[0]["rule"] == 0
    assert "restoration_coefficient" in rows[0]
