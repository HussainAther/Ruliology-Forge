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


def test_scan_repeats_and_seed_are_reproducible():
    kwargs = dict(
        width=31,
        steps=25,
        perturb_time=5,
        perturbation="random_mix",
        initial_condition="random",
        repeats=2,
        seed=123,
    )
    first = scan_rules([30], **kwargs)
    second = scan_rules([30], **kwargs)
    assert first == second
    assert len(first) == 2
    assert first[0]["repeat"] == 0
    assert "recovery_time" in first[0]


def test_after_update_marks_exact_perturbation_row():
    result = run_perturbation_experiment(
        rule=0,
        width=9,
        steps=8,
        perturb_time=3,
        perturbation="bit_flip",
        perturb_radius=0,
        perturbation_timing="after_update",
    )
    assert result.difference[:3].sum() == 0
    assert result.difference[3].sum() == 1


def test_seed_streams_are_separate_and_reproducible():
    kwargs = dict(
        rule=30,
        width=31,
        steps=20,
        perturb_time=5,
        initial_condition="random",
        perturbation="random_mix",
        seed=777,
    )
    first = run_perturbation_experiment(**kwargs)
    second = run_perturbation_experiment(**kwargs)
    assert first.initial_seed != first.perturbation_seed
    assert first.initial_seed == second.initial_seed
    assert first.perturbation_seed == second.perturbation_seed
    assert (first.perturbed == second.perturbed).all()
