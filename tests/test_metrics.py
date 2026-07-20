import numpy as np

from ruliology_forge import final_restoration, hamming_divergence, restoration_coefficient, xor_difference


def test_xor_difference():
    a = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    b = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.uint8)
    diff = xor_difference(a, b)
    assert diff.tolist() == [[0, 1, 0], [0, 1, 1]]


def test_hamming_divergence():
    a = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    b = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.uint8)
    div = hamming_divergence(a, b)
    assert np.allclose(div, [1 / 3, 2 / 3])


def test_restoration_coefficient():
    a = np.zeros((2, 4), dtype=np.uint8)
    b = np.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint8)
    assert restoration_coefficient(a, b, start=1) == 0.5
    assert final_restoration(a, b) == 0.5


def test_recovery_metrics():
    from ruliology_forge import divergence_auc, recovery_time, summarize_resilience

    control = np.zeros((6, 4), dtype=np.uint8)
    perturbed = control.copy()
    perturbed[1] = 1
    perturbed[2, :2] = 1

    assert recovery_time(control, perturbed, start=1, threshold=0.0, persistence=2) == 2
    assert divergence_auc(control, perturbed, start=1) > 0
    summary = summarize_resilience(
        control,
        perturbed,
        start=1,
        recovery_threshold=0.0,
        recovery_persistence=2,
    )
    assert summary.recovered is True
    assert summary.recovery_time == 2
    assert summary.peak_divergence == 1.0


def test_scar_metrics_and_shift_tolerance():
    from ruliology_forge import scar_metrics, shift_tolerant_divergence

    control = np.zeros((4, 7), dtype=np.uint8)
    control[:, 3] = 1
    shifted = np.zeros_like(control)
    shifted[:, 4] = 1

    exact = hamming_divergence(control, shifted)
    tolerant, shifts = shift_tolerant_divergence(control, shifted, max_shift=2)
    assert np.all(exact > 0)
    assert np.allclose(tolerant, 0)
    assert shifts[-1] == -1

    scars = scar_metrics(control, shifted, start=0)
    assert scars["final_scar_size"] == 2
    assert scars["scar_duration"] == 4
    assert scars["scar_spread"] == 2


def test_random_mix_noise_probability_zero_is_noop():
    from ruliology_forge import perturb_state

    state = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
    result = perturb_state(
        state,
        center=2,
        radius=2,
        kind="random_mix",
        seed=1,
        noise_probability=0.0,
    )
    assert np.array_equal(result, state)
