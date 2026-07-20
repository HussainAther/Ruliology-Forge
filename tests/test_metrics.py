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
