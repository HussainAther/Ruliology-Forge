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
