import numpy as np
import pytest

from ruliology_forge import ECARule, initial_state, simulate_eca


def test_rule_number_bounds():
    ECARule(0)
    ECARule(255)
    with pytest.raises(ValueError):
        ECARule(-1)
    with pytest.raises(ValueError):
        ECARule(256)


def test_initial_single_cell():
    state = initial_state(5)
    assert state.tolist() == [0, 0, 1, 0, 0]


def test_rule_0_goes_to_zero():
    trajectory = simulate_eca(0, width=7, steps=4)
    assert trajectory[1:].sum() == 0


def test_rule_90_from_single_cell():
    trajectory = simulate_eca(90, width=7, steps=3, boundary="fixed")
    assert np.array_equal(trajectory[0], np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.uint8))
    assert np.array_equal(trajectory[1], np.array([0, 0, 1, 0, 1, 0, 0], dtype=np.uint8))
