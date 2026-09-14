import numpy as np

from main import calculate_rms_difference


def test_calculate_rms_difference_basic():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 4.0, 2.0])
    expected = np.sqrt(((0.0 ** 2) + (-2.0 ** 2) + (1.0 ** 2)) / 3)
    assert np.isclose(calculate_rms_difference(actual, predicted), expected)
