import numpy as np
import pytest

from adc.align import to_8bits


@pytest.fixture
def create_random_data():
    a = ((np.random.rand(16, 16) + 1) * 400).astype("uint16")
    return a


def test_to_8bits(create_random_data):
    b = to_8bits(create_random_data)
    assert b.shape == create_random_data.shape
    assert all([250 < b.max() <= 255, 0 <= b.min() < 5])
    assert b.dtype == np.uint8
