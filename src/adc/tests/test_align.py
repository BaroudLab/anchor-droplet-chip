import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, rotate, shift

from adc import align

np.random.seed = 45654748


@pytest.fixture
def create_random_data():
    a = ((np.random.rand(16, 16) + 1) * 400).astype("uint16")
    return a


@pytest.fixture
def create_test_template_image():
    a = np.zeros((160, 360), dtype="uint16")
    a[40:-40, 50:-50] = 1
    a[:, 160:200] = 0
    b = np.pad(a, 100, mode="constant", constant_values=0)
    template, image = (gaussian_filter(x * 10000 + 30000, 2) for x in (a, b))
    image = rotate(image, (angle := 6.0), cval=30000)
    image = shift(image, (tr := (50.0, 30.0)), cval=30000)
    template, image = map(np.random.poisson, (template, image))
    tvec = {"tvec": np.array(tr, dtype="f"), "angle": angle}
    return image, template, tvec


def test_to_8bits(create_random_data):
    b = align.to_8bits(create_random_data)
    assert b.shape == create_random_data.shape
    assert all([250 < b.max() <= 255, 0 <= b.min() < 5])
    assert b.dtype == np.uint8
