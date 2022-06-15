import numpy as np
from scipy.ndimage import gaussian_filter
import pytest

@pytest.fixture
def create_test_data():
    data = np.zeros((16,16), dtype='uint16')
    data[5:6] = 1
    data[8,10] = 1
    data = data + 1
    data = data * 400
    
