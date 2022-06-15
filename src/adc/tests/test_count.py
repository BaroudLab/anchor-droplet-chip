import numpy as np
from scipy.ndimage import gaussian_filter
import pytest
from adc.count import get_cell_numbers
import pandas as pd
import matplotlib.pyplot as plt


@pytest.fixture
def create_test_data():
    data = np.zeros((16,16), dtype='uint16')
    data[5:6] = 1
    data[8,10] = 1
    data = data + 1
    data = data * 400
    data = gaussian_filter(data, 1.5)
    data = np.random.poisson(data)
    mask = np.zeros_like(data)
    mask[2:-2,2:-2] = 1

    bf = mask

    return {"bf":bf, "multiwell_image": data, "labels": mask}
    # return bf, data, mask

def test_count(create_test_data):
    table = get_cell_numbers(**create_test_data)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 1
    assert table.n_cells[0] == 2

if __name__ == "__main__":
    test_count()
