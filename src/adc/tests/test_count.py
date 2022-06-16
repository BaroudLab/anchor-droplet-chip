import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from adc.count import get_cell_numbers


@pytest.fixture
def create_test_data(plot=False):
    data = np.zeros((32, 32), dtype="uint16")
    data[8:11, 10] = 2.5
    data[15, 18:21] = 2.3
    data = data + 1
    data = data * 400
    data = gaussian_filter(data, 1.0)
    data = np.random.poisson(data)
    mask = np.zeros_like(data)
    mask[2:-2, 2:-2] = 1
    bf = mask
    out = {"bf": bf, "multiwell_image": data, "labels": mask}
    if plot:
        fig, ax = plt.subplots(ncols=3, figsize=(10, 5))
        for a, (k, v) in zip(ax, out.items()):
            a.imshow(v)
            # a.colorbar()
            a.set_title(k)

        plt.show()
    return out
    # return bf, data, mask


def test_count(create_test_data):
    table = get_cell_numbers(**create_test_data)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 1
    assert table.n_cells[0] == 2


if __name__ == "__main__":
    test_count()
