import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.DEBUG)

from adc import count


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
    table = count.get_cell_numbers(**create_test_data)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 1
    assert table.n_cells[0] == 2


def test_crop(create_test_data):
    crop = count.crop2d(create_test_data["bf"], center=(10, 10), size=5)
    assert crop.shape == (5, 5)


def test_count2d():
    pos = np.array([[10, 10], [10, 20]])
    locs, cnts = count.count2d(
        np.zeros((32, 32)),
        positions=pos,
        size=5,
        localizer=lambda fluo_data, center, size, **kwargs: [[0, 0]] * 5,
    )
    assert len(locs) == 10
    assert len(cnts) == 2


def test_recursive():
    n_per_well = 5
    pos = np.array([[10, 10], [10, 20]])

    def localizer(fluo_data, center, size, **kwargs):
        return [[0, 0]] * n_per_well

    loc_result, count_result, droplets_out, df = count.count_recursive(
        data=np.zeros((5, 4, 3, 32, 32)),
        positions=pos,
        size=5,
        localizer=localizer,
    )
    assert len(loc_result) == 5 * 4 * 3 * 2 * n_per_well
    assert len(count_result) == 5 * 4 * 3 * 2
    assert len(droplets_out) == 5 * 4 * 3 * 2
    assert len(df) == 5 * 4 * 3 * 2
    assert isinstance(df, pd.DataFrame)


if __name__ == "__main__":
    test_count()
