import dask.array as da
import numpy as np
from pytest import fixture

from adc._split_stack import SplitAlong


@fixture
def test_stack():
    sizes = {"T": 30, "P": 10, "C": 3, "Z": 15, "Y": 256, "X": 256}
    tpcz_stack = np.zeros(tuple(sizes.values()), dtype="uint16")
    return {
        "data": tpcz_stack,
        "metadata": {
            "dask_data": da.from_array(tpcz_stack),
            "sizes": sizes,
            "pixel_size_um": 0.3,
        },
        "channel_axis": None,
    }


def test_split_to_layers(make_napari_viewer, test_stack):
    viewer = make_napari_viewer()
    viewer.add_image(**test_stack)
    assert len(viewer.layers) == 1
    splitter = SplitAlong(viewer)
    splitter.split_selector.value = "layers"
    splitter.axis_selector.value = "C:3"
    splitter.split_data()
    assert len(viewer.layers) == 4
