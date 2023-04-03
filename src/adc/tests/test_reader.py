import logging
import os
import shutil

import dask.array as da
import numpy as np
from pytest import fixture
from zarr_tools import convert

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

ZPATH = "test.zarr"


@fixture
def zarr_dataset():
    if os.path.exists(ZPATH):
        shutil.rmtree(ZPATH)
    arr = np.random.random((5, 2**10, 2**10))
    dask_array = da.from_array(arr)
    path = convert.to_zarr(dask_array, ZPATH, channel_axis=0)
    logger.debug(f"{path} generated")
    return path


def test_read_zarr(zarr_dataset, make_napari_viewer):
    try:
        viewer = make_napari_viewer()
        zarr_path = zarr_dataset
        logger.debug(f"reading {zarr_path}")

        viewer.open(path=zarr_path, plugin="anchor-droplet-chip")
        assert len(viewer.layers) == 5
    finally:
        shutil.rmtree(ZPATH)
