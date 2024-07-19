import logging
import os
import shutil

import dask.array as da
import numpy as np
import tifffile as tf
from pytest import fixture
from zarr_tools import convert

logger = logging.getLogger(__name__)
# logger.setLevel("DEBUG")

ZPATH = "test.zarr"
BTPATH = "btest.tif"
STPATH = "stest.tif"
PLUGIN="anchor_droplet_chip"

@fixture
def big_tif_dataset():
    if os.path.exists(BTPATH):
        os.remove(BTPATH)
    arr = np.zeros((5, 2**13, 2**13), dtype="uint16")
    tf.imwrite(BTPATH, arr, imagej=True)
    logger.debug(f"{BTPATH} generated")
    return BTPATH


@fixture
def sm_tif_dataset():
    if os.path.exists(STPATH):
        os.remove(STPATH)
    arr = np.zeros((5, 2**5, 2**5), dtype="uint16")
    tf.imwrite(STPATH, arr, imagej=True)
    logger.debug(f"{STPATH} generated")
    return STPATH


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

        viewer.open(path=zarr_path, plugin=PLUGIN)
        assert len(viewer.layers) == 5
    finally:
        shutil.rmtree(ZPATH)


def test_read_small_tif(sm_tif_dataset, make_napari_viewer):
    viewer = make_napari_viewer()
    path = sm_tif_dataset
    logger.debug(f"reading {path}")

    viewer.open(path=path, plugin=PLUGIN)
    assert len(viewer.layers) == 5
    assert not viewer.layers[0].multiscale
    assert isinstance(viewer.layers[0].data, da.Array)


def test_read_big_tif(big_tif_dataset, make_napari_viewer):
    viewer = make_napari_viewer()
    path = big_tif_dataset
    logger.debug(f"reading {path}")

    viewer.open(path=path, plugin=PLUGIN)
    assert len(viewer.layers) == 5
    assert viewer.layers[0].multiscale
    assert isinstance(viewer.layers[0].data[0], da.Array)
