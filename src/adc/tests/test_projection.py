import os

import numpy as np
import tifffile as tf
from pytest import fixture

from adc._projection_stack import ProjectAlong

TIF_PATH = "test.tif"


@fixture
def tif_file():
    if os.path.exists(TIF_PATH):
        os.remove(TIF_PATH)
    stack = np.zeros((3, 4, 2**13, 2**10), dtype="uint16")  # z, c, y, x
    tf.imwrite(TIF_PATH, stack, imagej=True)
    return TIF_PATH


def test_projection(make_napari_viewer, tif_file):
    v = make_napari_viewer()
    layers = v.open(TIF_PATH, plugin="anchor-droplet-chip")
    assert len(layers) == 4
    p = ProjectAlong(v)
