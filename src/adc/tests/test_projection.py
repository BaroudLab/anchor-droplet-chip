import os

import numpy as np
import tifffile as tf
from pytest import fixture

from adc._projection_stack import ProjectAlong

TIF_PATH = "test.tif"
TIF_SHAPE = (3, 4, 2**13, 2**10)


@fixture
def tif_file():
    if os.path.exists(TIF_PATH):
        os.remove(TIF_PATH)
    stack = np.zeros(TIF_SHAPE, dtype="uint16")  # z, c, y, x
    tf.imwrite(TIF_PATH, stack, imagej=True)
    return TIF_PATH


def test_projection(make_napari_viewer, tif_file):
    v = make_napari_viewer()
    layers = v.open(TIF_PATH, plugin="anchor_droplet_chip")
    assert len(layers) == 4
    p = ProjectAlong(v)
    p.axis_selector.value = "Z:3"
    new_layers = p.make_projection()
    assert tuple(new_layers[0].metadata["sizes"].keys()) == tuple("CYX")
